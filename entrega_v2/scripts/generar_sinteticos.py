"""Generador de dataset sintetico de fichas de ajedrez.

Toma las dos plantillas del proyecto:

* ``plantilla/tablero.png``  -> tablero 8x8 impreso con 4 ArUco y los textos
  "LADO DE LAS NEGRAS" / "LADO DE LAS BLANCAS".
* ``plantilla/fichas.png``   -> hoja de stickers: por color, una fila de peones
  y una fila de piezas mayores.

y produce fotos sinteticas con etiquetas YOLO listas para entrenar:

    dataset/sintetico/images/sintetico_00000.jpg
    dataset/sintetico/labels/sintetico_00000.txt
    dataset/sintetico/manifesto.csv
    assets/previews/sintetico_00000_preview.jpg   (3 previews)

Los indices CONTINUAN desde la ultima imagen que ya exista en ``images/``
(``--inicio`` lo fuerza). El manifiesto se actualiza fila por fila: las filas
anteriores se conservan y solo se agregan o reemplazan las de esta corrida.
Cada fila guarda ``homografia_recorte``: la matriz 3x3 que lleva la imagen
original a la vista rectificada 800x800 del area jugable. Es la que consume
``recortar_sinteticos.py`` para no volver a detectar los ArUco.

Cada imagen renderiza el tablero a la resolucion nativa del perfil (640x480
para la ESP32, 1024x768 para un celular) y dibuja cada ficha con rotacion libre
de 0 a 360 grados, como pidio el usuario. La clase de cada ficha sale del orden
exacto de ``classes.txt``; ese archivo es la unica fuente de verdad.

Uso
---
    python generar_sinteticos.py --n 24 --semilla 42 --perfil tomo
    python generar_sinteticos.py --n 120 --perfil tomo --posiciones mezcla
    python generar_sinteticos.py --n 30 --posiciones jugada --semilla 7

No instala nada: usa cv2, numpy y python-chess del venv del proyecto v1.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from comun import cargar_clases

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLA_FICHAS = RAIZ / "plantilla" / "fichas.png"
PLANTILLA_TABLERO = RAIZ / "plantilla" / "tablero.png"
SALIDA_DEFECTO = RAIZ / "dataset" / "sintetico"
PREVIEWS = RAIZ / "assets" / "previews"
CLASES_TXT = RAIZ / "classes.txt"

# Lado de la vista rectificada del area jugable. Es el mismo 800 de la rejilla
# de comun.py (8 casillas de 100 px) y el que espera el clasificador por casilla.
LADO_RECORTE = 800

# Reparto objetivo del lote masivo (1520 imagenes). El lote cubre la base del
# clasificador por casilla: finales aportan casillas vacias, jugada aporta
# posiciones legales y dificiles aporta los bordes con dama/rey aislados.
MEZCLA_OBJETIVO = {
    "jugada": 610,
    "aleatoria": 380,
    "finales": 380,
    "dificil": 120,
    "vacio": 30,
}

CAMPOS_MANIFIESTO = [
    "imagen",
    "n_fichas",
    "perfil",
    "posiciones",
    "categoria",
    "semilla",
    "homografia_recorte",
    "ancho",
    "alto",
]

# Orden de la fila de piezas mayores en la hoja de stickers: torre, caballo,
# alfil, dama, rey, alfil, caballo, torre. Se valida contra la hoja al extraer.
BACK_RANK = ("torre", "caballo", "alfil", "dama", "rey", "alfil", "caballo", "torre")

FORMA_ES = {
    "P": "peon",
    "R": "torre",
    "N": "caballo",
    "B": "alfil",
    "Q": "dama",
    "K": "rey",
}

# Perfiles de camara. "tomo" elige uno al azar por imagen.
PERFILES = {
    "esp": dict(
        ancho=640,
        alto=480,
        relleno=(0.90, 1.00),      # fraccion del encuadre que ocupa la hoja
        angulo=2.5,                 # grados maximos de giro
        perspectiva=0.012,
        p_blur=0.92,
        blur=(0.35, 0.85),
        contraste=(1.02, 1.14),
        ganancia=(0.93, 1.05),
        offset=(-14.0, 6.0),
        ruido=(2.0, 6.0),
        jpeg=(65, 82),
    ),
    "celular": dict(
        ancho=1024,
        alto=768,
        relleno=(0.64, 0.84),
        angulo=6.0,
        perspectiva=0.05,
        p_blur=0.35,
        blur=(0.15, 0.55),
        contraste=(0.98, 1.08),
        ganancia=(0.90, 1.10),
        offset=(-18.0, 14.0),
        ruido=(1.5, 4.5),
        jpeg=(80, 93),
    ),
}

PALETA_FONDO = np.array(
    [
        [58, 60, 62],    # escritorio oscuro
        [40, 42, 46],
        [120, 112, 98],  # madera clara
        [92, 74, 58],
        [150, 155, 145], # tela verde grisacea
        [46, 52, 48],
        [172, 168, 158], # mesa clara
        [70, 60, 50],
    ],
    dtype=np.float32,
)

PALETA_CAJAS = [
    (56, 56, 220), (30, 160, 240), (60, 190, 60), (220, 160, 40),
    (200, 60, 200), (40, 200, 200), (120, 120, 240), (240, 240, 60),
    (60, 120, 240), (160, 60, 240), (60, 240, 160), (240, 100, 60),
]


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------

@dataclass
class Sprites:
    por_clase: dict[int, np.ndarray]
    nombres: dict[int, str]
    discos: list[tuple[int, int, int]]  # (x, y, radio) en la hoja


def _agrupar_por(valores, tolerancia: float) -> list[float]:
    """Separa valores 1D en grupos por saltos mayores que la tolerancia."""
    ordenados = np.sort(np.asarray(valores, dtype="float64"))
    if ordenados.size == 0:
        return []
    grupos: list[list[float]] = [[float(ordenados[0])]]
    for valor in ordenados[1:]:
        if float(valor) - grupos[-1][-1] <= tolerancia:
            grupos[-1].append(float(valor))
        else:
            grupos.append([float(valor)])
    return [float(np.mean(grupo)) for grupo in grupos]


def _detectar_discos(gris: np.ndarray) -> list[tuple[int, int, int]]:
    alto, ancho = gris.shape[:2]
    radio_min = max(6, int(round(0.025 * ancho)))
    radio_max = max(radio_min + 4, int(round(0.075 * ancho)))
    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(8, int(round(0.05 * ancho))),
        param1=70,
        param2=32,
        minRadius=radio_min,
        maxRadius=radio_max,
    )
    if circulos is None:
        raise ValueError(
            "no encontre los discos de fichas.png; revisa que la hoja tenga "
            "las fichas separadas y con el circulo completo"
        )
    return [
        (int(round(x)), int(round(y)), int(round(r)))
        for x, y, r in np.round(circulos[0]).astype(int)
    ]


def _agrupar_en_filas(
    discos: list[tuple[int, int, int]]
) -> list[list[tuple[int, int, int]]]:
    """Agrupa discos en 4 filas de 8 por clustering 1D de y despues de x."""
    radio = float(np.median([r for _, _, r in discos]))
    tolerancia = 0.7 * 2.0 * radio
    centros_y = _agrupar_por([y for _, y, _ in discos], tolerancia)

    filas: list[list[tuple[int, int, int]]] = [[] for _ in centros_y]
    for disco in discos:
        cercano = int(np.argmin([abs(disco[1] - cy) for cy in centros_y]))
        filas[cercano].append(disco)

    # Una fila real tiene al menos 6 discos; el antialias del texto de la
    # cabecera puede colar un circulo suelto y esas filas se descartan.
    filas = [fila for fila in filas if len(fila) >= 6]
    filas.sort(key=lambda fila: float(np.mean([y for _, y, _ in fila])))
    if len(filas) != 4:
        raise ValueError(
            f"esperaba 4 filas de discos en fichas.png y encontre {len(filas)}"
        )

    resultado = []
    for fila in filas:
        centros_x = _agrupar_por([x for x, _, _ in fila], tolerancia)
        columnas: list[list[tuple[int, int, int]]] = [[] for _ in centros_x]
        for disco in fila:
            cercano = int(np.argmin([abs(disco[0] - cx) for cx in centros_x]))
            columnas[cercano].append(disco)
        columnas = [col for col in columnas if col]
        if len(columnas) != 8:
            raise ValueError(
                "esperaba 8 discos por fila en fichas.png y encontre "
                f"{len(columnas)} en una fila"
            )
        # Cada columna debe tener un solo disco (el mas votado por cercania).
        elegidos = []
        for columna in columnas:
            columna.sort(key=lambda d: d[2], reverse=True)
            elegidos.append(columna[0])
        elegidos.sort(key=lambda d: d[0])
        resultado.append(elegidos)
    return resultado


def _brillo_disco(gris: np.ndarray, x: int, y: int, radio: int) -> float:
    """Brillo del RELLENO del disco, no del simbolo.

    Muestrear el centro mide el simbolo (blanco sobre disco negro), no el disco,
    y eso invierte blancas con negras. El anillo entre 0.65 y 0.88 del radio es
    siempre relleno: ~0 en los discos negros y ~255 en los blancos.
    """
    alto, ancho = gris.shape[:2]
    valores = []
    for fraccion in np.linspace(0.65, 0.88, 5):
        for angulo in np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False):
            px = int(round(x + fraccion * radio * math.cos(angulo)))
            py = int(round(y + fraccion * radio * math.sin(angulo)))
            if 0 <= px < ancho and 0 <= py < alto:
                valores.append(int(gris[py, px]))
    return float(np.median(valores)) if valores else 255.0


def _recortar_con_pad(imagen: np.ndarray, x0: int, y0: int, lado: int) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    canales = imagen.shape[2] if imagen.ndim == 3 else 1
    forma = (lado, lado, canales) if imagen.ndim == 3 else (lado, lado)
    salida = np.zeros(forma, dtype=imagen.dtype)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(ancho, x0 + lado)
    sy1 = min(alto, y0 + lado)
    if sx0 < sx1 and sy0 < sy1:
        salida[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = imagen[sy0:sy1, sx0:sx1]
    return salida


def _radio_real(gris: np.ndarray, x: int, y: int, radio_aprox: int,
                disco_negro: bool) -> float:
    """Refina el radio del disco por su perfil radial.

    Hough da un radio entero con error de 1-2 px. Con el radio equivocado el
    recorte de un disco negro arrastra un anillo blanco, y el de un disco
    blanco corta el contorno. Aqui se mide: en un disco negro el brillo salta de
    oscuro a claro en el borde; en un disco blanco hay un minimo en el contorno.
    """
    alto, ancho = gris.shape[:2]
    if disco_negro:
        radios = np.arange(max(2.0, radio_aprox * 0.60), radio_aprox * 1.50, 0.25)
    else:
        radios = np.arange(max(2.0, radio_aprox * 0.88), radio_aprox * 1.45, 0.25)
    if radios.size == 0:
        return float(radio_aprox)

    angulos = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    perfil = []
    for radio in radios:
        muestras = []
        for angulo in angulos:
            px = int(round(x + radio * math.cos(angulo)))
            py = int(round(y + radio * math.sin(angulo)))
            if 0 <= px < ancho and 0 <= py < alto:
                muestras.append(int(gris[py, px]))
        perfil.append(float(np.median(muestras)) if muestras else 255.0)
    perfil = np.asarray(perfil)

    if disco_negro:
        for i in range(len(radios) - 1):
            if perfil[i] < 120.0 <= perfil[i + 1]:
                # Medio pixel adentro para no arrastrar el antialias blanco.
                return float((radios[i] + radios[i + 1]) / 2.0 - 0.6)
        return float(radio_aprox)
    indice = int(np.argmin(perfil))
    return float(radios[indice])


def _crop_gris(gris: np.ndarray, x: int, y: int, radio: int, lado: int = 48) -> np.ndarray:
    recorte = _recortar_con_pad(gris, int(round(x - radio)), int(round(y - radio)),
                                2 * int(round(radio)) + 1)
    return cv2.resize(recorte, (lado, lado), interpolation=cv2.INTER_AREA)


def _sprite(bgr: np.ndarray, gris: np.ndarray, x: int, y: int, radio: float) -> np.ndarray:
    """Recorta el disco con mascara circular usando un radio ya medido."""
    lado = max(8, int(math.ceil(2.0 * radio)) + 1)
    x0 = int(math.floor(x - radio))
    y0 = int(math.floor(y - radio))
    recorte = _recortar_con_pad(bgr, x0, y0, lado)
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - (x - x0), yy - (y - y0))
    alpha = np.clip(radio - distancia + 0.5, 0.0, 1.0) * 255.0
    return np.dstack([recorte, alpha.astype(np.uint8)])


def extraer_sprites(ruta, clases: list[str]) -> Sprites:
    """Recorta las 32 fichas de la hoja y las asocia a las 12 clases."""
    bgra = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise FileNotFoundError(f"no pude leer la hoja de fichas: {ruta}")
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR) if bgra.shape[2] == 4 else bgra
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    discos = _detectar_discos(gris)
    filas = _agrupar_en_filas(discos)

    metadatos = []
    for fila in filas:
        brillo = float(np.median([_brillo_disco(gris, x, y, r) for x, y, r in fila]))
        metadatos.append((fila, brillo))

    # Disco negro = ficha blanca (b). Disco blanco = ficha negra (n).
    negros = [m for m in metadatos if m[1] < 110.0]
    blancos = [m for m in metadatos if m[1] >= 110.0]
    if len(negros) != 2 or len(blancos) != 2:
        raise ValueError(
            "la hoja de fichas no tiene 2 filas de discos negros y 2 de discos "
            f"blancos (negros={len(negros)}, blancos={len(blancos)})"
        )

    sprites: dict[int, np.ndarray] = {}
    nombres: dict[int, str] = {}

    # Todos los discos de la hoja miden lo mismo. El radio se mide una sola vez
    # sobre los discos negros, donde el borde es un cruce limpio de oscuro a
    # claro; medirlo en los discos blancos falla porque su simbolo negro puede
    # tapar el perfil y confundirse con el contorno.
    radios_negros = [
        _radio_real(gris, x, y, r, disco_negro=True)
        for fila, _ in negros for x, y, r in fila
    ]
    radio_disco = float(np.median(radios_negros))

    # 'b' = fichas blancas (disco negro), 'n' = fichas negras (disco blanco).
    for color, grupo in (("b", negros), ("n", blancos)):
        recortes = [
            [_crop_gris(gris, x, y, r) for x, y, r in fila]
            for fila, _ in grupo
        ]
        # La fila de peones tiene 8 simbolos identicos: su variacion interna es
        # casi cero. La fila de piezas mayores varia en cada columna.
        variaciones = []
        for fila_recortes in recortes:
            base = fila_recortes[0].astype("float32")
            variaciones.append(
                float(np.mean([np.abs(recorte.astype("float32") - base).mean()
                               for recorte in fila_recortes]))
            )
        orden = np.argsort(variaciones)
        if variaciones[int(orden[0])] >= variaciones[int(orden[1])] * 0.75:
            raise ValueError(
                f"no pude distinguir la fila de peones de la de piezas mayores "
                f"en las fichas {color} (variaciones {variaciones})"
            )
        fila_peones = grupo[int(orden[0])][0]
        fila_mayores = grupo[int(orden[1])][0]

        for x, y, r in fila_peones:
            nombre = f"peon_{color}"
            clase = clases.index(nombre)
            sprites[clase] = _sprite(bgr, gris, x, y, radio_disco)
            nombres[clase] = nombre
        for columna, (x, y, r) in enumerate(fila_mayores):
            nombre = f"{BACK_RANK[columna]}_{color}"
            clase = clases.index(nombre)
            sprites[clase] = _sprite(bgr, gris, x, y, radio_disco)
            nombres[clase] = nombre

    faltan = [clases[i] for i in range(len(clases)) if i not in sprites]
    if faltan or len(sprites) != len(clases):
        raise ValueError(f"No complete las 12 clases; faltan: {faltan}")
    return Sprites(por_clase=sprites, nombres=nombres, discos=discos)


# ---------------------------------------------------------------------------
# Posiciones
# ---------------------------------------------------------------------------

def _clase_de_pieza(pieza, clases: list[str]) -> int:
    forma = FORMA_ES[pieza.symbol().upper()]
    color = "b" if pieza.color else "n"
    return clases.index(f"{forma}_{color}")


def _piezas_jugada(rng: np.random.Generator, clases: list[str],
                   minimo: int = 4, maximo: int = 40) -> list[tuple[int, int, int]]:
    """Posicion legal: partida desde el inicio y N jugadas aleatorias."""
    import chess

    tablero = chess.Board()
    jugadas = int(rng.integers(minimo, maximo + 1))
    for _ in range(jugadas):
        legales = list(tablero.legal_moves)
        if not legales:
            break
        tablero.push(legales[int(rng.integers(len(legales)))])

    piezas = []
    for casilla, pieza in tablero.piece_map().items():
        fila = 7 - chess.square_rank(casilla)     # fila 0 = rank 8 (arriba)
        columna = chess.square_file(casilla)      # columna 0 = file a
        piezas.append((fila, columna, _clase_de_pieza(pieza, clases)))
    return piezas


def _piezas_aleatorias(rng: np.random.Generator, clases: list[str],
                       dificil: bool = False) -> list[tuple[int, int, int]]:
    """Fichas sueltas en casillas al azar; puede haber casillas vacias."""
    if dificil:
        cantidad = int(rng.integers(2, 6))
        if rng.random() < 0.5:
            # Bordes y esquinas: el caso que mas confunde al detector.
            casillas_borde = [f * 8 + c for f in range(8) for c in range(8)
                              if f in (0, 7) or c in (0, 7)]
            elegidas = rng.choice(len(casillas_borde), size=cantidad, replace=False)
            indices = [casillas_borde[int(i)] for i in elegidas]
        else:
            indices = [int(i) for i in rng.choice(64, size=cantidad, replace=False)]
        # Reinas y reyes aislados: el caso que el README llama 04_dificiles.
        candidatas = [clases.index(n) for n in ("dama_b", "rey_b", "dama_n", "rey_n")]
    else:
        # El tablero tiene que repartirse entre inicial, medio juego y finales,
        # no llenarse siempre. Una binomial pura se pega al maximo de 32 y deja
        # el dataset sin casillas vacias, que es justo lo que ensena "vacia".
        dado = float(rng.random())
        if dado < 0.45:
            cantidad = int(rng.integers(2, 9))       # finales / dificiles
        elif dado < 0.72:
            cantidad = int(rng.integers(9, 21))      # medio juego
        else:
            cantidad = int(rng.integers(21, 33))     # posicion inicial
        indices = [int(i) for i in rng.choice(64, size=cantidad, replace=False)]
        candidatas = list(range(len(clases)))

    piezas = []
    for indice in indices:
        fila, columna = divmod(int(indice), 8)
        clase = int(candidatas[int(rng.integers(len(candidatas)))])
        piezas.append((fila, columna, clase))
    return piezas


def _categoria(piezas: list[tuple[int, int, int]], clases: list[str]) -> str:
    cantidad = len(piezas)
    if cantidad == 0:
        return "05_tablero_vacio"
    formas = {clases[clase].rsplit("_", 1)[0] for _, _, clase in piezas}
    if cantidad <= 6 and formas <= {"dama", "rey"}:
        return "04_dificiles"
    if cantidad <= 8:
        return "03_finales"
    if cantidad <= 20:
        return "02_medio_juego"
    return "01_inicial"


def _piezas_finales(rng: np.random.Generator, clases: list[str],
                    minimo: int = 4, maximo: int = 8) -> list[tuple[int, int, int]]:
    """Final de partida: 4-8 fichas sueltas y muchas casillas vacias.

    Se garantiza que al menos una ficha NO sea dama ni rey. Sin eso, la
    categoria derivada podria caer en 04_dificiles (formas <= {dama, rey}) y el
    reparto del lote dejaria de cuadrar con MEZCLA_OBJETIVO.
    """
    cantidad = int(rng.integers(minimo, maximo + 1))
    indices = [int(i) for i in rng.choice(64, size=cantidad, replace=False)]
    formas_no_mayores = [
        clases.index(nombre)
        for nombre in ("peon_b", "peon_n", "torre_b", "torre_n",
                       "caballo_b", "caballo_n", "alfil_b", "alfil_n")
    ]
    elegidas = [int(rng.integers(len(clases))) for _ in range(cantidad)]
    elegidas[0] = formas_no_mayores[int(rng.integers(len(formas_no_mayores)))]
    piezas = []
    for indice, clase in zip(indices, elegidas):
        fila, columna = divmod(int(indice), 8)
        piezas.append((fila, columna, clase))
    return piezas


def _modo_mezcla(rng: np.random.Generator) -> str:
    """Un modo de mezcla sorteado con los pesos de MEZCLA_OBJETIVO."""
    modos = list(MEZCLA_OBJETIVO)
    pesos = np.array([MEZCLA_OBJETIVO[m] for m in modos], dtype="float64")
    return modos[int(rng.choice(len(modos), p=pesos / pesos.sum()))]


def _cuotas_mezcla(cantidad: int, rng: np.random.Generator) -> list[str]:
    """Lista de modos con el reparto exacto (o casi) de MEZCLA_OBJETIVO.

    Para 1520 salen exactamente 610/380/380/120/30. Para otros tamanos se
    redondea por el metodo del mayor resto, se mezcla con la semilla y se
    devuelve una lista de modo por imagen. Se hace antes del bucle para que el
    reparto no dependa del orden de generacion.
    """
    total = sum(MEZCLA_OBJETIVO.values())
    crudos = {modo: cantidad * peso / total for modo, peso in MEZCLA_OBJETIVO.items()}
    cuotas = {modo: int(valor // 1) for modo, valor in crudos.items()}
    restantes = cantidad - sum(cuotas.values())
    orden = sorted(crudos, key=lambda m: crudos[m] - cuotas[m], reverse=True)
    for modo in orden[:restantes]:
        cuotas[modo] += 1
    modos = np.array([modo for modo, n in cuotas.items() for _ in range(n)], dtype=object)
    rng.shuffle(modos)
    return [str(modo) for modo in modos]


def _generar_posicion(rng: np.random.Generator, clases: list[str], modo: str):
    """Devuelve (piezas, modo_real) segun el modo pedido."""
    if modo == "mezcla":
        return _generar_posicion(rng, clases, _modo_mezcla(rng))
    if modo == "vacio":
        return [], "vacio"
    if modo == "dificil":
        return _piezas_aleatorias(rng, clases, dificil=True), "dificil"
    if modo == "finales":
        return _piezas_finales(rng, clases), "finales"
    if modo == "jugada":
        return _piezas_jugada(rng, clases), "jugada"
    return _piezas_aleatorias(rng, clases, dificil=False), "aleatoria"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _compositar(destino: np.ndarray, fuente: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-composite de fuente RGBA sobre destino BGRA in-place."""
    alto_d, ancho_d = destino.shape[:2]
    alto_f, ancho_f = fuente.shape[:2]
    ax0 = max(0, x0)
    ay0 = max(0, y0)
    ax1 = min(ancho_d, x0 + ancho_f)
    ay1 = min(alto_d, y0 + alto_f)
    if ax0 >= ax1 or ay0 >= ay1:
        return
    sx0 = ax0 - x0
    sy0 = ay0 - y0
    sx1 = sx0 + (ax1 - ax0)
    sy1 = sy0 + (ay1 - ay0)
    origen = fuente[sy0:sy1, sx0:sx1].astype("float32")
    alpha = origen[..., 3:4] / 255.0
    base = destino[ay0:ay1, ax0:ax1, :3].astype("float32")
    destino[ay0:ay1, ax0:ax1, :3] = np.clip(
        origen[..., :3] * alpha + base * (1.0 - alpha), 0, 255
    ).astype(np.uint8)


def _pegar_ficha(plano: np.ndarray, sprite: np.ndarray, cx: float, cy: float,
                 diametro: float, angulo: float) -> None:
    lado = max(6, int(round(diametro)))
    escalado = cv2.resize(sprite, (lado, lado), interpolation=cv2.INTER_LANCZOS4)
    diagonal = int(math.ceil(lado * math.sqrt(2.0))) + 2
    lienzo = np.zeros((diagonal, diagonal, 4), np.uint8)
    margen = (diagonal - lado) // 2
    lienzo[margen:margen + lado, margen:margen + lado] = escalado
    matriz = cv2.getRotationMatrix2D((diagonal / 2.0, diagonal / 2.0), angulo, 1.0)
    rotado = cv2.warpAffine(
        lienzo, matriz, (diagonal, diagonal),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    _compositar(plano, rotado, int(round(cx - diagonal / 2.0)),
                int(round(cy - diagonal / 2.0)))


def _fondo(rng: np.random.Generator, ancho: int, alto: int) -> np.ndarray:
    base = PALETA_FONDO[int(rng.integers(len(PALETA_FONDO)))].copy()
    pequeno = rng.normal(0.0, 1.0, (alto // 24 + 2, ancho // 24 + 2)).astype("float32")
    textura = cv2.resize(pequeno, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    textura = cv2.GaussianBlur(textura, (0, 0), 6.0)
    textura = textura / (np.abs(textura).max() + 1e-6) * float(rng.uniform(6.0, 18.0))
    imagen = base[None, None, :] + textura[:, :, None]

    yy, xx = np.mgrid[0:alto, 0:ancho].astype("float32")
    cx = ancho * float(rng.uniform(0.30, 0.70))
    cy = alto * float(rng.uniform(0.30, 0.70))
    distancia = np.hypot((xx - cx) / max(ancho, 1), (yy - cy) / max(alto, 1))
    luz = 1.0 - float(rng.uniform(0.12, 0.38)) * np.clip(distancia * 1.4, 0.0, 1.0) ** 1.6
    return np.clip(imagen * luz[:, :, None], 0, 255).astype(np.uint8)


def _colocar(ancho_hoja: int, alto_hoja: int, ancho: int, alto: int,
             rng: np.random.Generator, perfil: dict,
             perspectiva: float | None) -> np.ndarray:
    """Esquinas destino de la hoja dentro del encuadre (TL,TR,BR,BL).

    Se construye un rectangulo girado y con perspectiva leve, y se garantiza
    que las 4 esquinas del tablero queden dentro del encuadre con margen: es el
    requisito del protocolo de captura.
    """
    margen = max(4, int(round(0.015 * min(ancho, alto))))
    escala_ajuste = min(
        (ancho - 2 * margen) / max(ancho_hoja, 1),
        (alto - 2 * margen) / max(alto_hoja, 1),
    )
    escala = escala_ajuste * float(rng.uniform(*perfil["relleno"]))
    angulo = math.radians(float(rng.uniform(-perfil["angulo"], perfil["angulo"])))

    ancho_local = ancho_hoja * escala
    alto_local = alto_hoja * escala
    locales = np.array(
        [
            [-ancho_local / 2.0, -alto_local / 2.0],
            [ancho_local / 2.0, -alto_local / 2.0],
            [ancho_local / 2.0, alto_local / 2.0],
            [-ancho_local / 2.0, alto_local / 2.0],
        ],
        dtype="float64",
    )
    coseno, seno = math.cos(angulo), math.sin(angulo)
    rotacion = np.array([[coseno, -seno], [seno, coseno]])
    puntos = locales @ rotacion.T

    factor = perfil["perspectiva"] if perspectiva is None else float(perspectiva)
    if factor > 0:
        lado = min(ancho_local, alto_local)
        puntos = puntos + rng.normal(0.0, factor * lado, size=(4, 2))

    def extents(pts):
        return (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())

    minx, maxx, miny, maxy = extents(puntos)
    disp_ancho = maxx - minx
    disp_alto = maxy - miny
    ajuste = min(
        1.0,
        (ancho - 2 * margen) / max(disp_ancho, 1e-6),
        (alto - 2 * margen) / max(disp_alto, 1e-6),
    )
    puntos *= ajuste
    minx, maxx, miny, maxy = extents(puntos)
    disp_ancho = maxx - minx
    disp_alto = maxy - miny

    cx_min = margen + disp_ancho / 2.0
    cx_max = ancho - margen - disp_ancho / 2.0
    cy_min = margen + disp_alto / 2.0
    cy_max = alto - margen - disp_alto / 2.0
    if cx_min > cx_max:
        cx_min = cx_max = ancho / 2.0
    if cy_min > cy_max:
        cy_min = cy_max = alto / 2.0
    centro_x = float(rng.uniform(cx_min, cx_max))
    centro_y = float(rng.uniform(cy_min, cy_max))

    centro_actual = np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
    puntos += np.array([centro_x, centro_y]) - centro_actual
    return puntos.astype("float32")


def _componer(deformada: np.ndarray, fondo: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Pega la hoja deformada sobre el fondo con una sombra suave."""
    alpha = deformada[..., 3].astype("float32") / 255.0
    alto, ancho = alpha.shape
    sigma = max(2.0, 0.012 * min(ancho, alto))
    sombra = cv2.GaussianBlur(alpha, (0, 0), sigma)
    desplazamiento_x = int(round(float(rng.uniform(0.003, 0.010)) * ancho))
    desplazamiento_y = int(round(float(rng.uniform(0.003, 0.010)) * alto))
    sombra = np.roll(sombra, (desplazamiento_y, desplazamiento_x), axis=(0, 1))
    base = fondo.astype("float32") * (1.0 - 0.5 * sombra[:, :, None])
    base = np.clip(base, 0, 255)
    compuesta = (
        deformada[..., :3].astype("float32") * alpha[:, :, None]
        + base * (1.0 - alpha[:, :, None])
    )
    return np.clip(compuesta, 0, 255).astype(np.uint8)


def _fotometria(imagen: np.ndarray, rng: np.random.Generator, perfil: dict) -> np.ndarray:
    img = imagen.astype("float32")
    contraste = float(rng.uniform(*perfil["contraste"]))
    img = (img - 128.0) * contraste + 128.0
    img = img * float(rng.uniform(*perfil["ganancia"])) + float(rng.uniform(*perfil["offset"]))
    img = img * np.array(
        [float(rng.uniform(0.97, 1.03)) for _ in range(3)], dtype="float32"
    )
    img = np.clip(img, 0, 255)
    if float(rng.random()) < perfil["p_blur"]:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(*perfil["blur"])))
    sigma = float(rng.uniform(*perfil["ruido"]))
    if sigma > 0:
        img = img + rng.normal(0.0, sigma, img.shape).astype("float32")
    return np.clip(img, 0, 255).astype(np.uint8)


def destino_rectificado(lado: int = LADO_RECORTE) -> np.ndarray:
    """Esquinas destino de la vista rectificada (TL, TR, BR, BL) en pixeles."""
    ultimo = float(lado - 1)
    return np.array(
        [[0.0, 0.0], [ultimo, 0.0], [ultimo, ultimo], [0.0, ultimo]],
        dtype="float32",
    )


def matriz_recorte(homografia_hoja: np.ndarray,
                   caja_area: tuple[int, int, int, int],
                   lado: int = LADO_RECORTE) -> np.ndarray:
    """Matriz imagen original -> vista rectificada del area jugable.

    ``homografia_hoja`` es la que uso el render para proyectar la hoja del
    tablero (coordenadas de ``plantilla/tablero.png``) a la foto. Se invierte y
    se compone con la transformacion de la caja del area jugable a un lienzo
    cuadrado de ``lado`` pixeles. Asi el recorte hereda exactamente la misma
    geometria que las etiquetas: no hay una segunda deteccion que introduzca
    error.
    """
    x0, y0, x1, y1 = caja_area
    origen_area = np.array(
        [[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1]],
        dtype="float32",
    )
    area_a_vista = cv2.getPerspectiveTransform(
        origen_area, destino_rectificado(lado)
    )
    return area_a_vista @ np.linalg.inv(np.asarray(homografia_hoja, dtype="float64"))


def matriz_a_texto(matriz: np.ndarray) -> str:
    """Serializa una 3x3 con precision round-trip (17 digitos significativos)."""
    arreglo = np.asarray(matriz, dtype="float64").reshape(3, 3)
    return ";".join(" ".join(f"{float(v):.17g}" for v in fila) for fila in arreglo)


def texto_a_matriz(texto: str) -> np.ndarray:
    """Inversa de ``matriz_a_texto``; valida forma 3x3 y valores finitos."""
    filas = [[float(v) for v in fila.split()] for fila in str(texto).split(";")]
    matriz = np.array(filas, dtype="float64")
    if matriz.shape != (3, 3):
        raise ValueError(f"matriz de recorte con forma {matriz.shape}, se esperaba 3x3")
    if not np.isfinite(matriz).all():
        raise ValueError("matriz de recorte con valores no finitos")
    return matriz


def render(hoja: np.ndarray, sprites: Sprites, caja_area: tuple[int, int, int, int],
           piezas: list[tuple[int, int, int]], perfil: dict,
           rng: np.random.Generator, perspectiva: float | None = None):
    """Devuelve (imagen BGR, cajas en pixeles (x0,y0,x1,y1,clase), matriz recorte)."""
    x0, y0, x1, y1 = caja_area
    ancho_casilla = (x1 - x0 + 1) / 8.0
    alto_casilla = (y1 - y0 + 1) / 8.0

    plano = hoja.copy()
    cajas_planas = []
    for fila, columna, clase in piezas:
        cx = x0 + (columna + 0.5) * ancho_casilla
        cy = y0 + (fila + 0.5) * alto_casilla
        # Jitter chico: una ficha real no queda perfectamente centrada.
        cx += float(rng.normal(0.0, 0.030 * ancho_casilla))
        cy += float(rng.normal(0.0, 0.030 * alto_casilla))
        diametro = ancho_casilla * float(rng.uniform(0.74, 0.88))
        angulo = float(rng.uniform(0.0, 360.0))
        _pegar_ficha(plano, sprites.por_clase[clase], cx, cy, diametro, angulo)
        cajas_planas.append(
            (cx - diametro / 2.0, cy - diametro / 2.0,
             cx + diametro / 2.0, cy + diametro / 2.0, clase)
        )

    alto_hoja, ancho_hoja = plano.shape[:2]
    destino = _colocar(ancho_hoja, alto_hoja, perfil["ancho"], perfil["alto"],
                       rng, perfil, perspectiva)
    origen = np.array(
        [[0, 0], [ancho_hoja, 0], [ancho_hoja, alto_hoja], [0, alto_hoja]],
        dtype="float32",
    )
    homografia = cv2.getPerspectiveTransform(origen, destino)
    deformada = cv2.warpPerspective(
        plano, homografia, (perfil["ancho"], perfil["alto"]),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    fondo = _fondo(rng, perfil["ancho"], perfil["alto"])
    compuesta = _componer(deformada, fondo, rng)

    cajas = []
    for (bx0, by0, bx1, by1, clase) in cajas_planas:
        esquinas = np.array(
            [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype="float32"
        )
        proyectadas = cv2.perspectiveTransform(
            esquinas.reshape(-1, 1, 2), homografia
        ).reshape(-1, 2)
        nx0 = max(0.0, float(proyectadas[:, 0].min()))
        nx1 = min(perfil["ancho"] - 1.0, float(proyectadas[:, 0].max()))
        ny0 = max(0.0, float(proyectadas[:, 1].min()))
        ny1 = min(perfil["alto"] - 1.0, float(proyectadas[:, 1].max()))
        if nx1 - nx0 < 1.5 or ny1 - ny0 < 1.5:
            continue
        cajas.append((nx0, ny0, nx1, ny1, clase))

    imagen = _fotometria(compuesta, rng, perfil)
    return imagen, cajas, matriz_recorte(homografia, caja_area)


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _linea_yolo(caja, ancho: int, alto: int) -> str:
    x0, y0, x1, y1, clase = caja
    centro_x = (x0 + x1) / 2.0 / ancho
    centro_y = (y0 + y1) / 2.0 / alto
    w = (x1 - x0) / ancho
    h = (y1 - y0) / alto
    centro_x = min(max(centro_x, 0.0), 1.0)
    centro_y = min(max(centro_y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return f"{clase} {centro_x:.6f} {centro_y:.6f} {w:.6f} {h:.6f}"


def _anotar(imagen: np.ndarray, cajas, clases: list[str],
            titulo: str) -> np.ndarray:
    anotada = imagen.copy()
    alto, ancho = anotada.shape[:2]
    grosor = max(1, int(round(min(alto, ancho) / 400)))
    for x0, y0, x1, y1, clase in cajas:
        color = PALETA_CAJAS[clase % len(PALETA_CAJAS)]
        cv2.rectangle(anotada, (int(x0), int(y0)), (int(x1), int(y1)), color, grosor)
        etiqueta = clases[clase]
        y_texto = max(12, int(y0) - 3)
        cv2.putText(anotada, etiqueta, (int(x0), y_texto), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.3, min(alto, ancho) / 1400.0), (20, 20, 20), grosor * 2,
                    cv2.LINE_AA)
        cv2.putText(anotada, etiqueta, (int(x0), y_texto), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.3, min(alto, ancho) / 1400.0), color, grosor, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, alto - 10), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, min(alto, ancho) / 1200.0), (25, 25, 25), grosor * 3, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, alto - 10), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, min(alto, ancho) / 1200.0), (255, 255, 255), grosor, cv2.LINE_AA)
    return anotada


def _leer_hoja(ruta) -> np.ndarray:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise FileNotFoundError(f"no pude leer el tablero: {ruta}")
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGRA)
    elif imagen.shape[2] == 3:
        alpha = np.full(imagen.shape[:2] + (1,), 255, np.uint8)
        imagen = np.dstack([imagen, alpha])
    return imagen


def _indices_preview(cantidad: int) -> set[int]:
    """Las 3 primeras imagenes de la corrida, para revision humana rapida."""
    if cantidad <= 0:
        return set()
    return set(range(min(3, cantidad)))


def _siguiente_indice(carpeta_imagenes: Path) -> int:
    """Primer indice libre: continua despues de la ultima sintetico_NNNNN.jpg."""
    mayor = -1
    for ruta in carpeta_imagenes.glob("sintetico_*.jpg"):
        partes = ruta.stem.split("_")
        if len(partes) == 2 and partes[1].isdigit():
            mayor = max(mayor, int(partes[1]))
    return mayor + 1


def _leer_manifiesto(ruta: Path) -> list[dict[str, str]]:
    """Lee el manifiesto existente conservando el orden de las filas."""
    if not ruta.exists():
        return []
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        return [dict(fila) for fila in csv.DictReader(archivo)]


def _escribir_manifiesto(ruta: Path, previas: list[dict[str, str]],
                         nuevas: list[dict]) -> tuple[int, int]:
    """Actualiza por nombre de imagen y conserva las filas anteriores.

    Devuelve (filas_actualizadas, filas_totales). Una fila que ya existia se
    reemplaza con los datos nuevos; las demas quedan intactas aunque no tengan
    las columnas nuevas (homografia_recorte, ancho, alto).
    """
    por_nombre: dict[str, dict] = {fila["imagen"]: dict(fila) for fila in previas}
    orden = [fila["imagen"] for fila in previas]
    actualizadas = 0
    for fila in nuevas:
        nombre = fila["imagen"]
        if nombre in por_nombre:
            actualizadas += 1
        else:
            orden.append(nombre)
        por_nombre[nombre] = {**por_nombre.get(nombre, {}), **fila}
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=CAMPOS_MANIFIESTO, restval="", extrasaction="ignore"
        )
        escritor.writeheader()
        for nombre in orden:
            escritor.writerow({campo: por_nombre[nombre].get(campo, "")
                               for campo in CAMPOS_MANIFIESTO})
    return actualizadas, len(orden)


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Genera fotos sinteticas con etiquetas YOLO desde las plantillas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--n", type=int, default=24,
                            help="cantidad de imagenes a generar")
    analizador.add_argument("--plantilla-fichas", default=str(PLANTILLA_FICHAS),
                            help="hoja de stickers de fichas")
    analizador.add_argument("--plantilla-tablero", default=str(PLANTILLA_TABLERO),
                            help="tablero impreso con los 4 ArUco")
    analizador.add_argument("--perfil", default="tomo",
                            choices=("esp", "celular", "celulo", "tomo"),
                            help="camara simulada; 'tomo' mezcla esp y celular")
    analizador.add_argument("--semilla", type=int, default=None,
                            help="semilla para reproducir una corrida")
    analizador.add_argument("--salida", default=str(SALIDA_DEFECTO),
                            help="carpeta de salida del dataset sintetico")
    analizador.add_argument("--posiciones", default="mezcla",
                            choices=("aleatoria", "jugada", "juego", "finales", "mezcla"),
                            help="origen de las posiciones; 'mezcla' reparte segun "
                                 "MEZCLA_OBJETIVO (jugada/aleatoria/finales/dificil/vacio)")
    analizador.add_argument("--inicio", type=int, default=None,
                            help="indice inicial; si se omite continua desde la "
                                 "ultima sintetico_NNNNN.jpg que ya exista")
    analizador.add_argument("--sin-previews", action="store_true",
                            help="no escribe las 3 previews de assets/previews "
                                 "(util para corridas de prueba)")
    analizador.add_argument("--perspectiva", type=float, default=None,
                            help="factor de perspectiva; si se omite usa el del perfil")
    argumentos = analizador.parse_args(argv)

    if argumentos.n <= 0:
        print("nada que generar: --n debe ser mayor que 0")
        return 2

    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    import esquinas_aruco  # se importa aca para que --help no cargue la deteccion

    inicio = time.time()
    sprites = extraer_sprites(argumentos.plantilla_fichas, clases)
    plantilla = esquinas_aruco.analizar_plantilla(argumentos.plantilla_tablero)
    hoja = _leer_hoja(argumentos.plantilla_tablero)

    semilla = argumentos.semilla
    if semilla is None:
        semilla = int(np.random.SeedSequence().entropy % (2 ** 31))
    rng_maestro = np.random.default_rng(int(semilla))

    salida = Path(argumentos.salida)
    carpeta_imagenes = salida / "images"
    carpeta_etiquetas = salida / "labels"
    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    carpeta_etiquetas.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    print(f"clases: {clases}")
    print(f"plantilla tablero: {argumentos.plantilla_tablero}")
    print(f"  diccionario: {plantilla.diccionario}")
    print(f"  ids por lado: {plantilla.lado_a_id}")
    print(f"  area jugable (x0,y0,x1,y1): {plantilla.caja_area}")
    print(f"fichas recortadas: {len(sprites.por_clase)} clases, {len(sprites.discos)} discos")
    print(f"semilla: {semilla}")

    if argumentos.inicio is None:
        indice_inicial = _siguiente_indice(carpeta_imagenes)
    else:
        indice_inicial = int(argumentos.inicio)
    if indice_inicial < 0:
        print("nada que generar: --inicio no puede ser negativo")
        return 2
    print(f"indices: sintetico_{indice_inicial:05d} .. "
          f"sintetico_{indice_inicial + argumentos.n - 1:05d}")

    previsualizaciones = _indices_preview(argumentos.n)
    modo_pedido = "jugada" if argumentos.posiciones == "juego" else argumentos.posiciones
    modos_mezcla = (
        _cuotas_mezcla(argumentos.n, rng_maestro)
        if modo_pedido == "mezcla" else None
    )
    filas = []
    conteo_perfil: dict[str, int] = {}
    conteo_modo: dict[str, int] = {}
    conteo_categoria: dict[str, int] = {}
    for indice in range(argumentos.n):
        rng = np.random.default_rng(int(rng_maestro.integers(0, 2 ** 63)))
        if argumentos.perfil == "tomo":
            nombre_perfil = "esp" if float(rng.random()) < 0.5 else "celular"
        elif argumentos.perfil == "celulo":
            nombre_perfil = "celular"
        else:
            nombre_perfil = argumentos.perfil
        perfil = PERFILES[nombre_perfil]

        modo = modos_mezcla[indice] if modos_mezcla is not None else modo_pedido
        piezas, modo_real = _generar_posicion(rng, clases, modo)
        imagen, cajas, matriz = render(hoja, sprites, plantilla.caja_area, piezas,
                                       perfil, rng, argumentos.perspectiva)
        categoria = _categoria(piezas, clases)

        numero = indice_inicial + indice
        nombre = f"sintetico_{numero:05d}"
        calidad = int(round(float(rng.uniform(*perfil["jpeg"]))))
        cv2.imwrite(str(carpeta_imagenes / f"{nombre}.jpg"), imagen,
                    [cv2.IMWRITE_JPEG_QUALITY, calidad])

        lineas = [_linea_yolo(caja, perfil["ancho"], perfil["alto"]) for caja in cajas]
        with open(carpeta_etiquetas / f"{nombre}.txt", "w", encoding="utf-8") as archivo:
            for linea in lineas:
                archivo.write(linea + "\n")

        filas.append({
            "imagen": f"{nombre}.jpg",
            "n_fichas": len(cajas),
            "perfil": nombre_perfil,
            "posiciones": modo_real,
            "categoria": categoria,
            "semilla": semilla,
            "homografia_recorte": matriz_a_texto(matriz),
            "ancho": perfil["ancho"],
            "alto": perfil["alto"],
        })
        conteo_perfil[nombre_perfil] = conteo_perfil.get(nombre_perfil, 0) + 1
        conteo_modo[modo_real] = conteo_modo.get(modo_real, 0) + 1
        conteo_categoria[categoria] = conteo_categoria.get(categoria, 0) + 1

        if indice in previsualizaciones and not argumentos.sin_previews:
            titulo = (f"{nombre}  perfil={nombre_perfil}  modo={modo_real}  "
                      f"fichas={len(cajas)}  semilla={semilla}")
            anotada = _anotar(imagen, cajas, clases, titulo)
            cv2.imwrite(str(PREVIEWS / f"{nombre}_preview.jpg"), anotada,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (indice + 1) % 10 == 0 or indice + 1 == argumentos.n:
            print(f"  {indice + 1}/{argumentos.n} imagenes")

    manifiesto = salida / "manifesto.csv"
    previas = _leer_manifiesto(manifiesto)
    actualizadas, total_filas = _escribir_manifiesto(manifiesto, previas, filas)

    segundos = time.time() - inicio
    print()
    print(f"listo: {argumentos.n} imagenes en {segundos:.1f} s")
    print(f"  imagenes: {carpeta_imagenes}")
    print(f"  etiquetas: {carpeta_etiquetas}")
    print(f"  manifiesto: {manifiesto} ({total_filas} filas, "
          f"{actualizadas} actualizadas, {len(previas)} previas)")
    print(f"  previews: "
          f"{'omitidas (--sin-previews)' if argumentos.sin_previews else sorted(previsualizaciones)}"
          f" -> {PREVIEWS}")
    print(f"  por perfil: {conteo_perfil}")
    print(f"  por posicion: {conteo_modo}")
    print(f"  por categoria: {conteo_categoria}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
