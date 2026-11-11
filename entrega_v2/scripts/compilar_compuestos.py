"""Compone tableros NUEVOS con texturas REALES a partir de los sprites extraidos.

Que hace distinto de ``generar_sinteticos.py``
----------------------------------------------
El sintetico usa la plantilla impresa y las fichas de la hoja de stickers: sale
"demasiado perfecto" y ensena el arte de la plantilla, no las fichas fisicas. Aca
cada ficha es un recorte RGBA de una foto real (``extraer_sprites.py``) y cada
fondo es un recorte real del tablero vacio sobre un mantel tambien real. Sobre
eso se construyen POSICIONES que no existen en ninguna foto.

Cubre los tres pedidos del usuario:

* ROTACION: cada ficha se rota de 0 a 360 grados antes de pegar. En estas fichas
  el contorno es un disco, asi que el giro se ve en el simbolo (rey y dama son
  los que mas lo evidencian).
* POSICIONES nuevas: 50 % legales con ``python-chess`` (se reutiliza
  ``generar_sinteticos._piezas_jugada``, 4-40 jugadas desde la inicial) y 50 %
  aleatorias con las clases raras (dama/rey) sobrerrepresentadas 8-12 por tablero.
* PERFIL ESP32 primero: 70 % 640x480 con desenfoque/ruido/JPEG bajo y 30 %
  celular 1024x768, que es el reparto de captura real del proyecto.

Reglas de composicion
---------------------
* Una ficha por casilla y maximo 32 por tablero: el muestreo es sin reemplazo.
* Escala 0.92-1.08 del diametro REAL del disco (el radio del tablero medido en
  la extraccion, no el del sprite recortado 2 px adentro).
* Armonizacion de color: se transfiere la iluminacion del cuadrado de origen de
  cada sprite al cuadrado de destino, en Lab, con la media y la desviacion del
  fondo. Se aplica sobre el fondo, no sobre el disco, para no lavar el contraste.
* Perspectiva de hasta 8 grados mas un jitter de esquinas: mas fuerte que el del
  sintetico, que es lo que pide el pedido ("mayor que el jitter leve").
* El fondo del marco es un recorte de una foto CRUDA real (mantel/mesa), para
  que aparezca una franja superior/inferior real detras del tablero.

Salidas
-------
``dataset/reales_compuesto/images/compuesto_<perfil>_<NNNN>.jpg``
``dataset/reales_compuesto/labels/compuesto_<perfil>_<NNNN>.txt``   (YOLO)
``dataset/reales_compuesto/casillas/clasificador/<clase>/*.png``    (muestra)
``dataset/reales_compuesto/manifiesto.csv``
``dataset/reales_compuesto/etiquetas_casillas.csv``
``dataset/reales_compuesto/rotaciones.csv``
``assets/previews/compuesto_muestra_1..5.jpg``

Uso
---
    python compilar_compuestos.py --n 1560 --semilla 11
    python compilar_compuestos.py --n 8 --semilla 11 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from comun import cargar_clases
from extraer_sprites import CASILLA, LADO as LADO_PLANO, _estadisticas_cuadrado
from generar_sinteticos import (
    _anotar,
    _colocar,
    _fotometria,
    _linea_yolo,
    _pegar_ficha,
    _piezas_jugada,
    destino_rectificado,
    matriz_a_texto,
)

RAIZ = Path(__file__).resolve().parent.parent
SPRITES = RAIZ / "dataset" / "reales_compuesto" / "sprites"
RECORTES = RAIZ / "dataset" / "reales" / "recortes" / "800x800"
CRUDO = RAIZ / "dataset" / "crudo"
SALIDA = RAIZ / "dataset" / "reales_compuesto"
PREVIEWS = RAIZ / "assets" / "previews"

PERFILES = {
    "esp": dict(
        ancho=640, alto=480,
        relleno=(0.84, 0.98),
        angulo=8.0,
        perspectiva=0.030,
        p_blur=0.92,
        blur=(0.45, 0.95),
        contraste=(1.02, 1.14),
        ganancia=(0.93, 1.05),
        offset=(-14.0, 6.0),
        ruido=(2.0, 6.0),
        jpeg=(62, 82),
    ),
    "celular": dict(
        ancho=1024, alto=768,
        relleno=(0.80, 0.94),
        angulo=8.0,
        perspectiva=0.040,
        p_blur=0.35,
        blur=(0.15, 0.55),
        contraste=(0.98, 1.08),
        ganancia=(0.90, 1.10),
        offset=(-18.0, 14.0),
        ruido=(1.5, 4.5),
        jpeg=(80, 93),
    ),
}

# 70 % ESP32 / 30 % celular: el reparto real de las capturas del proyecto.
CUOTA_PERFIL = {"esp": 0.70, "celular": 0.30}
# 50 % legal / 50 % aleatoria.
CUOTA_MODO = {"legal": 0.50, "aleatoria": 0.50}
# Densidad ESCASA: el lote original quedo casi todo denso (mediana 30 fichas) y
# el usuario lo noto. Esto cubre finales y tableros de poca pieza.
CUOTA_MODO_ESCASA = {
    "finales": 0.40,   # final legal-plausible: 2 reyes + menores/peones
    "escasa": 0.30,    # aleatoria con 4-12 fichas y clase rara reforzada
    "legal": 0.20,     # partida temprana (pocas jugadas)
    "vacio": 0.10,     # tablero en blanco
}

JITTER_CELDA = 0.10          # fraccion de casilla a la redonda del centro
ESCALA_FICHA = (0.92, 1.08)
ESCALA_SOMBRA = (0.10, 0.22)

CAMPOS_MANIFIESTO = [
    "imagen", "perfil", "modo", "n_fichas", "semilla", "ancho", "alto",
    "fondo", "homografia_recorte",
]
CAMPOS_CASILLAS = ["imagen", "casilla", "clase", "fila", "columna", "perfil"]
CAMPOS_ROTACION = [
    "imagen", "casilla", "clase", "sprite", "angulo", "diametro", "centro_x",
    "centro_y",
]

ROYALES = ("dama_b", "rey_b", "dama_n", "rey_n")
MENORES = ("peon_b", "peon_n", "torre_b", "torre_n",
           "caballo_b", "caballo_n", "alfil_b", "alfil_n")


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------

def cargar_sprites(ruta: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    """Agrupa los sprites por clase con sus datos de origen."""
    manifiesto = ruta / "sprites_manifesto.csv"
    if not manifiesto.exists():
        raise SystemExit(
            f"No encuentro {manifiesto}. Corre antes extraer_sprites.py"
        )
    por_clase: dict[str, list[dict]] = {}
    filas: list[dict] = []
    with open(manifiesto, "r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            sprite = cv2.imread(str(ruta / fila["archivo"]), cv2.IMREAD_UNCHANGED)
            if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
                continue
            datos = {
                "archivo": fila["archivo"],
                "imagen": sprite,
                "radio": float(fila["radio"]),
                "radio_tablero": float(fila.get("radio_tablero") or fila["radio"]),
                "fondo": (
                    float(fila["fondo_l"]), float(fila["fondo_a"]),
                    float(fila["fondo_b"]), float(fila["fondo_l_std"]),
                ),
            }
            por_clase.setdefault(fila["clase"], []).append(datos)
            filas.append(datos)
    faltan = [clase for clase in ("peon_b", "rey_n") if clase not in por_clase]
    if faltan:
        raise SystemExit(f"Faltan clases en los sprites: {faltan}")
    return por_clase, filas


def cargar_fondos() -> list[np.ndarray]:
    """Recortes reales del tablero vacio (800x800)."""
    fondos = []
    for ruta in sorted((RECORTES / "05_tablero_vacio").glob("*.jpg")):
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            continue
        if imagen.shape[0] != LADO_PLANO or imagen.shape[1] != LADO_PLANO:
            imagen = cv2.resize(imagen, (LADO_PLANO, LADO_PLANO),
                                interpolation=cv2.INTER_AREA)
        fondos.append((ruta.name, imagen))
    if not fondos:
        raise SystemExit(f"No hay recortes de tablero vacio en {RECORTES}")
    return fondos


def cargar_manteles(maximo: int = 8) -> list[np.ndarray]:
    """Fotos crudas grandes para muestrear el mantel/mesa real del marco.

    Se descartan las de 640x480: al recortar una esquina y estirarla al marco
    quedarian muy borrosas. Se guardan reducidas a 1400 px de lado mayor para no
    gastar memoria con las 134 fotos.
    """
    manteles: list[np.ndarray] = []
    for ruta in sorted(CRUDO.glob("*/*.jpg")):
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            continue
        alto, ancho = imagen.shape[:2]
        if min(alto, ancho) < 900:
            continue
        escala = 1400.0 / max(alto, ancho)
        manteles.append(cv2.resize(
            imagen, (int(ancho * escala), int(alto * escala)),
            interpolation=cv2.INTER_AREA,
        ))
        if len(manteles) >= maximo:
            break
    if not manteles:
        raise SystemExit(f"No encontre fotos crudas grandes en {CRUDO}")
    return manteles


# ---------------------------------------------------------------------------
# Posiciones
# ---------------------------------------------------------------------------

def _piezas_raras(rng: np.random.Generator, clases: list[str]
                  ) -> list[tuple[int, int, int]]:
    """Tablero aleatorio con damas y reyes sobrerrepresentados (8-12 por foto).

    La posicion legal hereda la escasez real del ajedrez (una dama por bando).
    Este modo rompe esa escasez a proposito: el clasificador y el detector v1
    tenian 33 y 49 cajas de dama/rey y no alcanzaban para aprender la clase.
    """
    total = int(rng.integers(20, 33))          # 20..32 fichas
    reinas = min(int(rng.integers(8, 13)), total)   # 8..12 damas/reyes
    indices = [int(i) for i in rng.choice(64, size=total, replace=False)]

    reales = [clases.index(nombre) for nombre in ROYALES]
    orden = [reales[int(i)] for i in rng.permutation(len(reales))]
    elegidas = [orden[i % len(orden)] for i in range(reinas)]

    menores = list(range(len(clases)))  # el resto, de las 12 clases
    for _ in range(total - reinas):
        elegidas.append(int(menores[int(rng.integers(len(menores)))]))

    piezas = []
    for indice, clase in zip(indices, elegidas):
        fila, columna = divmod(indice, 8)
        piezas.append((fila, columna, int(clase)))
    return piezas


def _piezas_escasa(rng: np.random.Generator, clases: list[str]
                   ) -> list[tuple[int, int, int]]:
    """Aleatoria pero ESCASA: 4-12 fichas, damas/reyes 2-5 (clase rara)."""
    total = int(rng.integers(4, 13))
    reinas = min(int(rng.integers(2, 6)), total)
    indices = [int(i) for i in rng.choice(64, size=total, replace=False)]

    reales = [clases.index(nombre) for nombre in ROYALES]
    elegidas = [reales[int(i)] for i in rng.integers(0, len(reales), size=reinas)]
    menores = list(range(len(clases)))
    for _ in range(total - reinas):
        elegidas.append(int(menores[int(rng.integers(len(menores)))]))

    piezas = []
    for indice, clase in zip(indices, elegidas):
        fila, columna = divmod(indice, 8)
        piezas.append((fila, columna, int(clase)))
    return piezas


def _piezas_finales(rng: np.random.Generator, clases: list[str]
                    ) -> list[tuple[int, int, int]]:
    """Final Plausible: 2 reyes SIEMPRE, 0-1 dama, menores y peones.

    El modo 'legal' genera aperturas con muchas piezas y el 'escasa' no sabe de
    ajedrez. Este modo cubre el TERCER tiempo del tablero: pocos efectivos, los
    reyes obligatorios (como manda el juego) y separados. La legalidad estricta
    no es necesaria porque el detector y el clasificador son locales.
    """
    rey_b = clases.index("rey_b")
    rey_n = clases.index("rey_n")
    dama_ids = [clases.index(n) for n in ("dama_b", "dama_n")]
    peon_ids = [clases.index(n) for n in ("peon_b", "peon_n")]
    menores = [clases.index(n) for n in MENORES]

    total = int(rng.integers(4, 15))          # 4..14 casillas ocupadas
    casillas = [int(i) for i in rng.choice(64, size=total, replace=False)]

    # Reyes no pegados: en una partida real nunca quedan adyacentes.
    reyes_idx = (casillas[0], casillas[-1])
    for _ in range(60):
        a, b = (int(casillas[int(rng.integers(len(casillas)))]),
                int(casillas[int(rng.integers(len(casillas)))]))
        (fa, ca), (fb, cb) = divmod(a, 8), divmod(b, 8)
        if abs(fa - fb) + abs(ca - cb) >= 2:
            reyes_idx = (a, b)
            break

    restantes = [i for i in casillas if i not in reyes_idx]
    elegidas = [(reyes_idx[0], rey_b), (reyes_idx[1], rey_n)]
    if restantes and rng.random() < 0.65:
        casilla = restantes.pop(int(rng.integers(len(restantes))))
        elegidas.append((casilla, dama_ids[int(rng.integers(2))]))
    for casilla in restantes:
        dado = rng.random()
        if dado < 0.40:
            elegidas.append((casilla, int(menores[int(rng.integers(6))])))
        elif dado < 0.75:
            elegidas.append((casilla, peon_ids[int(rng.integers(2))]))
        elif dado < 0.85:
            elegidas.append((casilla, dama_ids[int(rng.integers(2))]))
        else:
            elegidas.append((casilla, int(menores[int(rng.integers(6))])))

    piezas = []
    for indice, clase in elegidas:
        fila, columna = divmod(indice, 8)
        piezas.append((fila, columna, clase))
    return piezas


def _posicion(rng: np.random.Generator, clases: list[str], modo: str
              ) -> list[tuple[int, int, int]]:
    if modo == "legal":
        return _piezas_jugada(rng, clases)
    if modo == "finales":
        return _piezas_finales(rng, clases)
    if modo == "escasa":
        return _piezas_escasa(rng, clases)
    if modo == "vacio":
        return []
    return _piezas_raras(rng, clases)


# ---------------------------------------------------------------------------
# Fondo del marco
# ---------------------------------------------------------------------------

def _fondo_real(rng: np.random.Generator, ancho: int, alto: int,
                manteles: list[np.ndarray]) -> np.ndarray:
    """Recorte de esquina de una foto real, estirado al marco del perfil."""
    foto = manteles[int(rng.integers(len(manteles)))]
    alto_foto, ancho_foto = foto.shape[:2]
    lado = max(64, int(min(alto_foto, ancho_foto) * float(rng.uniform(0.42, 0.62))))
    x0 = 0 if rng.random() < 0.5 else ancho_foto - lado
    y0 = 0 if rng.random() < 0.5 else alto_foto - lado
    parche = foto[y0:y0 + lado, x0:x0 + lado].copy()
    if rng.random() < 0.5:
        parche = cv2.flip(parche, 1)
    giros = int(rng.integers(4))
    if giros:
        parche = np.ascontiguousarray(np.rot90(parche, giros))

    escala = max(ancho / parche.shape[1], alto / parche.shape[0])
    parche = cv2.resize(
        parche, (max(ancho, int(parche.shape[1] * escala) + 1),
                 max(alto, int(parche.shape[0] * escala) + 1)),
        interpolation=cv2.INTER_AREA,
    )
    sobra_x = parche.shape[1] - ancho
    sobra_y = parche.shape[0] - alto
    x0 = int(rng.integers(0, sobra_x + 1)) if sobra_x > 0 else 0
    y0 = int(rng.integers(0, sobra_y + 1)) if sobra_y > 0 else 0
    fondo = parche[y0:y0 + alto, x0:x0 + ancho]

    fondo = cv2.GaussianBlur(fondo, (0, 0), float(rng.uniform(0.5, 1.6)))
    imagen = fondo.astype("float32") * float(rng.uniform(0.82, 1.10))
    imagen = imagen + float(rng.uniform(-12.0, 12.0))
    tono = np.array([float(rng.uniform(0.97, 1.03)) for _ in range(3)], "float32")
    return np.clip(imagen * tono, 0, 255).astype(np.uint8)


def _variante_tablero(plano: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brillo/contraste/gamma al tablero real vacio para variar la iluminacion."""
    imagen = plano.astype("float32")
    contraste = float(rng.uniform(0.90, 1.10))
    imagen = (imagen - 128.0) * contraste + 128.0 + float(rng.uniform(-14.0, 14.0))
    gamma = float(rng.uniform(0.85, 1.18))
    imagen = np.power(np.clip(imagen, 0.0, 255.0) / 255.0, gamma) * 255.0
    tono = np.array([float(rng.uniform(0.96, 1.04)) for _ in range(3)], "float32")
    imagen = np.clip(imagen * tono, 0, 255).astype(np.uint8)
    # El tablero vacio esta mas nitido que las fichas (vienen de fotos mas
    # blandas). Un desenfoque chico iguala la nitidez y evita que el pegado se
    # note por contraste de foco.
    return cv2.GaussianBlur(imagen, (0, 0), float(rng.uniform(0.30, 0.70)))


# ---------------------------------------------------------------------------
# Armonizacion y sombra
# ---------------------------------------------------------------------------

def _transferir_luz(sprite: np.ndarray, origen: tuple[float, float, float, float],
                    destino: tuple[float, float, float, float],
                    fuerza: float) -> np.ndarray:
    """Lleva la ILUMINACION del cuadrado de origen al del destino (solo luz).

    No se toca el croma a/b: el color del cuadrado (azul o gris) es pintura del
    tablero, no luz. Transferirlo tiñe el disco entero de azul y borra el simbolo
    (se probo y se vio en ``compuesto_muestra_1``). Lo que se iguala es la
    luminancia del parche de fondo del que salio la ficha contra el parche de
    destino: si el tablero nuevo esta mas oscuro o mas claro, la ficha entera se
    escala igual y deja de verse pegada. El balance de blancos global lo aplica
    despues ``_fotometria`` a la imagen completa, que es como lo hace la camara.
    """
    lab = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2LAB).astype("float32")
    escala_l = 1.0
    if origen[3] > 1.5 and destino[3] > 1.5:
        escala_l = float(np.clip(destino[3] / origen[3], 0.80, 1.25))
    nuevo = (lab[..., 0] - origen[0]) * escala_l + destino[0]
    lab[..., 0] = lab[..., 0] * (1.0 - fuerza) + nuevo * fuerza
    bgr = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype("uint8"), cv2.COLOR_LAB2BGR)
    return np.dstack([bgr, sprite[..., 3]])


def _sombra(plano: np.ndarray, centro_x: float, centro_y: float,
            diametro: float, rng: np.random.Generator) -> None:
    """Sombra difusa bajo la ficha, del lado contrario a la luz.

    Se dibuja solo en la ventana de la ficha: hacerlo sobre los 800x800 completos
    cuesta 5-8 ms por ficha y con 30 fichas por tablero domina el tiempo total.
    """
    fuerza = float(rng.uniform(*ESCALA_SOMBRA))
    desplazamiento_x = diametro * float(rng.uniform(0.02, 0.08))
    desplazamiento_y = diametro * float(rng.uniform(0.03, 0.10))
    mitad = diametro * 0.75
    x0 = max(0, int(round(centro_x - mitad)))
    y0 = max(0, int(round(centro_y - mitad)))
    x1 = min(plano.shape[1], int(round(centro_x + mitad)))
    y1 = min(plano.shape[0], int(round(centro_y + mitad)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    capa = np.zeros((y1 - y0, x1 - x0), "float32")
    cv2.ellipse(
        capa,
        (int(round(centro_x + desplazamiento_x - x0)),
         int(round(centro_y + desplazamiento_y - y0))),
        (max(2, int(diametro * 0.48)), max(2, int(diametro * 0.40))),
        0.0, 0.0, 360.0, 1.0, -1,
    )
    capa = cv2.GaussianBlur(capa, (0, 0), max(1.0, diametro * 0.09))
    ventana = plano[y0:y1, x0:x1]
    factor = (1.0 - 0.55 * fuerza * capa)[..., None]
    plano[y0:y1, x0:x1] = np.clip(
        ventana.astype("float32") * factor, 0, 255
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Recortes por casilla
# ---------------------------------------------------------------------------

def _recortar_casillas(imagen: np.ndarray, homografia: np.ndarray,
                       clases_casilla: list[list[str]]) -> list[tuple[str, np.ndarray]]:
    """64 recortes de 100x100 alineados con la rejilla rectificada del tablero."""
    destino = destino_rectificado(CASILLA)
    salida = []
    for fila in range(8):
        for columna in range(8):
            x0, y0 = columna * CASILLA, fila * CASILLA
            origen = np.array(
                [[x0, y0], [x0 + CASILLA, y0],
                 [x0 + CASILLA, y0 + CASILLA], [x0, y0 + CASILLA]],
                dtype="float32",
            )
            en_frame = cv2.perspectiveTransform(
                origen.reshape(-1, 1, 2), homografia
            ).reshape(-1, 2).astype("float32")
            matriz = cv2.getPerspectiveTransform(en_frame, destino)
            recorte = cv2.warpPerspective(
                imagen, matriz, (CASILLA, CASILLA),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            salida.append((clases_casilla[fila][columna], recorte))
    return salida


def _hoja_casillas(recortes: list[tuple[str, np.ndarray]], titulo: str) -> np.ndarray:
    """Hoja de contacto 8x8 con la clase escrita arriba de cada casilla."""
    lado = CASILLA
    hoja = np.zeros((8 * lado + 26, 8 * lado, 3), np.uint8)
    for indice, (clase, recorte) in enumerate(recortes[:64]):
        fila, columna = divmod(indice, 8)
        y0 = 26 + fila * lado
        x0 = columna * lado
        hoja[y0:y0 + lado, x0:x0 + lado] = recorte
        etiqueta = "vacia" if clase == "vacia" else clase[:6]
        color = (60, 60, 60) if clase == "vacia" else (40, 200, 40)
        cv2.putText(hoja, etiqueta, (x0 + 3, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(hoja, etiqueta, (x0 + 3, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    cv2.putText(hoja, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return hoja


# ---------------------------------------------------------------------------
# Reparto de cuotas
# ---------------------------------------------------------------------------

def _cuotas(pesos: dict[str, float], cantidad: int,
            rng: np.random.Generator) -> list[str]:
    """Reparte ``cantidad`` con los pesos, por mayor resto, y lo mezcla."""
    if cantidad <= 0:
        return []
    total = sum(pesos.values())
    crudos = {clave: cantidad * peso / total for clave, peso in pesos.items()}
    cuotas = {clave: int(valor // 1) for clave, valor in crudos.items()}
    restantes = cantidad - sum(cuotas.values())
    orden = sorted(crudos, key=lambda clave: crudos[clave] - cuotas[clave],
                   reverse=True)
    for clave in orden[:restantes]:
        cuotas[clave] += 1
    lista = np.array([clave for clave, n in cuotas.items() for _ in range(n)],
                     dtype=object)
    rng.shuffle(lista)
    return [str(clave) for clave in lista]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _volcar_csv(ruta: Path, campos: list[str], filas: list[dict],
                append: bool) -> None:
    """Escribe o AGREGA filas a un CSV.

    En modo append NO se reescribe el archivo: un fallo a mitad de corrida no
    puede borrar las filas vivas, solo duplicar el encabezado en el peor caso
    (y eso se limpia al regenerar).
    """
    agregar = append and ruta.exists()
    with ruta.open("a" if agregar else "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        if not agregar:
            escritor.writeheader()
        for fila in filas:
            limpios = {c: fila.get(c, "") for c in campos}
            escritor.writerow(limpios)


def salidas_texto(salida: Path, filas_manifiesto: list[dict],
                  filas_casillas: list[dict], filas_rotacion: list[dict],
                  prevista_agregar: bool) -> tuple[Path, Path, Path]:
    manifiesto = salida / "manifiesto.csv"
    _volcar_csv(manifiesto, CAMPOS_MANIFIESTO, filas_manifiesto, prevista_agregar)

    csv_casillas = salida / "etiquetas_casillas.csv"
    _volcar_csv(csv_casillas, CAMPOS_CASILLAS, filas_casillas, prevista_agregar)

    csv_rotacion = salida / "rotaciones.csv"
    _volcar_csv(csv_rotacion, CAMPOS_ROTACION, filas_rotacion, prevista_agregar)
    return manifiesto, csv_casillas, csv_rotacion


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Compone tableros nuevos con sprites y fondos reales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--n", type=int, default=1560,
                            help="cantidad de tableros a componer")
    analizador.add_argument("--semilla", type=int, default=11)
    analizador.add_argument("--densidad", choices=("densa", "escasa"), default="densa",
                            help="escasa: mix de finales/escasa/legal/vacio "
                                 "(CORTA el sesgo denso del lote original)")
    analizador.add_argument("--agregar", action="store_true",
                            help="continuar el dataset: retoma numeracion y "
                                 "conserva las filas previas de los CSV")
    analizador.add_argument("--sprites", default=str(SPRITES))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--casillas-cada", type=int, default=0,
                            help="guardar 64 recortes de casilla cada N tableros "
                                 "(0 = solo el primero de cada perfil, para no "
                                 "llenar el disco)")
    analizador.add_argument("--sin-casillas", action="store_true",
                            help="no guardar recortes de casilla")
    analizador.add_argument("--sin-previews", action="store_true")
    argumentos = analizador.parse_args(argv)

    if argumentos.n <= 0:
        print("nada que componer: --n debe ser mayor que 0")
        return 2

    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    sprites, todos = cargar_sprites(Path(argumentos.sprites))
    fondos = cargar_fondos()
    manteles = cargar_manteles()
    print(f"clases: {clases}")
    print(f"sprites: {len(todos)} en {len(sprites)} clases -> {argumentos.sprites}")
    print(f"fondos de tablero real: {[nombre for nombre, _ in fondos]}")
    print(f"fotos de mantel: {len(manteles)}")

    salida = Path(argumentos.salida)
    imagenes_dir = salida / "images"
    etiquetas_dir = salida / "labels"
    casillas_dir = salida / "casillas" / "clasificador"
    imagenes_dir.mkdir(parents=True, exist_ok=True)
    etiquetas_dir.mkdir(parents=True, exist_ok=True)
    for clase in clases + ["vacia"]:
        (casillas_dir / clase).mkdir(parents=True, exist_ok=True)

    rng_maestro = np.random.default_rng(int(argumentos.semilla))
    perfiles = _cuotas(CUOTA_PERFIL, argumentos.n, rng_maestro)
    cuota_modo = CUOTA_MODO_ESCASA if argumentos.densidad == "escasa" else CUOTA_MODO
    modos = _cuotas(cuota_modo, argumentos.n, rng_maestro)
    contadores: Counter = Counter()

    # --agregar: retoma la numeracion desde los ARCHIVOS en disco (no releer
    # los CSV) y al final los CSV se escriben en MODO APPEND. Los CSV de
    # corridas previas jamas se reescriben: un write parcial no puede perder
    # datos. La numeracion solo necesita maximo de imagen por perfil.
    existentes_manifiesto: list[dict] = []
    existentes_casillas: list[dict] = []
    existentes_rotacion: list[dict] = []
    if argumentos.agregar:
        maximo: dict[str, int] = {}
        for ruta in imagenes_dir.glob("compuesto_*_*.jpg"):
            coincidencia = re.match(r"compuesto_([a-z]+)_(\d+)\.jpg", ruta.name)
            if coincidencia:
                maximo[coincidencia.group(1)] = max(
                    maximo.get(coincidencia.group(1), 0), int(coincidencia.group(2))
                )
        for perfil_nombre, valor in maximo.items():
            contadores[perfil_nombre] = valor
        print(f"--agregar: numeracion retomada {maximo}")

    inicio = time.time()
    filas_manifiesto: list[dict] = list(existentes_manifiesto)
    filas_casillas: list[dict] = list(existentes_casillas)
    filas_rotacion: list[dict] = list(existentes_rotacion)
    conteo_perfil: Counter = Counter()
    conteo_modo: Counter = Counter()
    conteo_cajas: Counter = Counter()
    min_ratio_distancia = float("inf")
    violaciones_superposicion = 0
    hojas: dict[str, list[tuple[str, np.ndarray]]] = {}
    previews: list[Path] = []
    anotadas = 0
    conteo_anotadas: Counter = Counter()

    for indice in range(argumentos.n):
        perfil_nombre = perfiles[indice]
        modo = modos[indice]
        perfil = PERFILES[perfil_nombre]
        semilla = int(rng_maestro.integers(0, 2 ** 63))
        rng = np.random.default_rng(semilla)

        nombre_fondo, tablero = fondos[int(rng.integers(len(fondos)))]
        plano = _variante_tablero(tablero, rng)
        piezas = _posicion(rng, clases, modo)

        clases_casilla = [["vacia"] * 8 for _ in range(8)]
        ocupacion: list[tuple[float, float, float, str]] = []
        rotaciones_imagen: list[dict] = []
        for fila, columna, clase in piezas:
            clase_nombre = clases[clase]
            sprites_clase = sprites[clase_nombre]
            sprite = sprites_clase[int(rng.integers(len(sprites_clase)))]
            escala = float(rng.uniform(*ESCALA_FICHA))
            # Se escala el diametro REAL del disco; el sprite trae 2 px menos.
            diametro = (2.0 * sprite["radio_tablero"] * escala
                        * (2.0 * sprite["radio"] + 2.0) / (2.0 * sprite["radio"]))
            # El jitter se limita para que el disco nunca pase el borde de su
            # casilla: asi dos fichas vecinas no pueden acercarse a menos de la
            # suma de sus radios (verificacion 3: cero superposiciones).
            radio_disco = diametro / 2.0
            holgura = max(0.0, 0.95 * (CASILLA / 2.0 - radio_disco))
            jitter = min(JITTER_CELDA * CASILLA, holgura)
            centro_x = (columna + 0.5) * CASILLA + float(
                rng.uniform(-jitter, jitter)
            )
            centro_y = (fila + 0.5) * CASILLA + float(
                rng.uniform(-jitter, jitter)
            )
            angulo = float(rng.uniform(0.0, 360.0))
            destino = _estadisticas_cuadrado(plano, centro_x, centro_y, diametro / 2.0)
            armonizado = _transferir_luz(
                sprite["imagen"], sprite["fondo"], destino,
                fuerza=float(rng.uniform(0.55, 0.85)),
            )
            _sombra(plano, centro_x, centro_y, diametro, rng)
            _pegar_ficha(plano, armonizado, centro_x, centro_y, diametro, angulo)
            clases_casilla[fila][columna] = clase_nombre
            ocupacion.append((centro_x, centro_y, diametro / 2.0, clase_nombre))
            rotaciones_imagen.append({
                "imagen": None,
                "casilla": f"{chr(ord('a') + columna)}{8 - fila}",
                "clase": clase_nombre,
                "sprite": sprite["archivo"],
                "angulo": f"{angulo:.1f}",
                "diametro": f"{diametro:.1f}",
                "centro_x": f"{centro_x:.1f}",
                "centro_y": f"{centro_y:.1f}",
            })

        # Anti-superposicion: una ficha por casilla, separadas por el paso.
        for i in range(len(ocupacion)):
            for j in range(i + 1, len(ocupacion)):
                distancia = math.hypot(ocupacion[i][0] - ocupacion[j][0],
                                       ocupacion[i][1] - ocupacion[j][1])
                ratio = distancia / max(ocupacion[i][2] + ocupacion[j][2], 1e-6)
                min_ratio_distancia = min(min_ratio_distancia, ratio)
                if ratio < 0.9:
                    violaciones_superposicion += 1

        # Proyeccion al marco del perfil.
        destino_marco = _colocar(LADO_PLANO, LADO_PLANO, perfil["ancho"],
                                 perfil["alto"], rng, perfil, None)
        origen = np.array(
            [[0, 0], [LADO_PLANO, 0], [LADO_PLANO, LADO_PLANO], [0, LADO_PLANO]],
            dtype="float32",
        )
        homografia = cv2.getPerspectiveTransform(origen, destino_marco)
        deformado = cv2.warpPerspective(
            plano, homografia, (perfil["ancho"], perfil["alto"]),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mascara = cv2.warpPerspective(
            np.full((LADO_PLANO, LADO_PLANO), 255, np.uint8), homografia,
            (perfil["ancho"], perfil["alto"]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mascara = cv2.GaussianBlur(mascara, (0, 0), 0.8).astype("float32") / 255.0
        mantel = _fondo_real(rng, perfil["ancho"], perfil["alto"], manteles)
        imagen = (
            deformado.astype("float32") * mascara[..., None]
            + mantel.astype("float32") * (1.0 - mascara[..., None])
        )
        imagen = _fotometria(np.clip(imagen, 0, 255).astype(np.uint8), rng, perfil)

        # Cajas YOLO en el marco completo.
        cajas = []
        for centro_x, centro_y, radio, clase_nombre in ocupacion:
            esquinas = np.array([
                [centro_x - radio, centro_y - radio],
                [centro_x + radio, centro_y - radio],
                [centro_x + radio, centro_y + radio],
                [centro_x - radio, centro_y + radio],
            ], dtype="float32")
            proyectadas = cv2.perspectiveTransform(
                esquinas.reshape(-1, 1, 2), homografia
            ).reshape(-1, 2)
            x0 = max(0.0, float(proyectadas[:, 0].min()))
            x1 = min(perfil["ancho"] - 1.0, float(proyectadas[:, 0].max()))
            y0 = max(0.0, float(proyectadas[:, 1].min()))
            y1 = min(perfil["alto"] - 1.0, float(proyectadas[:, 1].max()))
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            clase = clases.index(clase_nombre)
            cajas.append((x0, y0, x1, y1, clase))
            conteo_cajas[clase_nombre] += 1

        contadores[perfil_nombre] += 1
        numero = contadores[perfil_nombre]
        nombre = f"compuesto_{perfil_nombre}_{numero:04d}"
        calidad = int(round(float(rng.uniform(*perfil["jpeg"]))))
        cv2.imwrite(str(imagenes_dir / f"{nombre}.jpg"), imagen,
                    [cv2.IMWRITE_JPEG_QUALITY, calidad])

        lineas = [_linea_yolo(caja, perfil["ancho"], perfil["alto"])
                  for caja in cajas]
        with open(etiquetas_dir / f"{nombre}.txt", "w", encoding="utf-8") as archivo:
            for linea in lineas:
                archivo.write(linea + "\n")

        for fila_rotacion in rotaciones_imagen:
            fila_rotacion["imagen"] = f"{nombre}.jpg"
        filas_rotacion.extend(rotaciones_imagen)

        filas_manifiesto.append({
            "imagen": f"{nombre}.jpg",
            "perfil": perfil_nombre,
            "modo": modo,
            "n_fichas": len(cajas),
            "semilla": semilla,
            "ancho": perfil["ancho"],
            "alto": perfil["alto"],
            "fondo": nombre_fondo,
            "homografia_recorte": matriz_a_texto(np.linalg.inv(homografia)),
        })
        conteo_perfil[perfil_nombre] += 1
        conteo_modo[modo] += 1

        # Recortes de casilla: solo los tableros elegidos, para no llenar el disco.
        guardar_casillas = not argumentos.sin_casillas and (
            numero == 1
            or (argumentos.casillas_cada > 0
                and numero % argumentos.casillas_cada == 0)
        )
        if guardar_casillas:
            recortes = _recortar_casillas(imagen, homografia, clases_casilla)
            for posicion, (clase_casilla, recorte) in enumerate(recortes):
                fila, columna = divmod(posicion, 8)
                archivo = casillas_dir / clase_casilla / (
                    f"{nombre}_{chr(ord('a') + columna)}{8 - fila}.jpg"
                )
                cv2.imwrite(str(archivo), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                filas_casillas.append({
                    "imagen": f"{nombre}.jpg",
                    "casilla": f"{chr(ord('a') + columna)}{8 - fila}",
                    "clase": clase_casilla,
                    "fila": fila,
                    "columna": columna,
                    "perfil": perfil_nombre,
                })
            hojas[perfil_nombre] = recortes

        if anotadas < 3 and not argumentos.sin_previews:
            # 2 del perfil ESP y 1 del celular: el revisor ve los dos.
            cuota = 2 if perfil_nombre == "esp" else 1
            if conteo_anotadas[perfil_nombre] < cuota:
                titulo = (f"{nombre}  perfil={perfil_nombre}  modo={modo}  "
                          f"fichas={len(cajas)}")
                cv2.imwrite(
                    str(PREVIEWS / f"compuesto_muestra_{anotadas + 1}.jpg"),
                    _anotar(imagen, cajas, clases, titulo),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                previews.append(PREVIEWS / f"compuesto_muestra_{anotadas + 1}.jpg")
                anotadas += 1
                conteo_anotadas[perfil_nombre] += 1

        if (indice + 1) % 50 == 0 or indice + 1 == argumentos.n:
            print(f"  {indice + 1}/{argumentos.n} tableros", flush=True)

    # ---- salidas de texto ----
    manifiesto, csv_casillas, csv_rotacion = salidas_texto(
        salida, filas_manifiesto, filas_casillas, filas_rotacion,
        bool(argumentos.agregar),
    )

    if not argumentos.sin_previews:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        for indice_hoja, perfil_nombre in enumerate(sorted(hojas)):
            numero_hoja = 4 + indice_hoja
            ruta = PREVIEWS / f"compuesto_muestra_{numero_hoja}.jpg"
            cv2.imwrite(str(ruta), _hoja_casillas(
                hojas[perfil_nombre], f"casillas {perfil_nombre}"), 
                [cv2.IMWRITE_JPEG_QUALITY, 90])
            previews.append(ruta)

    # ---- informe ----
    segundos = time.time() - inicio
    tamano = sum(p.stat().st_size for p in salida.rglob("*") if p.is_file())
    imagenes = len(list(imagenes_dir.glob("*.jpg")))
    etiquetas = len(list(etiquetas_dir.glob("*.txt")))
    angulos = np.array([float(f["angulo"]) for f in filas_rotacion])
    total_cajas = sum(conteo_cajas.values())

    print()
    print(f"listo: {argumentos.n} tableros en {segundos:.1f} s "
          f"({segundos / max(argumentos.n, 1) * 1000:.0f} ms/tablero)")
    print(f"  imagenes: {imagenes} -> {imagenes_dir}")
    print(f"  etiquetas: {etiquetas} -> {etiquetas_dir}"
          f"  ({'OK' if imagenes == etiquetas else 'DESCUADRE'})")
    print(f"  por perfil: {dict(conteo_perfil)}")
    print(f"  por modo: {dict(conteo_modo)}")
    print(f"  tamano del dataset: {tamano / 1e6:.1f} MB")
    print(f"  cajas por clase (total {total_cajas}): "
          f"{dict(sorted(conteo_cajas.items()))}")
    print(f"  cajas por imagen: {total_cajas / max(len(filas_manifiesto), 1):.1f}")
    print(f"  recortes 100x100: {len(filas_casillas)} -> {casillas_dir}")
    print(f"  anti-superposicion: distancia minima entre centros / suma de "
          f"radios = {min_ratio_distancia:.2f} (umbral 0.90), "
          f"violaciones = {violaciones_superposicion}")
    print(f"  rotacion: {len(angulos)} fichas, min {angulos.min():.1f}, "
          f"max {angulos.max():.1f}, media {angulos.mean():.1f}, "
          f"llenado de 8 bins = {np.histogram(angulos, bins=8, range=(0, 360))[0].tolist()}")
    print(f"  manifiesto: {manifiesto}")
    print(f"  etiquetas de casilla: {csv_casillas}")
    print(f"  rotaciones: {csv_rotacion}")
    print(f"  previews: {[p.name for p in previews]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
