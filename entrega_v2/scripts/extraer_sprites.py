"""Extrae sprites RGBA de las fichas REALES a partir de los recortes rectificados.

Entrada
-------
``dataset/reales/recortes/800x800`` (los recortes ya rectificados del area
jugable) y ``dataset/reales/etiquetas_clasificador.csv`` (lo que produce
``autoetiquetar_reales.py``). Solo se usan los recortes de ``01_inicial`` con
``estado = auto``: en esos la posicion es exactamente la inicial y por lo tanto
cada casilla ocupada se sabe de que clase es. Los recortes de
``05_tablero_vacio`` no se tocan aqui; el compositor los usa como fondo real.

Como se separa la ficha del tablero
-----------------------------------
El tablero impreso es un damero azul/gris; las fichas son discos neutros
(blancos o negros) con un simbolo. Una simple umbralizacion por brillo falla en
dos casos reales: la ficha BLANCA sobre la casilla GRIS (contraste ~20) y la
ficha NEGRA sobre la casilla AZUL (el azul es oscuro en el canal rojo).

Por eso el fondo se modela y se resta. En la vista rectificada cada casilla mide
exactamente 100 px y la iluminacion cambia de forma suave dentro de una sesion,
asi que un polinomio cuadratico por canal y por PARIDAD de casilla (clara /
oscura) ajustado con los anillos de borde de las 32 casillas de esa paridad
predice el color que tendria esa casilla VACIA en cada pixel. La diferencia en
Lab entre la foto y ese modelo es alta en la ficha y ~6 en una casilla vacia
(medido: fichas 22-53 de mediana, vacias 6).

Sobre ese mapa de diferencia se corre HoughCircles. El tablero rectificado tiene
una sola escala, asi que todas las fichas de una foto comparten radio: se toma
la mediana de los radios detectados como radio del tablero y se usa el mismo en
las 32 casillas. Si Hough no ve una casilla, el centro se busca maximizando la
respuesta de un anillo del radio conocido en una ventana de +-16 px.

El sprite se recorta con mascara circular y feather de ~2 px. Si el anillo
exterior trae color de tablero (saturacion alta del azul impreso), el radio se
encoge 1.5 px y se vuelve a medir, hasta 4 veces; el numero de ajustes queda en
el manifiesto. Ese es el caso que el pedido llama "sprite con fondo de casilla
incluido cerca del borde".

Salidas
-------
``dataset/reales_compuesto/sprites/sprite_<clase>_<indice>.png``
      PNG RGBA por ficha (el indice es global y estable por clase).
``dataset/reales_compuesto/sprites/sprites_manifesto.csv``
      archivo fuente, casilla, clase, radio, centro y ajustes.
``assets/previews/sprite_muestra.jpg``
      12 sprites al azar para revision humana.

Uso
---
    python extraer_sprites.py
    python extraer_sprites.py --limite 3 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from aumentar_reales import leer_etiquetas
from comun import cargar_clases

RAIZ = Path(__file__).resolve().parent.parent
RECORTES = RAIZ / "dataset" / "reales" / "recortes" / "800x800"
ETIQUETAS = RAIZ / "dataset" / "reales" / "etiquetas_clasificador.csv"
SALIDA = RAIZ / "dataset" / "reales_compuesto" / "sprites"
PREVIEWS = RAIZ / "assets" / "previews"

LADO = 800
CASILLA = LADO // 8
UMBRAL_DIFERENCIA = 24.0   # Lab: por encima esta la ficha, por debajo la casilla
RADIO_MIN, RADIO_MAX = 30, 52
BUSQUEDA_CENTRO = 16       # px de tolerancia alrededor del centro de la casilla
FEATHER = 0.8              # sigma del desenfoque del alpha (~2 px visibles)
AJUSTE_RADIO = 1.5
AJUSTES_MAXIMOS = 4
SATURACION_TABLERO = 38.0  # el azul impreso supera 55-90; los discos < 20

CAMPOS = [
    "archivo", "fuente", "casilla", "clase", "radio", "radio_tablero",
    "ajustes", "fuga_fondo", "centro_x", "centro_y", "fondo_l", "fondo_a",
    "fondo_b", "fondo_l_std", "recortado",
]


# ---------------------------------------------------------------------------
# Modelo de fondo del tablero
# ---------------------------------------------------------------------------

def modelo_paridad(imagen: np.ndarray) -> dict[int, np.ndarray]:
    """Coeficientes del polinomio cuadratico de fondo por paridad de casilla.

    Se muestrean solo anillos de borde (a 3-6 px del borde de cada casilla) para
    que las fichas no contaminen el ajuste. La paridad (fila+columna) % 2 separa
    las casillas claras de las oscuras, que son dos colores distintos.
    """
    alto, ancho = imagen.shape[:2]
    puntos: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    valores: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for fila in range(8):
        for columna in range(8):
            x0, y0 = columna * CASILLA, fila * CASILLA
            par = (fila + columna) % 2
            for margen in (3, 4, 5, 6):
                for x in np.arange(x0 + margen, x0 + CASILLA - margen, 3):
                    for y in (y0 + margen, y0 + CASILLA - 1 - margen):
                        puntos[par].append((int(x), int(y)))
                        valores[par].append(imagen[int(y), int(x)].astype("float32"))
                for y in np.arange(y0 + margen + 3, y0 + CASILLA - margen - 3, 3):
                    for x in (x0 + margen, x0 + CASILLA - 1 - margen):
                        puntos[par].append((int(x), int(y)))
                        valores[par].append(imagen[int(y), int(x)].astype("float32"))

    modelos: dict[int, np.ndarray] = {}
    for par in (0, 1):
        coords = np.array(puntos[par], dtype="float64")
        muestras = np.array(valores[par], dtype="float64")
        xn, yn = coords[:, 0] / ancho, coords[:, 1] / alto
        diseno = np.stack(
            [np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=1
        )
        coeficientes, *_ = np.linalg.lstsq(diseno, muestras, rcond=None)
        modelos[par] = coeficientes
    return modelos


def fondo_predicho(imagen: np.ndarray, modelos: dict[int, np.ndarray]) -> np.ndarray:
    """Color BGR que tendria el tablero VACIO en cada pixel."""
    alto, ancho = imagen.shape[:2]
    yy, xx = np.mgrid[0:alto, 0:ancho].astype("float64")
    xn, yn = xx / ancho, yy / alto
    diseno = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=-1)
    predicho = np.zeros((alto, ancho, 3), dtype="float32")
    paridad = ((xx // CASILLA + yy // CASILLA).astype(int)) % 2
    for par in (0, 1):
        ajuste = np.tensordot(diseno, modelos[par], axes=([-1], [0]))
        predicho[paridad == par] = ajuste[paridad == par]
    return predicho


def mapa_diferencia(imagen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distancia Lab a la casilla vacia esperada y el fondo predicho en BGR."""
    predicho = fondo_predicho(imagen, modelo_paridad(imagen))
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB).astype("float32")
    lab_fondo = cv2.cvtColor(predicho.astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")
    return np.linalg.norm(lab - lab_fondo, axis=2), predicho


# ---------------------------------------------------------------------------
# Locales de las fichas
# ---------------------------------------------------------------------------

def _detectar_circulos(diferencia: np.ndarray, celdas: set[tuple[int, int]]
                       ) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Un circulo por casilla ocupada, eligiendo el mas cercano al centro."""
    realzada = np.clip(diferencia * 3.0, 0.0, 255.0).astype(np.uint8)
    realzada = cv2.GaussianBlur(realzada, (5, 5), 0)
    circulos = cv2.HoughCircles(
        realzada, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=60, param2=18, minRadius=RADIO_MIN, maxRadius=RADIO_MAX,
    )
    candidatos: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    if circulos is not None:
        for x, y, radio in np.round(circulos[0]).astype(int):
            celda = (int(y) // CASILLA, int(x) // CASILLA)
            if celda not in celdas:
                continue
            centro_x = (celda[1] + 0.5) * CASILLA
            centro_y = (celda[0] + 0.5) * CASILLA
            distancia = math.hypot(x - centro_x, y - centro_y)
            candidatos.setdefault(celda, []).append(
                (float(distancia), float(x), float(y), float(radio))
            )

    elegidos: dict[tuple[int, int], tuple[float, float, float]] = {}
    for celda, lista in candidatos.items():
        lista.sort(key=lambda t: t[0])
        distancia, x, y, radio = lista[0]
        if distancia <= BUSQUEDA_CENTRO:
            elegidos[celda] = (x, y, radio)
    return elegidos


def _mapa_borde(diferencia: np.ndarray) -> np.ndarray:
    """Modulo del gradiente del mapa de diferencia: maximo en el borde del disco."""
    suave = cv2.GaussianBlur(diferencia, (0, 0), 1.5)
    gradiente_x = cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(suave, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gradiente_x, gradiente_y)


def _kernel_anillo(radio: float, grosor: float = 1.6) -> tuple[np.ndarray, int]:
    lado = int(math.ceil(2.0 * (radio + 3.0 * grosor))) + 3
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - lado / 2.0 + 0.5, yy - lado / 2.0 + 0.5)
    kernel = np.exp(-0.5 * ((distancia - radio) / grosor) ** 2)
    kernel[kernel < 0.02] = 0.0
    return (kernel / kernel.sum()).astype("float32"), lado


def _centro_mascara(diferencia: np.ndarray, fila: int, columna: int
                    ) -> tuple[float, float] | None:
    """Centro de la ficha por el centroide de su mancha en el mapa de diferencia.

    En una ficha OSCURA la mancha es el disco entero y el centroide es el centro
    exacto. En una ficha CLARA sobre casilla gris la mancha es solo el simbolo,
    pero los simbolos se imprimen centrados en el disco. Se descartan los
    componentes que tocan el marco de la casilla: esos son las lineas de la
    rejilla, no la ficha.
    """
    x0, y0 = columna * CASILLA, fila * CASILLA
    celda = diferencia[y0:y0 + CASILLA, x0:x0 + CASILLA]
    mascara = (celda > UMBRAL_DIFERENCIA).astype(np.uint8)
    mascara[:6] = 0
    mascara[-6:] = 0
    mascara[:, :6] = 0
    mascara[:, -6:] = 0
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara)
    candidatos = [i for i in range(1, cantidad) if estadisticas[i, 4] > 200]
    if not candidatos:
        return None
    indice = max(candidatos, key=lambda j: estadisticas[j, 4])
    x, y, ancho, alto = estadisticas[indice, :4]
    if x <= 1 or y <= 1 or x + ancho >= CASILLA - 1 or y + alto >= CASILLA - 1:
        return None
    ys, xs = np.nonzero(etiquetas == indice)
    return (float(x0 + xs.mean()), float(y0 + ys.mean()))


def _centros(imagen: np.ndarray, diferencia: np.ndarray,
             celdas: dict[tuple[int, int], str]) -> tuple[dict, float]:
    """Centro y radio de cada ficha. Devuelve (centros, radio del tablero).

    Hough da el radio fiable: en una foto rectificada todas las fichas comparten
    escala, asi que la mediana de los radios detectados es el radio del tablero.
    El centro sale del centroide de la mancha de diferencia; si esa mancha no es
    un componente limpio (ficha clara sobre casilla clara) se cae al maximo de un
    anillo del radio conocido sobre el gradiente. El radio del sprite queda 2 px
    adentro del medido para no arrastrar el borde de la casilla.
    """
    detectados = _detectar_circulos(diferencia, set(celdas))
    if not detectados:
        raise ValueError("Hough no encontro ninguna ficha en el recorte")
    radio_tablero = float(np.median([radio for _, _, radio in detectados.values()]))
    radio_sprite = radio_tablero - 2.0

    kernel, _ = _kernel_anillo(radio_tablero)
    respuesta = cv2.filter2D(_mapa_borde(diferencia), -1, kernel,
                             borderType=cv2.BORDER_REPLICATE)
    margen = int(math.ceil(radio_tablero)) + 2

    centros: dict[tuple[int, int], tuple[float, float, float]] = {}
    for fila, columna in celdas:
        centro = _centro_mascara(diferencia, fila, columna)
        if centro is None:
            centro_x = int(round((columna + 0.5) * CASILLA))
            centro_y = int(round((fila + 0.5) * CASILLA))
            x0 = max(margen, centro_x - BUSQUEDA_CENTRO)
            x1 = min(LADO - margen, centro_x + BUSQUEDA_CENTRO + 1)
            y0 = max(margen, centro_y - BUSQUEDA_CENTRO)
            y1 = min(LADO - margen, centro_y + BUSQUEDA_CENTRO + 1)
            ventana = respuesta[y0:y1, x0:x1]
            indice = int(np.argmax(ventana))
            desplazamiento_y, desplazamiento_x = divmod(indice, ventana.shape[1])
            if float(ventana[desplazamiento_y, desplazamiento_x]) > 0.0:
                centro = (float(x0 + desplazamiento_x), float(y0 + desplazamiento_y))
            else:
                centro = (float(centro_x), float(centro_y))
        centros[(fila, columna)] = (centro[0], centro[1], radio_sprite)
    return centros, radio_tablero


# ---------------------------------------------------------------------------
# Recorte del sprite
# ---------------------------------------------------------------------------

def _recorte_replicado(imagen: np.ndarray, x0: int, y0: int, lado: int) -> np.ndarray:
    """Recorta un cuadrado replicando el borde (nunca rellena con negro)."""
    alto, ancho = imagen.shape[:2]
    izquierda, derecha = max(0, -x0), max(0, x0 + lado - ancho)
    arriba, abajo = max(0, -y0), max(0, y0 + lado - alto)
    rellenada = cv2.copyMakeBorder(
        imagen, arriba, abajo, izquierda, derecha, cv2.BORDER_REPLICATE
    )
    return rellenada[y0 + arriba:y0 + arriba + lado, x0 + izquierda:x0 + izquierda + lado]


def _alpha(lado: int, centro_x: float, centro_y: float, radio: float) -> np.ndarray:
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - centro_x, yy - centro_y)
    duro = np.clip(radio - distancia + 0.5, 0.0, 1.0)
    return cv2.GaussianBlur(duro, (0, 0), FEATHER)


def _fuga_tablero(recorte: np.ndarray, alfa: np.ndarray, lado: int,
                  centro_x: float, centro_y: float, radio: float,
                  fondo: np.ndarray | None) -> float:
    """Fraccion del anillo exterior que todavia parece TABLERO, no ficha.

    Se compara cada pixel del anillo contra el color que tendria esa casilla
    vacia (``fondo_predicho``) y contra el color del disco. Si el pixel esta mas
    cerca del tablero que del disco, es fondo que se colo en el sprite. Detecta
    tanto el azul impreso como el gris de una casilla clara; el umbral por
    saturacion solo veia el azul.
    """
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - centro_x, yy - centro_y)
    anillo = (distancia >= 0.80 * radio) & (distancia <= 0.96 * radio) & (alfa > 0.5)
    if not anillo.any():
        return 0.0
    interior = distancia < 0.55 * radio
    lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB).astype("float32")
    if fondo is not None and interior.any() and anillo.any():
        lab_fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2LAB).astype("float32")
        color_disco = np.median(lab[interior], axis=0)
        distancia_fondo = np.linalg.norm(lab - lab_fondo, axis=2)
        distancia_disco = np.linalg.norm(lab - color_disco, axis=2)
        fuga = float((distancia_fondo[anillo] < distancia_disco[anillo]).mean())
        if fuga > 0.0:
            return fuga
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV).astype("float32")
    saturacion = hsv[..., 1][anillo]
    return float((saturacion > SATURACION_TABLERO).mean())


def _estadisticas_cuadrado(imagen: np.ndarray, centro_x: float, centro_y: float,
                           radio: float) -> tuple[float, float, float, float]:
    """Lab del tablero alrededor de la ficha (el cuadrado bajo ella).

    Solo se muestrea el marco de la casilla, fuera del disco: es el color del
    tablero con esa iluminacion y sirve para transferir la luz al componer.
    """
    x0 = int(round(centro_x - CASILLA / 2))
    y0 = int(round(centro_y - CASILLA / 2))
    x1, y1 = x0 + CASILLA, y0 + CASILLA
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(LADO, x1), min(LADO, y1)
    celda = imagen[y0:y1, x0:x1]
    alto, ancho = celda.shape[:2]
    yy, xx = np.mgrid[0:alto, 0:ancho].astype("float32")
    distancia = np.hypot(xx - (centro_x - x0), yy - (centro_y - y0))
    marco = (distancia > radio + 2.0) & (distancia < 0.5 * CASILLA)
    if marco.sum() < 30:
        marco = np.ones((alto, ancho), bool)
    lab = cv2.cvtColor(celda, cv2.COLOR_BGR2LAB).astype("float32")
    valores = lab[marco]
    mediana = np.median(valores, axis=0)
    return (float(mediana[0]), float(mediana[1]), float(mediana[2]),
            float(np.std(valores[:, 0])))


def extraer_uno(imagen: np.ndarray, fondo: np.ndarray, centro_x: float,
                centro_y: float, radio: float) -> tuple[np.ndarray, int, float]:
    """Recorta una ficha con alpha circular, encogiendo si arrastra tablero."""
    ajustes = 0
    fuga = 0.0
    while True:
        lado = max(8, int(math.ceil(2.0 * radio)) + 2)
        x0 = int(round(centro_x - lado / 2))
        y0 = int(round(centro_y - lado / 2))
        recorte = _recorte_replicado(imagen, x0, y0, lado)
        fondo_recorte = _recorte_replicado(fondo, x0, y0, lado)
        alfa = _alpha(lado, centro_x - x0, centro_y - y0, radio)
        fuga = _fuga_tablero(recorte, alfa, lado, centro_x - x0,
                             centro_y - y0, radio, fondo_recorte)
        if fuga <= 0.20 or ajustes >= AJUSTES_MAXIMOS:
            break
        radio -= AJUSTE_RADIO
        ajustes += 1
    sprite = np.dstack([recorte, (alfa * 255.0).astype(np.uint8)])
    return sprite, ajustes, fuga


def recortado(imagen: np.ndarray, centro_x: float, centro_y: float,
              radio: float) -> bool:
    """True si el disco se sale del recorte (fichas de las esquinas)."""
    return (centro_x - radio < 1.0 or centro_y - radio < 1.0
            or centro_x + radio > LADO - 2.0 or centro_y + radio > LADO - 2.0)


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _preview(sprites: list[np.ndarray], etiquetas: list[str]) -> np.ndarray:
    """Fila de 12 sprites sobre damero gris para que se vea el alpha."""
    lado = 110
    celdas = []
    for indice, (sprite, etiqueta) in enumerate(zip(sprites, etiquetas)):
        lado_sprite = sprite.shape[0]
        lienzo = np.zeros((lado, lado, 3), np.uint8)
        paso = 18
        for fila in range(0, lado, paso):
            for columna in range(0, lado, paso):
                color = 92 if ((fila // paso) + (columna // paso)) % 2 else 132
                lienzo[fila:fila + paso, columna:columna + paso] = color
        escala = (lado - 8) / lado_sprite
        reducido = cv2.resize(sprite, (int(lado_sprite * escala),
                                       int(lado_sprite * escala)),
                              interpolation=cv2.INTER_AREA)
        dy = (lado - reducido.shape[0]) // 2
        dx = (lado - reducido.shape[1]) // 2
        bgr = reducido[..., :3].astype("float32")
        alfa = reducido[..., 3:4].astype("float32") / 255.0
        base = lienzo[dy:dy + reducido.shape[0], dx:dx + reducido.shape[1]].astype("float32")
        lienzo[dy:dy + reducido.shape[0], dx:dx + reducido.shape[1]] = np.clip(
            bgr * alfa + base * (1.0 - alfa), 0, 255
        ).astype(np.uint8)
        cv2.putText(lienzo, etiqueta, (3, lado - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(lienzo, etiqueta, (3, lado - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (255, 255, 255), 1, cv2.LINE_AA)
        celdas.append(lienzo)
        if len(celdas) == 12:
            break
    while len(celdas) < 12:
        celdas.append(np.full((lado, lado, 3), 40, np.uint8))
    return np.hstack(celdas)


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Extrae sprites RGBA de las fichas reales autoetiquetadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--recortes", default=str(RECORTES))
    analizador.add_argument("--etiquetas", default=str(ETIQUETAS))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--limite", type=int, default=0,
                            help="usar solo N recortes iniciales (0 = todos)")
    analizador.add_argument("--sin-previews", action="store_true")
    argumentos = analizador.parse_args(argv)

    clases = cargar_clases()
    etiquetado = leer_etiquetas(Path(argumentos.etiquetas))
    iniciales = {
        nombre: celdas for nombre, celdas in etiquetado.items()
        if nombre.startswith("01_inicial")
    }
    if not iniciales:
        raise SystemExit(
            "No hay recortes de 01_inicial con 64 casillas 'auto'. "
            "Corre antes autoetiquetar_reales.py"
        )
    nombres = sorted(iniciales)
    if argumentos.limite > 0:
        nombres = nombres[:argumentos.limite]

    base = Path(argumentos.recortes)
    salida = Path(argumentos.salida)
    salida.mkdir(parents=True, exist_ok=True)

    inicio = time.time()
    filas: list[dict] = []
    por_clase: Counter = Counter()
    radios: Counter = Counter()
    contadores: Counter = Counter()
    ajustes_totales = 0
    recortados = 0
    descartados = 0
    muestras: list[tuple[np.ndarray, str]] = []

    for nombre in nombres:
        imagen = cv2.imread(str(base / nombre),
                            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            print(f"  AVISO: no pude leer {nombre}")
            continue
        if imagen.shape[0] != LADO or imagen.shape[1] != LADO:
            imagen = cv2.resize(imagen, (LADO, LADO), interpolation=cv2.INTER_AREA)

        celdas = {
            divmod(indice, 8): clase
            for indice, clase in iniciales[nombre].items()
            if clase != "vacia"
        }
        diferencia, fondo_predicho = mapa_diferencia(imagen)
        try:
            centros, radio_tablero = _centros(imagen, diferencia, celdas)
        except ValueError as error:
            print(f"  AVISO: {nombre}: {error}")
            continue
        radios[round(radio_tablero, 1)] += 1

        for (fila, columna), clase in sorted(celdas.items()):
            centro_x, centro_y, radio = centros[(fila, columna)]
            sprite, ajustes, fuga = extraer_uno(
                imagen, fondo_predicho, centro_x, centro_y, radio
            )
            ajustes_totales += ajustes
            if fuga > 0.40:
                descartados += 1
                continue
            fondo_l, fondo_a, fondo_b, fondo_l_std = _estadisticas_cuadrado(
                imagen, centro_x, centro_y, radio
            )
            es_recortado = recortado(imagen, centro_x, centro_y, radio)
            if es_recortado:
                recortados += 1
            indice = contadores[clase]
            contadores[clase] += 1
            archivo = f"sprite_{clase}_{indice:03d}.png"
            cv2.imwrite(str(salida / archivo), sprite)
            por_clase[clase] += 1
            filas.append({
                "archivo": archivo,
                "fuente": nombre,
                "casilla": f"{chr(ord('a') + columna)}{8 - fila}",
                "clase": clase,
                "radio": f"{radio:.1f}",
                "radio_tablero": f"{radio_tablero:.1f}",
                "ajustes": ajustes,
                "fuga_fondo": f"{fuga:.3f}",
                "centro_x": f"{centro_x:.1f}",
                "centro_y": f"{centro_y:.1f}",
                "fondo_l": f"{fondo_l:.1f}",
                "fondo_a": f"{fondo_a:.1f}",
                "fondo_b": f"{fondo_b:.1f}",
                "fondo_l_std": f"{fondo_l_std:.1f}",
                "recortado": int(es_recortado),
            })
            if len(muestras) < 48:
                muestras.append((sprite, clase))

        print(f"  {nombre}: {len(celdas)} fichas, radio {radio_tablero:.1f} px",
              flush=True)

    manifiesto = salida / "sprites_manifesto.csv"
    with open(manifiesto, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)

    preview = None
    if not argumentos.sin_previews and muestras:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        elegidas = [int(i) for i in np.random.default_rng(0).choice(
            len(muestras), size=min(12, len(muestras)), replace=False
        )] if len(muestras) >= 12 else list(range(len(muestras)))
        preview = PREVIEWS / "sprite_muestra.jpg"
        cv2.imwrite(
            str(preview),
            _preview([muestras[i][0] for i in elegidas],
                     [muestras[i][1] for i in elegidas]),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    segundos = time.time() - inicio
    valores_radio = [float(f["radio"]) for f in filas]
    print()
    print(f"listo: {len(filas)} sprites de {len(nombres)} recortes en {segundos:.1f} s")
    print(f"  sprites: {salida}")
    print(f"  manifiesto: {manifiesto}")
    print(f"  por clase: {dict(sorted(por_clase.items()))}")
    vacias = [clase for clase in clases if por_clase.get(clase, 0) < 10]
    if vacias:
        print(f"  AVISO: clases con menos de 10 sprites: "
              f"{[(c, por_clase.get(c, 0)) for c in vacias]}")
    if valores_radio:
        print(f"  radio del sprite (px en el recorte 800): "
              f"min {min(valores_radio):.1f}, mediana "
              f"{float(np.median(valores_radio)):.1f}, max {max(valores_radio):.1f}")
    print(f"  radios por tablero: {dict(sorted(radios.items()))}")
    print(f"  ajustes de radio aplicados: {ajustes_totales}")
    print(f"  sprites con disco cortado por el borde del recorte: {recortados}")
    print(f"  descartados por fuga de fondo: {descartados}")
    print(f"  preview: {preview if preview else 'omitida'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
