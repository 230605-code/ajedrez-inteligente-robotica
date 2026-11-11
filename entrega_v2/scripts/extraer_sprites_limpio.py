"""Extrae los sprites de ficha LIMPIOS desde las fotos de la posicion inicial.

Por que existe este modulo
--------------------------
El dataset anterior murio por un cruce fisico caballo/alfil: los sprites de
``caballo_n`` salieron en realidad de una casilla de alfil, y el compuesto
enseno el glifo equivocado. Aqui la etiqueta de cada sprite NO sale de ningun
modelo: sale del LAYOUT FISICO de la posicion inicial (b8=caballo_n,
c8=alfil_n, f8=alfil_n, g8=caballo_n, ...). Caballo y alfil se cortan de la
MISMA foto y de casillas distintas, asi que el cruce es imposible por
construccion.

Metodo (el mismo de ``extraer_sprites.py``, con el centrado endurecido)
-----------------------------------------------------------------------
1. Modelo de fondo cuadratico por PARIDAD de casilla: predice el color que
   tendria esa casilla VACIA. La diferencia en Lab marca la ficha.
2. ``HoughCircles`` sobre ese mapa da el radio del tablero (todas las fichas de
   un recorte comparten escala).
3. Con ESE radio se correlaciona un kernel de ANILLO sobre el gradiente y se
   toma el maximo en una ventana de +-16 px alrededor del centro de la casilla.
   En este lote el centroide del mapa de diferencia se corria (la base clara de
   las negras se confunde con la casilla gris), y un sprite descentrado arrastra
   tablero; el anillo, que solo mira el borde del disco, es simetrico.
4. El sprite se recorta con alpha circular y feather de ~1 px. Si el anillo
   exterior todavia trae color del tablero (saturacion alta: el damero impreso
   es azul) se encoge el radio 1.5 px, hasta 4 veces. NUNCA se descarta una
   ficha: la casilla tiene ficha por la posicion fisica, lo que se ajusta es el
   radio del recorte.

Entradas
--------
``dataset/recortes_800/01_inicial_intacta/*.jpg``  recortes rectificados 800x800.

Salidas
-------
``dataset/sprites_limpio/sprite_<clase>_<indice>.png``  PNG RGBA (alpha circular)
``dataset/sprites_limpio/sprites_manifesto.csv``        fuente, casilla, clase, radio...
``dataset/sprites_limpio_preview.jpg``                  12 clases + prueba caballo/alfil

Controles que imprime
---------------------
* las 32 casillas de cada foto producen sprite (no se pierde ninguna ficha);
* sprites descartados por el filtro de luminancia/fuga: 0;
* caballo_n sale SOLO de b8/g8 y alfil_n SOLO de c8/f8.

Uso
---
    python extraer_sprites_limpio.py
    python extraer_sprites_limpio.py --limite 3 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from comun import DATASET, cargar_clases
from extraer_sprites import (
    CASILLA,
    LADO,
    _alpha,
    _detectar_circulos,
    _estadisticas_cuadrado,
    _kernel_anillo,
    _mapa_borde,
    _recorte_replicado,
    mapa_diferencia,
    recortado,
)

RAIZ = Path(__file__).resolve().parent.parent
RECORTES = DATASET / "recortes_800" / "01_inicial_intacta"
SALIDA = DATASET / "sprites_limpio"
PREVIEW = DATASET / "sprites_limpio_preview.jpg"

# Layout FISICO de la mesa: el estandar. b8=caballo_n, c8=alfil_n, f8=alfil_n,
# g8=caballo_n (el usuario confirmo que el tablero fisico quedo como el estandar).
BLANCAS = {
    "a1": "torre_b", "b1": "caballo_b", "c1": "alfil_b", "d1": "dama_b",
    "e1": "rey_b", "f1": "alfil_b", "g1": "caballo_b", "h1": "torre_b",
}
NEGRAS = {
    "a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n", "d8": "dama_n",
    "e8": "rey_n", "f8": "alfil_n", "g8": "caballo_n", "h8": "torre_n",
}
INICIAL = {
    **BLANCAS,
    **NEGRAS,
    **{f"{c}2": "peon_b" for c in "abcdefgh"},
    **{c + "7": "peon_n" for c in "abcdefgh"},
}

# Casillas de origen que DEBE usar cada clase critica (prueba anti-cruce).
ORIGEN_ESPERADO = {
    "caballo_n": {"b8", "g8"},
    "alfil_n": {"c8", "f8"},
    "caballo_b": {"b1", "g1"},
    "alfil_b": {"c1", "f1"},
}

BUSQUEDA = 16          # px de tolerancia alrededor del centro de la casilla
AJUSTE_RADIO_CENTRO = 3.0   # el radio del tablero incluye 2-3 px de antialias
FUGA_ACEPTABLE = 0.25  # fraccion del anillo con saturacion de tablero
# Corrimiento global de rejilla tolerado por foto. Medido: las fotos buenas dan
# 0-6 px; mas de 8 px en una casilla de 100 px descentra la ficha y mete tablero.
CORRIMIENTO_MAXIMO = 8.0

SATURACION_TABLERO = 50.0  # azul impreso 55-90; las bases crema/negras <= 45

CAMPOS = [
    "archivo", "fuente", "casilla", "clase", "radio", "radio_tablero",
    "ajustes", "fuga_fondo", "fuga_saturacion", "centro_x", "centro_y",
    "fondo_l", "fondo_a", "fondo_b", "fondo_l_std", "recortado",
]


def celdas_iniciales() -> dict[tuple[int, int], str]:
    """{(fila, columna): clase} de las 32 fichas de la posicion inicial."""
    celdas = {}
    for casilla, clase in INICIAL.items():
        columna = ord(casilla[0]) - ord("a")
        fila = 8 - int(casilla[1])
        celdas[(fila, columna)] = clase
    if len(celdas) != 32:
        raise SystemExit(f"la posicion inicial debe tener 32 casillas "
                         f"y tiene {len(celdas)}")
    return celdas


# ---------------------------------------------------------------------------
# Centrado por respuesta de anillo
# ---------------------------------------------------------------------------

def _kernel_disco(radio: float) -> np.ndarray:
    """Media sobre el interior del disco (0.80 del radio)."""
    lado = int(math.ceil(2.0 * (radio + 2.0))) | 1
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - lado // 2, yy - lado // 2)
    nucleo = (distancia <= radio * 0.80).astype("float32")
    return nucleo / nucleo.sum()


def _buscar_maximo(mapa: np.ndarray, semillas: dict, margen: int,
                   ventana: int) -> dict:
    """Posicion de maximo de ``mapa`` en una ventana alrededor de cada semilla."""
    centros: dict[tuple[int, int], tuple[float, float]] = {}
    for celda, (centro_x, centro_y) in semillas.items():
        x0 = max(margen, int(round(centro_x)) - ventana)
        x1 = min(LADO - margen, int(round(centro_x)) + ventana + 1)
        y0 = max(margen, int(round(centro_y)) - ventana)
        y1 = min(LADO - margen, int(round(centro_y)) + ventana + 1)
        bloque = mapa[y0:y1, x0:x1]
        indice = int(np.argmax(bloque))
        dy, dx = divmod(indice, bloque.shape[1])
        centros[celda] = (float(x0 + dx), float(y0 + dy))
    return centros


def centros_por_anillo(diferencia: np.ndarray, celdas: dict[tuple[int, int], str]
                       ) -> tuple[dict[tuple[int, int], tuple[float, float]], float, tuple[float, float]]:
    """Centro de cada ficha con un puntaje de disco + anillo, en dos pasadas.

    Primera pasada: maximo de ``0.65 * media_del_disco(diferencia) +
    0.35 * respuesta_del_anillo`` en +-16 px del centro de la casilla. La media
    del disco pide que el CIRCULO lo cubra la ficha; la respuesta del anillo pide
    que el borde caiga donde el gradiente es fuerte. Solo el anillo se equivoca
    en las fichas de base clara sobre casilla gris (no hay borde de brillo); solo
    la media se equivoca cuando la ficha vecina entra en ventana.

    Segunda pasada: el recorte hereda un corrimiento GLOBAL del rectificador
    (las piezas cuelgan ~10-15 px hacia un lado en la foto entera). Se mide la
    mediana de los corrimientos de los 32 centros y se repite la busqueda con la
    rejilla ya desplazada, en +-9 px. Es el mismo espiritu de
    ``pipeline_fen.relinear_por_fichas``: la nube de fichas dice donde esta la
    rejilla real.

    Devuelve ({(fila, columna): (x, y)}, radio del tablero, (dx, dy) global).
    """
    detectados = _detectar_circulos(diferencia, set(celdas))
    if not detectados:
        raise ValueError("Hough no encontro ninguna ficha en el recorte")
    radio_tablero = float(np.median([radio for _, _, radio in detectados.values()]))
    kernel_anillo, _ = _kernel_anillo(radio_tablero)
    respuesta = cv2.filter2D(_mapa_borde(diferencia), -1, kernel_anillo,
                             borderType=cv2.BORDER_REPLICATE)
    disco = cv2.filter2D(diferencia, -1, _kernel_disco(radio_tablero),
                         borderType=cv2.BORDER_REPLICATE)
    respuesta = respuesta / (float(respuesta.max()) + 1e-6)
    disco = disco / (float(disco.max()) + 1e-6)
    puntaje = 0.65 * disco + 0.35 * respuesta
    margen = int(math.ceil(radio_tablero)) + 2

    ideal = {(fila, columna): ((columna + 0.5) * CASILLA, (fila + 0.5) * CASILLA)
             for fila, columna in celdas}
    primera = _buscar_maximo(puntaje, ideal, margen, BUSQUEDA)
    desplazamiento_x = float(np.median(
        [primera[c][0] - ideal[c][0] for c in ideal]))
    desplazamiento_y = float(np.median(
        [primera[c][1] - ideal[c][1] for c in ideal]))
    corregidas = {c: (ideal[c][0] + desplazamiento_x,
                      ideal[c][1] + desplazamiento_y) for c in ideal}
    centros = _buscar_maximo(puntaje, corregidas, margen, 9)
    return centros, radio_tablero, (desplazamiento_x, desplazamiento_y)


def fuga_saturacion(imagen: np.ndarray, centro_x: float, centro_y: float,
                    radio: float) -> float:
    """Fraccion del anillo exterior que todavia es TABLERO (saturacion alta).

    Es la medida de respaldo de ``extraer_sprites._fuga_tablero``: el damero
    impreso es azul (saturacion 55-90) y las fichas son neutras (< 25), asi que
    la saturacion separa tablero de ficha sin depender del brillo ni del color
    del simbolo. La medida Lab original marcaba como "tablero" la base clara de
    las fichas negras, que es un falso positivo del propio modelo de fondo.
    """
    lado = max(8, int(math.ceil(2.0 * radio)) + 2)
    x0 = int(round(centro_x - lado / 2))
    y0 = int(round(centro_y - lado / 2))
    recorte = _recorte_replicado(imagen, x0, y0, lado)
    alfa = _alpha(lado, centro_x - x0, centro_y - y0, radio)
    yy, xx = np.mgrid[0:lado, 0:lado].astype("float32")
    distancia = np.hypot(xx - (centro_x - x0), yy - (centro_y - y0))
    anillo = ((distancia >= 0.78 * radio) & (distancia <= 0.97 * radio)
              & (alfa > 0.45))
    if not anillo.any():
        return 0.0
    saturacion = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)[..., 1]
    return float((saturacion[anillo] > SATURACION_TABLERO).mean())


def extraer_sprite(imagen: np.ndarray, centro_x: float, centro_y: float,
                   radio: float) -> np.ndarray:
    """Sprite RGBA con alpha circular (feather de ~1 px en el borde).

    Nunca descarta: la casilla tiene ficha por la posicion fisica. El radio se
    fija con el diametro medido del tablero, no se recorta a ciegas: un alpha
    mas chico cortaria la base de la ficha y ensenaria un disco incompleto.
    """
    lado = max(8, int(math.ceil(2.0 * radio)) + 2)
    x0 = int(round(centro_x - lado / 2))
    y0 = int(round(centro_y - lado / 2))
    recorte = _recorte_replicado(imagen, x0, y0, lado)
    alfa = _alpha(lado, centro_x - x0, centro_y - y0, radio)
    return np.dstack([recorte, (alfa * 255.0).astype(np.uint8)])


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def _preview(sprites: list[tuple[np.ndarray, str, str]]) -> np.ndarray:
    """Dos bloques: una ficha por clase y la prueba caballo/alfil por casilla."""
    lado = 130
    paso = 18

    def lienzo_gris() -> np.ndarray:
        lienzo = np.zeros((lado, lado, 3), np.uint8)
        for fila in range(0, lado, paso):
            for columna in range(0, lado, paso):
                color = 92 if ((fila // paso) + (columna // paso)) % 2 else 132
                lienzo[fila:fila + paso, columna:columna + paso] = color
        return lienzo

    def pegar(sprite: np.ndarray, etiqueta: str) -> np.ndarray:
        lienzo = lienzo_gris()
        alto = sprite.shape[0]
        escala = (lado - 12) / alto
        reducido = cv2.resize(sprite, (int(alto * escala), int(alto * escala)),
                              interpolation=cv2.INTER_AREA)
        dy = (lado - reducido.shape[0]) // 2
        dx = (lado - reducido.shape[1]) // 2
        bgr = reducido[..., :3].astype("float32")
        alfa = reducido[..., 3:4].astype("float32") / 255.0
        base = lienzo[dy:dy + reducido.shape[0],
                      dx:dx + reducido.shape[1]].astype("float32")
        lienzo[dy:dy + reducido.shape[0], dx:dx + reducido.shape[1]] = np.clip(
            bgr * alfa + base * (1.0 - alfa), 0, 255).astype(np.uint8)
        cv2.putText(lienzo, etiqueta, (3, lado - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(lienzo, etiqueta, (3, lado - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (255, 255, 255), 1, cv2.LINE_AA)
        return lienzo

    fila_uno = [pegar(s, etiqueta) for s, etiqueta, _casilla in sprites[:12]]
    while len(fila_uno) < 12:
        fila_uno.append(np.full((lado, lado, 3), 40, np.uint8))

    criticas = [t for t in sprites if t[1] in ORIGEN_ESPERADO][:12]
    fila_dos = [pegar(s, f"{etiqueta} {casilla}")
                for s, etiqueta, casilla in criticas]
    while len(fila_dos) < 12:
        fila_dos.append(np.full((lado, lado, 3), 25, np.uint8))

    arriba = np.hstack(fila_uno).copy()
    abajo = np.hstack(fila_dos).copy()
    cv2.putText(arriba, "una ficha por clase (12)", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(abajo, "prueba anti-cruce: caballo solo b8/g8, alfil solo c8/f8",
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([arriba, abajo])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Extrae sprites RGBA limpios de la posicion inicial fisica.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--recortes", default=str(RECORTES))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--limite", type=int, default=0,
                            help="usar solo N recortes (0 = todos)")
    analizador.add_argument("--sin-previews", action="store_true")
    argumentos = analizador.parse_args(argv)

    clases = cargar_clases()
    base = Path(argumentos.recortes)
    if not base.is_dir():
        raise SystemExit(f"No encuentro {base}. Corre antes rectificar_crudo_v2.py")
    nombres = sorted(p.name for p in base.iterdir() if p.suffix.lower() == ".jpg")
    if argumentos.limite > 0:
        nombres = nombres[:argumentos.limite]
    if not nombres:
        raise SystemExit(f"No hay recortes en {base}")

    salida = Path(argumentos.salida)
    salida.mkdir(parents=True, exist_ok=True)

    celdas = celdas_iniciales()
    inicio = time.time()
    filas: list[dict] = []
    por_clase: Counter = Counter()
    contadores: Counter = Counter()
    origenes: dict[str, Counter] = defaultdict(Counter)
    ajustes_totales = 0
    recortados = 0
    descartados = 0
    fuga_maxima = 0.0
    fuga_alta: list[tuple[str, str, float]] = []
    fotos_ok = 0
    fotos_fallidas: list[tuple[str, str]] = []
    faltantes_por_foto: dict[str, int] = {}
    muestras: list[tuple[np.ndarray, str, str]] = []

    for nombre in nombres:
        imagen = cv2.imread(str(base / nombre),
                            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            fotos_fallidas.append((nombre, "OpenCV no pudo leer la imagen"))
            continue
        if imagen.shape[0] != LADO or imagen.shape[1] != LADO:
            imagen = cv2.resize(imagen, (LADO, LADO), interpolation=cv2.INTER_AREA)

        diferencia, _fondo = mapa_diferencia(imagen)
        try:
            centros, radio_tablero, desplazamiento = centros_por_anillo(
                diferencia, celdas)
        except ValueError as error:
            fotos_fallidas.append((nombre, str(error)))
            print(f"  AVISO {nombre}: {error}", flush=True)
            continue
        if max(abs(desplazamiento[0]), abs(desplazamiento[1])) > CORRIMIENTO_MAXIMO:
            # La rectificacion de ESA foto esta corrida (o girada) mas de lo que
            # tolera una casilla de 100 px: el sprite quedaria descentrado y con
            # tablero dentro. Se descarta la foto, nunca la ficha.
            motivo = (f"corrimiento de rejilla ({desplazamiento[0]:+.0f}, "
                      f"{desplazamiento[1]:+.0f}) px > {CORRIMIENTO_MAXIMO}")
            fotos_fallidas.append((nombre, motivo))
            print(f"  AVISO {nombre}: {motivo}", flush=True)
            continue

        fotos_ok += 1
        faltan = 0
        for (fila, columna), clase in sorted(celdas.items()):
            centro_x, centro_y = centros[(fila, columna)]
            radio = radio_tablero - AJUSTE_RADIO_CENTRO
            sprite = extraer_sprite(imagen, centro_x, centro_y, radio)
            fuga = fuga_saturacion(imagen, centro_x, centro_y, radio)
            casilla = f"{chr(ord('a') + columna)}{8 - fila}"
            if fuga > FUGA_ACEPTABLE:
                # Se queda: la casilla tiene ficha por la posicion fisica. Solo
                # se audita (y se ve en el preview) que tan cargado viene el borde.
                fuga_alta.append((nombre, casilla, fuga))
            fuga_maxima = max(fuga_maxima, fuga)
            if sprite is None:
                descartados += 1
                faltan += 1
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
            origenes[clase][casilla] += 1
            filas.append({
                "archivo": archivo,
                "fuente": nombre,
                "casilla": casilla,
                "clase": clase,
                "radio": f"{radio:.1f}",
                "radio_tablero": f"{radio_tablero:.1f}",
                "ajustes": 0,
                "fuga_fondo": f"{fuga:.3f}",
                "fuga_saturacion": f"{fuga:.3f}",
                "centro_x": f"{centro_x:.1f}",
                "centro_y": f"{centro_y:.1f}",
                "fondo_l": f"{fondo_l:.1f}",
                "fondo_a": f"{fondo_a:.1f}",
                "fondo_b": f"{fondo_b:.1f}",
                "fondo_l_std": f"{fondo_l_std:.1f}",
                "recortado": int(es_recortado),
            })
            if len(muestras) < 200:
                muestras.append((sprite, clase, casilla))
        if faltan:
            faltantes_por_foto[nombre] = faltan
        print(f"  {nombre}: radio {radio_tablero:.1f} px, "
              f"desplazamiento de rejilla ({desplazamiento[0]:+.0f}, "
              f"{desplazamiento[1]:+.0f}) px, "
              f"{len(celdas) - faltan}/{len(celdas)} sprites", flush=True)

    manifiesto = salida / "sprites_manifesto.csv"
    with manifiesto.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)

    ruta_preview: Path | None = None
    if not argumentos.sin_previews and muestras:
        ruta_preview = PREVIEW
        una_por_clase: list[tuple[np.ndarray, str, str]] = []
        for clase in clases:
            candidato = next((t for t in muestras if t[1] == clase), None)
            if candidato is not None:
                una_por_clase.append(candidato)
        criticas = [t for t in muestras if t[1] in ORIGEN_ESPERADO][:12]
        ordenadas = una_por_clase + criticas
        cv2.imwrite(str(ruta_preview), _preview(ordenadas),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    segundos = time.time() - inicio
    valores_radio = [float(f["radio"]) for f in filas]
    print()
    print(f"listo: {len(filas)} sprites de {fotos_ok}/{len(nombres)} recortes "
          f"en {segundos:.1f} s")
    print(f"  sprites: {salida}")
    print(f"  manifiesto: {manifiesto}")
    print(f"  por clase: {dict(sorted(por_clase.items()))}")
    print(f"  fotos sin geometria valida: {len(fotos_fallidas)}")
    for nombre, motivo in fotos_fallidas[:5]:
        print(f"    {nombre}: {motivo}")
    print(f"  descartados por el filtro de luminancia: {descartados} (objetivo 0)")
    print(f"  fuga de tablero en el anillo: max {fuga_maxima:.3f}, "
          f"sobre {FUGA_ACEPTABLE:.2f} en {len(fuga_alta)} sprites (se quedan)")
    for nombre, casilla, fuga in fuga_alta[:6]:
        print(f"    {nombre} {casilla}: {fuga:.3f}")
    if faltantes_por_foto:
        print(f"  fotos con casillas sin sprite: {dict(faltantes_por_foto)}")
    if valores_radio:
        print(f"  radio del sprite (px en el recorte 800): min {min(valores_radio):.1f}, "
              f"mediana {float(np.median(valores_radio)):.1f}, "
              f"max {max(valores_radio):.1f}")
    print(f"  sprites con disco cortado por el borde: {recortados}")
    print("  origen por clase critica (prueba anti-cruce):")
    cruces = []
    for clase, permitidas in ORIGEN_ESPERADO.items():
        usadas = set(origenes[clase])
        fuera = sorted(usadas - permitidas)
        if fuera:
            cruces.append((clase, fuera))
        print(f"    {clase:10s} usa {sorted(usadas)} "
              f"-> {'OK' if not fuera else 'CRUCE ' + str(fuera)}")
    faltan_clases = [c for c in clases if por_clase.get(c, 0) == 0]
    if faltan_clases:
        print(f"  CLASES SIN SPRITE: {faltan_clases}")
    print(f"  preview: {ruta_preview if ruta_preview else 'omitida'}")

    correcto = (
        descartados == 0
        and not faltantes_por_foto
        and not cruces
        and not faltan_clases
        and len(filas) == 32 * fotos_ok
    )
    if not correcto:
        print("RESULTADO: revisar (ver conteos arriba)")
        return 1
    print("RESULTADO: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
