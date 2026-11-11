"""Rectifica las 252 fotos del crudo NUEVO y deriva sus etiquetas de verdad.

Este modulo cubre los pasos A y B del dataset limpio:

A. RECTIFICACION
   Por cada ``dataset/crudo/<carpeta>/*.jpg`` se corre
   ``esquinas_aruco.detectar_robusto`` (ArUco multi-escala -> cuadriculado ->
   piezas) y se deforma el area jugable a la vista canonica 800x800 (blancas
   abajo, casillas de 100 px) en ``dataset/recortes_800/<carpeta>/<archivo>``.
   Las fotos sin esquinas validas NO se inventan: van a
   ``dataset/recortes_800/fallos.csv`` con su motivo y se saltan.

B. ETIQUETAS
   La verdad sale del MANIFIESTO de cada carpeta (``archivo,posicion,fen,fuente``),
   NO del modelo de v1: por eso este dataset no puede heredar el cruce
   caballo/alfil que contamino el lote anterior.

   * ``01_inicial_intacta`` -> FEN inicial ESTANDAR fisico (b8=caballo_n,
     c8=alfil_n, f8=alfil_n, g8=caballo_n). El usuario confirmo que el tablero
     fisico quedo como el estandar.
   * ``05_tablero_vacio``   -> ``todo_vacia`` (0 fichas).
   * ``02/03/04``           -> la FEN declarada en el manifiesto.

   De cada recorte se derivan:

   ``<carpeta>/labels/<archivo>.txt``  YOLO: una caja por casilla ocupada,
       centrada en la casilla y con lado 0.85 de casilla (= 0.10625 en las
       coordenadas normalizadas del 800, la misma medida del dataset v2 medido).
   ``dataset/casillas/clasificador/<clase>/<stem>_<casilla>.jpg``  recorte
       100x100 de CADA una de las 64 posiciones (13 clases con ``vacia``).
   ``dataset/recortes_800/manifesto_recortes.csv``  archivo, carpeta_origen,
       fen, casilla_ok, cantidad_fichas, fallo.
   ``dataset/recortes_800/fallos.csv``  archivo, motivo.

Controles que imprime al final: recortes+fallos == fotos, imagenes == labels,
cajas == fichas del FEN en CADA recorte y casillas == 64 * recortes.

Uso
---
    python rectificar_crudo_v2.py
    python rectificar_crudo_v2.py --limite 3 --sin-piezas --sin-previews
    python rectificar_crudo_v2.py --carpetas 02_inicial_con_jugadas
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import esquinas_aruco
from comun import CARPETAS, CRUDO, DATASET, cargar_clases
from generar_sinteticos import destino_rectificado

RAIZ = Path(__file__).resolve().parent.parent
RECORTES = DATASET / "recortes_800"
CASILLAS = DATASET / "casillas" / "clasificador"

LADO = 800
CASILLA = LADO // 8
# Lado de la caja YOLO como fraccion del lado de la casilla. Medido sobre el
# dataset v2: sintetico 0.891, compuesto 0.851 -> 0.85 es la escala real de una
# ficha en el tablero impreso. En coords normalizadas del 800: 0.85/8.
BOX_CASILLA = 0.85
LADO_CAJA = BOX_CASILLA / 8.0

CLASE_VACIA = "vacia"
ETIQUETA_VACIA = "todo_vacia"
ETIQUETA_INICIAL = "inicial_estandar"

# Posicion inicial ESTANDAR fisica de la mesa (confirmada por el usuario: el
# tablero quedo como el estandar, sin el cruce del lote anterior).
BLANCAS = {
    "a1": "torre_b", "b1": "caballo_b", "c1": "alfil_b", "d1": "dama_b",
    "e1": "rey_b", "f1": "alfil_b", "g1": "caballo_b", "h1": "torre_b",
}
NEGRAS = {
    "a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n", "d8": "dama_n",
    "e8": "rey_n", "f8": "alfil_n", "g8": "caballo_n", "h8": "torre_n",
}
INICIAL_ESTANDAR = {
    **BLANCAS,
    **NEGRAS,
    **{f"{c}2": "peon_b" for c in "abcdefgh"},
    **{f"{c}7": "peon_n" for c in "abcdefgh"},
    **{f"{c}{f}": CLASE_VACIA for c in "abcdefgh" for f in (3, 4, 5, 6)},
}

# FEN: caracter -> forma y color. MAYUSCULA = blanca (sufijo _b), minuscula = negra.
FORMAS_FEN = {"P": "peon", "R": "torre", "N": "caballo", "B": "alfil",
              "Q": "dama", "K": "rey"}

CAMPOS_MANIFIESTO = ["archivo", "carpeta_origen", "fen", "casilla_ok",
                     "cantidad_fichas", "fallo"]
CAMPOS_FALLOS = ["archivo", "motivo"]


# ---------------------------------------------------------------------------
# Verdad: manifiesto -> ocupacion por casilla
# ---------------------------------------------------------------------------

def casilla_a_fila_columna(casilla: str) -> tuple[int, int]:
    columna = ord(casilla[0]) - ord("a")
    fila = 8 - int(casilla[1])
    if not (0 <= columna < 8 and 0 <= fila < 8):
        raise ValueError(f"casilla fuera del tablero: {casilla}")
    return fila, columna


def fila_columna_a_casilla(fila: int, columna: int) -> str:
    return f"{chr(ord('a') + columna)}{8 - fila}"


def ocupacion_de_fen(texto: str) -> dict[str, str]:
    """{casilla: clase} de una FEN (o de las etiquetas ``inicial_estandar`` /
    ``todo_vacia``).

    Solo se usa el campo de piezas (el primero). Lanza ``ValueError`` si la FEN
    no tiene 8 filas de 8 casillas: una FEN mal escrita no puede producir
    etiquetas, y fallar aqui es mejor que etiquetar basura.
    """
    limpio = (texto or "").strip()
    if not limpio or limpio == "(SIN DECLARAR)":
        raise ValueError("la fila no declara FEN")
    if limpio == ETIQUETA_INICIAL:
        return dict(INICIAL_ESTANDAR)
    if limpio in (ETIQUETA_VACIA, "todo_vacía"):
        return {fila_columna_a_casilla(f, c): CLASE_VACIA
                for f in range(8) for c in range(8)}

    campo = limpio.split()[0]
    filas = campo.split("/")
    if len(filas) != 8:
        raise ValueError(f"la FEN tiene {len(filas)} filas, se esperaban 8")
    ocupacion: dict[str, str] = {}
    for indice_fila, fila in enumerate(filas):
        columna = 0
        for simbolo in fila:
            if simbolo.isdigit():
                for _ in range(int(simbolo)):
                    if columna > 7:
                        raise ValueError(f"la fila {indice_fila} se pasa de 8 casillas")
                    ocupacion[fila_columna_a_casilla(indice_fila, columna)] = CLASE_VACIA
                    columna += 1
                continue
            forma = FORMAS_FEN.get(simbolo.upper())
            if forma is None:
                raise ValueError(f"simbolo de FEN desconocido: {simbolo!r}")
            if columna > 7:
                raise ValueError(f"la fila {indice_fila} se pasa de 8 casillas")
            color = "b" if simbolo.isupper() else "n"
            ocupacion[fila_columna_a_casilla(indice_fila, columna)] = f"{forma}_{color}"
            columna += 1
        if columna != 8:
            raise ValueError(f"la fila {indice_fila} tiene {columna} casillas")
    if len(ocupacion) != 64:
        raise ValueError(f"la FEN describio {len(ocupacion)} casillas, se esperaban 64")
    return ocupacion


def leer_manifiesto(carpeta: Path) -> dict[str, str]:
    """{archivo: fen} del manifiesto de la carpeta."""
    ruta = carpeta / "manifesto.csv"
    if not ruta.exists():
        raise SystemExit(f"No encuentro el manifiesto de {carpeta}")
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        return {
            (fila.get("archivo") or "").strip(): (fila.get("fen") or "").strip()
            for fila in csv.DictReader(archivo)
            if (fila.get("archivo") or "").strip()
        }


# ---------------------------------------------------------------------------
# Etiquetas
# ---------------------------------------------------------------------------

def linea_yolo(clase: str, casilla: str, clases: list[str]) -> str:
    """Caja cuadrada centrada en la casilla, en coords normalizadas del 800."""
    if clase not in clases:
        raise ValueError(f"clase desconocida {clase!r}; no esta en classes.txt")
    fila, columna = casilla_a_fila_columna(casilla)
    x = (columna + 0.5) / 8.0
    y = (fila + 0.5) / 8.0
    return (f"{clases.index(clase)} {x:.6f} {y:.6f} "
            f"{LADO_CAJA:.6f} {LADO_CAJA:.6f}")


def recortar_casillas(recorte: np.ndarray, ocupacion: dict[str, str]
                      ) -> list[tuple[str, str, np.ndarray]]:
    """64 recortes 100x100 en orden fila/columna con su clase."""
    salida = []
    for fila in range(8):
        for columna in range(8):
            casilla = fila_columna_a_casilla(fila, columna)
            x0, y0 = columna * CASILLA, fila * CASILLA
            salida.append((
                casilla,
                ocupacion[casilla],
                recorte[y0:y0 + CASILLA, x0:x0 + CASILLA],
            ))
    return salida


# ---------------------------------------------------------------------------
# Rectificacion
# ---------------------------------------------------------------------------

def rectificar(imagen: np.ndarray, esquinas: np.ndarray) -> np.ndarray:
    """Deforma el area jugable a la vista canonica 800x800."""
    matriz = cv2.getPerspectiveTransform(
        np.asarray(esquinas, dtype="float32"), destino_rectificado(LADO)
    )
    return cv2.warpPerspective(
        imagen, matriz, (LADO, LADO), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _dibujar_muestra(recorte: np.ndarray, ocupacion: dict[str, str],
                     clases: list[str], titulo: str) -> np.ndarray:
    anotada = recorte.copy()
    for fila in range(8):
        for columna in range(8):
            casilla = fila_columna_a_casilla(fila, columna)
            clase = ocupacion[casilla]
            if clase == CLASE_VACIA:
                continue
            color = (60, 200, 60) if clase.endswith("_b") else (60, 60, 220)
            lado_px = int(round(BOX_CASILLA * CASILLA))
            cx, cy = int((columna + 0.5) * CASILLA), int((fila + 0.5) * CASILLA)
            cv2.rectangle(anotada, (cx - lado_px // 2, cy - lado_px // 2),
                          (cx + lado_px // 2, cy + lado_px // 2), color, 2)
    for i in range(9):
        p = min(int(round(i * CASILLA)), LADO - 1)
        cv2.line(anotada, (p, 0), (p, LADO - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(anotada, (0, p), (LADO - 1, p), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, LADO - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, LADO - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return anotada


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Rectifica el crudo nuevo y deriva labels YOLO + casillas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--crudo", default=str(CRUDO))
    analizador.add_argument("--salida", default=str(RECORTES))
    analizador.add_argument("--casillas", default=str(CASILLAS))
    analizador.add_argument("--carpetas", nargs="*", default=None,
                            help="subconjunto de carpetas del crudo (default: todas)")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo las primeras N fotos por carpeta")
    analizador.add_argument("--sin-piezas", action="store_true",
                            help="no usar el modelo YOLO de v1 como ultimo recurso")
    analizador.add_argument("--muestras", type=int, default=3,
                            help="tableros anotados de muestra (0 = ninguno)")
    argumentos = analizador.parse_args(argv)

    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    raiz_crudo = Path(argumentos.crudo)
    salida = Path(argumentos.salida)
    raiz_casillas = Path(argumentos.casillas)
    salida.mkdir(parents=True, exist_ok=True)
    for clase in clases + [CLASE_VACIA]:
        (raiz_casillas / clase).mkdir(parents=True, exist_ok=True)

    carpetas = argumentos.carpetas or [c for c in CARPETAS
                                       if (raiz_crudo / c).is_dir()]
    plantilla = esquinas_aruco.analizar_plantilla()

    inicio = time.time()
    filas_manifesto: list[dict] = []
    filas_fallos: list[dict] = []
    conteo_metodo: Counter = Counter()
    conteo_carpeta: Counter = Counter()
    conteo_fichas: Counter = Counter()
    conteo_fallos: Counter = Counter()
    discrepancias_cajas: list[str] = []
    discrepancias_casillas: list[str] = []
    casillas_escritas = 0
    labels_escritos = 0
    total_fotos = 0
    muestras: list[tuple[np.ndarray, dict[str, str], str]] = []
    muestras_carpetas: set[str] = set()

    for carpeta in carpetas:
        directorio = raiz_crudo / carpeta
        manifiesto = leer_manifiesto(directorio)
        fotos = sorted(p for p in directorio.iterdir()
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        if argumentos.limite > 0:
            fotos = fotos[:argumentos.limite]
        destino_carpeta = salida / carpeta
        destino_labels = destino_carpeta / "labels"
        destino_labels.mkdir(parents=True, exist_ok=True)
        total_fotos += len(fotos)
        print(f"[{carpeta}] {len(fotos)} fotos", flush=True)

        for numero, ruta in enumerate(fotos, start=1):
            fen = manifiesto.get(ruta.name)
            imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR
                                | cv2.IMREAD_IGNORE_ORIENTATION)
            if imagen is None:
                filas_fallos.append({"archivo": ruta.name,
                                     "motivo": "OpenCV no pudo leer la imagen"})
                conteo_fallos["ilegible"] += 1
                continue

            try:
                ocupacion = ocupacion_de_fen(fen or "")
            except ValueError as error:
                filas_fallos.append({"archivo": ruta.name,
                                     "motivo": f"FEN invalida: {error}"})
                conteo_fallos["fen"] += 1
                continue

            esquinas, metodo, detalle = esquinas_aruco.detectar_robusto(
                imagen, plantilla, usar_piezas=not argumentos.sin_piezas
            )
            if esquinas is None:
                motivo = f"{metodo}: {detalle.get('motivo') or 'sin detalle'}"
                filas_fallos.append({"archivo": ruta.name, "motivo": motivo})
                conteo_fallos[metodo] += 1
                continue

            recorte = rectificar(imagen, esquinas)
            cv2.imwrite(str(destino_carpeta / ruta.name), recorte,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            ocupadas = [c for c, clase in ocupacion.items() if clase != CLASE_VACIA]
            lineas = [linea_yolo(ocupacion[c], c, clases) for c in sorted(ocupadas)]
            destino_labels.joinpath(ruta.stem + ".txt").write_text(
                "".join(linea + "\n" for linea in lineas), encoding="utf-8"
            )
            labels_escritos += 1
            if len(lineas) != len(ocupadas):
                discrepancias_cajas.append(ruta.name)

            recortes = recortar_casillas(recorte, ocupacion)
            escritas_aqui = 0
            for casilla, clase, parche in recortes:
                if parche.shape[0] != CASILLA or parche.shape[1] != CASILLA:
                    continue
                # El nombre lleva la CARPETA: ``capture (N).jpg`` existe en
                # varias carpetas del crudo y sin el prefijo se sobreescribian
                # las casillas de fotos distintas.
                destino = raiz_casillas / clase / f"{carpeta}_{ruta.stem}_{casilla}.jpg"
                cv2.imwrite(str(destino), parche,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                escritas_aqui += 1
            casillas_escritas += escritas_aqui
            if escritas_aqui != 64:
                discrepancias_casillas.append(f"{ruta.name}: {escritas_aqui}")

            conteo_metodo[metodo] += 1
            conteo_carpeta[carpeta] += 1
            conteo_fichas[carpeta] += len(ocupadas)
            filas_manifesto.append({
                "archivo": ruta.name,
                "carpeta_origen": carpeta,
                "fen": fen or "",
                "casilla_ok": escritas_aqui,
                "cantidad_fichas": len(ocupadas),
                "fallo": "",
            })
            if (len(muestras) < max(argumentos.muestras, 0)
                    and carpeta not in muestras_carpetas):
                muestras.append((recorte, ocupacion,
                                 f"{ruta.stem}  {carpeta}  "
                                 f"fichas={len(ocupadas)}  {metodo}"))
                muestras_carpetas.add(carpeta)

            if numero % 20 == 0 or numero == len(fotos):
                print(f"  {numero}/{len(fotos)}  ({metodo}, {len(ocupadas)} fichas)",
                      flush=True)

    # ---- muestras anotadas
    rutas_muestras: list[str] = []
    if muestras:
        carpeta_muestras = salida / "muestras"
        carpeta_muestras.mkdir(parents=True, exist_ok=True)
        for indice, (recorte, ocupacion, titulo) in enumerate(muestras, start=1):
            destino = carpeta_muestras / f"muestra_{indice}.jpg"
            cv2.imwrite(str(destino), _dibujar_muestra(
                recorte, ocupacion, clases, titulo), [cv2.IMWRITE_JPEG_QUALITY, 90])
            rutas_muestras.append(str(destino))

    # ---- CSV
    ruta_manifesto = salida / "manifesto_recortes.csv"
    with ruta_manifesto.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_MANIFIESTO)
        escritor.writeheader()
        escritor.writerows(filas_manifesto)
    ruta_fallos = salida / "fallos.csv"
    with ruta_fallos.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_FALLOS)
        escritor.writeheader()
        escritor.writerows(filas_fallos)

    # ---- informe
    recortes_ok = len(filas_manifesto)
    total_cajas = sum(f["cantidad_fichas"] for f in filas_manifesto)
    labels_disco = sum(len(list((salida / c / "labels").glob("*.txt")))
                       for c in carpetas if (salida / c / "labels").is_dir())
    segundos = time.time() - inicio
    cobertura = 100.0 * recortes_ok / max(total_fotos, 1)

    print()
    print(f"listo: {recortes_ok} recortes, {len(filas_fallos)} fallos "
          f"de {total_fotos} fotos en {segundos:.0f} s")
    print(f"  cobertura: {cobertura:.1f}% (objetivo > 85%)")
    print(f"  por carpeta: {dict(sorted(conteo_carpeta.items()))}")
    print(f"  por metodo:  {dict(sorted(conteo_metodo.items()))}")
    print(f"  fallos por motivo: {dict(sorted(conteo_fallos.items()))}")
    print(f"  fichas del FEN: {dict(sorted(conteo_fichas.items()))} "
          f"(total {total_cajas})")
    print(f"  labels en disco: {labels_disco} "
          f"({'OK' if labels_disco == recortes_ok else 'DESCUADRE'})")
    print(f"  casillas 100x100: {casillas_escritas} "
          f"(== 64 * {recortes_ok} = {64 * recortes_ok})")
    print(f"  recortes con cajas != fichas: {len(discrepancias_cajas)}")
    print(f"  recortes con casillas != 64: {len(discrepancias_casillas)}")
    print(f"  manifiesto: {ruta_manifesto}")
    print(f"  fallos: {ruta_fallos}")
    print(f"  muestras: {rutas_muestras}")

    correcto = (
        labels_disco == recortes_ok
        and casillas_escritas == 64 * recortes_ok
        and not discrepancias_cajas
        and not discrepancias_casillas
    )
    if not correcto:
        print("RESULTADO: revisar (ver conteos arriba)")
        return 1
    print("RESULTADO: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
