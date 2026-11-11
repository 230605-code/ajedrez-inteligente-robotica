"""Deriva del dataset compuesto: recortes 800x800 y las 64 casillas 100x100.

El dataset compuesto (``compilar_compuestos.py``) guarda cada tablero tal como lo
veria el DETECTOR YOLO: con fondo, marco de mesa y perspectiva. Este script
genera el derivado del CLASIFICADOR por casilla, que es otra cosa:

    dataset/reales_compuesto/recortes/800x800/<compuesto>.jpg
    dataset/reales_compuesto/casillas/clasificador/<clase>/<compuesto>_<casilla>.jpg
    dataset/reales_compuesto/etiquetas_casillas.csv

La geometria NO se vuelve a detectar: sale de ``homografia_recorte`` de
``manifiesto.csv``, la misma matriz con la que el compositor proyecto el plano
del tablero. La vista rectificada mide 800 px (8 casillas de 100 px), igual que
``dataset/sintetico/recortes/800x800``, asi que el clasificador recibe el mismo
tamano venga del sintetico, de las fotos reales o del compuesto.

Las clases de cada casilla salen de ``rotaciones.csv`` (una fila por ficha con
su casilla y clase, generada al componer). Todo lo que no aparece ahi es
``vacia``. Eso hace el etiquetado exacto por construccion: la etiqueta viaja con
la geometria, no se infiere de la imagen.

Los 128 PNG de muestra que dejo el compositor en ``casillas/clasificador`` se
conservan intactos: el derivado completo escribe JPG con el MISMO nombre base,
asi que no colisiona y no borra la muestra.

Uso
---
    python recortar_compuestos.py                       # los 1560 compuestos
    python recortar_compuestos.py --inicio 1 --fin 200  # lote de trabajo
    python recortar_compuestos.py --inicio 1 --fin 4 --sin-previews  # prueba
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from compilar_compuestos import _hoja_casillas
from comun import cargar_clases
from generar_sinteticos import LADO_RECORTE, texto_a_matriz
from recortar_sinteticos import fraccion_borde

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "dataset" / "reales_compuesto"
PREVIEWS = RAIZ / "assets" / "previews"

CASILLA = LADO_RECORTE // 8
CALIDAD_JPEG = 92

CAMPOS = ["archivo", "imagen", "casilla", "clase", "fila", "columna", "perfil"]
PATRON_CASILLA = re.compile(r"^[a-h][1-8]$")


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def _leer_manifiesto(ruta: Path) -> list[dict[str, str]]:
    """Filas del manifiesto en su orden de generacion (por perfil y numero)."""
    if not ruta.exists():
        raise SystemExit(f"No encuentro el manifiesto: {ruta}")
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise SystemExit(f"El manifiesto esta vacio: {ruta}")
    return filas


def _leer_rotaciones(ruta: Path, clases: list[str]) -> dict[tuple[str, str], str]:
    """(imagen, casilla) -> clase, y valida que la clase exista."""
    if not ruta.exists():
        raise SystemExit(f"No encuentro {ruta}. Corre antes compilar_compuestos.py")
    validas = set(clases)
    por_celda: dict[tuple[str, str], str] = {}
    invalidas = 0
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            clase = fila["clase"]
            if clase not in validas or not PATRON_CASILLA.match(fila["casilla"]):
                invalidas += 1
                continue
            por_celda[(fila["imagen"], fila["casilla"])] = clase
    if invalidas:
        raise SystemExit(f"{ruta}: {invalidas} filas con clase o casilla invalida")
    return por_celda


def _casilla(fila: int, columna: int) -> str:
    """Nombre de casilla del plano: fila 0 = rank 8, columna 0 = columna a."""
    return f"{chr(ord('a') + columna)}{8 - fila}"


def _matriz(fila: dict[str, str]) -> np.ndarray | None:
    texto = (fila.get("homografia_recorte") or "").strip()
    if not texto:
        return None
    try:
        return texto_a_matriz(texto)
    except ValueError as exc:
        print(f"  AVISO: homografia invalida en {fila.get('imagen')}: {exc}")
        return None


def _leer_csv_existente(ruta: Path, perfiles: dict[str, str],
                        en_lote: set[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Filas de imagenes FUERA del lote, apuntando al archivo que existe.

    Las filas de imagenes del lote no se cargan: se reescriben completas. Asi un
    lote parcial conserva el CSV de lo ya derivado sin arrastrar rutas rotas
    (la muestra vieja apunta a PNG; el derivado completo escribe JPG).
    """
    filas: dict[tuple[str, str], dict[str, str]] = {}
    if not ruta.exists():
        return filas
    base = ruta.parent / "casillas" / "clasificador"
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            imagen, casilla_ = fila.get("imagen", ""), fila.get("casilla", "")
            clase = fila.get("clase", "")
            if (not imagen or Path(imagen).stem in en_lote
                    or not PATRON_CASILLA.match(casilla_) or not clase):
                continue
            tallo = base / clase / f"{Path(imagen).stem}_{casilla_}"
            if (tallo.with_suffix(".jpg")).is_file():
                archivo_rel = f"casillas/clasificador/{clase}/{tallo.name}.jpg"
            elif (tallo.with_suffix(".png")).is_file():
                archivo_rel = f"casillas/clasificador/{clase}/{tallo.name}.png"
            else:
                continue
            fila["archivo"] = archivo_rel
            fila["perfil"] = fila.get("perfil") or perfiles.get(imagen, "")
            filas[(imagen, casilla_)] = {campo: fila.get(campo, "") for campo in CAMPOS}
    return filas


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

def _tamano(ruta: Path) -> int:
    if not ruta.exists():
        return 0
    return sum(p.stat().st_size for p in ruta.rglob("*") if p.is_file())


def _estadisticas_brillo(gris: np.ndarray, fila: int, columna: int,
                         acumulados: dict[str, float]) -> None:
    """Suma media y varianza pooled de la casilla (escala de grises)."""
    x0, y0 = columna * CASILLA, fila * CASILLA
    celda = gris[y0:y0 + CASILLA, x0:x0 + CASILLA]
    acumulados["n"] += celda.size
    acumulados["suma"] += float(celda.sum())
    acumulados["suma2"] += float(np.square(celda).sum())


def _guardar_casilla(recorte: np.ndarray, fila: int, columna: int,
                     clase: str, casillas_dir: Path, nombre: str) -> dict[str, str]:
    x0, y0 = columna * CASILLA, fila * CASILLA
    celda = recorte[y0:y0 + CASILLA, x0:x0 + CASILLA]
    if celda.shape != (CASILLA, CASILLA, 3):
        raise SystemExit(
            f"{nombre}: casilla con forma {celda.shape}, se esperaba "
            f"({CASILLA}, {CASILLA}, 3)"
        )
    casilla_ = _casilla(fila, columna)
    archivo = casillas_dir / clase / f"{nombre}_{casilla_}.jpg"
    cv2.imwrite(str(archivo), celda, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
    return {
        "archivo": f"casillas/clasificador/{clase}/{nombre}_{casilla_}.jpg",
        "imagen": f"{nombre}.jpg",
        "casilla": casilla_,
        "clase": clase,
        "fila": str(fila),
        "columna": str(columna),
    }


# ---------------------------------------------------------------------------
# Verificacion
# ---------------------------------------------------------------------------

def _verificar_csv(ruta: Path, clases: list[str], limite_lectura: int,
                   rng: np.random.Generator) -> tuple[int, int, int, str]:
    """Valida filas, archivos y una muestra de JPG 100x100x3.

    Devuelve (filas, invalidas, leidas, detalle). Un archivo que falta en una
    fila cuenta como invalido: el CSV es el indice del dataset, no una promesa.
    """
    validas = set(clases) | {"vacia"}
    filas = 0
    invalidas: list[str] = []
    leibles: list[tuple[Path, str]] = []
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        for numero, fila in enumerate(csv.DictReader(archivo), start=2):
            filas += 1
            casilla_ = fila.get("casilla", "")
            clase = fila.get("clase", "")
            if clase not in validas:
                invalidas.append(f"linea {numero}: clase '{clase}' fuera de rango")
                continue
            if not PATRON_CASILLA.match(casilla_):
                invalidas.append(f"linea {numero}: casilla '{casilla_}' invalida")
                continue
            try:
                fila_n = int(fila.get("fila") or -1)
                columna_n = int(fila.get("columna") or -1)
            except ValueError:
                invalidas.append(f"linea {numero}: fila/columna no numericas")
                continue
            if not (0 <= fila_n < 8 and 0 <= columna_n < 8) or \
                    _casilla(fila_n, columna_n) != casilla_:
                invalidas.append(f"linea {numero}: fila/columna no cuadran con {casilla_}")
                continue
            archivo = ruta.parent / fila.get("archivo", "")
            if not archivo.is_file():
                invalidas.append(f"linea {numero}: falta {archivo.name}")
                continue
            leibles.append((archivo, clase))

    leidas = 0
    if leibles and limite_lectura > 0:
        indices = rng.choice(len(leibles), size=min(limite_lectura, len(leibles)),
                             replace=False)
        for indice in indices:
            archivo, _clase = leibles[int(indice)]
            celda = cv2.imread(str(archivo), cv2.IMREAD_COLOR)
            if celda is None or celda.shape != (CASILLA, CASILLA, 3):
                invalidas.append(f"{archivo.name}: no carga 100x100x3")
                continue
            leidas += 1
    return filas, len(invalidas), leidas, "; ".join(invalidas[:5])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Recorta los compuestos al tablero y separa sus 64 casillas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--inicio", type=int, default=1,
                            help="primer compuesto del lote (1-based, inclusive)")
    analizador.add_argument("--fin", type=int, default=0,
                            help="ultimo compuesto del lote (0 = hasta el final)")
    analizador.add_argument("--manifiesto", default=str(SALIDA / "manifiesto.csv"))
    analizador.add_argument("--rotaciones", default=str(SALIDA / "rotaciones.csv"))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--semilla", type=int, default=23,
                            help="semilla de previews y muestra verificada")
    analizador.add_argument("--verificar", type=int, default=200,
                            help="casillas al azar que se vuelven a leer (0 desactiva)")
    analizador.add_argument("--sin-previews", action="store_true")
    argumentos = analizador.parse_args(argv)

    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    manifiesto = _leer_manifiesto(Path(argumentos.manifiesto))
    rotaciones = _leer_rotaciones(Path(argumentos.rotaciones), clases)

    salida = Path(argumentos.salida)
    imagenes_dir = salida / "images"
    recortes_dir = salida / "recortes" / f"{LADO_RECORTE}x{LADO_RECORTE}"
    casillas_dir = salida / "casillas" / "clasificador"
    csv_ruta = salida / "etiquetas_casillas.csv"

    for carpeta in [recortes_dir] + [casillas_dir / c for c in clases + ["vacia"]]:
        carpeta.mkdir(parents=True, exist_ok=True)
    if not argumentos.sin_previews:
        PREVIEWS.mkdir(parents=True, exist_ok=True)

    inicio_lote = max(1, argumentos.inicio)
    fin_lote = argumentos.fin if argumentos.fin > 0 else len(manifiesto)
    fin_lote = min(fin_lote, len(manifiesto))
    if inicio_lote > fin_lote:
        print(f"lote vacio: inicio={inicio_lote} fin={fin_lote}")
        return 2
    lote = manifiesto[inicio_lote - 1:fin_lote]

    perfiles = {fila["imagen"]: fila.get("perfil", "") for fila in manifiesto}
    nombres_lote = {Path(fila["imagen"]).stem for fila in lote}
    previas = _leer_csv_existente(csv_ruta, perfiles, nombres_lote)
    rng = np.random.default_rng(int(argumentos.semilla))

    candidatas_preview = [
        Path(fila["imagen"]).stem for fila in lote
        if int(fila.get("n_fichas") or 0) >= 6
        and (imagenes_dir / fila["imagen"]).is_file()
    ] or [Path(fila["imagen"]).stem for fila in lote
          if (imagenes_dir / fila["imagen"]).is_file()]
    elegidas_preview: set[str] = set()
    if candidatas_preview and not argumentos.sin_previews:
        indices = rng.choice(len(candidatas_preview),
                             size=min(3, len(candidatas_preview)), replace=False)
        elegidas_preview = {candidatas_preview[int(i)] for i in indices}

    inicio = time.time()
    filas_nuevas: dict[tuple[str, str], dict[str, str]] = {}
    celdas_preview: dict[str, list[tuple[str, np.ndarray]]] = {}
    por_clase: Counter = Counter()
    brillo: dict[str, dict[str, float]] = {
        clase: {"n": 0.0, "suma": 0.0, "suma2": 0.0} for clase in clases + ["vacia"]
    }
    procesadas: list[str] = []
    recortes_escritos = 0
    sin_homografia: list[str] = []
    borde_maximo = 0.0
    borde_imagen = ""

    for indice, fila in enumerate(lote, start=1):
        nombre = Path(fila["imagen"]).stem
        ruta = imagenes_dir / fila["imagen"]
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            print(f"  AVISO: no pude leer {ruta}")
            continue
        matriz = _matriz(fila)
        if matriz is None:
            sin_homografia.append(fila["imagen"])
            continue

        recorte = cv2.warpPerspective(
            imagen, matriz, (LADO_RECORTE, LADO_RECORTE),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        fraccion = fraccion_borde(imagen, matriz, LADO_RECORTE)
        if fraccion > borde_maximo:
            borde_maximo, borde_imagen = fraccion, fila["imagen"]

        cv2.imwrite(str(recortes_dir / f"{nombre}.jpg"), recorte,
                    [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
        recortes_escritos += 1
        procesadas.append(fila["imagen"])

        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY).astype("float64")
        celdas_imagen: list[tuple[str, np.ndarray]] = []
        for fila_celda in range(8):
            for columna in range(8):
                casilla_ = _casilla(fila_celda, columna)
                clase = rotaciones.get((fila["imagen"], casilla_), "vacia")
                fila_csv = _guardar_casilla(
                    recorte, fila_celda, columna, clase, casillas_dir, nombre
                )
                fila_csv["perfil"] = fila.get("perfil", "")
                filas_nuevas[(fila["imagen"], casilla_)] = fila_csv
                por_clase[clase] += 1
                _estadisticas_brillo(gris, fila_celda, columna, brillo[clase])
                if nombre in elegidas_preview:
                    x0, y0 = columna * CASILLA, fila_celda * CASILLA
                    celdas_imagen.append(
                        (clase, recorte[y0:y0 + CASILLA, x0:x0 + CASILLA].copy())
                    )
        if nombre in elegidas_preview:
            celdas_preview[nombre] = celdas_imagen

        if indice % 100 == 0 or indice == len(lote):
            print(f"  {indice}/{len(lote)} compuestos", flush=True)

    # ---- CSV completo: se conservan las filas de imagenes fuera del lote ----
    filas_finales = dict(previas)
    filas_finales.update(filas_nuevas)
    conservadas = len(set(filas_finales) - set(filas_nuevas))
    orden = {fila["imagen"]: i for i, fila in enumerate(manifiesto)}
    indice_casilla = {_casilla(f, c): f * 8 + c for f in range(8) for c in range(8)}
    with open(csv_ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for (_imagen, _casilla_csv), fila in sorted(
            filas_finales.items(),
            key=lambda par: (orden.get(par[0][0], 10 ** 9), par[0][0],
                             indice_casilla[par[0][1]]),
        ):
            escritor.writerow(fila)

    # ---- previews: hoja 8x8 con la clase de cada casilla ----
    previews: list[Path] = []
    if not argumentos.sin_previews:
        for numero, nombre in enumerate(sorted(celdas_preview), start=1):
            fila = next(f for f in manifiesto if Path(f["imagen"]).stem == nombre)
            titulo = (f"{nombre}  perfil={fila.get('perfil', '?')}  "
                      f"modo={fila.get('modo', '?')}  fichas={fila.get('n_fichas', '?')}")
            destino = PREVIEWS / f"casilla_muestra_{numero:02d}.jpg"
            cv2.imwrite(str(destino), _hoja_casillas(celdas_preview[nombre], titulo),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            previews.append(destino)

    # ---- verificacion ----
    verificacion = _verificar_csv(csv_ruta, clases, argumentos.verificar, rng)
    filas_csv, invalidas_csv, leidas, detalle_invalidas = verificacion
    procesadas_set = {Path(nombre).stem for nombre in procesadas}
    ocupadas_esperadas = sum(
        1 for imagen, _casilla_rot in rotaciones
        if Path(imagen).stem in procesadas_set
    )
    fichas_manifiesto = sum(
        int(fila.get("n_fichas") or 0) for fila in lote
        if Path(fila["imagen"]).stem in procesadas_set
    )
    vacias_esperadas = len(procesadas) * 64 - ocupadas_esperadas
    recortes_disco = len(list(recortes_dir.glob("*.jpg")))

    faltan_clase = [c for c in clases + ["vacia"] if por_clase.get(c, 0) == 0]
    texto_vacias = ""
    if brillo["vacia"]["n"] > 0:
        ocupadas = [dato for clase, dato in brillo.items() if clase != "vacia"]
        suma_n = sum(d["n"] for d in ocupadas)
        media_ocupadas = sum(d["suma"] for d in ocupadas) / max(suma_n, 1.0)
        var_ocupadas = (sum(d["suma2"] for d in ocupadas) / max(suma_n, 1.0)
                        - media_ocupadas ** 2)
        media_vacia = brillo["vacia"]["suma"] / brillo["vacia"]["n"]
        var_vacia = brillo["vacia"]["suma2"] / brillo["vacia"]["n"] - media_vacia ** 2
        texto_vacias = (
            f"desv 'vacia' {np.sqrt(max(var_vacia, 0.0)):.2f} vs "
            f"ocupadas {np.sqrt(max(var_ocupadas, 0.0)):.2f} -> "
            f"{'OK' if var_vacia < var_ocupadas else 'AVISO'}"
        )
    if faltan_clase:
        texto_vacias += f"  AVISO: clases sin casillas {faltan_clase}"

    segundos = time.time() - inicio
    tamano_recortes = _tamano(recortes_dir)
    tamano_casillas = _tamano(casillas_dir)

    print()
    print(f"listo: lote {inicio_lote}..{fin_lote} en {segundos:.1f} s "
          f"({segundos / max(len(lote), 1) * 1000:.0f} ms/compuesto)")
    print(f"  recortes 800x800: {recortes_escritos} en el lote, "
          f"{recortes_disco} en disco -> {recortes_dir}")
    print(f"  casillas 100x100: {sum(por_clase.values())} en el lote -> {casillas_dir}")
    print(f"  por clase: {dict(sorted(por_clase.items()))}")
    print(f"  CSV: {filas_csv} filas ({conservadas} de imagenes fuera del lote) "
          f"-> {csv_ruta}")
    print(f"  filas invalidas en el CSV: {invalidas_csv} {detalle_invalidas}")
    print(f"  casillas releidas: {leidas} de "
          f"{min(argumentos.verificar, filas_csv)} pedidas (esperado 100x100x3)")
    print(f"  ocupadas por rotaciones: {ocupadas_esperadas}, "
          f"n_fichas del manifiesto: {fichas_manifiesto}, "
          f"vacias: {vacias_esperadas}")
    print(f"  brillo (media, desv) por clase:")
    for clase in clases + ["vacia"]:
        dato = brillo[clase]
        if dato["n"] <= 0:
            print(f"    {clase:<10} sin casillas")
            continue
        media = dato["suma"] / dato["n"]
        varianza = max(dato["suma2"] / dato["n"] - media ** 2, 0.0)
        print(f"    {clase:<10} media {media:6.2f}  desv {np.sqrt(varianza):5.2f}"
              f"  n {int(dato['n'] // (CASILLA * CASILLA))}")
    if texto_vacias:
        print(f"  {texto_vacias}")
    print(f"  borde inventado: max {borde_maximo * 100:.4f}% "
          f"({borde_imagen or 'ninguna'})")
    print(f"  sin homografia: {len(sin_homografia)} {sin_homografia[:5]}")
    print(f"  tamano: recortes {tamano_recortes / 1e6:.1f} MB, "
          f"casillas {tamano_casillas / 1e6:.1f} MB")
    print(f"  previews: {[p.name for p in previews] or 'omitidas'}")

    correcto = (
        recortes_escritos == len(lote)
        and len(procesadas) == len(lote)
        and not sin_homografia
        and invalidas_csv == 0
        and filas_csv == len(procesadas) * 64 + conservadas
        and por_clase.get("vacia", 0) == vacias_esperadas
        and por_clase.get("vacia", 0) > sum(
            por_clase.get(c, 0) for c in clases
        )
        and borde_maximo < 1e-9
    )
    print(f"RESULTADO: {'ok' if correcto else 'revisar (ver conteos arriba)'}")
    return 0 if correcto else 1


if __name__ == "__main__":
    raise SystemExit(main())
