"""Auditoria del dataset crudo: mide cada foto sin depender del modelo.

Recorre las 5 carpetas de `dataset/crudo/`, mide cada imagen (resolucion,
fuente deducida, brillo, nitidez, EXIF y estado del tablero) y deja un
manifiesto en `dataset/auditoria/manifesto.csv`.

Dedupe exacto por SHA-256 de los BYTES del archivo entre todas las carpetas.
Los duplicados NO se borran ni se mueven: se COPIAN a
`dataset/auditoria/duplicados/` y quedan anotados en el CSV con su original.

Reglas:
  - No se modifica, renombra ni mueve ninguna foto original: solo se copia.
  - El estado del tablero reutiliza `detectar_tablero` de
    `diagnostico_esp32.py`; no se carga ningun modelo.
  - Sin PIL, la columna `exif_rotacion` queda en "-".

Uso:
    python scripts/auditar_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import CARPETAS, imagenes  # noqa: E402
from diagnostico_esp32 import detectar_tablero  # noqa: E402

CRUDO = RAIZ / "dataset" / "crudo"
AUDITORIA = RAIZ / "dataset" / "auditoria"
DUPLICADOS = AUDITORIA / "duplicados"
MANIFIESTO = AUDITORIA / "manifesto.csv"

COLUMNAS = [
    "archivo",
    "carpeta_origen",
    "fuente",
    "anchoxalto",
    "brillo",
    "nitidez",
    "exif_rotacion",
    "tablero",
    "duplicado_de",
    "nota",
]

# Ancho maximo de la ESP32-CAM en su modo mas alto (UXGA). Un nombre que
# sugiere esp32 con una imagen mucho mas grande viene de otra camara.
ANCHO_MAX_ESP32 = 1600

NOMBRE_EXIF_ORIENTACION = 274


def clave_natural(ruta: Path):
    """Orden de captura: capture (2) antes que capture (10)."""
    return [
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r"(\d+)", ruta.name)
    ]


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def deducir_fuente(nombre: str) -> str:
    """Fuente por el nombre: pipeline > esp32 > celular > desconocido.

    El pipeline renombra a `01_inicial_001.jpg`; las descargas de la ESP32
    llegan como `capture*.jpg` o `cap*.jpg`; el celular deja su marca de tiempo
    `YYYYMMDD_HHMMSS.jpg`.
    """
    base = nombre.lower()
    if base.startswith(tuple(c.lower() for c in CARPETAS)):
        return "pipeline"
    if base.startswith(("capture", "cap")):
        return "esp32"
    if re.match(r"^\d{8}_\d{6}", base):
        return "celular"
    return "desconocido"


def leer_exif_rotacion(ruta: Path) -> str:
    """Valor EXIF de orientacion (274), o "-" si no hay PIL o no existe."""
    try:
        from PIL import Image
    except Exception:  # sin PIL el campo se omite
        return "-"
    try:
        with Image.open(ruta) as imagen:
            valor = imagen.getexif().get(NOMBRE_EXIF_ORIENTACION)
    except Exception:
        return "-"
    return str(valor) if valor is not None else "-"


def contar_casillas(imagen) -> int:
    """Casillas cuadradas de la rejilla detectadas como contornos de 4 lados.

    Es solo una corroboracion: `detectar_tablero` necesita un cuadrilatero
    cerrado y el margen blanco del tablero impreso se pierde contra fondos
    claros, asi que devuelve "ausente" en fotos donde el tablero si esta. Un
    tablero real deja decenas de casillas de 4 lados; un fondo sin tablero, no.
    """
    import cv2

    alto, ancho = imagen.shape[:2]
    area_imagen = alto * ancho
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 40, 120)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    casillas = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro <= 0 or not (area_imagen * 0.0003 < area < area_imagen * 0.03):
            continue
        aproximado = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
        if len(aproximado) != 4:
            continue
        _, _, ancho_caja, alto_caja = cv2.boundingRect(aproximado)
        if ancho_caja < 8 or alto_caja < 8:
            continue
        if 0.6 < ancho_caja / alto_caja < 1.7:
            casillas += 1
    return casillas


def medir(ruta: Path) -> dict[str, str]:
    """Una fila del manifiesto. Devuelve `tablero="ilegible"` si no abre."""
    import cv2

    fila = {
        "archivo": ruta.name,
        "carpeta_origen": ruta.parent.name,
        "fuente": deducir_fuente(ruta.name),
        "anchoxalto": "-",
        "brillo": "-",
        "nitidez": "-",
        "exif_rotacion": leer_exif_rotacion(ruta),
        "tablero": "-",
        "duplicado_de": "",
        "nota": "",
    }
    notas: list[str] = []

    imagen = cv2.imread(str(ruta))
    if imagen is None:
        fila["tablero"] = "ilegible"
        notas.append("no se pudo decodificar con OpenCV")
        fila["nota"] = "; ".join(notas)
        return fila

    alto, ancho = imagen.shape[:2]
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    brillo = float(gris.mean())
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    fila["anchoxalto"] = f"{ancho}x{alto}"
    fila["brillo"] = f"{brillo:.1f}"
    fila["nitidez"] = f"{nitidez:.1f}"
    fila["tablero"] = detectar_tablero(imagen)
    if fila["tablero"] == "ausente":
        casillas = contar_casillas(imagen)
        if casillas >= 8:
            notas.append(
                f"rejilla: {casillas} casillas detectadas -> tablero presente "
                "(la heuristica de contornos da ausente)"
            )

    if fila["fuente"] == "esp32" and ancho > ANCHO_MAX_ESP32:
        notas.append(
            f"el nombre sugiere esp32 pero mide {ancho}px de ancho "
            f"(>{ANCHO_MAX_ESP32}px de la ESP32)"
        )
    if brillo < 40:
        notas.append("muy oscura")
    elif brillo > 190:
        notas.append("muy sobreexpuesta")
    if fila["exif_rotacion"] not in ("-", "1"):
        notas.append(
            f"EXIF orientacion={fila['exif_rotacion']} (espurio en este lote): "
            "los pixeles ya estan derechos; NO aplicar exif_transpose, "
            "normalizar el giro con los marcadores ArUco del tablero"
        )

    fila["nota"] = "; ".join(notas)
    return fila


def copiar_duplicado(ruta: Path, carpeta: Path) -> Path:
    """Copia un duplicado a auditoria/duplicados con nombre que no colisione."""
    destino = carpeta / f"{ruta.parent.name}__{ruta.name}"
    contador = 1
    while destino.exists() and destino.stat().st_size != ruta.stat().st_size:
        contador += 1
        destino = carpeta / f"{ruta.parent.name}__{ruta.stem}_{contador}{ruta.suffix}"
    shutil.copy2(ruta, destino)
    return destino


def main() -> int:
    AUDITORIA.mkdir(parents=True, exist_ok=True)
    DUPLICADOS.mkdir(parents=True, exist_ok=True)

    filas: list[dict[str, str]] = []
    hashes: dict[str, Path] = {}
    total = 0
    duplicados = 0
    ilegibles = 0

    for nombre_carpeta in CARPETAS:
        carpeta = CRUDO / nombre_carpeta
        if not carpeta.is_dir():
            print(f"AVISO: no existe {carpeta}")
            continue
        for ruta in sorted(imagenes(carpeta), key=clave_natural):
            total += 1
            fila = medir(ruta)
            digest = sha256(ruta)

            original = hashes.get(digest)
            if original is None:
                hashes[digest] = ruta
            else:
                duplicados += 1
                fila["duplicado_de"] = f"{original.parent.name}/{original.name}"
                copia = copiar_duplicado(ruta, DUPLICADOS)
                extra = f"duplicado exacto de {original.parent.name}/{original.name}"
                extra += f"; copia en auditoria/duplicados/{copia.name}"
                fila["nota"] = f"{fila['nota']}; {extra}" if fila["nota"] else extra

            if fila["tablero"] == "ilegible":
                ilegibles += 1
            filas.append(fila)

    with MANIFIESTO.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS, delimiter=";")
        escritor.writeheader()
        escritor.writerows(filas)

    por_carpeta = Counter(f["carpeta_origen"] for f in filas)
    por_fuente = Counter(f["fuente"] for f in filas)
    por_tablero = Counter(f["tablero"] for f in filas)
    por_resolucion = Counter(f["anchoxalto"] for f in filas)

    print("Auditoria del dataset crudo")
    print(f"Fotos procesadas: {total}")
    print(f"Unicas por SHA-256: {len(hashes)}")
    print(f"Duplicadas exactas: {duplicados} (copiadas a {DUPLICADOS})")
    print(f"Ilegibles: {ilegibles}")

    print("\nPor carpeta:")
    for nombre_carpeta in CARPETAS:
        print(f"  {nombre_carpeta}: {por_carpeta.get(nombre_carpeta, 0)}")
    print("\nPor fuente:")
    for fuente, cuenta in sorted(por_fuente.items()):
        print(f"  {fuente}: {cuenta}")
    print("\nPor resolucion:")
    for resolucion, cuenta in por_resolucion.most_common():
        print(f"  {resolucion}: {cuenta}")
    print("\nTablero:")
    for estado in ("completo", "cortado", "ausente", "ilegible"):
        print(f"  {estado}: {por_tablero.get(estado, 0)}")
    print(f"\nManifiesto: {MANIFIESTO}  ({len(filas)} filas)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
