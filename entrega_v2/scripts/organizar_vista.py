"""Vista organizada del dataset v2: por pipeline y por categoria.

Estructura que se genera (HARDLINKS: 0 MB extra, NTFS apunta al mismo bytes):

  dataset/organizado/
  ├── deteccion/          frames COMPLETOS con fondo (para el detector YOLO)
  │   ├── 01_inicial/     {sint_*, comp_*, real_*}.jpg
  │   ├── 02_medio_juego/
  │   ├── 03_finales/
  │   ├── 04_dificiles/
  │   └── 05_tablero_vacio/
  ├── recortes_tablero/   SOLO el tablero 800x800 rectificado (para el
  │   └── 01..05/         clasificador por casilla)
  └── README.md           mapa de todo (donde vive cada cosa, incl. casillas)

Esa estructura espejo el网球 del flujo: el detector entrena con fotos tal cual
sale de la camara y el clasificador con tableros rectificados.

Mapeo de categoria:
  * sintetico: columna 'categoria' del manifiesto (del generador).
  * compuestos: modo -> categoria:
      legal -> 02_medio_juego | finales -> 03_finales | escasa -> 04_dificiles
      vacio -> 05_tablero_vacio | aleatoria -> 04_dificiles
  * reales: la carpeta de crudo donde capturo el usuario (T4b la deja asi).
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
DESTINO = RAIZ / "dataset" / "organizado"

MAPEO_COMPUESTO = {
    "legal": "02_medio_juego",
    "finales": "03_finales",
    "escasa": "04_dificiles",
    "vacio": "05_tablero_vacio",
    "aleatoria": "04_dificiles",
}

FUENTES = {
    "sintetico": (RAIZ / "dataset" / "sintetico" / "images",
                  RAIZ / "dataset" / "sintetico" / "manifesto.csv",
                  RAIZ / "dataset" / "sintetico" / "recortes" / "800x800"),
    "compuesto": (RAIZ / "dataset" / "reales_compuesto" / "images",
                  RAIZ / "dataset" / "reales_compuesto" / "manifiesto.csv",
                  RAIZ / "dataset" / "reales_compuesto" / "recortes" / "800x800"),
}

CARPETAS_CAT = ["01_inicial", "02_medio_juego", "03_finales",
                "04_dificiles", "05_tablero_vacio"]


def _vinculo(origen: Path, destino: Path) -> bool:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return True
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/H", str(destino), str(origen)],
            check=True, capture_output=True,
        )
        return destino.exists()
    except Exception:
        try:
            shutil.copy2(origen, destino)
            return True
        except Exception:
            return False


def limpiar_vista() -> None:
    if DESTINO.exists():
        # Elimina solo los LINKS/copias de la vista; los originales no se tocan.
        shutil.rmtree(DESTINO)


def organizar() -> None:
    limpiar_vista()
    resumen: Counter = Counter()

    for fuente, (dir_imagenes, ruta_manifiesto, dir_recortes) in FUENTES.items():
        if not dir_imagenes.exists():
            continue
        categorias: dict[str, str] = {}
        with ruta_manifiesto.open(encoding="utf-8-sig", newline="") as f:
            for fila in csv.DictReader(f):
                nombre = fila.get("imagen", "")
                if fuente == "sintetico":
                    categorias[nombre] = fila.get("categoria", "")
                else:
                    categorias[nombre] = MAPEO_COMPUESTO.get(fila.get("modo", ""), "")
        for nombre, categoria in categorias.items():
            origen = dir_imagenes / nombre
            destino = DESTINO / "deteccion" / categoria / f"{fuente[:4]}_{nombre}"
            if origen.exists() and _vinculo(origen, destino):
                resumen[f"deteccion/{categoria}"] += 1
            destino_recorte = (DESTINO / "recortes_tablero" / categoria
                               / f"{fuente[:4]}_{nombre}")
            recorte = dir_recortes / nombre
            if recorte.exists() and _vinculo(recorte, destino_recorte):
                resumen[f"recortes/{categoria}"] += 1

    # Reales: ya viven en carpetas de categoria, y sirven para AMBOS pipelines.
    for origen in (RAIZ / "dataset" / "reales" / "recortes" / "800x800").glob("*/*.jpg"):
        categoria = origen.parent.name
        destino = DESTINO / "deteccion" / categoria / f"real_{origen.name}"
        if _vinculo(origen, destino):
            resumen[f"deteccion/{categoria}"] += 1
        destino_recorte = (DESTINO / "recortes_tablero" / categoria
                           / f"real_{origen.name}")
        if _vinculo(origen, destino_recorte):
            resumen[f"recortes/{categoria}"] += 1

    README_DESTINO = DESTINO / "recortes_tablero" / CARPETAS_CAT[0] / "placeholder"
    _ = README_DESTINO
    _escribir_readme(resumen)

    print("Vista organizada en dataset/organizado/:")
    for clave in sorted(resumen):
        print(f"  {clave:34s} {resumen[clave]:6d}")
    print(f"  TOTAL {sum(resumen.values())}")


def _escribir_readme(resumen: Counter) -> None:
    por_deteccion: Counter = Counter()
    por_recorte: Counter = Counter()
    for clave, n in resumen.items():
        destino, categoria = clave.split("/")
        (por_deteccion if destino == "deteccion" else por_recorte)[categoria] += n

    texto = []
    texto.append("# Mapa del dataset v2 (archivo legible, generado por organizar_vista.py)")
    texto.append("")
    texto.append("Cada archivo esta TAMBIEN en su carpeta origen (hardlink): no hay")
    texto.append("copias dobles. Prefix del archivo = fuente:")
    texto.append("  sint_ = sintetico de plantilla | comp_ = compuesto con sprites reales")
    texto.append("  real_ = foto rectificada de la camara del usuario")
    texto.append("")
    texto.append("## deteccion/ -> frames COMPLETOS con fondo: entrenamiento del YOLO detector")
    for categoria in CARPETAS_CAT:
        texto.append(f"  {categoria:18s} {por_deteccion[categoria]:6d}")
    texto.append("")
    texto.append("## recortes_tablero/ -> SOLO tablero 800x800 rectificado: clasificador")
    for categoria in CARPETAS_CAT:
        texto.append(f"  {categoria:18s} {por_recorte[categoria]:6d}")
    texto.append("")
    texto.append("## donde vive cada cosa dentro del proyecto:")
    texto.append("  dataset/crudo/<categoria>             fotos SIN procesar del usuario")
    texto.append("  dataset/reales/recortes/800x800/<cat> recortes de fotos REALES 800x800")
    texto.append("  dataset/reales_compuesto/images       tableros compuestos (fondo real)")
    texto.append("  dataset/reales_compuesto/recortes/800x800    version tablero-only")
    texto.append("  dataset/reales_compuesto/casillas/clasificador/<clase>  100x100 por CASILLA")
    texto.append("       13 carpetas: peon_b..rey_n + vacia. Labels: etiquetas_casillas.csv")
    texto.append("  dataset/reales/etiquetas_clasificador.csv    labels de las casillas reales")
    texto.append("  dataset/sintetico/{images,labels}     tableros sinteticos + labels YOLO")
    texto.append("  dataset/sintetico/recortes/800x800    version tablero-only de los sinteticos")
    texto.append("  dataset/reales/augmentado/images/<cat> variantes de luz de los reales")
    texto.append("  dataset/auditoria/  auditoria de capturas + manifiesto.csv (+ detalles)")
    texto.append("  odd/tasks/ajedrez-v2.md                  plan del proyecto (spec y progreso)")
    ruta = DESTINO / "README.md"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(texto), encoding="utf-8")


if __name__ == "__main__":
    organizar()
