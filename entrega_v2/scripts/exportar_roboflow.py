"""Descarga una version del dataset de Roboflow en formato YOLO para ajedrez-v2.

Uso:
    python scripts/exportar_roboflow.py --version 1
    python scripts/exportar_roboflow.py --version 1 --formato yolov8
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import EXPORTS, PROYECTO, WORKSPACE, api_key, cargar_clases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True, help="Numero de version en Roboflow")
    parser.add_argument("--formato", default="yolov8", help="Formato de exportacion (default: yolov8)")
    parser.add_argument("--workspace", default=WORKSPACE, help=f"Workspace Roboflow (default: {WORKSPACE})")
    parser.add_argument("--proyecto", default=PROYECTO, help=f"Proyecto Roboflow (default: {PROYECTO})")
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key())
    project = rf.workspace(args.workspace).project(args.proyecto)
    version = project.version(args.version)

    EXPORTS.mkdir(parents=True, exist_ok=True)
    destino = EXPORTS / f"{args.proyecto}-v{args.version}"
    print(f"Descargando {args.workspace}/{args.proyecto} v{args.version} como {args.formato}...")
    dataset = version.download(args.formato, location=str(destino), overwrite=True)
    raiz = Path(dataset.location)

    clases = cargar_clases()
    conteo = {}
    for split in ("train", "valid", "test"):
        carpeta = raiz / split / "labels"
        conteo[split] = len(list(carpeta.glob("*.txt"))) if carpeta.exists() else 0

    entrega = raiz / "ENTREGA.md"
    entrega.write_text(
        "\n".join(
            [
                "# Entrega del dataset (ajedrez-v2)",
                "",
                f"- Proyecto: `{args.workspace}/{args.proyecto}`",
                f"- Version: **{args.version}** (formato `{args.formato}`)",
                f"- Fecha: {date.today().isoformat()}",
                "",
                "## Clases",
                "",
                *[f"{i}. `{c}`" for i, c in enumerate(clases)],
                "",
                "## Imagenes por split",
                "",
                *[f"- `{split}`: {n} imagenes etiquetadas" for split, n in conteo.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nDataset descargado en: {raiz}")
    print(f"Resumen generado en: {entrega}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
