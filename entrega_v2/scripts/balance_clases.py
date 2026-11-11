"""Cuenta cajas e imagenes por clase en un dataset YOLO de ajedrez-v2.

Sirve para detectar:
  - Clases subrepresentadas (pocas cajas en total).
  - Clases que aparecen en pocas imagenes (poca variedad de contexto).

Uso:
    python scripts/balance_clases.py
    python scripts/balance_clases.py --dataset dataset/sintetico
    python scripts/balance_clases.py --dataset dataset/reales_compuesto
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import cargar_clases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=RAIZ / "dataset" / "sintetico",
                        help="Ruta al directorio del dataset (con subcarpeta labels/)")
    args = parser.parse_args()

    clases = cargar_clases()
    labels_dir = args.dataset / "labels" if (args.dataset / "labels").exists() else args.dataset
    labels = sorted(labels_dir.glob("*.txt"))
    if not labels:
        print(f"No hay archivos .txt en {labels_dir}")
        return 1

    cajas: collections.Counter[int] = collections.Counter()
    imagenes: collections.Counter[int] = collections.Counter()
    for ruta in labels:
        vistas = set()
        for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
            linea_str = linea.strip()
            if not linea_str:
                continue
            partes = linea_str.split()
            try:
                clase = int(partes[0])
            except ValueError:
                continue
            cajas[clase] += 1
            vistas.add(clase)
        for clase in vistas:
            imagenes[clase] += 1

    total = sum(cajas.values())
    print(f"Dataset: {args.dataset}")
    print(f"Imagenes: {len(labels)} | Cajas totales: {total}")
    print(f"{'Clase':14s} {'ID':>3s} {'Cajas':>7s} {'%':>7s} {'Imagenes':>9s}")
    print("-" * 45)
    flojas = []
    for i, clase in enumerate(clases):
        porcentaje = 100 * cajas[i] / total if total else 0.0
        print(f"{clase:14s} {i:3d} {cajas[i]:7d} {porcentaje:6.1f}% {imagenes[i]:9d}")
        if imagenes[i] < 20:
            flojas.append(clase)

    if flojas:
        print(f"\nAVISO: clases presentes en menos de 20 imagenes -> {', '.join(flojas)}")
        return 1
    print("\nOK: todas las clases aparecen en al menos 20 imagenes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
