"""Entrenamiento local del detector YOLO en ajedrez-v2.

Permite entrenar o afinar el modelo detector sobre CPU o GPU localmente
sin depender de Google Colab.

Uso:
    python scripts/entrenar_local_detector.py --epocas 30 --imgsz 640
    python scripts/entrenar_local_detector.py --data dataset/colab_bundle/detector_v2_clean --epocas 10
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import cargar_clases  # noqa: E402

DATASET_DEFAULT = RAIZ / "dataset" / "colab_bundle" / "detector_v2_clean"
RUNS = RAIZ / "runs" / "detector"


def preparar_data_yaml(directorio: Path) -> Path:
    """Asegura un data_local.yaml con la ruta absoluta local."""
    clases = cargar_clases()
    nombres_dict = {i: c for i, c in enumerate(clases)}
    yaml_lines = [
        f"path: {directorio.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(clases)}",
        "names:",
    ]
    for i, c in nombres_dict.items():
        yaml_lines.append(f"  {i}: {c}")

    yaml_path = directorio / "data_local.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET_DEFAULT,
                        help="Directorio del dataset con images/ y labels/")
    parser.add_argument("--modelo", default="yolo11n.pt",
                        help="Pesos base o checkpoint inicial (yolo11n.pt o modelos/best.pt)")
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--paciencia", type=int, default=15)
    parser.add_argument("--nombre", default=None)
    parser.add_argument("--device", default=None, help="'cpu', '0', etc.")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: no existe el directorio de datos {args.data}")
        return 1

    yaml_config = preparar_data_yaml(args.data)

    import torch
    from ultralytics import YOLO

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"

    nombre = args.nombre or time.strftime("detector_%Y%m%d_%H%M%S")
    print(f"Entrenamiento local YOLO ajedrez-v2")
    print(f"Modelo: {args.modelo} | Epocas: {args.epocas} | Imgsz: {args.imgsz} | Device: {args.device}")
    print(f"Dataset: {yaml_config}")

    modelo = YOLO(args.modelo)
    resultados = modelo.train(
        data=str(yaml_config),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.lote,
        workers=args.workers,
        patience=args.paciencia,
        device=args.device,
        project=str(RUNS),
        name=nombre,
        plots=True,
        val=True,
        exist_ok=False,
        verbose=True,
    )

    guardado = Path(resultados.save_dir)
    mejor = guardado / "weights" / "best.pt"
    print("\n=== Resumen de Entrenamiento ===")
    caja = resultados.box
    nombres = resultados.names if hasattr(resultados, "names") else modelo.names
    print(f"mAP50:    {caja.map50:.4f}")
    print(f"mAP50-95: {caja.map:.4f}")
    for i, nombre_clase in nombres.items():
        ap = caja.ap50[i] if i < len(caja.ap50) else float("nan")
        print(f"  {nombre_clase:12s} AP50: {ap:.4f}")

    if mejor.exists():
        destino = RAIZ / "modelos" / "best_local_detector.pt"
        shutil.copy2(mejor, destino)
        print(f"\nMejor modelo guardado en: {destino}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
