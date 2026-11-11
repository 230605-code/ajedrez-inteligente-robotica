#!/usr/bin/env python
"""Smoke test local (CPU) del detector v2 empaquetado para Colab.

No entrena nada: carga los pesos del bundle, comprueba que el orden de clases
es el de `classes.txt` y corre un `val` diminuto sobre unas pocas imagenes REALES
del split `test` del propio bundle. Sirve para detectar el paquete roto (clases
cruzadas, labels desalineados, zip incompleto) antes de gastar GPU en Colab.

Uso:
    python scripts/verificar_detector_v2.py --bundle dataset/colab_bundle/detector_v2.zip
    python scripts/verificar_detector_v2.py --bundle ... --n 5 --imgsz 320 --batch 2
    python scripts/verificar_detector_v2.py --dir dataset/colab_bundle/detector

El zip solo se descomprime en un directorio temporal fuera del proyecto y se
borra al terminar (salvo `--conservar`).
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import yaml

from comun import RAIZ

PESO_DENTRO_DEL_ZIP = "weights/modelo_sintetico.pt"


def extraer(zip_bundle: Path, destino: Path, split: str, n: int) -> tuple[Path, Path]:
    """Descomprime data.yaml, pesos y las primeras `n` imagenes (+labels) del split."""
    with zipfile.ZipFile(zip_bundle) as zf:
        nombres = zf.namelist()
        for obligatorio in ("data.yaml", PESO_DENTRO_DEL_ZIP):
            if obligatorio not in nombres:
                raise SystemExit(f"{zip_bundle} no contiene {obligatorio}")

        zf.extract("data.yaml", destino)
        zf.extract(PESO_DENTRO_DEL_ZIP, destino)

        prefijo_img = f"images/{split}/"
        imagenes = sorted(n for n in nombres if n.startswith(prefijo_img) and n.endswith(".jpg"))
        if not imagenes:
            raise SystemExit(f"El bundle no tiene imagenes en {prefijo_img}")
        elegidas = imagenes[:n]
        for imagen in elegidas:
            zf.extract(imagen, destino)
            etiqueta = f"labels/{split}/{Path(imagen).stem}.txt"
            if etiqueta not in nombres:
                raise SystemExit(f"{imagen} no tiene etiqueta en el bundle")
            zf.extract(etiqueta, destino)

    pesas = destino / PESO_DENTRO_DEL_ZIP
    yaml_origen = destino / "data.yaml"
    return pesas, yaml_origen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="detector_v2.zip")
    parser.add_argument("--dir", type=Path, help="carpeta detector/ ya descomprimida")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--n", type=int, default=3, help="imagenes del smoke (default 3)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--conservar", action="store_true",
                        help="no borra el directorio temporal al terminar")
    args = parser.parse_args()

    if not args.bundle and not args.dir:
        parser.error("indica --bundle <zip> o --dir <carpeta detector>")

    temporal = Path(tempfile.mkdtemp(prefix="verif_detector_v2_"))
    inicio = time.time()
    try:
        if args.dir:
            base = args.dir
            pesos = base / PESO_DENTRO_DEL_ZIP
            yaml_origen = base / "data.yaml"
            split_dir = base / "images" / args.split
            if not pesos.exists() or not yaml_origen.exists():
                raise SystemExit(f"{base} no parece un bundle del detector")
            imagenes = sorted(split_dir.glob("*.jpg"))[:args.n]
            if not imagenes:
                raise SystemExit(f"No hay imagenes en {split_dir}")
            n_imagenes = len(imagenes)
        else:
            if not args.bundle.exists():
                raise SystemExit(f"No encuentro {args.bundle}")
            pesos, yaml_origen = extraer(args.bundle, temporal, args.split, args.n)
            n_imagenes = len(sorted((temporal / "images" / args.split).glob("*.jpg")))

        cfg = yaml.safe_load(yaml_origen.read_text(encoding="utf-8"))
        nombres = [cfg["names"][i] if isinstance(cfg["names"], dict) else cfg["names"][i]
                   for i in range(cfg["nc"])]
        esperadas = [
            linea.strip() for linea in (RAIZ / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
            if linea.strip() and not linea.lstrip().startswith("#")
        ]
        print(f"nc={cfg['nc']}  names={nombres}")
        if len(nombres) != 12:
            raise SystemExit(f"El data.yaml declara {len(nombres)} clases, se esperaban 12")
        if esperadas and nombres != esperadas:
            raise SystemExit(f"Orden de clases distinto de classes.txt\n  bundle: {nombres}\n  esperado: {esperadas}")
        print("Orden de clases: OK (identico a classes.txt)")

        # data.yaml del smoke: solo las imagenes copiadas
        if args.dir:
            conjunto = yaml_origen
        else:
            smoke = temporal / "smoke"
            (smoke / "images" / args.split).mkdir(parents=True)
            (smoke / "labels" / args.split).mkdir(parents=True)
            for imagen in sorted((temporal / "images" / args.split).glob("*.jpg")):
                shutil.copyfile(imagen, smoke / "images" / args.split / imagen.name)
                etiqueta = temporal / "labels" / args.split / (imagen.stem + ".txt")
                shutil.copyfile(etiqueta, smoke / "labels" / args.split / etiqueta.name)
            cfg_smoke = dict(cfg)
            cfg_smoke["path"] = str(smoke)
            cfg_smoke["val"] = f"images/{args.split}"
            cfg_smoke["test"] = f"images/{args.split}"
            conjunto = temporal / "smoke.yaml"
            conjunto.write_text(yaml.safe_dump(cfg_smoke, sort_keys=False), encoding="utf-8")

        from ultralytics import YOLO

        modelo = YOLO(str(pesos))
        print(f"Pesos: {pesos.name} ({pesos.stat().st_size / 1e6:.1f} MB)")
        print(f"names del modelo: {[modelo.names[i] for i in sorted(modelo.names)]}")
        if [modelo.names[i] for i in sorted(modelo.names)] != nombres:
            raise SystemExit("Los names del .pt no coinciden con data.yaml")

        metricas = modelo.val(
            data=str(conjunto),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            plots=False,
            verbose=False,
            project=str(temporal / "runs"),
            name="smoke",
            exist_ok=True,
        )
        segundos = time.time() - inicio
        print()
        print(f"SMOKE {args.split}: imagenes={n_imagenes} imgsz={args.imgsz} batch={args.batch}")
        print(f"  mAP50    : {metricas.box.map50:.4f}")
        print(f"  mAP50-95 : {metricas.box.map:.4f}")
        print(f"  precision: {metricas.box.mp:.4f}")
        print(f"  recall   : {metricas.box.mr:.4f}")
        print(f"  segundos : {segundos:.1f}")
        if segundos > 180:
            print("  AVISO: mas de 3 minutos en CPU; mide en Colab, no aqui.")
        print("Smoke OK: el bundle carga, las clases estan en orden y el val corre.")
        return 0
    finally:
        if args.conservar:
            print(f"Directorio temporal conservado: {temporal}")
        else:
            shutil.rmtree(temporal, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
