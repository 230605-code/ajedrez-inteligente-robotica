"""Genera un clasificador LIGERO para Colab gratuito.

Problema medido: el clasificador tenia 174,000 casillas y el 55% era 'vacia'
(95,000 casillas de vacia en el train): demasiadas, y el balance hace que la
GPU se corte antes de terminar las 20 epocas en Colab gratuito.

Reduce 'vacia' en el train a ~18,000 (0.85 aleatorias), deja intacto el resto,
y forma un zip nuevo con las misma casillas de esta muestra y de test/valid.

Salida:
    dataset\colab_bundle\clasificador_ligero_v2.zip
"""
from __future__ import annotations

import random
import zipfile
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
ORIGEN = RAIZ / "dataset" / "colab_bundle" / "clasificador_v2_clean.zip"
SALIDA = RAIZ / "dataset" / "colab_bundle" / "clasificador_ligero_v2.zip"
VACIAS_TRAIN_MAX = 18000

random.seed(42)


def main() -> int:
    z_origen = zipfile.ZipFile(ORIGEN)
    splits = {}
    for split in ("train", "valid", "test"):
        lineas = z_origen.read(f"split/{split}.txt").decode("utf-8").splitlines()
        splits[split] = [l.strip() for l in lineas if l.strip()]

    total_vacias_train = sum(1 for l in splits["train"] if "/vacia/" in l)
    print(f"original train: {len(splits['train'])} items "
          f"(vacias: {total_vacias_train})")

    # Downsampler de 'vacia' en TRAIN
    nuevas_train = []
    vacias_restantes = 0
    for item in splits["train"]:
        if "/vacia/" in item:
            if vacias_restantes < VACIAS_TRAIN_MAX:
                nuevas_train.append(item)
                vacias_restantes += 1
            # si excede: lo saltamos
        else:
            nuevas_train.append(item)
    splits["train"] = sorted(nuevas_train)
    print(f"nuevo train: {len(splits['train'])} items "
          f"(vacias: {sum(1 for l in splits['train'] if '/vacia/' in l)})")

    # Archivos referenciados por los splits (unicamente los necesarios);
    # el resto de casillas no entra al zip (peso bajo).
    necesarios: set[str] = set()
    for split in ("train", "valid", "test"):
        for item in splits[split]:
            necesarios.add(item[len("casillas/"):] if item.startswith("casillas/")
                          else item)

    escritor = zipfile.ZipFile(SALIDA, "w", zipfile.ZIP_DEFLATED)
    for nombre in z_origen.namelist():
        if nombre.startswith("casillas/"):
            casilla_rel = nombre[len("casillas/"):]
            if casilla_rel not in necesarios:
                continue
        elif nombre.endswith(".txt") and "split" in nombre:
            if nombre.endswith("split/train.txt"):
                data = "\n".join(splits["train"]).encode("utf-8")
            elif nombre.endswith("split/valid.txt"):
                data = "\n".join(splits["valid"]).encode("utf-8")
            elif nombre.endswith("split/test.txt"):
                data = "\n".join(splits["test"]).encode("utf-8")
            else:
                data = z_origen.read(nombre)
            escritor.writestr(nombre, data)
            continue
        escritor.writestr(nombre, z_origen.read(nombre))
    escritor.close()

    print(f"\nclasificador LIGERO: {SALIDA}")
    print(f"peso: {SALIDA.stat().st_size / 1024 / 1024:.1f} MB "
          f"(antes: {ORIGEN.stat().st_size / 1024 / 1024:.1f} MB)")
    for split in ("train", "valid", "test"):
        print(f"  {split}: {len(splits[split])} casillas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
