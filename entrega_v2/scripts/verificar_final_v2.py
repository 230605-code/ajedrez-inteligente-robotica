"""Verificacion FINAL del entrenamiento desde cero (pesos de Colab nov-2026).

Mide contra el MISMO test real:
  * detector: best.pt sobre las 110 fotos reales de detector_v2_clean/images/test
  * clasificador: best_clasificador.pt sobre 600 casillas muestreadas del
    split/test.txt de clasificador_v2_clean.zip
Y los compara contra los numeros de la corrida anterior (mAP50 0.55 / 0.71,
clasificador 99.3%).
"""
from __future__ import annotations

import random
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

BUNDLE_DET = RAIZ / "dataset" / "colab_bundle" / "detector_v2_clean"
BUNDLE_CLA = RAIZ / "dataset" / "colab_bundle" / "clasificador_v2_clean.zip"


def verificar_detector() -> None:
    from ultralytics import YOLO
    from comun import cargar_clases

    clases = cargar_clases()
    modelo = YOLO(str(RAIZ / "modelos" / "best.pt"))
    nombres = [modelo.names[i] for i in range(len(modelo.names))]
    if nombres != clases:
        print("ORDEN MAL — no evaluar")
        return
    print("detector: 12 clases en el orden correcto OK")

    temporal = BUNDLE_DET / "_val_final"
    temporal.mkdir(parents=True, exist_ok=True)
    yaml = BUNDLE_DET / "data.yaml"
    texto = yaml.read_text(encoding="utf-8")
    texto = re.sub(r"^path: .*$", f"path: {BUNDLE_DET.as_posix()}", texto,
                   flags=re.M)
    destino = temporal / "data.yaml"
    destino.write_text(texto, encoding="utf-8")

    for split in ("test", "val"):
        metricas = modelo.val(data=str(destino), split=split, imgsz=640,
                              batch=8, verbose=False,
                              project=str(RAIZ / "resultados" / "final_v2"),
                              name=f"final_{split}", exist_ok=True)
        print(f"[{split}] mAP50 {metricas.box.map50:.4f}  "
              f"mAP50-95 {metricas.box.map:.4f}  "
              f"P {metricas.box.mp:.4f}  R {metricas.box.mr:.4f}")


def verificar_clasificador() -> None:
    import torch
    import torchvision
    import torchvision.transforms as T

    transformar = T.Compose([T.ToTensor(),
                             T.Resize((128, 128), antialias=True),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    ck = torch.load(RAIZ / "modelos" / "best_clasificador.pt",
                    map_location="cpu", weights_only=False)
    clases = ck["clases"]
    net = torchvision.models.mobilenet_v3_small(weights=None)
    import torch.nn as nn
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(clases))
    net.load_state_dict(ck["modelo"])
    net.eval()

    z = zipfile.ZipFile(BUNDLE_CLA)
    test = [l.strip() for l in z.read("split/test.txt").decode("utf-8")
            .splitlines() if l.strip()]
    random.seed(42)
    muestra = random.sample(test, min(1200, len(test)))

    aciertos = total = 0
    por_clase: dict[str, list] = {}
    for item in muestra:
        clase_verdad = item.split("/")[-2]
        datos = np.frombuffer(z.read(item), np.uint8)
        img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        if img is None:
            continue
        tensor = transformar(cv2.cvtColor(np.ascontiguousarray(img),
                                          cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            sal = net(tensor)[0]
        prob = int(sal.argmax().item())
        total += 1
        d = por_clase.setdefault(clase_verdad, {"n": 0, "ok": 0})
        d["n"] += 1
        if prob == clases.index(clase_verdad):
            aciertos += 1
            d["ok"] += 1
    print(f"\nclasificador: {aciertos}/{total} "
          f"({100.0 * aciertos / max(total, 1):.2f}%)")
    for clase, d in sorted(por_clase.items()):
        print(f"  {clase:10s} {100.0 * d['ok'] / max(d['n'], 1):5.1f}% "
              f"({d['ok']}/{d['n']})")


if __name__ == "__main__":
    print("=== DETECTOR ===")
    verificar_detector()
    print("\n=== CLASIFICADOR ===")
    verificar_clasificador()
