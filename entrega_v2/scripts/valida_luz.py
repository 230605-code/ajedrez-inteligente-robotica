"""Valida el modelo 'luz' contra el modelo viejo en 2 dominios:
1. muestras del test compuesto (que no retroceda otra clase)
2. las casillas del minidataset (la luz real de la ESP32, vista negra)
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))


def cargar(peso: Path):
    import torch.nn as nn

    ck = torch.load(peso, map_location="cpu", weights_only=False)
    clases = ck["clases"]
    net = torchvision.models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(clases))
    net.load_state_dict(ck["modelo"])
    net.eval()
    return net, clases


def main() -> int:
    import torchvision.transforms as T

    pre = T.Compose([T.ToTensor(), T.Resize((128, 128), antialias=True),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    viejo, clases = cargar(RAIZ / "modelos" / "best_clasificador.pt")
    nuevo, _ = cargar(RAIZ / "modelos" / "best_clasificador_luz.pt")
    indice = {c: i for i, c in enumerate(clases)}

    def clasificar(net, ruta: Path) -> int:
        img = cv2.imread(str(ruta))
        if img is None:
            return -1
        tensor = pre(cv2.cvtColor(np.ascontiguousarray(img),
                                  cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            salida = net(tensor)[0]
        return int(salida.argmax().item()), clases[int(salida.argmax().item())]

    random.seed(11)

    # 1) test compuesto (600 casillas)
    test_txt = (RAIZ / "dataset" / "colab_bundle" / "clasificador_split" /
                "test.txt").read_text(encoding="utf-8").splitlines()
    muestra = random.sample([l.strip() for l in test_txt if l.strip()], 600)
    print("== test compuesto (600) ==")
    for etiqueta, net_obj in (("viejo", viejo), ("nuevo", nuevo)):
        aciertos = total = 0
        por_clase = Counter()
        por_clase_t = Counter()
        for virtual in muestra:
            clase_verdad = virtual.split("/")[-2]
            ruta = RAIZ / "dataset" / "reales_compuesto" / virtual.replace(
                "casillas/", "casillas/clasificador/", 1)
            if ruta.exists() is False:
                ruta = RAIZ / "dataset" / "reales" / "clasificador" / virtual
                if ruta.exists() is False:
                    continue
            resultado = clasificar(net_obj, ruta)
            total += 1
            por_clase_t[clase_verdad] += 1
            if resultado[0] == indice[clase_verdad]:
                aciertos += 1
                por_clase[clase_verdad] += 1
        print(f"  {etiqueta}: {aciertos}/{total} "
              f"({100.0 * aciertos / max(total, 1):.2f}%)")
        visto = {}
        for clase in clases:
            n = por_clase_t[clase]
            if n and clase in ("caballo_n", "caballo_b", "alfil_n", "alfil_b"):
                visto[clase] = f"{100.0 * por_clase[clase] / n:.0f}%({n})"
        print("    " + str(visto))

    # 2) minidataset (luz real, vista negra y caseillario total)
    print("\n== minidataset (todo: luz real) ==")
    for etiqueta, net_obj in (("viejo", viejo), ("nuevo", nuevo)):
        aciertos = total = 0
        fallos: Counter = Counter()
        for clase in clases:
            carpeta = RAIZ / "dataset" / "minidataset" / "casillas" / clase
            for ruta in sorted(carpeta.glob("*.jpg")):
                resultado = clasificar(net_obj, ruta)
                total += 1
                if resultado[0] == indice[clase]:
                    aciertos += 1
                else:
                    fallos[f"{clase}->{resultado[1]}"] += 1
        print(f"  {etiqueta}: {aciertos}/{total} "
              f"({100.0 * aciertos / max(total, 1):.2f}%)")
        for par, n in fallos.most_common(5):
            print(f"    fallo {par}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
