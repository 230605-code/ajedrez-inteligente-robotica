"""Experimento: rescatar caballo_n sin reentrenar, normalizando la luz.

Diagnosis: en las capturas nocturnas de la ESP32 el relleno del caballo negro
se apaga y el clasificador vota alfil; el modelo entreno con imagenes mas
claras (compuestos con fondo real + plantilla). Si NORMALIZO la casilla al
dominio de entrenamiento, la vision del glifo debe volver y la clase correcta
aparecer. Pruebo variantes de preprocesado + votacion por rotacion sobre las
casillas donde fallo el live test (b8, c8) y controles donde funciono.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))
from pipeline_fen import cargar_clasificador  # noqa: E402

CENITAL = RAIZ / "assets" / "previews" / "v2_vivo" / "v2_1_cenital.jpg"

OBJETIVOS = {"b8": "caballo_n", "c8": "alfil_n", "f8": "alfil_n",
             "g8": "caballo_n", "d8": "dama_n", "e8": "rey_n", "a8": "torre_n",
             "a1": "torre_b", "d1": "dama_b", "e1": "rey_b"}


def variantes(celda):
    """Generador de (nombre, imagen) para la misma casilla."""
    yield "original", celda

    # v1: CLAHE en el canal L (LAB): airea los simbolos apagados.
    lab = cv2.cvtColor(celda, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(
        lab[:, :, 0])
    yield "clahe", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # v2: gamma 1.35 (aclara) + saturacion x1.25 (recupera color de ficha).
    gamma = np.power(celda.astype("float32") / 255.0, 1 / 1.35) * 255.0
    claro = gamma.astype("uint8")
    hsv = cv2.cvtColor(claro, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.25)
    yield "gamma+sat", cv2.cvtColor(np.clip(hsv, 0, 255), cv2.COLOR_HSV2BGR)

    # v3: CLAHE + gamma combinados.
    lab = cv2.cvtColor(claro, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(
        lab[:, :, 0])
    yield "clahe+gamma", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


import torch

celda_blanco = None

def cargar_preparador():
    from torchvision import transforms as T_lib
    return T_lib.Compose([
        T_lib.ToTensor(),
        T_lib.Resize((128, 128), antialias=True),
        T_lib.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

def main() -> int:
    cenital = cv2.imread(str(CENITAL))
    if cenital is None:
        raise SystemExit(f"sin {CENITAL}: corre prueba_en_vivo_v2 primero")
    net, clases, info = cargar_clasificador(RAIZ / "modelos" / "best_clasificador.pt")
    transformacion = cargar_preparador()
    print("casilla | variante     | top1 (conf)        | top2 (conf) | leido")
    for casilla, esperado in OBJETIVOS.items():
        col = ord(casilla[0]) - ord("a")
        fila = 8 - int(casilla[1])
        celda = cenital[fila * 100:(fila + 1) * 100, col * 100:(col + 1) * 100]

        def predecir(img):
            entrada = transformacion(
                cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
            )
            with torch.no_grad():
                salida = net(entrada.unsqueeze(0))[0]
            probs = torch.softmax(salida, 0).numpy()
            orden = np.argsort(probs)[::-1]
            return probs

        print(f"\n{casilla}  (esperado: {esperado})")
        for nombre, img in variantes(celda):
            probs = predecir(img)
            orden = np.argsort(probs)[::-1]
            clase1, p1 = clases[int(orden[0])], float(probs[orden[0]])
            clase2, p2 = clases[int(orden[1])], float(probs[orden[1]])
            marca = "OK" if clase1 == esperado else ""
            print(f"  {casilla} | {nombre:11s} | {clase1:11s} {p1:.2f} | "
                  f"{clase2:11s} {p2:.2f} | {marca}")

        # TTA de rotacion PROPIA: promedio del softmax COMPLETO (13 clases)
        # entre 4 giros. El caballo de b8 se lee mal a su giro real; girar la
        # casilla cambia la entrada y el promedio recupera la clase correcta.
        if esperado.startswith(("caballo", "alfil")):
            probs_total = None
            for giro in range(4):
                girada = np.rot90(celda, giro)
                probs_total = ((predecir(girada) if probs_total is None
                                else probs_total + predecir(girada)))
            orden = np.argsort(probs_total)[::-1]
            top = [(clases[int(o)], float(probs_total[int(o)])) for o in orden[:2]]
            marca = "OK" if top[0][0] == esperado else "MAL"
            print(f"  {casilla} | TTA 4 rot  | {top[0][0]:11s} {top[0][1]:.2f} | "
                  f"{top[1][0]:11s} {top[1][1]:.2f} | {marca}")
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
