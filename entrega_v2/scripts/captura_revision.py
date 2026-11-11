"""Sesion de revision en vivo: tu mueves fichas, yo capturo y etiqueto.

Cada N segundos:
  captura -> ArUco (rectificacion) -> clasificador + fusion YOLO -> FEN
  y guarda en assets/previews/revista_viva/prueba_NN/:
      crudo.jpg     el frame de la camara
      cenital.jpg   el tablero 800x800 con CADA casilla etiquetada en grande
                    (verde=confianza alta, amarillo=media, rojo=baja)
      fen.txt       la FEN, el conteo y el detalle por casilla
Las etiquetas se pintan GORDAS a proposito: el objetivo es que YOU mires y
mortos si el modelo acertó.

Uso:
    python scripts/captura_revision.py --ip 10.251.192.200 --n 6 --segundo 10
entre captura y captura tu mueves las fichas como quieras.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from esquinas_aruco import detectar_robusto  # noqa: E402
from pipeline_fen import (  # noqa: E402
    clasificar_recorte, construir_fen, cargar_clasificador,
    descartar_fantasmas_baja_confianza, legalizar,
)

SALIDA = RAIZ / "assets" / "previews" / "revista_viva"
LETTER = {"vacia": ".", "peon_b": "P", "torre_b": "R", "caballo_b": "N",
          "alfil_b": "B", "dama_b": "Q", "rey_b": "K",
          "peon_n": "p", "torre_n": "r", "caballo_n": "n",
          "alfil_n": "b", "dama_n": "q", "rey_n": "k"}

GOLES = {"vacia": ".", "peon_b": "P", "torre_b": "R", "caballo_b": "N",
         "alfil_b": "B", "dama_b": "Q", "rey_b": "K",
         "peon_n": "p", "torre_n": "r", "caballo_n": "n",
         "alfil_n": "b", "dama_n": "q", "rey_n": "k"}

AMARILLO = (40, 220, 220)
VERDE = (60, 220, 60)
ROJO = (40, 60, 240)


def capturar(url: str) -> np.ndarray | None:
    import requests

    try:
        respuesta = requests.get(url, timeout=15)
        respuesta.raise_for_status()
        return cv2.imdecode(np.frombuffer(respuesta.content, np.uint8),
                            cv2.IMREAD_COLOR)
    except requests.RequestException as exc:
        print(f"    ERROR: {type(exc).__name__}: {str(exc)[:80]}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.4.1")
    parser.add_argument("--url")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--segundo", type=float, default=10.0,
                        help="segundos entre capturas (aqui migren las piezas")
    args = parser.parse_args()

    url = args.url or f"http://{args.ip}/capture"
    modelo, clases, info = cargar_clasificador(RAIZ / "modelos" / "best_clasificador.pt")
    from ultralytics import YOLO
    detector = YOLO(str(RAIZ / "modelos" / "best.pt"))
    print(f"clasificador lista: {info}")

    for indice in range(1, args.n + 1):
        carpeta = SALIDA / f"prueba_{indice:02d}"
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"\n--- prueba {indice}/{args.n}")
        frame = capturar(url)
        if frame is None:
            (carpeta / "FALLO.txt").write_text("no llego imagen de la camara",
                                               encoding="utf-8")
            continue
        cv2.imwrite(str(carpeta / "crudo.jpg"), frame)

        esquinas, metodo, detalle = detectar_robusto(frame)
        if esquinas is None:
            (carpeta / "FALLO.txt").write_text(
                f"sin esquinas: {metodo} {detalle}", encoding="utf-8")
            print(f"    sin esquinas ({detalle or metodo})")
            time.sleep(max(1.0, args.segundo))
            continue
        print(f"    esquinas: {metodo} ({detalle})")

        destino = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")
        matriz = cv2.getPerspectiveTransform(
            np.asarray(esquinas, dtype="float32"), destino)
        cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        datos = clasificar_recorte(modelo, clases, cenital)

        cenital_480 = cv2.resize(cenital, (480, 480),
                                 interpolation=cv2.INTER_AREA)
        cenital_640 = cv2.resize(cenital, (640, 640),
                                 interpolation=cv2.INTER_AREA)
        resultado_yolo = detector.predict([cenital_480, cenital_640],
                                          conf=0.25, agnostic_nms=True,
                                          verbose=False)
        piezas_yolo: dict[str, tuple[str, float]] = {}
        for resultado in resultado_yolo:
            factor = resultado.orig_shape[0] / 800.0
            for i in range(len(resultado.boxes)):
                nombre = resultado.names[int(resultado.boxes.cls[i].item())]
                caja = resultado.boxes.xyxy[i]
                x_c = (float(caja[0]) + float(caja[2])) / 2.0 / factor
                y_c = (float(caja[1]) + float(caja[3])) / 2.0 / factor
                conf = float(resultado.boxes.conf[i].item())
                columna = min(7, max(0, int(x_c // (800 / 8))))
                fila_ = min(7, max(0, int(y_c // (800 / 8))))
                casilla = f"{chr(ord('a') + columna)}{8 - fila_}"
                if casilla not in piezas_yolo or conf > piezas_yolo[casilla][1]:
                    piezas_yolo[casilla] = (nombre, conf)

        marcado = cenital.copy()
        detalles = []
        fusiones = 0
        for fila_ in range(8):
            for col_ in range(8):
                casilla = f"{chr(ord('a') + col_)}{8 - fila_}"
                dato = datos[casilla]
                clase, conf = dato["elegida"], dato["confianza"]
                yolo_termino = piezas_yolo.get(casilla)
                fusion = ""
                if yolo_termino and yolo_termino[0] != "vacia" \
                        and yolo_termino[0] != clase:
                    if yolo_termino[1] > conf * 0.9:
                        clase, conf = yolo_termino
                        fusion = " (+yolo)"
                        fusiones += 1
                color = (VERDE if conf >= 0.90 else
                         AMARILLO if conf >= 0.70 else ROJO)
                x0, y0 = col_ * 100, fila_ * 100
                cv2.rectangle(marcado, (x0, y0), (x0 + 99, y0 + 99),
                              color, 1)
                cv2.putText(marcado, casilla, (x0 + 4, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(marcado, LETTER.get(clase, "?"),
                            (x0 + 34, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, color, 3, cv2.LINE_AA)
                cv2.putText(marcado, f"{conf:.0%}", (x0 + 6, y0 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                            cv2.LINE_AA)
                detalles.append(f"{casilla}: {clase} ({conf:.0f}){fusion}")

        planas = {c: d["elegida"] for c, d in datos.items()}
        fen_crudo = construir_fen(planas)
        espectros = descartar_fantasmas_baja_confianza(datos)
        planas = {c: d["elegida"] for c, d in datos.items()}
        fen_final, _ = legalizar(datos)

        cv2.imwrite(str(carpeta / "cenital.jpg"), marcado)
        (carpeta / "fen.txt").write_text(
            "FEN SIN filtro : " + fen_crudo + "\n"
            "FEN CON filtro : " + fen_final + "\n"
            f"fusiones con YOLO: {fusiones}\n"
            f"espectros descartados: {', '.join(espectros) or '-'}\n\n"
            "detalle por casilla:\n" + "\n".join(detalles),
            encoding="utf-8")
        print(f"    fichas: {sum(1 for c in planas.values() if c != 'vacia')}   "
              f"fusiones yolo: {fusiones}")
        print(f"    FEN con filtro: {fen_final}")
        print(f"    guardado en {carpeta}")
        if indice < args.n:
            time.sleep(max(1.0, args.segundo))

    print(f"\nlas {args.n} pruebas quedan en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
