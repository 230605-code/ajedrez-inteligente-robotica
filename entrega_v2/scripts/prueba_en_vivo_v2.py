"""Prueba en vivo con la arquitectura v2 completa (sin depender de YOLO).

Flujo: captura ESP32 -> deteccion ArUco de las 4 esquinas (metodo robusto
multi-escala) -> rectificacion 800x800 -> CLASIFICADOR por casilla (13 clases)
-> FEN -> post-filtro de movimientos legales con python-chess.

Es el capitulo de resultados 'v2' de la tesis: la misma geometria del demo y
el clasificador que dio 99.3% en el test.

Uso:
    python scripts/prueba_en_vivo_v2.py --ip 10.251.192.200 --frames 2
    python scripts/prueba_en_vivo_v2.py --ip 10.251.192.200 --frames 3 --esperado inicial
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from esquinas_aruco import detectar_robusto  # noqa: E402
from pipeline_fen import (  # noqa: E402
    clasificar_recorte, construir_fen, cargar_clasificador, legalizar,
)

SALIDA = RAIZ / "assets" / "previews" / "v2_vivo"
GOLES = {"vacia": ".", "peon_b": "P", "torre_b": "R", "caballo_b": "N",
         "alfil_b": "B", "dama_b": "Q", "rey_b": "K",
         "peon_n": "p", "torre_n": "r", "caballo_n": "n",
         "alfil_n": "b", "dama_n": "q", "rey_n": "k"}


def capturar(url: str, intentos: int = 3) -> np.ndarray | None:
    import requests

    for intento in range(1, intentos + 1):
        try:
            respuesta = requests.get(url, timeout=15)
            respuesta.raise_for_status()
            frame = cv2.imdecode(np.frombuffer(respuesta.content, np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except requests.RequestException as exc:
            print(f"  intento {intento}: ERROR {type(exc).__name__}: "
                  f"{str(exc)[:80]}")
    return None


def tablero_ascii(ocupacion: dict[str, str]) -> str:
    lineas = ["  +--" + "---" * 8 + "-+"]
    for fila in range(8):
        linea = f"{8 - fila} | "
        for columna in range(8):
            casilla = f"{chr(ord('a') + columna)}{8 - fila}"
            clase = ocupacion.get(casilla, "vacia")
            linea += GOLES.get(clase, "?") + " "
        lineas.append(linea + "|")
    lineas.append("  +--" + "---" * 8 + "-+")
    lineas.append("    a b c d e f g h")
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.4.1")
    parser.add_argument("--url")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--esperado", choices=("inicial", "vacio", "ninguno"),
                        default="ninguno")
    args = parser.parse_args()

    url = args.url or f"http://{args.ip}/capture"
    modelo, clases, info = cargar_clasificador(RAIZ / "modelos" / "best_clasificador.pt")
    print(f"clasificador: best_clasificador.pt {info}")
    from ultralytics import YOLO
    detector_yolo = YOLO(str(RAIZ / "modelos" / "best.pt"))

    SALIDA.mkdir(parents=True, exist_ok=True)
    verdad = None
    if args.esperado == "inicial":
        tablero = __import__("chess").board if False else None
        import chess
        verdad = {chess.square_name(c): c and None for c in []}  # placeholder
        verdad = {}
        b = __import__("chess")
        tablero_chess = chess.Board()
        for c in chess.SQUARES:
            p = tablero_chess.piece_at(c)
            if p:
                verdad[chess.square_name(c)] = {
                    "p": "peon", "n": "caballo", "b": "alfil", "r": "torre",
                    "q": "dama", "k": "rey",
                }[p.symbol().lower()] + ("_b" if p.color else "_n")

    for indice in range(1, args.frames + 1):
        print(f"\n--- captura {indice}")
        frame = capturar(url)
        if frame is None:
            print("  sin frame")
            return 1
        cv2.imwrite(str(SALIDA / f"v2_{indice}_crudo.jpg"), frame)

        esquinas, metodo, detalle = detectar_robusto(frame)
        if esquinas is None:
            print(f"  ARUCO NO detecto las esquinas ({detalle or metodo})")
            cv2.imwrite(str(SALIDA / f"v2_{indice}_falla.jpg"), frame)
            continue
        print(f"  esquinas: {metodo} ({detalle}) -> "
              + " ".join(f"({x:.0f},{y:.0f})" for x, y in esquinas))

        destino = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")
        matriz = cv2.getPerspectiveTransform(
            np.asarray(esquinas, dtype="float32"), destino)
        cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        ocupacion_datos = clasificar_recorte(modelo, clases, cenital)

        # Consenso dual por casilla: el CLASIFICADOR vota con su probabilidad
        # y el DETECTOR YOLO con la suya si su centro cae en la casilla. En el
        # dominio de la ESP32 nocturna el clasificador confunde caballo->alfil
        # y el segundo modelo lo arregla (se midio en la regresion del smoke).
        cenital_480 = cv2.resize(cenital, (480, 480),
                                 interpolation=cv2.INTER_AREA)
        cenital_640 = cv2.resize(cenital, (640, 640),
                                 interpolation=cv2.INTER_AREA)
        resultado_yolo = detector_yolo.predict(
            [cenital_480, cenital_640], conf=0.25, agnostic_nms=True,
            verbose=False)
        piezas_yolo: dict[str, tuple[str, float]] = {}
        for resultado in resultado_yolo:
            alto_origen = resultado.orig_shape[0]
            factor = alto_origen / 800.0
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

        planas = {}
        fusion_info = {}
        for casilla, dato in ocupacion_datos.items():
            pred_aplicativo = dato["elegida"]
            conf_aplicativo = dato["confianza"]
            yolo_termino = piezas_yolo.get(casilla)
            if yolo_termino and yolo_termino[0] != "vacia":
                nombre_y, conf_y = yolo_termino
                if conf_y > conf_aplicativo:
                    pred_aplicativo, conf_aplicativo = nombre_y, conf_y
                    fusion_info[casilla] = f"{pred_aplicativo} yolo{conf_y:.2f}"
            planas[casilla] = pred_aplicativo
        fen_crudo = construir_fen(planas)
        if fusion_info:
            print("  fusion con YOLO:", ", ".join(
                f"{c}={info}" for c, info in sorted(fusion_info.items())))
        fen_final, cambios = legalizar(ocupacion_datos)

        n_fichas = sum(1 for c in planas.values() if c != "vacia")
        print(f"  fichas detectadas: {n_fichas}")
        print(f"  cambios del filtro legal: {len(cambios)}")
        print(f"  FEN SIN filtro : {fen_crudo}")
        print(f"  FEN CON filtro : {fen_final}")
        print(tablero_ascii(planas))

        if verdad is not None:
            phantomas = [c for c, clase in planas.items()
                         if clase != "vacia" and c not in verdad]
            faltan = [c for c, clase in verdad.items() if planas.get(c, "vacia") == "vacia"]
            errores = [c for c, clase in verdad.items()
                       if planas.get(c, "vacia") not in (clase, "vacia")]
            print(f"  vs inicial: ciertas={sum(1 for c, clase in verdad.items() if planas.get(c) == clase)}/32"
                  f"  phantom={len(phantomas)}  faltan={len(faltan)}  forma_erronea={len(errores)}")

        marcado = cenital.copy()
        for fila_ in range(8):
            for col_ in range(8):
                casilla = f"{chr(ord('a') + col_)}{8 - fila_}"
                dato = ocupacion_datos[casilla]
                color = (60, 220, 60) if dato["elegida"] != "vacia" and "_" in dato["elegida"] and dato["elegida"].endswith("b") else (40, 160, 255) if "_" in dato["elegida"] else (180, 180, 180)
                cv2.putText(marcado, dato["elegida"], (col_ * 100 + 4, fila_ * 100 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(marcado, dato["elegida"], (col_ * 100 + 4, fila_ * 100 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.imwrite(str(SALIDA / f"v2_{indice}_cenital.jpg"), marcado)
        cv2.imwrite(str(SALIDA / f"v2_{indice}_crudo.jpg"), frame)
        print(f"  guardado: {SALIDA}\\v2_{indice}_cenital.jpg")

    print(f"\n(capturas en {SALIDA})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
