"""Prueba en vivo contra una FEN declarada por el usuario.

Captura frames de la camara, corre el pipeline dual (clasificador +
detector YOLO), abstrae la FEN y la compara casilla por casilla contra
la FEN declarada.

Uso:
    python scripts/prueba_fen_declared.py --ip 10.236.255.200 \
        --fen "r1bq1rk1/1p2bppp/p1np1n2/4p3/3PP3/2N2N2/PPQ2PPP/R1B2RK1 w" \
        --frames 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from esquinas_aruco import detectar_robusto  # noqa: E402
from pipeline_fen import (  # noqa: E402
    clasificar_recorte,
    construir_fen,
    cargar_clasificador,
    legalizar,
)

SALIDA = RAIZ / "resultados" / "prueba_fen_declared"

GOLES = {"vacia": ".", "peon_b": "P", "torre_b": "R", "caballo_b": "N",
         "alfil_b": "B", "dama_b": "Q", "rey_b": "K",
         "peon_n": "p", "torre_n": "r", "caballo_n": "n",
         "alfil_n": "b", "dama_n": "q", "rey_n": "k"}

NOMBRE = {"p": "peon", "n": "caballo", "b": "alfil",
          "r": "torre", "q": "dama", "k": "rey"}


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
        linea += "|"
        lineas.append(linea)
    lineas.append("  +--" + "---" * 8 + "-+")
    lineas.append("     a  b  c  d  e  f  g  h")
    return "\n".join(lineas)


def verdad_desde_fen(fen: str) -> tuple[dict[str, str], chess.Board]:
    tablero = chess.Board(fen)
    verdad: dict[str, str] = {}
    for casilla in chess.SQUARES:
        pieza = tablero.piece_at(casilla)
        if pieza:
            nombre = NOMBRE[pieza.symbol().lower()]
            verdad[chess.square_name(casilla)] = (
                nombre + ("_b" if pieza.color else "_n"))
    return verdad, tablero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.236.255.200")
    parser.add_argument("--url")
    parser.add_argument("--fen", required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    url = args.url or f"http://{args.ip}/capture"

    try:
        verdad, _ = verdad_desde_fen(args.fen)
    except ValueError as exc:
        print(f"FEN declarada INVALIDA: {exc}")
        return 1
    print(f"FEN declarada: {args.fen}")
    print(f"fichas esperadas: {len(verdad)}")

    modelo, clases, info = cargar_clasificador(
        RAIZ / "modelos" / "best_clasificador.pt")
    print(f"clasificador: best_clasificador.pt {info}")
    from ultralytics import YOLO  # noqa: E402
    detector_yolo = YOLO(str(RAIZ / "modelos" / "best.pt"))

    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados_votacion: list[dict[str, str]] = []

    for indice in range(1, args.frames + 1):
        print(f"\n--- captura {indice}")
        frame = capturar(url)
        if frame is None:
            print("  sin frame")
            continue
        cv2.imwrite(str(SALIDA / f"declared_{indice}_crudo.jpg"), frame)

        esquinas, metodo, detalle = detectar_robusto(frame)
        if esquinas is None:
            print(f"  ARUCO NO detecto las esquinas ({detalle or metodo})")
            cv2.imwrite(str(SALIDA / f"declared_{indice}_falla.jpg"), frame)
            continue
        print(f"  esquinas: {metodo} ({detalle})")

        destino = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")
        matriz = cv2.getPerspectiveTransform(
            np.asarray(esquinas, dtype="float32"), destino)
        cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        cv2.imwrite(str(SALIDA / f"declared_{indice}_cenital.jpg"), cenital)

        ocupacion_datos = clasificar_recorte(modelo, clases, cenital)

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
            pred = dato["elegida"]
            conf = dato["confianza"]
            yolo = piezas_yolo.get(casilla)
            if yolo and yolo[0] != "vacia" and yolo[1] > conf:
                planas[casilla] = yolo[0]
                fusion_info[casilla] = f"yolo {yolo[0]} {yolo[1]:.2f}"
            else:
                planas[casilla] = pred
        if fusion_info:
            print("  fusion YOLO:", ", ".join(
                f"{c}: {d}" for c, d in sorted(fusion_info.items())))

        resultados_votacion.append(planas)
        n_fichas = sum(1 for c in planas.values() if c != "vacia")
        print(f"  fichas detectadas: {n_fichas}")

    if not resultados_votacion:
        print("\nNINGUNA captura funciono (revisa la camara/esquinas)")
        return 1

    # Votacion por camara: mayoria simple por casilla
    planas = {}
    for casilla in resultados_votacion[0]:
        votos = [r[casilla] for r in resultados_votacion]
        planas[casilla] = max(set(votos), key=votos.count)
    print(f"\nvotacion de {len(resultados_votacion)} capturas")

    fen_crudo = construir_fen(planas)
    print(f"FEN construida (sin filtro legal): {fen_crudo}")

    ciertas = [c for c, clase in verdad.items() if planas.get(c) == clase]
    phantomas = [c for c, clase in planas.items()
                 if clase != "vacia" and c not in verdad]
    faltan = [c for c, clase in verdad.items()
              if planas.get(c, "vacia") == "vacia"]
    distintas = [(c, planas.get(c), clase) for c, clase in verdad.items()
                 if planas.get(c, "vacia") not in (clase, "vacia")
                 and planas.get(c, "vacia") != "vacia"]

    total = len(verdad)
    print(f"\n=== RESULTADO contra FEN declarada ===")
    print(f"coincidentes: {len(ciertas)}/{total}")
    if phantomas:
        print(f"phantomas ({len(phantomas)}): " + ", ".join(
            f"{c}={planas[c]}" for c in phantomas))
    if faltan:
        print(f"faltan ({len(faltan)}): " + ", ".join(
            f"{c} clasificada como "
            f"{planas.get(c, 'vacia')}" for c in faltan))
    if distintas:
        print("clase distinta:")
        for c, detectada, esperada in distintas[:12]:
            print(f"  {c}: esperada {esperada}  detectada {detectada}")

    if phantomas or faltan or distintas:
        print("\nplano detectado:")
        print(tablero_ascii(planas))
        return 2
    print("\nPERFECTO: las 64 casillas coinciden con la FEN declarada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
