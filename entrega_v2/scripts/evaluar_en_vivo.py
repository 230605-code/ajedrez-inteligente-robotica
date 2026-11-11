"""Compara la deteccion en vivo contra la posicion inicial, pieza por pieza.

Responde con numeros a "¿esta detectando bien?". Separa los tres errores que se
confunden entre si y que se arreglan de formas distintas:

  * geometria: la pieza aparece en la casilla equivocada, falta o sobra
  * color:     acierta la casilla y la forma, pero cambia el bando
  * forma:     acierta la casilla y el bando, pero cambia el tipo de pieza

Uso:
    python scripts/evaluar_en_vivo.py --ip 10.35.119.200 --frames 5
    python scripts/evaluar_en_vivo.py --imagen assets/previews/esp32/frame_01_crudo.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (  # noqa: E402
    RAIZ, cargar_modulos_equipo, corregir_color, esquinas_por_rejilla,
    modelo_por_defecto, necesita_giro, normalizar_detecciones,
    ordenar_esquinas, radio_estimado,
)


def esperado_inicial() -> dict[str, str]:
    import chess

    tablero = chess.Board()
    return {
        chess.square_name(c): tablero.piece_at(c).symbol()
        for c in chess.SQUARES if tablero.piece_at(c)
    }


def a_simbolo(clase: str) -> str:
    """white_rook -> R, black_rook -> r. Mismo criterio que el FEN."""
    import chess

    from board import CLASE_A_PIEZA

    pieza = CLASE_A_PIEZA.get(clase)
    if pieza is None:
        return "?"
    caracter = {
        chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b",
        chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k",
    }[pieza]
    return caracter.upper() if clase.startswith("white") else caracter


def analizar_frame(board, detect_mod, modelo, imagen, conf: float,
                   esquinas_manual=None):
    import cv2

    detecciones = normalizar_detecciones(
        detect_mod.detect(imagen, model=modelo, conf=conf, agnostic_nms=True)
    )
    if not detecciones:
        return None, "sin detecciones"

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    detecciones, _ = corregir_color(gris, detecciones, radio_estimado(detecciones))

    if necesita_giro(detecciones):
        imagen = cv2.rotate(imagen, cv2.ROTATE_180)
        detecciones = normalizar_detecciones(
            detect_mod.detect(imagen, model=modelo, conf=conf, agnostic_nms=True)
        )
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        detecciones, _ = corregir_color(gris, detecciones, radio_estimado(detecciones))

    if esquinas_manual is None:
        esquinas, motivo = esquinas_por_rejilla(detecciones)
        if esquinas is None:
            return None, f"rejilla rechazada ({motivo})"
    else:
        esquinas, motivo = esquinas_manual, "esquinas manuales"

    esquinas = ordenar_esquinas(esquinas)
    _, homografia = board.rectificar_tablero(imagen, esquinas)
    _, ocupacion = board.construir_fen(detecciones, homografia)
    obtenido = {c: a_simbolo(d["clase"]) for c, d in ocupacion.items()}
    return obtenido, f"{len(detecciones)} detecciones, {len(obtenido)} casillas"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.35.119.200")
    parser.add_argument("--url")
    parser.add_argument("--imagen", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--modelo", type=Path, default=modelo_por_defecto())
    parser.add_argument(
        "--esquinas",
        help="Opcional: 8 numeros x1,y1,...,x4,y4 si la rejilla inferida falla. "
             "Se aplican tras el giro automatico de 180 grados.",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np
    import requests
    from ultralytics import YOLO

    board, detect_mod, _ = cargar_modulos_equipo()
    if not args.modelo.exists():
        print(f"No existe el modelo: {args.modelo}")
        return 1
    modelo = YOLO(str(args.modelo))
    print(f"Modelo: {args.modelo}")
    esperado = esperado_inicial()
    print(f"Comparando contra la posicion inicial ({len(esperado)} piezas)\n")

    if args.imagen:
        frame = cv2.imread(str(args.imagen))
        imagenes = [frame] * args.frames
    else:
        url = args.url or f"http://{args.ip}/capture"
        imagenes = []
        try:
            for _ in range(args.frames):
                respuesta = requests.get(url, timeout=15)
                respuesta.raise_for_status()
                imagenes.append(
                    cv2.imdecode(np.frombuffer(respuesta.content, np.uint8),
                                 cv2.IMREAD_COLOR)
                )
        except requests.RequestException as exc:
            print(f"ERROR al capturar de {url}: {type(exc).__name__}: {exc}")
            return 1

    totales = {"faltan": 0, "sobran": 0, "color": 0, "forma": 0}
    validos = 0

    esquinas_manual = None
    if args.esquinas:
        valores = [float(v) for v in args.esquinas.split(",")]
        if len(valores) != 8:
            print("--esquinas necesita 8 numeros (4 puntos x 2 coordenadas).")
            return 1
        esquinas_manual = np.array(valores, dtype="float32").reshape(4, 2)

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"frame {i + 1}: no llego como JPEG")
            continue

        obtenido, nota = analizar_frame(
            board, detect_mod, modelo, imagen, args.conf, esquinas_manual
        )
        if obtenido is None:
            print(f"frame {i + 1}: {nota}")
            continue

        validos += 1
        faltan = sorted(c for c in esperado if c not in obtenido)
        sobran = sorted(c for c in obtenido if c not in esperado)
        err_color = err_forma = 0
        for casilla, simbolo in esperado.items():
            detectado = obtenido.get(casilla)
            if detectado is None:
                continue
            # El color es la caja (mayuscula = blanca); la forma es la letra.
            # Son dos errores distintos que se arreglan de forma distinta, asi
            # que se cuentan por separado y pueden coincidir en una pieza.
            if detectado.isupper() != simbolo.isupper():
                err_color += 1
            if detectado.lower() != simbolo.lower():
                err_forma += 1

        totales["faltan"] += len(faltan)
        totales["sobran"] += len(sobran)
        totales["color"] += err_color
        totales["forma"] += err_forma
        print(f"frame {i + 1}: {nota}")
        print(f"  casillas ocupadas  : {len(obtenido)}/{len(esperado)}"
              + (f"  faltan {', '.join(faltan)}" if faltan else "")
              + (f"  sobran {', '.join(sobran)}" if sobran else ""))
        print(f"  geometria: {len(faltan)} faltan / {len(sobran)} sobran   "
              f"color: {err_color}   forma: {err_forma}")

    if not validos:
        return 1

    correctas = len(esperado) - (
        totales["faltan"] + totales["color"] + totales["forma"]
    ) / validos
    correctas = max(0.0, correctas)
    print(f"\nMedia de {validos} frames:")
    print(f"  piezas totalmente correctas: {correctas:.1f}/{len(esperado)} "
          f"({100 * correctas / len(esperado):.0f}%)")
    print(f"  error de geometria: {totales['faltan'] / validos:.1f} faltantes, "
          f"{totales['sobran'] / validos:.1f} sobrantes")
    print(f"  errores de color: {totales['color'] / validos:.1f}   "
          f"errores de forma: {totales['forma'] / validos:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
