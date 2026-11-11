"""Prueba en vivo no interactiva contra la ESP32-CAM.

Captura frames, rectifica el tablero con homografia, detecta piezas y guarda
imagenes anotadas. Sirve para revisar resultados sin depender de una ventana
que hay que cerrar a mano.

El orden importa: primero se detectan las piezas y RECIEN despues se buscan las
esquinas. Las piezas estan sobre casillas conocidas, asi que sus centros dan el
paso exacto de la rejilla; buscar el borde del tablero por contornos falla
cuando el tablero y la mesa tienen el mismo brillo.

Uso:
    python scripts/prueba_en_vivo.py --ip 10.35.119.200 --frames 3
    python scripts/prueba_en_vivo.py --ip 10.35.119.200 --esquinas 52,28,522,20,525,415,50,420
    python scripts/prueba_en_vivo.py --imagen assets/previews/esp32_raw.jpg
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

SALIDA = RAIZ / "assets" / "previews" / "esp32"


def esquinas_robustas(frame):
    """Cuadrilatero del tablero por umbral adaptativo.

    `board.detectar_esquinas_tablero` usa Canny y exige un contorno cerrado; en
    una foto con el borde del tablero oscuro sobre fondo oscuro no lo encuentra.
    El tablero, en cambio, es la region clara mas grande: umbralizar y tomar el
    mayor contorno de 4 vertices es mucho mas estable para una camara fija.
    """
    import cv2
    import numpy as np

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (7, 7), 0)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(
        binaria, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    area_imagen = frame.shape[0] * frame.shape[1]
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contorno) < area_imagen * 0.15:
            continue
        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aproximado) == 4:
            return aproximado.reshape(4, 2).astype("float32")
    return None


def describir_tablero(fen: str) -> None:
    """Imprime el tablero como texto para leerlo de un vistazo."""
    import chess

    tablero = chess.Board(fen)
    print("\nTablero reconstruido:")
    print("  +" + "-" * 17 + "+")
    for fila in range(7, -1, -1):
        linea = f"{fila + 1} | "
        for columna in range(8):
            pieza = tablero.piece_at(chess.square(columna, fila))
            linea += (pieza.symbol() if pieza else ".") + " "
        print(linea + "|")
    print("  +" + "-" * 17 + "+")
    print("    a b c d e f g h")
    print(f"\nFEN: {fen}")


def dibujar_ocupacion(warped, ocupacion: dict):
    """Marca en la vista cenital cada casilla ocupada y su clase."""
    import cv2
    import chess

    imagen = warped.copy()
    lado = imagen.shape[1] / 8.0
    for casilla, dato in ocupacion.items():
        square = chess.parse_square(casilla)
        columna = chess.square_file(square)
        fila_visual = 7 - chess.square_rank(square)
        x = int((columna + 0.5) * lado)
        y = int((fila_visual + 0.5) * lado)
        color = (60, 220, 60) if dato["color"] == chess.WHITE else (40, 160, 255)
        cv2.circle(imagen, (x, y), int(lado * 0.30), color, 2)
        cv2.putText(
            imagen, casilla, (x - 14, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            imagen, casilla, (x - 14, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return imagen


def dibujar_detecciones(frame, detecciones, radio: int):
    """Anota el frame con las clases YA corregidas por brillo.

    No uso `modelo.plot()`: dibuja la salida cruda del modelo, que en esta
    camara etiqueta todo como blanco. La imagen tiene que mostrar lo que el
    pipeline realmente usa, no lo que adivino la red.
    """
    import cv2

    LETRAS = {"pawn": "P", "knight": "N", "bishop": "B",
              "rook": "R", "queen": "Q", "king": "K"}
    imagen = frame.copy()
    for deteccion in detecciones:
        clase, cx, cy = deteccion[0], int(deteccion[1]), int(deteccion[2])
        color, _, forma = clase.partition("_")
        letra = LETRAS.get(forma, "?")
        tono = (60, 200, 60) if color == "white" else (40, 140, 255)
        cv2.circle(imagen, (cx, cy), radio + 5, tono, 2)
        cv2.putText(imagen, letra, (cx - 7, cy + 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(imagen, letra, (cx - 7, cy + 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, tono, 2, cv2.LINE_AA)
    return imagen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.35.119.200")
    parser.add_argument("--url")
    parser.add_argument("--imagen", type=Path,
                        help="Usa una imagen local en vez de capturar de la camara")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--modelo", type=Path, default=modelo_por_defecto())
    parser.add_argument("--conf", type=float, default=0.22,
                        help="Umbral de confianza. Medido en vivo: 0.22 recupera "
                             "la pieza que se pierde a 0.30 sin casi falsos positivos")
    parser.add_argument(
        "--esquinas",
        help="8 numeros separados por coma: x1,y1,x2,y2,x3,y3,x4,y4 "
             "(superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda)",
    )
    parser.add_argument("--rotar", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Gira el frame antes de procesar (0, 90, 180 o 270)")
    parser.add_argument("--sin-color", action="store_true",
                        help="No corregir el color de las piezas por brillo")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    board, detect_mod, esp32_mod = cargar_modulos_equipo()

    url = args.url or f"http://{args.ip}/capture"
    SALIDA.mkdir(parents=True, exist_ok=True)

    if not args.modelo.exists():
        print(f"No existe el modelo: {args.modelo}")
        print("Pasa uno con --modelo o deja un best.pt en modelos/.")
        return 1

    if args.imagen:
        frame = cv2.imread(str(args.imagen))
        if frame is None:
            print(f"No pude leer la imagen: {args.imagen}")
            return 1
        frames = [frame.copy() for _ in range(args.frames)]
        print(f"Imagen local: {args.imagen} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        print(f"Capturando de {url} ...")
        frames = []
        try:
            for i in range(args.frames):
                frames.append(esp32_mod.capturar_esp32(url))
                print(f"  frame {i + 1}: {frames[-1].shape[1]}x{frames[-1].shape[0]}")
        except ConnectionError as exc:
            print(f"\nERROR al capturar: {exc}")
            return 1

    def girar(lote, grados: int):
        if not grados:
            return lote
        codigo = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                  270: cv2.ROTATE_90_COUNTERCLOCKWISE}[grados]
        return [cv2.rotate(f, codigo) for f in lote]

    frames = girar(frames, args.rotar)
    if args.rotar:
        print(f"Rotacion aplicada: {args.rotar} grados")

    modelo = YOLO(str(args.modelo))
    print(f"Modelo: {args.modelo.name}   conf={args.conf}")

    def aplicar_color(lote_detecciones, lote_frames, radio_px: int):
        if args.sin_color:
            return
        for j, fotograma in enumerate(lote_frames):
            gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
            lote_detecciones[j], cambiadas = corregir_color(
                gris, lote_detecciones[j], radio_px
            )
            print(f"  color medido en {len(lote_detecciones[j])} piezas "
                  f"({cambiadas} cambiadas por el modelo)")

    def detectar():
        return [
            normalizar_detecciones(
                detect_mod.detect(f, model=modelo, conf=args.conf, agnostic_nms=True)
            )
            for f in frames
        ]

    # 1) Detectar primero: la rejilla se deduce de las piezas.
    detecciones = detectar()
    print(f"\nFrame 1: {len(detecciones[0])} detecciones")

    # 2) El color se mide en la imagen. El modelo lo confunde a menudo y sin
    #    esto la orientacion automatica nunca se activa (todo sale "white").
    aplicar_color(detecciones, frames, radio_estimado(detecciones[0]))

    if necesita_giro(detecciones[0]):
        print("Las blancas salen arriba: giro de 180 grados para que queden en las filas 1-2.")
        frames = girar(frames, 180)
        detecciones = detectar()
        print(f"Frame 1 tras el giro: {len(detecciones[0])} detecciones")
        aplicar_color(detecciones, frames, radio_estimado(detecciones[0]))

    for i, frame in enumerate(frames):
        cv2.imwrite(str(SALIDA / f"frame_{i + 1:02d}_crudo.jpg"), frame)

    # 3) Esquinas: manual > rejilla inferida > contornos.
    if args.esquinas:
        valores = [float(v) for v in args.esquinas.split(",")]
        if len(valores) != 8:
            print("--esquinas necesita 8 numeros (4 puntos x 2 coordenadas).")
            return 1
        esquinas = np.array(valores, dtype="float32").reshape(4, 2)
        origen = "manuales (--esquinas)"
    else:
        esquinas = None
        origen = ""
        for nombre, funcion in (
            ("rejilla inferida de las piezas",
             lambda: esquinas_por_rejilla(detecciones[0])),
            ("contornos (Otsu)", lambda: esquinas_robustas(frames[0])),
            ("contornos (Canny del repo)",
             lambda: board.detectar_esquinas_tablero(frames[0])),
        ):
            resultado = funcion()
            if isinstance(resultado, tuple):
                candidatas, motivo = resultado
            else:
                candidatas, motivo = resultado, nombre
            if candidatas is not None:
                esquinas, origen = candidatas, nombre
                print(f"\nEsquinas: {nombre} -> {motivo}")
                break
            print(f"  descartado {nombre}: {motivo}")
        if esquinas is None:
            print("\nNo pude detectar las esquinas del tablero.")
            print("Pasalas a mano con --esquinas x1,y1,x2,y2,x3,y3,x4,y4")
            return 2

    esquinas = ordenar_esquinas(esquinas)
    print(f"Esquinas ({origen}):")
    print("  " + "  ".join(f"({x:.0f},{y:.0f})" for x, y in esquinas))

    # Paso de casilla en pixeles: da la escala de la pieza para medir el color.
    lado_x = np.linalg.norm(esquinas[1] - esquinas[0]) / 8.0
    lado_y = np.linalg.norm(esquinas[2] - esquinas[1]) / 8.0
    lado = float(np.clip((lado_x + lado_y) / 2.0, 8.0, 400.0))
    radio = max(5, int(round(lado * 0.25)))
    print(f"Casilla: {lado:.0f} px -> radio de medida del color: {radio} px")

    # Afino la medida del color con el paso real de casilla.
    aplicar_color(detecciones, frames, radio)

    for i, frame in enumerate(frames):
        warped, homografia = board.rectificar_tablero(frame, esquinas)

        clases: dict[str, int] = {}
        for nombre, *_ in detecciones[i]:
            clases[nombre] = clases.get(nombre, 0) + 1
        print(f"\n--- frame {i + 1}: {len(detecciones[i])} detecciones")
        for nombre, cantidad in sorted(clases.items()):
            print(f"      {nombre:14s} {cantidad}")

        fen, ocupacion = board.construir_fen(detecciones[i], homografia)
        print(f"      casillas ocupadas: {len(ocupacion)}")

        anotado = dibujar_detecciones(frame, detecciones[i], radio)
        cv2.polylines(anotado, [esquinas.astype(np.int32)], True, (255, 0, 255), 2)
        cv2.imwrite(str(SALIDA / f"frame_{i + 1:02d}_detecciones.jpg"), anotado)

        marcado = frame.copy()
        cv2.polylines(marcado, [esquinas.astype(np.int32)], True, (0, 255, 0), 2)
        for punto in esquinas:
            cv2.circle(marcado, (int(punto[0]), int(punto[1])), 7, (0, 0, 255), -1)
        cv2.imwrite(str(SALIDA / f"frame_{i + 1:02d}_esquinas.jpg"), marcado)

        cv2.imwrite(
            str(SALIDA / f"frame_{i + 1:02d}_cenital.jpg"),
            dibujar_ocupacion(warped, ocupacion),
        )

        if i == 0:
            describir_tablero(fen)
        else:
            print(f"      FEN: {fen}")

    print(f"\nImagenes en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
