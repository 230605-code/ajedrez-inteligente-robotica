"""CONECTOR PARA EL EQUIPO DE JUEGO EN TIEMPO REAL.

Este modulo es el CONTRATO entre:
  * la vision v2 (aqui lee la posicion del tablero fisico desde la ESP32-CAM)
  * el MOTOR de juego en tiempo real (el que elige y ejecuta las jugadas)

Garantia: `leer_tablero()` SIEMPRE devuelve un `chess.Board` con una posicion
REPRESENTABLE (o lanza RuntimeError con el motivo exacto: sin ArUco, sin
suficiente iluminacion, camara apagada), lista para usar con `python-chess`
(legal_moves, push_move, is_checkmate, san...) y con cualquier motor UCI
(Stockfish) via `chess.engine.SimpleEngine`.

NO MODIFIQUES el pipeline de vision sin avisar: los otros equipos usan el
contrato por ruta (ip -> chess.Board y sugerir_jugada -> UCI/SAN).

Uso minimo:
    from conector_juego import LectorFenV2
    lector = LectorFenV2(ip="10.251.192.200")
    tablero, fen, detalle = lector.leer_tablero()
    jugada_san = lector.mejor_jugada_humano(tablero)     # 'e4', 'Nf3' ...
    jugada_uci = lector.sugerir_jugada(tablero)          # 'e2e4'
"""
from __future__ import annotations

import sys
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

DESTINO_CENITAL = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")


def _ruta_stockfish() -> Path | None:
    candidatos = [
        RAIZ / "tools" / "stockfish" / "stockfish.exe",
        RAIZ / "tools" / "stockfish" /
            "stockfish-windows-x86-64-avx2.exe",
        RAIZ / "tools" / "stockfish-windows-x86-64-avx2.exe",
        RAIZ.parent / "ajedrez-yolo" / "tools" /
            "stockfish-windows-x86-64-avx2.exe",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    return None


class LectorFenV2:
    """Lee el tablero fisico y devuelve la posicion lista para el motor.

    La fusion de DOS modelos es la que dio 30/32 piezas correctas en la
    prueba en vivo: el CLASIFICADOR decide por casilla y el DETECTOR YOLO
    puede OVERTHROW cuando su confianza supera la del clasificador (es el
    que arregla la confusion caballo/alfil en la ESP32 con poca luz).
    """

    def __init__(self, ip: str = "10.251.192.200",
                 captura_url: str | None = None,
                 peso_clasificador: Path | None = None,
                 peso_detector: Path | None = None,
                 carpeta_salida: Path | None = None):
        import requests

        self.requests = requests
        self.url = captura_url or f"http://{ip}/capture"
        self.model, self.clases, self.info = cargar_clasificador(
            peso_clasificador or (RAIZ / "modelos" / "best_clasificador.pt"))
        from ultralytics import YOLO
        self.detector = YOLO(
            str(peso_detector or (RAIZ / "modelos" / "best.pt")))
        self.carpeta = Path(carpeta_salida or
                            (RAIZ / "assets" / "previews" / "integracion"))
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.contador = 0

    def capturar(self, intentos: int = 3) -> np.ndarray:
        for intento in range(1, intentos + 1):
            try:
                respuesta = self.requests.get(self.url, timeout=15)
                respuesta.raise_for_status()
                frame = cv2.imdecode(
                    np.frombuffer(respuesta.content, np.uint8),
                    cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            except self.requests.RequestException as exc:
                print(f"  intento {intento}: {type(exc).__name__}: "
                      f"{str(exc)[:80]}")
        raise RuntimeError(
            "la ESP32-CAM no respondio: revisa encendido, IP y red")

    def leer_tablero(self, guardar: bool = False, pausa: float = 0.0,
                     frame: np.ndarray | None = None,
                     cenital: np.ndarray | None = None
                     ) -> tuple["chess.Board", str, dict]:
        """Una lectura del tablero fisico. Lanza RuntimeError si falla.
        Devuelve (chess.Board, fen, detalle)."""
        import chess
        import time as _time

        if pausa > 0:
            _time.sleep(pausa)

        metodo = "cenital_aportada"  # si el llamador pasa la vista cenital,
        # no hay deteccion ArUco fresca en esta lectura
        if cenital is None:
            if frame is None:
                frame = self.capturar()
            self.contador += 1
            esquinas, metodo, detalle_esquinas = detectar_robusto(frame)
            if esquinas is None:
                raise RuntimeError(
                    "sin esquinas ArUco. Encuadra el tablero con SUS 4 esquinas "
                    "blancas visibles y entra buena luz.")

            matriz = cv2.getPerspectiveTransform(
                np.asarray(esquinas, dtype="float32"), DESTINO_CENITAL)
            cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))
        else:
            self.contador += 1

        datos = clasificar_recorte(self.model, self.clases, cenital)

        cenital_640 = cv2.resize(cenital, (640, 640),
                                 interpolation=cv2.INTER_AREA)
        resultado_yolo = self.detector.predict(
            cenital_640, conf=0.25, agnostic_nms=True, verbose=False)[0]
        factor = resultado_yolo.orig_shape[0] / 800.0
        piezas_yolo: dict[str, tuple[str, float]] = {}
        for i in range(len(resultado_yolo.boxes)):
            nombre = resultado_yolo.names[int(resultado_yolo.boxes.cls[i].item())]
            caja = resultado_yolo.boxes.xyxy[i]
            x_c = (float(caja[0]) + float(caja[2])) / 2.0 / factor
            y_c = (float(caja[1]) + float(caja[3])) / 2.0 / factor
            conf = float(resultado_yolo.boxes.conf[i].item())
            columna = min(7, max(0, int(x_c // (800 / 8))))
            fila_ = min(7, max(0, int(y_c // (800 / 8))))
            casilla = f"{chr(ord('a') + columna)}{8 - fila_}"
            previa = piezas_yolo.get(casilla)
            if previa is None or conf > previa[1]:
                piezas_yolo[casilla] = (nombre, conf)

        ocupacion: dict[str, dict] = {}
        fusiones = 0
        for casilla, dato in datos.items():
            clase = dato["elegida"]
            conf = dato["confianza"]
            yolo = piezas_yolo.get(casilla)
            if yolo and yolo[1] > conf * 0.9 and yolo[0] != clase:
                clase, conf = yolo
                fusiones += 1
            ocupacion[casilla] = {"elegida": clase, "confianza": conf,
                                  "ranking": dato["ranking"]}

        planas = {c: d["elegida"] for c, d in ocupacion.items()}
        fen_crudo = construir_fen(planas)
        espectros = descartar_fantasmas_baja_confianza(ocupacion)
        planas = {c: d["elegida"] for c, d in ocupacion.items()}
        fen_final, cambios = legalizar(ocupacion)
        tablero = chess.Board(fen_final)

        detalle = {
            "metodo_esquinas": metodo,
            "fen_sin_filtro": fen_crudo,
            "cambios_filtro_legal": cambios,
            "fusiones_yolo": fusiones,
            "espectros_descartados": espectros,
            "confianza_media": round(sum(d["confianza"] for d in
                                          ocupacion.values()) / 64, 3),
            "ocupadas": sum(1 for v in planas.values() if v != "vacia"),
        }
        if guardar:
            carpeta = self.carpeta / f"lectura_{self.contador:03d}"
            carpeta.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(carpeta / "crudo.jpg"), frame)
            marcado = cenital.copy()
            for fila_ in range(8):
                for col_ in range(8):
                    casilla = f"{chr(ord('a') + col_)}{8 - fila_}"
                    dato = ocupacion[casilla]
                    color = ((60, 220, 60) if dato["confianza"] >= 0.9
                             else (40, 220, 220) if dato["confianza"] >= 0.7
                             else (40, 60, 240))
                    cv2.putText(marcado,
                                f"{casilla} {dato['elegida'][:9]}",
                                (col_ * 100 + 4, fila_ * 100 + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                                cv2.LINE_AA)
            cv2.imwrite(str(carpeta / "cenital.jpg"), marcado)
        return tablero, fen_final, detalle

    def ruta_stockfish(self) -> Path | None:
        return _ruta_stockfish()

    def sugerir_jugada(self, tablero, ruta_stockfish: Path | None = None,
                       tiempo_ms: int = 300) -> str | None:
        """Mejor jugada en UCI ('e2e4'). Necesita Stockfish."""
        import chess.engine

        ruta = ruta_stockfish or _ruta_stockfish()
        if ruta is None:
            raise RuntimeError(
                "Falta Stockfish: descargalo (stockfishchess.org) y guardalo "
                "como tools/stockfish/stockfish.exe")
        with chess.engine.SimpleEngine.popen_uci(str(ruta)) as motor:
            resultado = motor.play(tablero,
                                   chess.engine.Limit(time=tiempo_ms / 1000.0))
        return resultado.move.uci() if resultado and resultado.move else None

    def mejor_jugada_humano(self, tablero) -> str | None:
        """Mejor jugada en SAN legible ('e4', 'Nf3', 'O-O', 'Qxf7#')."""
        import chess.engine

        ruta = _ruta_stockfish()
        if ruta is None:
            raise RuntimeError(
                "Falta Stockfish: pon el binario en tools/stockfish/stockfish.exe")
        with chess.engine.SimpleEngine.popen_uci(str(ruta)) as motor:
            resultado = motor.play(tablero, chess.engine.Limit(time=0.3))
        return tablero.san(resultado.move) if resultado and resultado.move else None

    def terminar(self):
        pass


if __name__ == "__main__":
    lector = LectorFenV2(ip="10.251.192.200")
    tablero, fen, detalle = lector.leer_tablero(guardar=True)
    print("FEN:", fen)
    print("detalle:", detalle)
    print(tablero)
    try:
        print("mejor jugada:", lector.mejor_jugada_humano(tablero))
    except RuntimeError as exc:
        print("sin Stockfish:", exc)
