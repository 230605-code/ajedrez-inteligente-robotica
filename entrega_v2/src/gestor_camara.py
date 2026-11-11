"""Gestor de cámara ESP32-CAM y detección de estabilidad / presencia de mano.

Monitorea el flujo cenital de la ESP32-CAM, analiza la variación entre cuadros
para detectar cuando la mano entra/sale del tablero y, tras un periodo de
estabilidad física, dispara la inferencia de FEN de forma 100% autónoma.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import chess
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from conector_juego import DESTINO_CENITAL, LectorFenV2  # noqa: E402
from esquinas_aruco import detectar_robusto  # noqa: E402


class DetectorEstabilidad:
    """Detecta movimiento (ej: mano del jugador) y mide estabilidad tras la jugada."""

    def __init__(self, umbral_diff: float = 3.5, umbral_pct: float = 3.0,
                 segundos_estables: float = 1.3) -> None:
        self.umbral_diff = umbral_diff
        self.umbral_pct = umbral_pct
        self.segundos_estables = segundos_estables

        self.prev_gray: np.ndarray | None = None
        self.tiempo_inicio_estable: float | None = None
        self.habia_movimiento: bool = False
        self.bloqueado_hasta_nuevo_movimiento: bool = False

    def reset(self) -> None:
        self.prev_gray = None
        self.tiempo_inicio_estable = None
        self.habia_movimiento = False
        self.bloqueado_hasta_nuevo_movimiento = False

    def procesar(self, frame: np.ndarray) -> tuple[bool, float, float, str]:
        """Evalúa un cuadro.

        Retorna:
            (disparar_analisis, diff_media, pct_cambio, estado_texto)
        """
        peq = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0, 0.0, "Calibrando sensor..."

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        diff_media = float(np.mean(diff))
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        pct_cambio = float((np.count_nonzero(thresh) / thresh.size) * 100.0)

        es_movimiento = (diff_media > self.umbral_diff) or (pct_cambio > self.umbral_pct)

        if es_movimiento:
            self.habia_movimiento = True
            self.bloqueado_hasta_nuevo_movimiento = False
            self.tiempo_inicio_estable = None
            return False, diff_media, pct_cambio, "Mano / Movimiento en tablero"

        # No hay movimiento en este cuadro
        if self.bloqueado_hasta_nuevo_movimiento:
            return False, diff_media, pct_cambio, "Esperando próxima jugada..."

        if not self.habia_movimiento:
            return False, diff_media, pct_cambio, "Tablero en reposo"

        # Había movimiento y ahora está quieto -> temporizador de estabilidad
        ahora = time.time()
        if self.tiempo_inicio_estable is None:
            self.tiempo_inicio_estable = ahora

        transcurrido = ahora - self.tiempo_inicio_estable
        if transcurrido < self.segundos_estables:
            pct_progreso = int((transcurrido / self.segundos_estables) * 100)
            return (False, diff_media, pct_cambio,
                    f"Estabilizando ({pct_progreso}%)...")

        # Se cumplió el tiempo requerido de quietud tras el movimiento
        self.bloqueado_hasta_nuevo_movimiento = True
        self.habia_movimiento = False
        self.tiempo_inicio_estable = None
        return True, diff_media, pct_cambio, "Tablero estable -> Leyendo..."


class HiloCamara(QThread):
    """Hilo trabajador que consulta la ESP32-CAM y genera eventos de visión."""

    sig_frame_raw = pyqtSignal(np.ndarray)
    sig_frame_cenital = pyqtSignal(np.ndarray)
    sig_estado_estabilidad = pyqtSignal(str, float, float)
    sig_iniciando_analisis = pyqtSignal(str)
    sig_analisis_completado = pyqtSignal(dict, str, dict, np.ndarray)
    sig_error = pyqtSignal(str)
    sig_conexion = pyqtSignal(bool, str)

    def __init__(self, ip: str = "10.236.255.200", parent=None) -> None:
        super().__init__(parent)
        self.ip = ip
        self.url = f"http://{ip}/capture"
        self._activo = True
        self._auto_deteccion = True
        self._solicitar_analisis = False
        self._imagen_offline: Path | None = None
        self._mutex = QMutex()

        self.session = requests.Session()
        self.detector_estabilidad = DetectorEstabilidad()
        self.lector: LectorFenV2 | None = None
        self._esquinas_cacheadas: list[tuple[float, float]] | None = None
        self._contador_frames = 0

    def configurar_ip(self, ip: str) -> None:
        with QMutexLocker(self._mutex):
            self.ip = ip
            self.url = f"http://{ip}/capture"
            if self.lector is not None:
                self.lector.url = self.url

    def set_auto_deteccion(self, activa: bool) -> None:
        with QMutexLocker(self._mutex):
            self._auto_deteccion = activa
            if not activa:
                self.detector_estabilidad.reset()

    def set_imagen_offline(self, ruta: Path | None) -> None:
        with QMutexLocker(self._mutex):
            self._imagen_offline = ruta

    def solicitar_analisis_inmediato(self) -> None:
        with QMutexLocker(self._mutex):
            self._solicitar_analisis = True

    def detener(self) -> None:
        with QMutexLocker(self._mutex):
            self._activo = False
        try:
            self.session.close()
        except Exception:
            pass
        self.wait(4000)

    def run(self) -> None:
        # Inicializar el Lector de FEN con los modelos en memoria
        try:
            self.lector = LectorFenV2(ip=self.ip)
            self.sig_conexion.emit(True, f"Modelos de visión cargados. Conectando a {self.ip}...")
        except Exception as exc:
            self.sig_error.emit(f"Error cargando modelos de visión: {exc}")
            return

        while True:
            with QMutexLocker(self._mutex):
                if not self._activo:
                    break
                url_actual = self.url
                offline = self._imagen_offline
                auto = self._auto_deteccion
                forzar = self._solicitar_analisis
                self._solicitar_analisis = False

            frame: np.ndarray | None = None

            # 1. Obtener el cuadro actual
            if offline and offline.exists():
                frame = cv2.imread(str(offline))
                time.sleep(0.15)
            else:
                try:
                    resp = self.session.get(url_actual, timeout=(2.0, 3.0))
                    if resp.status_code == 200:
                        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                        self.sig_conexion.emit(True, f"Conectado a {self.ip}")
                    else:
                        self.sig_conexion.emit(False, f"HTTP {resp.status_code} desde {self.ip}")
                except Exception as exc:
                    with QMutexLocker(self._mutex):
                        if not self._activo:
                            break
                    self.sig_conexion.emit(False, f"Fallo enlace con {self.ip}: {str(exc)[:40]}")
                    time.sleep(1.0)
                    continue

            if frame is None:
                time.sleep(0.2)
                continue

            self._contador_frames += 1
            self.sig_frame_raw.emit(frame)

            # 2. ArUco y corrección de perspectiva
            cenital: np.ndarray | None = None
            try:
                # Cada 10 frames o si no hay esquinas, recalcular ArUco
                if self._esquinas_cacheadas is None or (self._contador_frames % 15 == 0):
                    esquinas, metodo, _ = detectar_robusto(frame)
                    if esquinas is not None:
                        self._esquinas_cacheadas = esquinas

                if self._esquinas_cacheadas is not None:
                    matriz = cv2.getPerspectiveTransform(
                        np.asarray(self._esquinas_cacheadas, dtype="float32"), DESTINO_CENITAL)
                    cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT,
                                                  borderValue=(0, 0, 0))
                    self.sig_frame_cenital.emit(cenital)
            except Exception:
                pass

            # 3. Detección de estabilidad en el tablero
            eval_frame = cenital if cenital is not None else frame
            disparar, diff, pct, estado = self.detector_estabilidad.procesar(eval_frame)
            self.sig_estado_estabilidad.emit(estado, diff, pct)

            # 4. Decisión de inferencia
            debe_analizar = forzar or (auto and disparar)
            if debe_analizar:
                self.sig_iniciando_analisis.emit("Capturando y analizando tablero físico...")
                try:
                    # Usamos el cenital / frame ya capturado para máxima velocidad y evitar doble petición HTTP
                    tablero, fen, detalle = self.lector.leer_tablero(
                        guardar=False,
                        frame=frame,
                        cenital=cenital,
                    )
                    # Extraer ocupación plana
                    _TIPO_NOMBRE = {
                        chess.PAWN: "peon", chess.ROOK: "torre", chess.KNIGHT: "caballo",
                        chess.BISHOP: "alfil", chess.QUEEN: "dama", chess.KING: "rey",
                    }
                    ocupacion: dict[str, str] = {}
                    for sq in chess.SQUARES:
                        cas = chess.square_name(sq)
                        p = tablero.piece_at(sq)
                        if p:
                            nom = _TIPO_NOMBRE[p.piece_type]
                            color_p = "b" if p.color == chess.WHITE else "n"
                            ocupacion[cas] = f"{nom}_{color_p}"
                        else:
                            ocupacion[cas] = "vacia"

                    frame_cenital_out = cenital if cenital is not None else eval_frame
                    self.sig_analisis_completado.emit(ocupacion, fen, detalle, frame_cenital_out)
                except Exception as exc:
                    self.sig_error.emit(f"Error durante la lectura del tablero: {exc}")

            time.sleep(0.08)
