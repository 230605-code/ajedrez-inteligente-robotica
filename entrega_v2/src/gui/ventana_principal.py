"""Ventana principal que integra Tablero, Visor ESP32-CAM, Árbitro y Motor."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.svg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from arbitro import ArbitroAjedrez, ResultadoJugada  # noqa: E402
from gestor_camara import HiloCamara  # noqa: E402
from gui.componente_arbitro import PanelArbitro  # noqa: E402
from gui.componente_camara import VisorCamaraWidget  # noqa: E402
from gui.componente_tablero import ContenedorTablero  # noqa: E402


class VentanaPrincipalAjedrez(QMainWindow):
    """Aplicación de escritorio para arbitraje visual de ajedrez con ESP32-CAM."""

    def __init__(self, ip_camara: str = "10.236.255.200") -> None:
        super().__init__()
        self.setWindowTitle("Árbitro de Ajedrez Inteligente | ESP32-CAM & Stockfish")
        self.resize(1200, 780)
        self.setMinimumSize(980, 660)

        # 1. Instanciar Motor de Arbitraje
        self.arbitro = ArbitroAjedrez()

        # 2. Construir Componentes GUI
        self._construir_ui(ip_camara)

        # 3. Registrar Receptor para Eventos del Brazo Robótico
        self.arbitro.despachador_robot.registrar_handler(self.panel_arbitro.registrar_evento_robot)

        # 4. Iniciar Hilo de Cámara ESP32
        self.hilo_camara = HiloCamara(ip=ip_camara, parent=self)
        self._conectar_senales_camara()
        self.hilo_camara.start()

        # 5. Estado de Inicialización y Partida
        self._modo_escaneo_inicial = False
        self._historial_san: list[str] = []
        self._ultima_jugada_uci: str | None = None
        self._casillas_infraccion: list[int] = []

        # 6. Sincronizar UI con la posición base inicial
        self.panel_arbitro.actualizar_fen_display(self.arbitro.tablero.fen())
        self._refrescar_interfaz()

    def _construir_ui(self, ip_camara: str) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
            }
            QWidget {
                color: #f8fafc;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QScrollBar:vertical {
                background: #1e293b;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #1e293b;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #475569;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QStatusBar {
                background: #070a10;
                color: #94a3b8;
                font-size: 11px;
                border-top: 1px solid #1e293b;
            }
        """)

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(10, 10, 10, 10)
        layout_principal.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel Izquierdo: Tablero SVG + Barra de Evaluación
        self.contenedor_tablero = ContenedorTablero(self)
        splitter.addWidget(self.contenedor_tablero)

        # Panel Central: Visor ESP32-CAM y Sensor de Movimiento
        self.visor_camara = VisorCamaraWidget(ip_inicial=ip_camara, parent=self)
        splitter.addWidget(self.visor_camara)

        # Panel Derecho: Modos de Inicio, Árbitro, Stockfish, Alertas y Robot
        self.panel_arbitro = PanelArbitro(self)
        splitter.addWidget(self.panel_arbitro)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 4)

        layout_principal.addWidget(splitter)

        # Barra de estado inferior
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sistema listo. Selecciona cómo iniciar la partida o comienza a jugar.")

        # Conectar las 3 opciones de inicio claras
        self.panel_arbitro.sig_modo_estandar.connect(self._on_modo_estandar)
        self.panel_arbitro.sig_modo_escanear.connect(self._on_modo_escanear)
        self.panel_arbitro.sig_modo_fen.connect(self._on_modo_fen)

        # Conectar selector de turno
        self.panel_arbitro.sig_turno_seleccionado.connect(self._on_turno_seleccionado)

        # Conectar controles de la cámara
        self.visor_camara.sig_forzar_captura.connect(self._on_captura_manual)
        self.visor_camara.sig_cambio_auto.connect(self._on_cambio_auto_deteccion)
        self.visor_camara.sig_reconectar_ip.connect(self._on_reconectar_ip)

    def _conectar_senales_camara(self) -> None:
        self.hilo_camara.sig_frame_raw.connect(self.visor_camara.actualizar_frame_raw)
        self.hilo_camara.sig_frame_cenital.connect(self.visor_camara.actualizar_frame_cenital)
        self.hilo_camara.sig_estado_estabilidad.connect(self.visor_camara.actualizar_estabilidad)
        self.hilo_camara.sig_conexion.connect(self.visor_camara.actualizar_estado_conexion)
        self.hilo_camara.sig_iniciando_analisis.connect(self._on_inicio_analisis_camara)
        self.hilo_camara.sig_analisis_completado.connect(self._on_tablero_detectado)
        self.hilo_camara.sig_error.connect(self._on_error_camara)

    def _on_inicio_analisis_camara(self, mensaje: str) -> None:
        self.status_bar.showMessage(f"🔍 {mensaje}")

    def _on_error_camara(self, error: str) -> None:
        self.status_bar.showMessage(f"❌ {error}")

    def _on_captura_manual(self) -> None:
        self.status_bar.showMessage("Solicitando análisis manual inmediato...")
        self.hilo_camara.solicitar_analisis_inmediato()

    def _on_cambio_auto_deteccion(self, activa: bool) -> None:
        self.hilo_camara.set_auto_deteccion(activa)
        txt = "activada" if activa else "pausada (solo manual)"
        self.status_bar.showMessage(f"Detección manos libres {txt}.")

    def _on_reconectar_ip(self, ip: str) -> None:
        self.status_bar.showMessage(f"Reconectando a {ip}...")
        self.hilo_camara.configurar_ip(ip)

    # -----------------------------------------------------------------
    # OPCIÓN 1: EMPEZAR EN CERO (PARTIDA ESTÁNDAR)
    # -----------------------------------------------------------------
    def _on_modo_estandar(self) -> None:
        fen = self.arbitro.iniciar_nueva_partida(turno=chess.WHITE)
        self._historial_san.clear()
        self._ultima_jugada_uci = None
        self._casillas_infraccion.clear()
        self.panel_arbitro.resetear_historial()
        self.panel_arbitro.resetear_terminal_robot()
        self.panel_arbitro.actualizar_fen_display(fen)
        self.panel_arbitro.ocultar_alerta_ilegal()
        self._refrescar_interfaz()
        self.status_bar.showMessage("♟️ Partida reiniciada a la posición inicial estándar (Blancas juegan primero).")

    # -----------------------------------------------------------------
    # OPCIÓN 2: DETECTAR TABLERO FÍSICO CON LA CÁMARA
    # -----------------------------------------------------------------
    def _on_modo_escanear(self) -> None:
        self._modo_escaneo_inicial = True
        self.status_bar.showMessage("📷 Escaneando tablero físico con ESP32-CAM... Por favor mantén el tablero libre.")
        self.hilo_camara.solicitar_analisis_inmediato()

    # -----------------------------------------------------------------
    # OPCIÓN 3: DIGITAR O PEGAR FEN PERSONALIZADA
    # -----------------------------------------------------------------
    def _on_modo_fen(self) -> None:
        fen_actual = self.arbitro.tablero.fen()
        fen, ok = QInputDialog.getText(
            self, "Digitar / Pegar FEN", "Introduce la cadena FEN:", text=fen_actual
        )
        if ok and fen.strip():
            cadena_fen = fen.strip()
            try:
                temp_board = chess.Board(cadena_fen)
                fen_fijado = self.arbitro.fijar_posicion(cadena_fen, turno=temp_board.turn, reconstruir_enroque=False)
                self._historial_san.clear()
                self._ultima_jugada_uci = None
                self._casillas_infraccion.clear()
                self.panel_arbitro.resetear_historial()
                self.panel_arbitro.actualizar_fen_display(fen_fijado)
                self.panel_arbitro.ocultar_alerta_ilegal()
                self._refrescar_interfaz()
                self.status_bar.showMessage(f"📋 FEN cargada y fijada como base. Turno activo: {'Blancas' if temp_board.turn else 'Negras'}.")
            except Exception as exc:
                QMessageBox.warning(self, "FEN Inválida", f"No se pudo cargar la posición FEN:\n{exc}")

    # -----------------------------------------------------------------
    # SELECTOR DE TURNO EXPLÍCITO (BLANCAS / NEGRAS)
    # -----------------------------------------------------------------
    def _on_turno_seleccionado(self, es_blancas: bool) -> None:
        nuevo_turno = chess.WHITE if es_blancas else chess.BLACK
        fen_actualizado = self.arbitro.cambiar_turno(nuevo_turno)
        self.panel_arbitro.actualizar_fen_display(fen_actualizado)
        self._refrescar_interfaz()
        bando = "Blancas" if es_blancas else "Negras"
        self.status_bar.showMessage(f"🔄 Turno fijado para {bando}. Stockfish recalculó la mejor respuesta.")

    # -----------------------------------------------------------------
    # PROCESAMIENTO DE LECTURAS DE CÁMARA
    # -----------------------------------------------------------------
    def _on_tablero_detectado(self, ocupacion: dict[str, str], fen_leido: str,
                              detalle: dict, cenital) -> None:
        # Modo A: Escaneo de configuración inicial
        if self._modo_escaneo_inicial:
            self._modo_escaneo_inicial = False
            try:
                turno_actual = self.arbitro.tablero.turn
                fen_fijado = self.arbitro.fijar_posicion(fen_leido, turno=turno_actual, reconstruir_enroque=True)
                self._historial_san.clear()
                self._ultima_jugada_uci = None
                self._casillas_infraccion.clear()
                self.panel_arbitro.resetear_historial()
                self.panel_arbitro.actualizar_fen_display(fen_fijado)
                self.panel_arbitro.ocultar_alerta_ilegal()
                self._refrescar_interfaz()
                self.status_bar.showMessage(
                    f"✅ Tablero físico detectado ({detalle.get('ocupadas', '?')} piezas) y fijado como base. FEN: {fen_fijado}"
                )
            except Exception as exc:
                QMessageBox.warning(self, "Error de Escaneo", f"No se pudo registrar el tablero: {exc}")
            return

        # Modo B: Partida en curso -> Deducción contra la base confirmada
        resultado: ResultadoJugada = self.arbitro.deducir_y_validar_movimiento(ocupacion)

        # 0. Partida terminada: dejar de evaluar posiciones nuevas de cámara
        if resultado.tipo == "FIN_PARTIDA":
            self.status_bar.showMessage(f"🏁 {resultado.mensaje}")
            return

        # 1. Sin cambio respecto a la base confirmada
        if resultado.tipo == "SIN_CAMBIO":
            if self.panel_arbitro.card_alerta.isVisible():
                self._casillas_infraccion.clear()
                self.panel_arbitro.ocultar_alerta_ilegal()
                self._refrescar_interfaz()
                self.status_bar.showMessage("✅ Corrección física confirmada: pieza devuelta. El juego se reanuda.")
            else:
                self.status_bar.showMessage("Tablero en reposo. Esperando jugada.")
            return

        # 2. Ruido masivo de visión (cámara movida, sombras, variación transitoria)
        # Se descarta SIN cambiar el tablero y SIN lanzar falsas alarmas
        if resultado.tipo == "RUIDO_VISION":
            self.status_bar.showMessage(f"👁️ {resultado.mensaje}")
            return

        # 3. Pieza levantada en mano (esperando que sea apoyada)
        if resultado.tipo == "PIEZA_LEVANTADA":
            self.status_bar.showMessage(f"✋ {resultado.mensaje}")
            return

        # 4. Jugada legal aceptada
        if resultado.tipo == "LEGAL":
            self._casillas_infraccion.clear()
            self._ultima_jugada_uci = resultado.jugada_uci
            if resultado.jugada_san:
                self._historial_san.append(resultado.jugada_san)
                self.panel_arbitro.actualizar_historial_pgn(self._historial_san)
            self.panel_arbitro.actualizar_fen_display(self.arbitro.tablero.fen())
            self.panel_arbitro.ocultar_alerta_ilegal()
            self._refrescar_interfaz()
            self.status_bar.showMessage(f"♟️ {resultado.mensaje}")
            return

        # 5. Jugada ilegal genuina (1 pieza movida ilegalmente según las reglas de ajedrez)
        if resultado.tipo == "ILEGAL":
            self._casillas_infraccion = [chess.parse_square(c) for c in resultado.casillas_afectadas if len(c) == 2]
            self.panel_arbitro.mostrar_alerta_ilegal(resultado)
            self._refrescar_interfaz()
            self.status_bar.showMessage(f"⚠️ {resultado.mensaje}")

    def _refrescar_interfaz(self) -> None:
        tablero = self.arbitro.tablero
        analisis = self.arbitro.ultimo_analisis

        # 1. Flechas y resaltado de casillas
        flechas: list[chess.svg.Arrow] = []
        if analisis.origen_sq is not None and analisis.destino_sq is not None and not tablero.is_game_over():
            flechas.append(chess.svg.Arrow(analisis.origen_sq, analisis.destino_sq, color="#00d26a"))

        casillas_coloreadas: dict[int, str] = {}
        if self._casillas_infraccion:
            for sq in self._casillas_infraccion:
                casillas_coloreadas[sq] = "#ef444488"
        elif self._ultima_jugada_uci and len(self._ultima_jugada_uci) >= 4:
            try:
                sq_from = chess.parse_square(self._ultima_jugada_uci[:2])
                sq_to = chess.parse_square(self._ultima_jugada_uci[2:4])
                casillas_coloreadas[sq_from] = "#38bdf855"
                casillas_coloreadas[sq_to] = "#38bdf888"
            except Exception:
                pass

        # 2. Actualizar Tablero SVG
        self.contenedor_tablero.visor_tablero.actualizar_tablero(
            tablero=tablero,
            flechas=flechas,
            casillas=casillas_coloreadas,
            orientacion=chess.WHITE,
        )

        # 3. Actualizar Barra de Evaluación
        self.contenedor_tablero.barra_eval.actualizar_evaluacion(
            eval_cp=analisis.evaluacion_cp,
            es_mate=analisis.es_mate,
            mate_en=analisis.mate_en,
            pct_blancas=analisis.porcentaje_blancas,
        )

        # 4. Actualizar Panel de Árbitro y Motor
        self.panel_arbitro.sincronizar_turno_ui(
            es_blancas=(tablero.turn == chess.WHITE),
            estado=self.arbitro.estado_partida,
        )
        self.panel_arbitro.actualizar_motor(analisis)

    def closeEvent(self, event) -> None:
        self.status_bar.showMessage("Cerrando recursos y cámara...")
        self.hilo_camara.detener()
        self.arbitro.cerrar()
        event.accept()
