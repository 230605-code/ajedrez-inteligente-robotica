"""Componentes de arbitraje, alertas de jugadas ilegales, Stockfish y terminal de robot."""
from __future__ import annotations

import time
import chess
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication, QWheelEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from arbitro import AnalisisMotor, ResultadoJugada
from controlador_robot import EventoJugadaRobot


class TablaHistorialScrolleable(QTableWidget):
    """Tabla que propaga el scroll vertical al contenedor padre cuando no desborda."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        sb = self.verticalScrollBar()
        delta = event.angleDelta().y()
        if sb.maximum() == 0 or (delta > 0 and sb.value() <= sb.minimum()) or (delta < 0 and sb.value() >= sb.maximum()):
            event.ignore()
            return
        super().wheelEvent(event)


class TerminalRobotScrolleable(QPlainTextEdit):
    """Terminal que propaga el scroll vertical al contenedor padre cuando no desborda."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        sb = self.verticalScrollBar()
        delta = event.angleDelta().y()
        if sb.maximum() == 0 or (delta > 0 and sb.value() <= sb.minimum()) or (delta < 0 and sb.value() >= sb.maximum()):
            event.ignore()
            return
        super().wheelEvent(event)


class PanelArbitro(QWidget):
    """Muestra modos de inicio, FEN activo, selector de turno, motor y eventos de robot."""

    sig_modo_estandar = pyqtSignal()
    sig_modo_escanear = pyqtSignal()
    sig_modo_fen = pyqtSignal()
    sig_turno_seleccionado = pyqtSignal(bool)  # True = Blancas, False = Negras

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)
        self._construir_ui()

    def _construir_ui(self) -> None:
        # Layout principal del widget exterior
        layout_exterior = QVBoxLayout(self)
        layout_exterior.setContentsMargins(0, 0, 0, 0)
        layout_exterior.setSpacing(0)

        # Contenedor con Scroll para garantizar navegación fluida en cualquier pantalla
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        # Widget interno que contiene todos los elementos
        self.contenido = QWidget()
        self.contenido.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self.contenido)
        layout.setContentsMargins(8, 8, 12, 8)
        layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. TARJETA DE CONFIGURACIÓN Y MODOS DE INICIO
        # -------------------------------------------------------------
        self.card_inicio = QFrame()
        self.card_inicio.setStyleSheet("background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px;")
        layout_inicio = QVBoxLayout(self.card_inicio)
        layout_inicio.setSpacing(6)

        lbl_titulo_inicio = QLabel("1. MODO DE INICIO DE PARTIDA")
        lbl_titulo_inicio.setStyleSheet("font-size: 11px; font-weight: bold; color: #38bdf8; letter-spacing: 0.5px;")
        layout_inicio.addWidget(lbl_titulo_inicio)

        # Tres botones claros y destacados
        self.btn_modo_estandar = QPushButton("♟️ 1. Empezar en Cero (Estándar)")
        self.btn_modo_estandar.setStyleSheet(
            "background: #2563eb; color: white; font-weight: bold; padding: 7px; border-radius: 5px; font-size: 11px; text-align: left;"
        )
        self.btn_modo_estandar.clicked.connect(self.sig_modo_estandar.emit)

        self.btn_modo_escanear = QPushButton("📷 2. Detectar Tablero Físico (Cámara)")
        self.btn_modo_escanear.setStyleSheet(
            "background: #0284c7; color: white; font-weight: bold; padding: 7px; border-radius: 5px; font-size: 11px; text-align: left;"
        )
        self.btn_modo_escanear.clicked.connect(self.sig_modo_escanear.emit)

        self.btn_modo_fen = QPushButton("📋 3. Digitar / Pegar FEN")
        self.btn_modo_fen.setStyleSheet(
            "background: #475569; color: white; font-weight: bold; padding: 7px; border-radius: 5px; font-size: 11px; text-align: left;"
        )
        self.btn_modo_fen.clicked.connect(self.sig_modo_fen.emit)

        layout_inicio.addWidget(self.btn_modo_estandar)
        layout_inicio.addWidget(self.btn_modo_escanear)
        layout_inicio.addWidget(self.btn_modo_fen)

        # Caja de FEN Activo con botón de copiado
        fila_fen = QHBoxLayout()
        fila_fen.setSpacing(4)
        lbl_fen = QLabel("FEN:")
        lbl_fen.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")

        self.input_fen = QLineEdit()
        self.input_fen.setReadOnly(True)
        self.input_fen.setFont(QFont("Consolas", 8))
        self.input_fen.setStyleSheet("background: #0f172a; color: #38bdf8; border: 1px solid #334155; border-radius: 4px; padding: 3px;")

        self.btn_copiar_fen = QPushButton("Copiar")
        self.btn_copiar_fen.setMaximumWidth(52)
        self.btn_copiar_fen.setStyleSheet("background: #334155; color: #f8fafc; font-size: 10px; border-radius: 3px; padding: 3px; font-weight: 500;")
        self.btn_copiar_fen.clicked.connect(self._copiar_fen)

        fila_fen.addWidget(lbl_fen)
        fila_fen.addWidget(self.input_fen)
        fila_fen.addWidget(self.btn_copiar_fen)
        layout_inicio.addLayout(fila_fen)

        # Selector Explícito de Turno Inicial / Activo
        fila_turno = QHBoxLayout()
        fila_turno.setSpacing(6)
        lbl_sel_turno = QLabel("Turno Inicial:")
        lbl_sel_turno.setStyleSheet("color: #cbd5e1; font-weight: bold; font-size: 11px;")
        fila_turno.addWidget(lbl_sel_turno)

        self.btn_turno_blancas = QPushButton("⚪ Blancas")
        self.btn_turno_blancas.setCheckable(True)
        self.btn_turno_blancas.setChecked(True)
        self.btn_turno_blancas.setStyleSheet("""
            QPushButton { background: #334155; color: #f8fafc; font-weight: bold; padding: 5px; border-radius: 4px; font-size: 11px; }
            QPushButton:checked { background: #059669; color: white; border: 1px solid #34d399; }
        """)

        self.btn_turno_negras = QPushButton("⚫ Negras")
        self.btn_turno_negras.setCheckable(True)
        self.btn_turno_negras.setStyleSheet("""
            QPushButton { background: #334155; color: #94a3b8; font-weight: bold; padding: 5px; border-radius: 4px; font-size: 11px; }
            QPushButton:checked { background: #7c3aed; color: white; border: 1px solid #c084fc; }
        """)

        self.grupo_turno = QButtonGroup(self)
        self.grupo_turno.addButton(self.btn_turno_blancas, 1)
        self.grupo_turno.addButton(self.btn_turno_negras, 2)
        self.btn_turno_blancas.clicked.connect(lambda: self.sig_turno_seleccionado.emit(True))
        self.btn_turno_negras.clicked.connect(lambda: self.sig_turno_seleccionado.emit(False))

        fila_turno.addWidget(self.btn_turno_blancas)
        fila_turno.addWidget(self.btn_turno_negras)
        layout_inicio.addLayout(fila_turno)

        layout.addWidget(self.card_inicio)

        # -------------------------------------------------------------
        # 2. BANNER DE ESTADO Y ALERTA DE JUGADA ILEGAL
        # -------------------------------------------------------------
        self.card_estado = QFrame()
        self.card_estado.setStyleSheet("background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 6px;")
        layout_estado = QHBoxLayout(self.card_estado)

        self.lbl_turno = QLabel("TURNO ACTIVO: BLANCAS ⚪")
        self.lbl_turno.setStyleSheet("font-size: 13px; font-weight: bold; color: #f8fafc;")

        self.lbl_estado_partida = QLabel("EN JUEGO")
        self.lbl_estado_partida.setStyleSheet("background: #065f46; color: #34d399; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px;")

        layout_estado.addWidget(self.lbl_turno)
        layout_estado.addStretch()
        layout_estado.addWidget(self.lbl_estado_partida)
        layout.addWidget(self.card_estado)

        # Tarjeta de Alerta de Jugada Ilegal (Oculta por defecto)
        self.card_alerta = QFrame()
        self.card_alerta.setStyleSheet(
            "background: #450a0a; border: 2px solid #ef4444; border-radius: 6px; padding: 8px;"
        )
        layout_alerta = QVBoxLayout(self.card_alerta)
        layout_alerta.setSpacing(4)

        self.lbl_alerta_titulo = QLabel("⚠️ JUGADA ILEGAL DETECTADA")
        self.lbl_alerta_titulo.setStyleSheet("font-size: 12px; font-weight: bold; color: #fca5a5;")

        self.lbl_alerta_motivo = QLabel("Explicación de la infracción.")
        self.lbl_alerta_motivo.setStyleSheet("font-size: 11px; color: #fee2e2;")
        self.lbl_alerta_motivo.setWordWrap(True)

        self.lbl_alerta_instruccion = QLabel("Acción requerida de corrección física.")
        self.lbl_alerta_instruccion.setStyleSheet("font-size: 11px; font-weight: bold; color: #fef08a;")
        self.lbl_alerta_instruccion.setWordWrap(True)

        layout_alerta.addWidget(self.lbl_alerta_titulo)
        layout_alerta.addWidget(self.lbl_alerta_motivo)
        layout_alerta.addWidget(self.lbl_alerta_instruccion)

        self.card_alerta.setVisible(False)
        layout.addWidget(self.card_alerta)

        # -------------------------------------------------------------
        # 3. SUGERENCIA DE MOTOR (STOCKFISH)
        # -------------------------------------------------------------
        self.card_motor = QFrame()
        self.card_motor.setStyleSheet("background: #111827; border: 1px solid #1f2937; border-radius: 6px; padding: 8px;")
        layout_motor = QVBoxLayout(self.card_motor)
        layout_motor.setSpacing(4)

        fila_sug = QHBoxLayout()
        self.lbl_sugerencia_titulo = QLabel("💡 Sugerencia Stockfish:")
        self.lbl_sugerencia_titulo.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 12px;")

        self.lbl_mejor_jugada = QLabel("e2-e4 (e4)")
        self.lbl_mejor_jugada.setStyleSheet("color: #10b981; font-weight: bold; font-size: 14px;")

        self.lbl_eval_valor = QLabel("+0.30")
        self.lbl_eval_valor.setStyleSheet("background: #334155; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")

        fila_sug.addWidget(self.lbl_sugerencia_titulo)
        fila_sug.addWidget(self.lbl_mejor_jugada)
        fila_sug.addStretch()
        fila_sug.addWidget(self.lbl_eval_valor)
        layout_motor.addLayout(fila_sug)

        self.lbl_linea_pv = QLabel("Línea: Calculando...")
        self.lbl_linea_pv.setStyleSheet("color: #64748b; font-size: 11px;")
        layout_motor.addWidget(self.lbl_linea_pv)
        layout.addWidget(self.card_motor)

        # -------------------------------------------------------------
        # 4. HISTORIAL DE JUGADAS (PGN) - RESPONSIVO Y SCROLLEABLE
        # -------------------------------------------------------------
        lbl_hist_titulo = QLabel("Historial de Jugadas (PGN):")
        lbl_hist_titulo.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8; margin-top: 2px;")
        layout.addWidget(lbl_hist_titulo)

        self.tabla_historial = TablaHistorialScrolleable(0, 3)
        self.tabla_historial.setHorizontalHeaderLabels(["#", "Blancas", "Negras"])
        header = self.tabla_historial.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tabla_historial.setColumnWidth(0, 36)
        self.tabla_historial.verticalHeader().setVisible(False)
        self.tabla_historial.verticalHeader().setDefaultSectionSize(22)
        self.tabla_historial.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tabla_historial.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tabla_historial.setMinimumHeight(120)
        self.tabla_historial.setMaximumHeight(200)
        self.tabla_historial.setStyleSheet("""
            QTableWidget {
                background: #0f172a;
                color: #f8fafc;
                gridline-color: #334155;
                border: 1px solid #1e293b;
                border-radius: 4px;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #1e293b;
                color: #94a3b8;
                font-weight: bold;
                border: none;
                padding: 3px;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.tabla_historial)

        # -------------------------------------------------------------
        # 5. TERMINAL DE EVENTOS DE BRAZO ROBÓTICO - SCROLLEABLE
        # -------------------------------------------------------------
        lbl_robot_titulo = QLabel("Despacho de Eventos hacia Brazo Robótico:")
        lbl_robot_titulo.setStyleSheet("font-size: 11px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_robot_titulo)

        self.terminal_robot = TerminalRobotScrolleable()
        self.terminal_robot.setReadOnly(True)
        self.terminal_robot.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.terminal_robot.setMinimumHeight(95)
        self.terminal_robot.setMaximumHeight(160)
        self.terminal_robot.setFont(QFont("Consolas", 9))
        self.terminal_robot.setStyleSheet(
            "background: #050811; color: #38bdf8; border: 1px solid #1e293b; border-radius: 4px; padding: 4px;"
        )
        self.terminal_robot.appendPlainText("[ROBOT] Despachador listo. En espera de jugadas...")
        layout.addWidget(self.terminal_robot)

        # Configurar y añadir el scroll al layout principal
        self.scroll_area.setWidget(self.contenido)
        layout_exterior.addWidget(self.scroll_area)

    def _copiar_fen(self) -> None:
        texto = self.input_fen.text().strip()
        if texto:
            portapapeles = QGuiApplication.clipboard()
            if portapapeles:
                portapapeles.setText(texto)

    def actualizar_fen_display(self, fen: str) -> None:
        self.input_fen.setText(fen)

    def sincronizar_turno_ui(self, es_blancas: bool, estado: str) -> None:
        self.btn_turno_blancas.blockSignals(True)
        self.btn_turno_negras.blockSignals(True)
        self.btn_turno_blancas.setChecked(es_blancas)
        self.btn_turno_negras.setChecked(not es_blancas)
        self.btn_turno_blancas.blockSignals(False)
        self.btn_turno_negras.blockSignals(False)

        if es_blancas:
            self.lbl_turno.setText("TURNO ACTIVO: BLANCAS ⚪")
            self.lbl_turno.setStyleSheet("font-size: 13px; font-weight: bold; color: #f8fafc;")
        else:
            self.lbl_turno.setText("TURNO ACTIVO: NEGRAS ⚫")
            self.lbl_turno.setStyleSheet("font-size: 13px; font-weight: bold; color: #cbd5e1;")

        estilos_estado = {
            "EN_JUEGO": ("EN JUEGO", "background: #065f46; color: #34d399;"),
            "JAQUE": ("¡JAQUE! ", "background: #854d0e; color: #fde047;"),
            "JAQUE_MATE": ("¡JAQUE MATE! ", "background: #7f1d1d; color: #f87171;"),
            "TABLAS": ("TABLAS", "background: #334155; color: #94a3b8;"),
            "BLOQUEADO_ILEGAL": ("BLOQUEADO (ILEGAL)", "background: #991b1b; color: #fecaca;"),
        }
        texto, estilo = estilos_estado.get(estado, ("EN JUEGO", "background: #065f46; color: #34d399;"))
        self.lbl_estado_partida.setText(texto)
        self.lbl_estado_partida.setStyleSheet(f"{estilo} font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px;")

    def mostrar_alerta_ilegal(self, resultado: ResultadoJugada) -> None:
        self.lbl_alerta_motivo.setText(resultado.mensaje)
        self.lbl_alerta_instruccion.setText(f"👉 Acción Requerida: {resultado.instruccion_correccion}")
        self.card_alerta.setVisible(True)

    def ocultar_alerta_ilegal(self) -> None:
        self.card_alerta.setVisible(False)

    def actualizar_motor(self, analisis: AnalisisMotor) -> None:
        if analisis.mejor_jugada_san:
            self.lbl_mejor_jugada.setText(f"{analisis.mejor_jugada_uci} ({analisis.mejor_jugada_san})")
        else:
            self.lbl_mejor_jugada.setText("--")

        if analisis.es_mate and analisis.mate_en is not None:
            signo = "+" if analisis.mate_en > 0 else "-"
            self.lbl_eval_valor.setText(f"{signo}M{abs(analisis.mate_en)}")
        else:
            val = analisis.evaluacion_cp / 100.0
            self.lbl_eval_valor.setText(f"{val:+.2f}")

        if analisis.pv_uci:
            linea = " ".join(analisis.pv_uci[:5])
            self.lbl_linea_pv.setText(f"Línea (prof. {analisis.profundidad}): {linea}")
        else:
            self.lbl_linea_pv.setText("Sin análisis activo.")

    def registrar_evento_robot(self, evento: EventoJugadaRobot) -> None:
        t_str = time.strftime("%H:%M:%S", time.localtime(evento.timestamp))
        enroque_txt = f" [Enroque: {evento.torre_origen}->{evento.torre_destino}]" if evento.es_enroque else ""
        captura_txt = f" [Captura: {evento.casilla_captura}]" if evento.es_captura else ""
        linea = (
            f"[{t_str}] {evento.color.upper()} {evento.pieza}: "
            f"{evento.origen} -> {evento.destino} ({evento.san}){captura_txt}{enroque_txt}"
        )
        self.terminal_robot.appendPlainText(linea)

    def resetear_historial(self) -> None:
        self.tabla_historial.setRowCount(0)

    def resetear_terminal_robot(self) -> None:
        self.terminal_robot.clear()
        self.terminal_robot.appendPlainText("[ROBOT] Nueva partida iniciada. En espera de jugadas...")

    def actualizar_historial_pgn(self, movimientos: list[str]) -> None:
        self.tabla_historial.setRowCount(0)
        num_pares = (len(movimientos) + 1) // 2
        for i in range(num_pares):
            self.tabla_historial.insertRow(i)
            item_num = QTableWidgetItem(f"{i + 1}.")
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabla_historial.setItem(i, 0, item_num)

            m_blanca = movimientos[i * 2] if (i * 2) < len(movimientos) else ""
            item_b = QTableWidgetItem(m_blanca)
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabla_historial.setItem(i, 1, item_b)

            m_negra = movimientos[i * 2 + 1] if (i * 2 + 1) < len(movimientos) else ""
            item_n = QTableWidgetItem(m_negra)
            item_n.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabla_historial.setItem(i, 2, item_n)

        self.tabla_historial.scrollToBottom()
