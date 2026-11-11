"""Componente de visualización de cámara ESP32-CAM y control de captura."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class VisorCamaraWidget(QWidget):
    """Muestra el flujo de video (crudo o cenital) y los controles de visión."""

    sig_forzar_captura = pyqtSignal()
    sig_cambio_auto = pyqtSignal(bool)
    sig_reconectar_ip = pyqtSignal(str)

    def __init__(self, ip_inicial: str = "10.236.255.200", parent=None) -> None:
        super().__init__(parent)
        self.ip_actual = ip_inicial
        self._modo_cenital = True
        self._frame_crudo: np.ndarray | None = None
        self._frame_cenital: np.ndarray | None = None

        self._construir_ui()

    def _construir_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Barra de estado y conexión superior
        fila_sup = QHBoxLayout()
        self.lbl_conexion = QLabel("ESP32-CAM:")
        self.lbl_conexion.setStyleSheet("font-weight: bold; color: #9ca3af;")

        self.input_ip = QLineEdit(self.ip_actual)
        self.input_ip.setMaximumWidth(130)
        self.input_ip.setStyleSheet("background: #1f2937; color: #f3f4f6; border: 1px solid #374151; padding: 4px; border-radius: 4px;")

        self.btn_reconectar = QPushButton("Conectar")
        self.btn_reconectar.setStyleSheet("background: #2563eb; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 500;")
        self.btn_reconectar.clicked.connect(self._on_reconectar_click)

        self.badge_estado_cam = QLabel("🟢 En Línea")
        self.badge_estado_cam.setStyleSheet("color: #10b981; font-weight: bold;")

        fila_sup.addWidget(self.lbl_conexion)
        fila_sup.addWidget(self.input_ip)
        fila_sup.addWidget(self.btn_reconectar)
        fila_sup.addStretch()
        fila_sup.addWidget(self.badge_estado_cam)
        layout.addLayout(fila_sup)

        # Visor de imagen
        self.marco_imagen = QFrame()
        self.marco_imagen.setStyleSheet("background-color: #111827; border: 1px solid #374151; border-radius: 6px;")
        layout_marco = QVBoxLayout(self.marco_imagen)
        layout_marco.setContentsMargins(4, 4, 4, 4)

        self.lbl_imagen = QLabel("Esperando señal de la cámara...")
        self.lbl_imagen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_imagen.setMinimumSize(360, 270)
        self.lbl_imagen.setStyleSheet("color: #9ca3af; font-size: 13px;")
        layout_marco.addWidget(self.lbl_imagen)
        layout.addWidget(self.marco_imagen, stretch=1)

        # Badge flotante / barra de estabilidad de tablero
        fila_sensor = QHBoxLayout()
        self.badge_sensor = QLabel("Sensor de Movimiento: Esperando reposo...")
        self.badge_sensor.setStyleSheet("background: #1e293b; color: #38bdf8; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
        fila_sensor.addWidget(self.badge_sensor)
        layout.addLayout(fila_sensor)

        # Selector de vista (Cenital vs Cruda)
        fila_opciones = QHBoxLayout()
        self.radio_cenital = QRadioButton("Vista Cenital (800x800)")
        self.radio_cenital.setChecked(True)
        self.radio_cenital.toggled.connect(self._on_modo_vista_cambiado)
        self.radio_cenital.setStyleSheet("color: #d1d5db; font-size: 11px;")

        self.radio_cruda = QRadioButton("Cámara Cruda")
        self.radio_cruda.setStyleSheet("color: #d1d5db; font-size: 11px;")

        fila_opciones.addWidget(self.radio_cenital)
        fila_opciones.addWidget(self.radio_cruda)
        fila_opciones.addStretch()

        self.chk_auto = QCheckBox("Detección Automática (Manos Libres)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px;")
        self.chk_auto.toggled.connect(self.sig_cambio_auto.emit)
        fila_opciones.addWidget(self.chk_auto)

        layout.addLayout(fila_opciones)

        # Botón de captura manual inmediata
        self.btn_captura_manual = QPushButton("⚡ Capturar y Analizar Tablero Ahora")
        self.btn_captura_manual.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 8px; border-radius: 5px; font-size: 12px;"
        )
        self.btn_captura_manual.clicked.connect(self.sig_forzar_captura.emit)
        layout.addWidget(self.btn_captura_manual)

    def _on_reconectar_click(self) -> None:
        ip = self.input_ip.text().strip()
        if ip:
            self.ip_actual = ip
            self.sig_reconectar_ip.emit(ip)

    def _on_modo_vista_cambiado(self, checked: bool) -> None:
        self._modo_cenital = self.radio_cenital.isChecked()
        self._refrescar_display()

    def actualizar_frame_raw(self, frame: np.ndarray) -> None:
        self._frame_crudo = frame
        if not self._modo_cenital:
            self._refrescar_display()

    def actualizar_frame_cenital(self, cenital: np.ndarray) -> None:
        self._frame_cenital = cenital
        if self._modo_cenital:
            self._refrescar_display()

    def _refrescar_display(self) -> None:
        frame = self._frame_cenital if self._modo_cenital else self._frame_crudo
        if frame is None and self._frame_crudo is not None:
            frame = self._frame_crudo

        if frame is None:
            return

        alto, ancho, canales = frame.shape
        bytes_por_linea = canales * ancho
        q_img = QImage(frame.data, ancho, alto, bytes_por_linea, QImage.Format.Format_BGR888)

        tam_disp = self.lbl_imagen.size()
        pixmap = QPixmap.fromImage(q_img).scaled(
            tam_disp.width(), tam_disp.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_imagen.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refrescar_display()

    def actualizar_estado_conexion(self, conectada: bool, mensaje: str) -> None:
        if conectada:
            self.badge_estado_cam.setText("🟢 Conectado")
            self.badge_estado_cam.setStyleSheet("color: #10b981; font-weight: bold;")
        else:
            self.badge_estado_cam.setText("🔴 Sin señal")
            self.badge_estado_cam.setStyleSheet("color: #ef4444; font-weight: bold;")

    def actualizar_estabilidad(self, estado: str, score: float, pct: float) -> None:
        color = "#38bdf8"
        if "Mano" in estado or "Movimiento" in estado:
            color = "#f59e0b"
        elif "Estabilizando" in estado:
            color = "#a855f7"
        elif "estable" in estado.lower() or "leyendo" in estado.lower():
            color = "#10b981"

        self.badge_sensor.setText(f"Sensor: {estado} (var: {score:.1f})")
        self.badge_sensor.setStyleSheet(f"background: #1e293b; color: {color}; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500;")
