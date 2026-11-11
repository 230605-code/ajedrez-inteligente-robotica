"""Componentes gráficos del tablero de ajedrez y barra de evaluación."""
from __future__ import annotations

import chess
import chess.svg
from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QHBoxLayout, QWidget


class BarraEvaluacion(QWidget):
    """Barra vertical que ilustra la ventaja posicional según Stockfish."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(28)
        self.setMinimumHeight(320)
        self.pct_blancas: float = 50.0
        self.texto_eval: str = "+0.0"
        self.es_mate: bool = False

    def actualizar_evaluacion(self, eval_cp: int, es_mate: bool,
                              mate_en: int | None, pct_blancas: float) -> None:
        self.pct_blancas = max(0.0, min(100.0, pct_blancas))
        self.es_mate = es_mate
        if es_mate and mate_en is not None:
            self.texto_eval = f"M{abs(mate_en)}" if mate_en > 0 else f"-M{abs(mate_en)}"
        else:
            val = eval_cp / 100.0
            self.texto_eval = f"{val:+.1f}" if abs(val) >= 0.05 else "0.0"
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        ancho = self.width()
        alto = self.height()

        # Fondo negras (arriba)
        painter.setBrush(QBrush(QColor("#2c2b29")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, ancho, alto, 4, 4)

        # Franja blancas (abajo)
        alto_blancas = int(alto * (self.pct_blancas / 100.0))
        y_blancas = alto - alto_blancas
        painter.setBrush(QBrush(QColor("#f3f4f6")))
        painter.drawRoundedRect(0, y_blancas, ancho, alto_blancas, 4, 4)

        # Texto con la evaluación
        painter.setPen(QPen(QColor("#111827") if self.pct_blancas > 50 else QColor("#f9fafb")))
        fuente = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(fuente)

        # Posicionar el texto en el lado dominante
        rect_texto = QRectF(0, alto - 24 if self.pct_blancas > 50 else 4, ancho, 20)
        painter.drawText(rect_texto, Qt.AlignmentFlag.AlignCenter, self.texto_eval)


class VisorTableroSvg(QWidget):
    """Renderiza vectorialmente el tablero usando python-chess y chess.svg."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.renderer = QSvgRenderer()
        self.setMinimumSize(340, 340)
        self.tablero = chess.Board()
        self.flechas: list[chess.svg.Arrow] = []
        self.casillas_resaltadas: dict[int, str] = {}
        self.orientacion: chess.Color = chess.WHITE
        self._regenerar_svg()

    def actualizar_tablero(self, tablero: chess.Board,
                           flechas: list[chess.svg.Arrow] | None = None,
                           casillas: dict[int, str] | None = None,
                           orientacion: chess.Color = chess.WHITE) -> None:
        self.tablero = tablero
        self.flechas = flechas or []
        self.casillas_resaltadas = casillas or {}
        self.orientacion = orientacion
        self._regenerar_svg()
        self.update()

    def _regenerar_svg(self) -> None:
        # Estilos modernos para casillas y piezas
        svg_code = chess.svg.board(
            board=self.tablero,
            orientation=self.orientacion,
            arrows=self.flechas,
            fill=self.casillas_resaltadas,
            colors={
                "square light": "#f0d9b5",
                "square dark": "#b58863",
                "margin": "#262421",
                "coord": "#a3a19e",
            },
            size=None,
        )
        self.renderer.load(QByteArray(svg_code.encode("utf-8")))

    def paintEvent(self, event) -> None:
        if not self.renderer.isValid():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        lado = min(self.width(), self.height())
        offset_x = (self.width() - lado) // 2
        offset_y = (self.height() - lado) // 2
        rect = QRectF(offset_x, offset_y, lado, lado)
        self.renderer.render(painter, rect)


class ContenedorTablero(QWidget):
    """Agrupa la barra de evaluación y el tablero en una sola unidad responsiva."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.barra_eval = BarraEvaluacion(self)
        self.visor_tablero = VisorTableroSvg(self)

        layout.addWidget(self.barra_eval)
        layout.addWidget(self.visor_tablero, stretch=1)
