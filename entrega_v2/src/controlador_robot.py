"""Controlador y despacho de eventos para brazo robótico.

Estructura las órdenes físicas deducidas por la visión para que un futuro
microcontrolador (Arduino / ESP32 / Raspberry) pueda ejecutar los movimientos
físicos del brazo sin conocer la lógica de ajedrez.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class EventoJugadaRobot:
    uci: str
    san: str
    origen: str                          # ej: "e2"
    destino: str                         # ej: "e4"
    origen_col_fila: tuple[int, int]     # (0..7, 0..7) donde a=0..h=7, 1=0..8=7
    destino_col_fila: tuple[int, int]
    pieza: str                           # "peon", "caballo", "alfil", etc.
    color: str                           # "blanco" / "negro"
    es_captura: bool
    casilla_captura: str | None          # util para captura al paso
    es_enroque: bool
    torre_origen: str | None             # ej: "h1" para enroque corto blanco
    torre_destino: str | None            # ej: "f1"
    es_promocion: bool
    pieza_promocion: str | None
    fen_resultante: str
    timestamp: float

    def a_dict(self) -> dict:
        return asdict(self)

    def a_json(self) -> str:
        return json.dumps(self.a_dict(), ensure_ascii=False)


class DespachadorRobot:
    """Administra la cola de comandos y callbacks hacia el brazo robótico."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[EventoJugadaRobot], None]] = []
        self._historial: list[EventoJugadaRobot] = []

    def registrar_handler(self, handler: Callable[[EventoJugadaRobot], None]) -> None:
        """Registra un receptor de eventos (ej: interfaz grafica, puerto serie o socket)."""
        self._handlers.append(handler)

    def despachar(self, evento: EventoJugadaRobot) -> None:
        self._historial.append(evento)
        for handler in self._handlers:
            try:
                handler(evento)
            except Exception as exc:
                print(f"[ROBOT DISPATCH ERROR] Fallo handler al despachar {evento.uci}: {exc}")

    @property
    def ultimo_evento(self) -> EventoJugadaRobot | None:
        return self._historial[-1] if self._historial else None

    @property
    def historial(self) -> list[EventoJugadaRobot]:
        return list(self._historial)
