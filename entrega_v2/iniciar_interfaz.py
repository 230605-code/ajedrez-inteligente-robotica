"""Punto de entrada principal para la interfaz de escritorio de Ajedrez V2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from PyQt6.QtWidgets import QApplication  # noqa: E402
from gui.ventana_principal import VentanaPrincipalAjedrez  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Árbitro de Ajedrez con Visión ESP32-CAM y Stockfish")
    parser.add_argument("--ip", default="10.236.255.200", help="Dirección IP de la ESP32-CAM (default: 10.236.255.200)")
    parser.add_argument("--offline", type=str, default=None, help="Ruta a una imagen estática para pruebas offline")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Árbitro de Ajedrez Inteligente")

    ventana = VentanaPrincipalAjedrez(ip_camara=args.ip)
    if args.offline:
        ventana.hilo_camara.set_imagen_offline(Path(args.offline))

    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
