"""Captura fotos crudas de la ESP32-CAM al dataset, sin tocarlas.

Los bytes se guardan tal cual llegan de `/capture`: el protocolo del dataset
prohibe recortes, renombres y ajustes de brillo. Solo se comprueba que sean un
JPEG valido y que no repitan una foto ya guardada.

Una foto identica a otra significa que la escena no cambio (o que el firmware
devolvio el frame cacheado); se vuelve a pedir hasta 5 veces y, si sigue igual,
se descarta en vez de duplicar el dataset.

Uso:
    python scripts/capturar_dataset.py --ip 10.35.119.200 --carpeta 01_inicial --n 8
    python scripts/capturar_dataset.py --ip 10.35.119.200 --carpeta 03_finales --n 30 --intervalo 5
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CARPETAS, CRUDO, imagenes  # noqa: E402

TIMEOUT = 15.0
INTENTOS_RED = 3
MAX_REPETIDAS = 5


def hash_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def hashes_existentes(carpeta: Path) -> set[str]:
    """Hash de los JPG ya guardados, para no repetir fotos entre corridas."""
    return {hash_bytes(p.read_bytes()) for p in imagenes(carpeta)}


def siguiente_indice(carpeta: Path, prefijo: str) -> int:
    """Continua la numeracion existente en vez de sobreescribir fotos."""
    maximo = 0
    for ruta in carpeta.glob(f"{prefijo}_*.jpg"):
        try:
            maximo = max(maximo, int(ruta.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return maximo + 1


def descargar(url: str) -> bytes:
    """Pide un frame y valida que sea JPEG. Devuelve los bytes sin reencodear."""
    import cv2
    import numpy as np
    import requests

    ultimo_error: Exception | None = None
    for intento in range(1, INTENTOS_RED + 1):
        try:
            respuesta = requests.get(url, timeout=TIMEOUT)
            respuesta.raise_for_status()
            datos = respuesta.content
            if cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR) is None:
                raise ValueError("la respuesta no es un JPEG valido")
            return datos
        except (requests.RequestException, ValueError) as exc:
            ultimo_error = exc
            print(f"    red {intento}/{INTENTOS_RED}: {type(exc).__name__}: {str(exc)[:90]}")
            if intento < INTENTOS_RED:
                time.sleep(0.05)
    raise ConnectionError(f"no se pudo capturar de {url}: {ultimo_error}")


def capturar_una(url: str, vistos: set[str]) -> bytes | None:
    """Una foto nueva o None si tras MAX_REPETIDAS sigue siendo un duplicado."""
    for intento in range(1, MAX_REPETIDAS + 1):
        datos = descargar(url)
        if hash_bytes(datos) not in vistos:
            return datos
        print(f"    identica a una foto previa, reintento {intento}/{MAX_REPETIDAS}")
        time.sleep(0.05)
    return None


def resumen() -> None:
    """Conteo logrado por carpeta, para saber que falta del objetivo."""
    print("\nResumen del dataset crudo:")
    for nombre in CARPETAS:
        cantidad = len(imagenes(CRUDO / nombre))
        print(f"  {nombre:16s} {cantidad:3d} fotos")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.35.119.200", help="IP de la ESP32-CAM")
    parser.add_argument("--url", help="URL completa, gana sobre --ip")
    parser.add_argument("--carpeta", choices=CARPETAS, default="01_inicial",
                        help="Carpeta destino dentro de dataset/crudo")
    parser.add_argument("--n", type=int, default=8, help="Cuantas fotos nuevas bajar")
    parser.add_argument("--intervalo", type=float, default=3.0,
                        help="Segundos entre foto y foto para mover piezas")
    args = parser.parse_args()

    if args.n < 1:
        print("--n debe ser al menos 1.")
        return 1

    url = args.url or f"http://{args.ip}/capture"
    destino = CRUDO / args.carpeta
    destino.mkdir(parents=True, exist_ok=True)

    vistos = hashes_existentes(destino)
    indice = siguiente_indice(destino, args.carpeta)
    print(f"Capturando {args.n} fotos de {url}")
    print(f"Destino: {destino}   (ya hay {len(vistos)} fotos)\n")

    guardadas = 0
    for numero in range(1, args.n + 1):
        print(f"[{numero}/{args.n}]")
        try:
            datos = capturar_una(url, vistos)
        except ConnectionError as exc:
            print(f"\nERROR: {exc}")
            print("Revisa que la camara este encendida y en la misma red.")
            resumen()
            return 1

        if datos is None:
            print(f"  descartada: {MAX_REPETIDAS} intentos dieron la misma foto")
            continue

        ruta = destino / f"{args.carpeta}_{indice:03d}.jpg"
        ruta.write_bytes(datos)
        vistos.add(hash_bytes(datos))
        guardadas += 1
        print(f"  guardada {ruta.name}  {len(datos) / 1024:.1f} KB  "
              f"sha256={hash_bytes(datos)[:12]}")
        indice += 1

        if numero < args.n and args.intervalo > 0:
            print(f"  esperando {args.intervalo:g}s ...")
            time.sleep(args.intervalo)

    print(f"\nNuevas en {args.carpeta}: {guardadas}/{args.n}")
    resumen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
