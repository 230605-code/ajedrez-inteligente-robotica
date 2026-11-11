"""Diagnostico de la camara ESP32-CAM antes de lanzar el pipeline.

Comprueba lo que suele romper la prueba en vivo, en el orden en que conviene
arreglarlo:
  1. Que el endpoint responda con un JPEG.
  2. Cuanto tarda y en que resolucion entrega (afecta al tamano de cada pieza).
  3. Que el tablero este realmente en el encuadre.

Las detecciones con modelo son OPCIONALES (`--modelo`): el diagnostico tiene
que poder decir si la camara sirve aunque todavia no haya un modelo entrenado
para las fichas nuevas.

Uso:
    python scripts/diagnostico_esp32.py
    python scripts/diagnostico_esp32.py --ip 10.35.119.200
    python scripts/diagnostico_esp32.py --ip 10.35.119.200 --modelo modelos/best.pt
    python scripts/diagnostico_esp32.py --url http://10.35.119.200/capture
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (  # noqa: E402
    RAIZ, cargar_modulos_equipo, normalizar_detecciones,
)

SALIDA = RAIZ / "assets" / "previews" / "esp32_raw.jpg"

# Ancho minimo para que una pieza mida ~35 px en un tablero de 8x8 ocupando el
# 80% del encuadre: el dataset se entreno con piezas de ~40 px.
ANCHO_MINIMO = 640
ANCHO_RECOMENDADO = 800

CONF_DETECCION = 0.22


def probar_url(url: str, intentos: int = 3) -> tuple[list[float], object]:
    import cv2
    import numpy as np
    import requests

    tiempos: list[float] = []
    imagen = None

    for i in range(1, intentos + 1):
        inicio = time.perf_counter()
        try:
            respuesta = requests.get(url, timeout=15)
            respuesta.raise_for_status()
        except requests.RequestException as exc:
            print(f"  intento {i}: ERROR {type(exc).__name__}: {str(exc)[:90]}")
            continue

        transcurrido = time.perf_counter() - inicio
        imagen = cv2.imdecode(np.frombuffer(respuesta.content, np.uint8), cv2.IMREAD_COLOR)

        if imagen is None:
            print(f"  intento {i}: {transcurrido:5.2f}s  respuesta NO es un JPEG valido")
            print(f"             Content-Type: {respuesta.headers.get('Content-Type', '?')}")
            continue

        tiempos.append(transcurrido)
        alto, ancho = imagen.shape[:2]
        print(
            f"  intento {i}: {transcurrido:5.2f}s  "
            f"{len(respuesta.content) / 1024:6.1f} KB  {ancho}x{alto}"
        )

    return tiempos, imagen


def evaluar_resolucion(imagen) -> list[str]:
    import cv2

    alto, ancho = imagen.shape[:2]
    avisos: list[str] = []

    print(f"\nResolucion: {ancho}x{alto}")

    # Suponiendo que el tablero ocupa ~80% del lado corto y 32 piezas en 8 filas.
    lado = min(ancho, alto) * 0.80
    pieza_px = lado / 8
    print(f"Pieza estimada: {pieza_px:.0f} px de alto (el dataset tenia ~40 px)")

    if ancho < ANCHO_MINIMO:
        avisos.append(
            f"Resolucion por debajo de {ANCHO_MINIMO}px de ancho. "
            "Subela en la web del firmware (Resolution > VGA o SVGA > Set)."
        )
    elif ancho < ANCHO_RECOMENDADO:
        avisos.append(
            f"Resolucion usable, pero {ANCHO_RECOMENDADO}px de ancho daria margen."
        )

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    brillo = float(gris.mean())
    print(f"Brillo medio: {brillo:.0f}/255")
    if brillo > 190:
        avisos.append(
            "Imagen muy sobreexpuesta: probablemente apunta a una luz o al techo."
        )
    elif brillo < 40:
        avisos.append("Imagen muy oscura: la deteccion va a fallar.")

    return avisos


def detectar_tablero(imagen) -> str:
    """Devuelve 'completo', 'cortado' o 'ausente'.

    Distinguir 'cortado' importa: si el tablero se sale del encuadre no se
    pueden marcar sus 4 esquinas, y sin las 4 esquinas la homografia no existe.
    Es el fallo mas confuso de diagnosticar porque la imagen 'se ve bien'.
    """
    import cv2
    import numpy as np

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_imagen = imagen.shape[0] * imagen.shape[1]
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro <= 0:
            continue
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aproximado) == 4 and area > area_imagen * 0.20:
            return "completo"

    # Sin cuadrilatero cerrado: si hay muchos bordes pegados al marco, el
    # tablero existe pero se sale de la imagen.
    margen = 3
    franjas = [
        bordes[:margen, :], bordes[-margen:, :],
        bordes[:, :margen], bordes[:, -margen:],
    ]
    proporcion = sum(float((f > 0).mean()) for f in franjas) / len(franjas)
    if proporcion > 0.10:
        return "cortado"
    return "ausente"


def contar_detecciones(imagen, modelo: Path) -> None:
    """Cuenta piezas con el modelo, sin que un fallo opaque el diagnostico."""
    from ultralytics import YOLO

    try:
        _, detect_mod, _ = cargar_modulos_equipo()
        red = YOLO(str(modelo))
        detecciones = normalizar_detecciones(
            detect_mod.detect(imagen, model=red, conf=CONF_DETECCION, agnostic_nms=True)
        )
    except Exception as exc:  # el diagnostico de camara no depende del modelo
        print(f"\nDetecciones: no se pudo usar el modelo ({type(exc).__name__}: {exc})")
        return

    print(f"\nDetecciones: {len(detecciones)} piezas con conf>={CONF_DETECCION}")
    if len(detecciones) < 8:
        print("  AVISO: menos de 8 piezas; no alcanza para inferir la rejilla.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.35.119.200", help="IP de la ESP32-CAM")
    parser.add_argument("--url", help="URL completa, gana sobre --ip")
    parser.add_argument("--intentos", type=int, default=3)
    parser.add_argument("--modelo", type=Path,
                        help="Opcional: cuenta piezas con este modelo (no afecta al diagnostico)")
    args = parser.parse_args()

    url = args.url or f"http://{args.ip}/capture"
    print(f"Probando {url}\n")

    tiempos, imagen = probar_url(url, args.intentos)
    if imagen is None:
        print("\nNo se obtuvo ninguna imagen valida.")
        print("Revisa: camara encendida, misma red, y que el endpoint sea /capture.")
        return 1

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(SALIDA), imagen)
    print(f"\nFrame guardado en {SALIDA}")

    if tiempos:
        promedio = sum(tiempos) / len(tiempos)
        print(f"Latencia media: {promedio:.2f}s")
        if promedio > 4.0:
            print("  AVISO: cerca del timeout de 5s de src/config.py.")

    avisos = evaluar_resolucion(imagen)

    estado_tablero = detectar_tablero(imagen)
    if estado_tablero == "completo":
        print("Tablero: completo en el encuadre, las 4 esquinas son visibles.")
    elif estado_tablero == "cortado":
        print("Tablero: PRESENTE pero CORTADO por el encuadre.")
        avisos.append(
            "El tablero se sale de la imagen. Sin las 4 esquinas no hay homografia: "
            "separa la camara o subela hasta que quede todo el tablero con margen."
        )
    else:
        print("Tablero: ausente.")
        avisos.append("No se ve un tablero. Apunta la camara al tablero.")

    if args.modelo:
        contar_detecciones(imagen, args.modelo)

    if avisos:
        print("\nA revisar:")
        for aviso in avisos:
            print(f"  - {aviso}")
        return 2

    print("\nTodo listo para el pipeline en vivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
