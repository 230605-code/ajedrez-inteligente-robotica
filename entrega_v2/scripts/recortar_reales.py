"""Recorta las fotos REALES al area jugable (vista rectificada 800x800).

Es el dataset derivado del clasificador por casilla para las fotos de camara.
Lee de ``dataset/crudo`` (solo lectura: las fotos originales no se tocan),
detecta el area jugable con ``esquinas_aruco.detectar_robusto`` y guarda:

    dataset/reales/recortes/800x800/<carpeta>/<nombre>.jpg
    dataset/auditoria/recortes_reales.csv      (una fila por foto)

``detectar_robusto`` encadena tres estrategias (ArUco multi-escala, el
cuadriculado azul/gris del tablero y, como ultimo recurso, la rejilla de
piezas del modelo). Cualquiera de las tres tiene que pasar la misma validacion
independiente: que la rejilla rectificada acierte el damero impreso del
tablero. El CSV dice por foto que metodo funciono y con que evidencia.

Las fotos a las que ninguna estrategia les encuentra el tablero NO se inventan
una homografia: quedan como ``sin_recorte`` con el motivo exacto.

El giro no se corrige a ojo: los ids de los ArUco (o, si no hay, la oscuridad de
las fichas) fijan cual de las cuatro esquinas es cual, asi que la vista sale
siempre con las blancas abajo, venga girada la camara o no.

Nota sobre los originales de celular: la auditoria del dataset confirmo que
tienen EXIF=6 espurio (los pixeles ya estan derechos). Por eso se leen con
``IMREAD_IGNORE_ORIENTATION``: si OpenCV aplicara el EXIF, giraria una foto que
ya viene bien y no habria homografia posible.

Uso
---
    python recortar_reales.py
    python recortar_reales.py --metodo robusto --limite 10
    python recortar_reales.py --metodo aruco        (solo ArUco, como antes)
    python recortar_reales.py --sin-piezas          (sin el modelo YOLO de v1)
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import esquinas_aruco
from comun import EXTENSIONES
from generar_sinteticos import LADO_RECORTE, destino_rectificado
from recortar_sinteticos import dibujar_recorte, fraccion_borde

RAIZ = Path(__file__).resolve().parent.parent
CRUDO = RAIZ / "dataset" / "crudo"
SALIDA = RAIZ / "dataset" / "reales" / "recortes" / f"{LADO_RECORTE}x{LADO_RECORTE}"
AUDITORIA = RAIZ / "dataset" / "auditoria" / "recortes_reales.csv"
PREVIEWS = RAIZ / "assets" / "previews"

CAMPOS_CSV = [
    "archivo", "estado", "metodo", "entero", "ids", "marcadores", "confianza",
    "color", "orientacion", "cuadricula_inliers", "rms_px", "ancho", "alto",
    "desborde", "motivo",
]


def _archivos(carpeta: Path) -> list[Path]:
    """Imagenes de todas las subcarpetas, ordenadas por ruta relativa."""
    return sorted(
        ruta for ruta in carpeta.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


def _leer(ruta: Path) -> np.ndarray | None:
    """Lee sin aplicar EXIF: en este dataset el EXIF de celular es espurio."""
    return cv2.imread(str(ruta), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def _procesar(ruta: Path, plantilla, metodo: str, usar_piezas: bool):
    """Devuelve (imagen, esquinas|None, metodo, detalle)."""
    imagen = _leer(ruta)
    if imagen is None:
        return None, None, "ninguno", {"motivo": "no se pudo leer la imagen con OpenCV",
                                       "ids": []}
    if metodo == "aruco":
        # Camino historico: solo ArUco, exige los 4 marcadores.
        try:
            deteccion = esquinas_aruco.detectar(imagen, plantilla)
        except Exception as exc:
            return imagen, None, "aruco", {
                "motivo": f"{type(exc).__name__}: {exc}", "ids": [],
            }
        if not deteccion.esquinas.any():
            return imagen, None, "aruco", {
                "motivo": deteccion.mensaje or "sin esquinas del area jugable",
                "ids": sorted(deteccion.marcadores),
            }
        detalle = {
            "motivo": deteccion.mensaje,
            "ids": sorted(deteccion.marcadores),
            "marcadores": len(deteccion.marcadores),
            "confianza": deteccion.confianza,
            "rms_px": deteccion.rms_px,
            "color": "",
            "orientacion": "aruco",
            "cuadricula_inliers": 0,
        }
        return imagen, deteccion.esquinas, "aruco", detalle

    esquinas, metodo_real, detalle = esquinas_aruco.detectar_robusto(
        imagen, plantilla, usar_piezas=usar_piezas
    )
    return imagen, esquinas, metodo_real, detalle


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Recorta las fotos reales al area jugable en 800x800.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--crudo", default=str(CRUDO),
                            help="carpeta con las fotos reales (no se modifica)")
    analizador.add_argument("--salida", default=str(SALIDA),
                            help="carpeta de recortes")
    analizador.add_argument("--csv", default=str(AUDITORIA),
                            help="CSV de auditoria (una fila por foto)")
    analizador.add_argument("--metodo", choices=("robusto", "aruco"), default="robusto",
                            help="robusto = ArUco multi-escala + cuadriculado + piezas")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo las primeras N fotos (0 = todas)")
    analizador.add_argument("--sin-previews", action="store_true",
                            help="no escribir las previews en assets/previews")
    analizador.add_argument("--sin-piezas", action="store_true",
                            help="no usar el modelo YOLO como ultimo recurso")
    argumentos = analizador.parse_args(argv)

    crudo = Path(argumentos.crudo)
    if not crudo.exists():
        raise SystemExit(f"No encuentro {crudo}")
    salida = Path(argumentos.salida)
    csv_salida = Path(argumentos.csv)
    lado = LADO_RECORTE
    destino_base = destino_rectificado(lado)

    archivos = _archivos(crudo)
    if argumentos.limite > 0:
        archivos = archivos[:argumentos.limite]
    if not archivos:
        raise SystemExit(f"No hay imagenes en {crudo}")

    plantilla = esquinas_aruco.analizar_plantilla()
    salida.mkdir(parents=True, exist_ok=True)
    csv_salida.parent.mkdir(parents=True, exist_ok=True)

    inicio = time.time()
    filas = []
    exitos = []
    recortadas = 0
    fallidas = 0
    por_metodo = Counter()
    motivos = Counter()
    desbordes: list[tuple[str, float]] = []
    por_carpeta: dict[str, int] = {}
    reutilizados = 0

    for numero, ruta in enumerate(archivos, start=1):
        relativa = ruta.relative_to(crudo)
        carpeta = relativa.parent.name if relativa.parent != Path(".") else ""
        imagen, esquinas, metodo, detalle = _procesar(
            ruta, plantilla, argumentos.metodo, not argumentos.sin_piezas
        )
        alto = int(imagen.shape[0]) if imagen is not None else 0
        ancho = int(imagen.shape[1]) if imagen is not None else 0
        borde = 0.0
        ids = detalle.get("ids") or []

        if esquinas is not None and imagen is not None:
            matriz = cv2.getPerspectiveTransform(
                esquinas.astype("float32"), destino_base
            )
            recorte = cv2.warpPerspective(
                imagen, matriz, (lado, lado), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            destino = salida / relativa.parent / f"{ruta.stem}.jpg"
            destino.parent.mkdir(parents=True, exist_ok=True)
            if destino.exists():
                reutilizados += 1
            cv2.imwrite(str(destino), recorte, [cv2.IMWRITE_JPEG_QUALITY, 95])
            borde = fraccion_borde(imagen, matriz, lado)
            if borde > 1e-9:
                desbordes.append((relativa.as_posix(), borde))
            recortadas += 1
            por_metodo[metodo] += 1
            por_carpeta[carpeta] = por_carpeta.get(carpeta, 0) + 1
            exitos.append({
                "relativa": relativa,
                "carpeta": carpeta,
                "matriz": matriz,
                "borde": borde,
                "metodo": metodo,
            })
            filas.append({
                "archivo": relativa.as_posix(),
                "estado": "recortada",
                "metodo": metodo,
                "entero": int(borde <= 1e-9),
                "ids": " ".join(str(i) for i in ids),
                "marcadores": len(ids),
                "confianza": f"{float(detalle.get('color') or 0.0):.4f}",
                "color": f"{float(detalle.get('color') or 0.0):.4f}",
                "orientacion": detalle.get("orientacion", ""),
                "cuadricula_inliers": detalle.get("cuadricula_inliers", 0),
                "rms_px": ("" if detalle.get("rms_px") in (None, "")
                           else f"{float(detalle['rms_px']):.3f}"),
                "ancho": ancho,
                "alto": alto,
                "desborde": f"{borde * 100:.4f}%",
                "motivo": detalle.get("motivo", ""),
            })
        else:
            fallidas += 1
            motivo = detalle.get("motivo") or "sin esquinas del area jugable"
            motivos[motivo] += 1
            filas.append({
                "archivo": relativa.as_posix(),
                "estado": "sin_recorte",
                "metodo": metodo,
                "entero": 0,
                "ids": " ".join(str(i) for i in ids),
                "marcadores": len(ids),
                "confianza": "",
                "color": "",
                "orientacion": "",
                "cuadricula_inliers": "",
                "rms_px": "",
                "ancho": ancho,
                "alto": alto,
                "desborde": "",
                "motivo": motivo,
            })

        if numero % 10 == 0 or numero == len(archivos):
            print(f"  {numero}/{len(archivos)} fotos "
                  f"(recortadas={recortadas}, sin_recorte={fallidas})", flush=True)

    with open(csv_salida, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)

    # 3 previews: una por bloque si se puede; si no, las primeras recortadas.
    previews: list[str] = []
    if not argumentos.sin_previews and exitos:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        elegidas = []
        for carpeta in ("01_inicial", "02_medio_juego", "03_finales"):
            elegida = next((e for e in exitos if e["carpeta"] == carpeta), None)
            if elegida is not None and elegida not in elegidas:
                elegidas.append(elegida)
        for candidata in exitos:
            if len(elegidas) >= 3:
                break
            if candidata not in elegidas:
                elegidas.append(candidata)
        for indice, elegida in enumerate(elegidas[:3], start=1):
            imagen = _leer(crudo / elegida["relativa"])
            recorte = cv2.warpPerspective(
                imagen, elegida["matriz"], (lado, lado), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            titulo = (f"{elegida['relativa'].stem}  |  {elegida['metodo']}  |  "
                      f"borde={elegida['borde'] * 100:.2f}%")
            destino = PREVIEWS / f"recorte_muestra_real_{indice:02d}.jpg"
            cv2.imwrite(str(destino), dibujar_recorte(recorte, [], titulo),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            previews.append(destino.name)

    segundos = time.time() - inicio
    print()
    print(f"listo: {len(archivos)} fotos en {segundos:.1f} s")
    print(f"  recortadas: {recortadas} ({recortadas / len(archivos) * 100:.1f}%)")
    print(f"  sin recorte: {fallidas}")
    print(f"  por metodo: {dict(por_metodo)}")
    print(f"  por carpeta (recortadas): {por_carpeta}")
    print(f"  recortes ya existentes sobrescritos: {reutilizados}")
    print("  motivos (sin recorte):")
    for motivo, cantidad in motivos.most_common(10):
        print(f"    {cantidad:3d}  {motivo}")
    print(f"  desbordes: {len(desbordes)} (max "
          f"{max([f for _, f in desbordes], default=0.0) * 100:.4f}%)")
    print(f"  recortes: {salida}")
    print(f"  auditoria: {csv_salida}")
    print(f"  previews: {previews if previews else 'omitidas'}")
    print(f"RESULTADO: {recortadas} recortadas, {fallidas} sin recorte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
