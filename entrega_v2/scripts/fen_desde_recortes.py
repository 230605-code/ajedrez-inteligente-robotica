"""Smoke del pipeline v2 sobre los recortes reales verificados.

Prueba end-to-end: recorte rectificado 800x800 -> clasificador por casilla ->
FEN -> post-filtro de movimientos legales -> comparacion contra la verdad
(etiquetas del CSV de autoetiquetado: posicion inicial o tablero vacio).

 dispersion:
    python scripts/fen_desde_recortes.py           (20 verificadas + manual)
    python scripts/fen_desde_recortes.py --detallar
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pipeline_fen import (  # noqa: E402
    cargar_clasificador, clasificar_recorte, construir_fen, legalizar,
    realinear_por_fichas,
)

from comun import ordenar_esquinas  # noqa: E402  (para detectar_piezas)


def detectar_piezas(resultado) -> list[tuple[str, float, float, float]]:
    """[(clase, x_centro, y_centro, confianza), ...] de una salida YOLO.

    Salida en coordenadas del recorte. Agnostic NMS evita que la misma ficha
    salga dos veces con clases distintas.
    """
    piezas = []
    for i in range(len(resultado.boxes)):
        nombre = resultado.names[int(resultado.boxes.cls[i].item())]
        x1, y1, x2, y2 = [float(v) for v in resultado.boxes.xyxy[i].tolist()]
        piezas.append((nombre, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                       float(resultado.boxes.conf[i].item())))
    return piezas
    for i in range(len(resultado.boxes)):
        nombre = resultado.names[int(resultado.boxes.cls[i].item())]
        x1, y1, x2, y2 = [float(v) for v in resultado.boxes.xyxy[i].tolist()]
        piezas.append((nombre, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                       float(resultado.boxes.conf[i].item())))
    return piezas

RAIZ = Path(__file__).resolve().parent.parent
CSV_ETIQUETAS = RAIZ / "dataset" / "reales" / "etiquetas_clasificador.csv"
PESO = RAIZ / "modelos" / "best_clasificador.pt"
CARPETA = RAIZ / "dataset" / "reales" / "recortes" / "800x800"

VALOR_EN = {"peon": chr(0x265F), "torre": chr(0x265C), "caballo": chr(0x265E),
            "alfil": chr(0x265D), "dama": chr(0x265B), "rey": chr(0x265A)}


def cargar_verdad() -> dict[str, dict[str, str]]:
    """archivo_recorte -> {casilla: clase}; None si quedo 'desconocida'."""
    por_archivo: dict[str, dict[str, str]] = {}
    estado: dict[str, str] = {}
    with CSV_ETIQUETAS.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            archivo = fila["archivo_recorte"]
            estado.setdefault(archivo, fila.get("estado", ""))
            por_archivo.setdefault(archivo, {})[fila["casilla"]] = fila["clase"]
    verificado = {}
    for archivo, tabla in por_archivo.items():
        valido = {c: clase for c, clase in tabla.items()
                  if c and clase and clase != "desconocida"}
        if valido:
            verificado[archivo] = valido
    return verificado


def simbolo(clase: str) -> str:
    if clase == "vacia":
        return "."
    pieza, _, color = clase.partition("_")
    return VALOR_EN[pieza] if color == "n" else VALOR_EN[pieza].upper()


def main() -> int:
    detallar = "--detallar" in sys.argv
    verdad = cargar_verdad()
    print(f"recortes con verdad conocida: {len(verdad)}")

    net, clases, info = cargar_clasificador(PESO)
    print(f"clasificador cargado: {info}")

    from ultralytics import YOLO
    detector = YOLO(str(RAIZ / "modelos" / "best.pt"))

    def detectar(imagen):
        resultado = detector.predict(imagen, conf=0.25,
                                     agnostic_nms=True, verbose=False)[0]
        return detectar_piezas(resultado)

    aciertos = 0
    total = 0
    legales = []
    cambios_totales = 0
    fallos = []

    for indice, (archivo, tabla) in enumerate(sorted(verdad.items()), 1):
        nombre = archivo.split("/")[-1]
        ruta = CARPETA / archivo
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"[{indice}] {archivo}: no pude leer")
            continue

        # Realinear ANTES de clasificar si el recorte esta descentrado.
        alineada, motivo = realinear_por_fichas(imagen, detectar)
        if alineada is None:
            print(f"[{indice}] {nombre}: sin realineacion ({motivo}), sigo con "
                  "el recorte original")
        else:
            imagen = alineada
            print(f"[{indice}] {nombre}: {motivo}")

        ocupacion = clasificar_recorte(net, clases, imagen)
        # Algunos archivos 'manual' pueden tener tablas parciales: tolerar.
        aciertos_directo = sum(
            1 for casilla, dto in ocupacion.items()
            if casilla in tabla and dto["elegida"] == tabla[casilla]
        )
        celdas_verdad = max(len(tabla), len(ocupacion))
        clasificados = {c: dto["elegida"] for c, dto in ocupacion.items()}
        fen_crudo = construir_fen(clasificados)
        fen_final, cambios = legalizar(ocupacion)
        fen_anterior, _ = (fen_crudo, cambios)

        # La verdad es la posicion inicial o el vacio: el FEN final debe
        # coincidir pieza por pieza con las etiquetas.
        compare = fen_final
        legado = None
        import chess as chess_mod

        final = chess_mod.Board(compare)
        piezas_fen = {chess_mod.square_name(c):
                          (p.symbol() if p else ".")
                      for c in chess_mod.SQUARES
                      for p in [final.piece_at(c)]}
        aciertos_fen = sum(
            1 for c in tabla
            if (piezas_fen[c] != ".") == (tabla[c] != "vacia")
            and (tabla[c] == "vacia" or
                 piezas_fen[c].lower() == tabla[c][0].replace("peon", "p")[:1])
        )
        total += celdas_verdad
        aciertos += aciertos_directo
        cambios_totales += len(cambios)
        if len(cambios):
            import chess  # reuso: valida la posicion
            estado_final = chess.Board(fen_final).status()
        else:
            estado_final = None
        legales.append(estado_final is None)
        if aciertos_directo != celdas_verdad:
            errores = {c: (tabla[c], dto["elegida"]) for c, dto in ocupacion.items()
                       if dto["elegida"] != tabla[c]}
            fallos.append((archivo, errores))
        if detallar or aciertos_directo < len(tabla):
            print(f"[{indice}] {nombre:26s} casillas OK {aciertos_directo}/{len(tabla)}"
                  f"  cambios_legales={len(cambios)}  fen={compare[:22]}...")
            print(f"      piezas_FEN_correctas={aciertos_fen}/{len(tabla)}"
                  f"  legal={'si' if estado_final is None else 'no'}")
            for casilla, (correcta, pre) in list(errores.items())[:8]:
                print(f"      {casilla}: verdad {correcta:10s} dijo {pre:10s}")

    print()
    print(f"casillas evaluadas: {total}")
    print(f"aciertos del clasificador: {aciertos}/{total} "
          f"({100.0 * aciertos / max(total, 1):.2f}%)")
    print(f"fotos con FEN legal sin tocar: {sum(1 for l in legales if l)}/{len(legales)}")
    print(f"cambios del filtro legal: {cambios_totales}")
    if fallos:
        print(f"recortes con errores: {len(fallos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
