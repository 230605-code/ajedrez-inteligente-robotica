"""Revision humana de las 128 fotos reales: el modelo propone, tu corriges.

Genera (modo --generar):
  1. dataset/revision_humana/marcados/<carpeta>/<nombre>.jpg
     el recorte REALINEADO con la clase predecida por clasificador en cada
     casilla (verde = confianza alta; rojo = baja).
  2. dataset/revision_humana/revision.csv
     UNA FILA POR CASILLA: clase_en_csv (lo que decia la etiqueta), 
     clase_predecida, confianza, y una columna VERDICT vacia.
     TU editas VERDICT solo en las casillas malas escribiendo la clase real
     (peon_b..vacia) y listo: vacio = aceptar la prediccion.

Aplica modo --aplicar:
  fusiona el verde del usuario, reescribe etiquetas_clasificador.csv con
  estado 'revisada' y regenera los labels YOLO para el bundle del detector.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from pipeline_fen import cargar_clasificador, clasificar_recorte, realinear_por_fichas  # noqa: E402

CRUDO = RAIZ / "dataset" / "crudo"
REVISION = RAIZ / "dataset" / "revision_humana"
MARCADOS = REVISION / "marcados"
CSV_ETIQUETAS = RAIZ / "dataset" / "reales" / "etiquetas_clasificador.csv"
CSV_REVISION = REVISION / "revision.csv"
PESO = RAIZ / "modelos" / "best_clasificador.pt"

ROJO = (60, 60, 230)
VERDE = (90, 220, 90)


def recortes_reales() -> list[Path]:
    salto = []
    for base in sorted((RAIZ / "dataset" / "reales" / "recortes" / "800x800").glob("*/*.jpg")):
        salto.append(base)
    return salto


def clase_actual_de(archivo: str, casilla: str,
                    por_archivo: dict[str, dict[str, str]]) -> str:
    return por_archivo.get(archivo, {}).get(casilla, "desconocida")


def generar() -> None:
    from ultralytics import YOLO

    etiquetas: dict[str, dict[str, str]] = {}
    with CSV_ETIQUETAS.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            if not fila.get("casilla"):
                continue
            etiquetas.setdefault(fila["archivo_recorte"], {})[fila["casilla"]] = fila["clase"]

    detector = YOLO(str(RAIZ / "modelos" / "best.pt"))

    def detectar(imagen):
        res = detector.predict(imagen, conf=0.25, agnostic_nms=True, verbose=False)[0]
        return [
            (res.names[int(b.cls.item())],
             float((b.xyxy[0][0] + b.xyxy[0][2]) / 2.0),
             float((b.xyxy[0][1] + b.xyxy[0][3]) / 2.0),
             float(b.conf.item()))
            for b in res.boxes
        ]

    net, clases, _ = cargar_clasificador(PESO)
    MARCADOS.mkdir(parents=True, exist_ok=True)

    filas_csv = []
    archivos = recortes_reales()
    for indice, base in enumerate(archivos, 1):
        nombre = base.name
        padre = base.parent.name
        imagen = cv2.imread(str(base))
        if imagen is None:
            continue
        alineada, motivo = realinear_por_fichas(imagen, detectar)
        if alineada is not None:
            imagen = alineada
        volumen = clasificar_recorte(net, clases, imagen)
        marcado = imagen.copy()
        lado = 100
        # PROTOCOLO DE COLORES (para el usuario):
        #   VERDE   = la etiqueta existente y la prediccion CONCUERDAN: no mirar
        #   NARANJA = hay etiqueta y DIFIERE de la prediccion: MIRAR esta casilla
        #             (puede ser etiqueta mal (el swap fisico) o modelo mal)
        #   ROJO    = sin etiqueta en CSV y modelo INSEGURO (< 0.85): MIRAR
        #   CELESTE = sin etiqueta y modelo seguro: normalmente bueno
        CONFIANZA_OK = 0.85
        NARANJA = (30, 140, 255)
        for fila in range(8):
            for columna in range(8):
                casilla = f"{chr(ord('a') + columna)}{8 - fila}"
                dato = volumen[casilla]
                actual = clase_actual_de(f"{padre}/{nombre}", casilla, etiquetas)
                concuerda = (actual != "desconocida"
                             and actual == dato["elegida"])
                if concuerda:
                    color, clase_rol = VERDE, "coincide"
                elif actual != "desconocida":
                    color, clase_rol = NARANJA, "conflictivo"
                elif dato["confianza"] < CONFIANZA_OK:
                    color, clase_rol = ROJO, "baja_confianza"
                else:
                    color, clase_rol = (0, 200, 220), "propuesta"
                cv2.putText(marcado, casilla, (columna * lado + 4, fila * lado + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(marcado, dato["elegida"], (columna * lado + 4, fila * lado + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                cv2.putText(marcado, f"{100 * dato['confianza']:.0f}",
                            (columna * lado + 4, fila * lado + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
                filas_csv.append({
                    "archivo": f"{padre}/{nombre}",
                    "casilla": casilla,
                    "clase_en_csv": actual,
                    "clase_predecida": dato["elegida"],
                    "confianza": round(dato["confianza"], 3),
                    "rol_color": clase_rol,
                    "verdict": "",
                })
        destino = MARCADOS / padre
        destino.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino / nombre), marcado)
        print(f"[{indice}/{len(archivos)}] {padre}/{nombre}::{motivo}")

    with CSV_REVISION.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["archivo", "casilla",
                                                "clase_en_csv",
                                                "clase_predecida",
                                                "confianza", "rol_color",
                                                "verdict"])
        escritor.writeheader()
        escritor.writerows(filas_csv)
    print(f"\nrevision.csv listo: {len(filas_csv)} filas")
    print(f"marcados: {MARCADOS}")
    print("Edita VERDICT con la clase real SOLO donde el modelo se equivoco.")


def aplicar() -> None:
    if not CSV_REVISION.exists():
        raise SystemExit(f"No encuentro {CSV_REVISION}; corre --generar primero")
    filas = list(csv.DictReader(CSV_REVISION.open(encoding="utf-8-sig", newline="")))
    por_archivo: dict[str, dict[str, str]] = {}
    estados_archivo: dict[str, str] = {}
    verdicts: Counter = Counter()
    descartados: set[str] = set()
    for fila in filas:
        vered = (fila.get("verdict") or "").strip()
        archivo = fila["archivo"]
        if vered == "DESCARTADA":
            descartados.add(archivo)
            estados_archivo[archivo] = "descartada"
            continue
        por_archivo.setdefault(archivo, {})[fila["casilla"]] = \
            vered if vered else fila["clase_predecida"]
        verdicts["corregida" if vered else "ok"] += 1
    print(f"verdicts: {dict(verdicts)}   descartados: {sorted(descartados)}")

    actuales: dict[str, dict[str, str]] = {}
    estados: dict[str, dict[str, str]] = {}
    with CSV_ETIQUETAS.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            if not fila.get("casilla"):
                continue
            estados.setdefault(fila["archivo_recorte"], {})[fila["casilla"]] = fila.get("estado", "")
            actuales.setdefault(fila["archivo_recorte"], {})[fila["casilla"]] = fila["clase"]
    escritores = []
    todo = sorted(set(actuales) | set(por_archivo) | descartados)
    for archivo in todo:
        estado_de = estados_archivo.get(archivo, "")
        if archivo in descartados:
            # Descartadas por el humano: sus filas quedan con estado
            # descartada y clase vacia para que TODO el pipeline las excluya.
            for numero in range(11):
                casilla = (f"{chr(ord('a') + numero)}" if numero == 10 else "")
            for casilla in sorted(actuales.get(archivo,
                                               por_archivo.get(archivo, {}))):
                escritores.append({"archivo_recorte": archivo,
                                   "casilla": casilla, "clase": "",
                                   "estado": "descartada",
                                   "motivo": "orientacion no diagnosticada"})
            continue
        if archivo in por_archivo:
            nueva_tabla = por_archivo[archivo]
            previa = actuales.get(archivo, {})
            viejos_estados = estados.get(archivo, {})
            cambiadas = {c for c in nueva_tabla
                         if nueva_tabla[c] != previa.get(c, '')}
            for casilla, clase in sorted(nueva_tabla.items()):
                cambio = casilla in cambiadas
                escritores.append({'archivo_recorte': archivo,
                                   'casilla': casilla, 'clase': clase,
                                   'estado': 'revisada' if cambio else
                                             viejos_estados.get(casilla, ''),
                                   'motivo': 'revision humana v2' if cambio
                                             else ''})

