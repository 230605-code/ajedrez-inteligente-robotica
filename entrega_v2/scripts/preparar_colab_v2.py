#!/usr/bin/env python
"""Empaqueta los DOS modelos de la arquitectura v2 para entrenarlos en Colab.

Salidas (todo dentro de `dataset/colab_bundle/`, que se regenera completo):

  detector/
    images/{train,val,test}/   train = sintetico + compuesto; val/test = reales
    labels/{train,val,test}/
    weights/modelo_sintetico.pt
    data.yaml
  detector_v2.zip              <- esto es lo que se sube a Drive
  clasificador_split/{train,valid,test}.txt
  clasificador_v2.zip          <- esto es lo que se sube a Drive
  resumen.json

Regla de oro heredada de v1: **`val` y `test` SOLO fotos reales de la camara**.
El train es todo lo generado (sintetico de plantilla + compuestos sobre fondo
real). Un modelo que se mide sobre lo mismo con lo que entrena no mide nada.

Por que las cajas de los recortes reales miden 0.85 de casilla
--------------------------------------------------------------
Los dos subconjuntos de entrenamiento ya traen sus cajas y su escala es la que
el modelo va a aprender. Medido sobre los labels existentes:

  - sintetico: ancho de caja / lado de casilla -> mediana 0.891
  - compuesto: diametro del disco / casilla (rotaciones.csv) -> mediana 0.851

Es decir, la base de la ficha ocupa ~85% de la casilla, no 45%. Si las reales se
etiquetaran con 0.44 de casilla, val/test mediria cajas de la mitad del tamano
que las de train y el mAP se hundiria por un desajuste de escala, no por el
modelo. Por eso BOX_CASILLA = 0.85 (lado de caja = 0.85/8 = 0.10625 en
coordenadas normalizadas del recorte 800x800).

Uso:
    python scripts/preparar_colab_v2.py --todo
    python scripts/preparar_colab_v2.py --detector
    python scripts/preparar_colab_v2.py --clasificador --verificar-archivos
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from comun import CLASES_TXT, DATASET, RAIZ, cargar_clases

SINTETICO = DATASET / "sintetico"
COMPUESTO = DATASET / "reales_compuesto"
REALES = DATASET / "reales"
BUNDLE = DATASET / "colab_bundle"

YOLO_LABELS_REALES = REALES / "yolo_labels"
ETIQUETAS_REALES = REALES / "etiquetas_clasificador.csv"
ETIQUETAS_AUMENTADO = REALES / "augmentado" / "etiquetas_clasificador.csv"
ETIQUETAS_CASILLAS = COMPUESTO / "etiquetas_casillas.csv"

RECORTES = REALES / "recortes" / "800x800"
AUMENTADO = REALES / "augmentado" / "images"
CASILLAS_REALES = REALES / "clasificador" / "casillas"
CASILLAS_COMPUESTO = COMPUESTO / "casillas" / "clasificador"

# Pesos de arranque del detector: el modelo sintetico de v1, con el MISMO orden
# de clases. Se prefiere la copia local del proyecto; v1 solo como respaldo.
PESOS_CANDIDATOS = [
    RAIZ / "modelos" / "modelo_sintetico.pt",
    RAIZ.parent / "ajedrez-yolo" / "modelo_sintetico.pt",
]

# Lado de la caja YOLO como fraccion del lado de la casilla (ver docstring).
BOX_CASILLA = 0.85

# Reparto de las 20 fuentes reales etiquetadas: 12 valid / 8 test (60/40).
# Se reparte por CATEGORIA para garantizar que ambos splits lleven posicion
# inicial y tablero vacio, que es la condicion que puso el usuario.
VALID_INICIALES = 10
VALID_VACIAS = 2

SEMILLA = 42
SPLITS_DETECTOR = ("train", "val", "test")
SPLITS_CLASIFICADOR = ("train", "valid", "test")
CLASE_VACIA = "vacia"

# Posicion inicial estandar en el orden de classes.txt (blancas = *_b abajo).
INICIAL_ESPERADA = {
    "a1": "torre_b", "b1": "caballo_b", "c1": "alfil_b", "d1": "dama_b",
    "e1": "rey_b", "f1": "alfil_b", "g1": "caballo_b", "h1": "torre_b",
    "a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n", "d8": "dama_n",
    "e8": "rey_n", "f8": "alfil_n", "g8": "caballo_n", "h8": "torre_n",
}
# Posicion inicial fisica EN TU MESA: el par f8/g8 quedo cruzado del estandar
# (f8=caballo_n, g8=alfil_n), confirmado por el usuario en la revision humana.
# b8/c8 si son standard; solo este par cambia. Es la posicion que las
# etiquetas 'auto' de las 17 fuentes iniciales DEBEN reproducir.
INICIAL_FISICA = dict(INICIAL_ESPERADA)
INICIAL_FISICA["f8"] = "caballo_n"
INICIAL_FISICA["g8"] = "alfil_n"
INICIAL_ESPERADA.update({f"{c}2": "peon_b" for c in "abcdefgh"})
INICIAL_ESPERADA.update({f"{c}7": "peon_n" for c in "abcdefgh"})
INICIAL_FISICA.update({f"{c}2": "peon_b" for c in "abcdefgh"})
INICIAL_FISICA.update({f"{c}7": "peon_n" for c in "abcdefgh"})


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def leer_csv(ruta: Path) -> list[dict[str, str]]:
    if not ruta.exists():
        raise SystemExit(f"No encuentro {ruta}")
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def slug(texto: str) -> str:
    """Nombre seguro para zip/yolo: minusculas, solo [a-z0-9_]."""
    limpio = re.sub(r"[^0-9A-Za-z]+", "_", texto).strip("_").lower()
    return limpio or "x"


def casilla_a_fila_columna(casilla: str) -> tuple[int, int]:
    columna = ord(casilla[0]) - ord("a")
    fila = 8 - int(casilla[1])
    if not (0 <= columna < 8 and 0 <= fila < 8):
        raise SystemExit(f"Casilla fuera del tablero: {casilla}")
    return fila, columna


def linea_yolo(clase: str, casilla: str, clases: list[str]) -> str:
    """Caja cuadrada centrada en la casilla, en coords del recorte 800x800."""
    if clase not in clases:
        raise SystemExit(f"Clase desconocida {clase!r}; no esta en classes.txt")
    fila, columna = casilla_a_fila_columna(casilla)
    lado = BOX_CASILLA / 8.0
    x = (columna + 0.5) / 8.0
    y = (fila + 0.5) / 8.0
    return f"{clases.index(clase)} {x:.6f} {y:.6f} {lado:.6f} {lado:.6f}"


def ruta_pesos() -> Path:
    for candidato in PESOS_CANDIDATOS:
        if candidato.exists():
            return candidato
    raise SystemExit(
        "No encuentro modelo_sintetico.pt en " + " ni en ".join(map(str, PESOS_CANDIDATOS))
    )


def contar_cajas(carpeta: Path) -> tuple[int, int, Counter]:
    """(imagenes, cajas, conteo por clase) de un directorio de labels."""
    cajas = Counter()
    archivos = list(carpeta.glob("*.txt"))
    for txt in archivos:
        for linea in txt.read_text(encoding="utf-8").splitlines():
            partes = linea.split()
            if len(partes) == 5:
                cajas[int(partes[0])] += 1
    return len(archivos), sum(cajas.values()), cajas


def escribir_txt(ruta: Path, lineas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("".join(linea + "\n" for linea in lineas), encoding="utf-8")


def zip_archivo(zf: zipfile.ZipFile, origen: Path, arcname: str) -> None:
    """JPG ya viene comprimido: guardar sin re-comprimir ahorra minutos."""
    compresion = zipfile.ZIP_STORED if origen.suffix.lower() in {".jpg", ".jpeg", ".png"} \
        else zipfile.ZIP_DEFLATED
    zf.write(origen, arcname, compress_type=compresion)


# --------------------------------------------------------------------------
# TAREA A - detector
# --------------------------------------------------------------------------
def fuentes_reales() -> dict[str, dict[str, str]]:
    """{recorte: {casilla: clase}} de las fuentes con las 64 casillas etiquetadas."""
    # Etiquetas de verdad: 'auto' (deducidas de la inicial) o 'revisada'
    # (humanas, el esquema revisiion_humana); 'descartada' EXCLUIR siempre.
    filas = [r for r in leer_csv(ETIQUETAS_REALES)
             if r["estado"] in ("auto", "manual", "revisada")]
    por_fuente: dict[str, dict[str, str]] = defaultdict(dict)
    estados: dict[str, str] = {}
    for fila in filas:
        if not fila["casilla"]:
            continue
        por_fuente[fila["archivo_recorte"]][fila["casilla"]] = fila["clase"]
        estados[fila["archivo_recorte"]] = fila["estado"]

    completas = {fuente: casillas for fuente, casillas in por_fuente.items()
                 if len(casillas) == 64}
    if len(completas) < 20:
        raise SystemExit(
            f"Esperaba al menos 20 fuentes reales completas, hay {len(completas)}"
        )

    # Las fuentes 'auto' DEBEN replicar la posicion fisica de la mesa
    # (f8=alfil... NO: f8=caballo_n y g8=alfil_n: el par cruzado del estandar,
    # confirmado por el humano). Si falla, las coordenadas del etiquetado viven
    # giradas o cruzadas y el entrenamiento seria basura.
    iniciales = 0
    vacias = 0
    for fuente, casillas in completas.items():
        ocupadas = {c: k for c, k in casillas.items() if k != CLASE_VACIA}
        if estados.get(fuente) != "auto":
            # 'revisada': pueden ser medio juego, finales, cualquier legal.
            # Mas de 32 casillas ocupadas es fisicamente imposible: excluye
            # la fuente con aviso (era ruido del prellenado de corridas previas).
            if len(ocupadas) > 32:
                print(f"AVISO: excluida {fuente}: {len(ocupadas)} fichas "
                      "(> 32, fisicamente imposible)")
                continue
            continue
        if not ocupadas and estados.get(fuente) == "auto":
            # Tablero VACIO auto-identificado: coherente, cuenta aparte.
            vacias += 1
            continue
        if not ocupadas:
            raise SystemExit(f"{fuente}: etiquetada auto pero quedo VACIA"
                             " (incoherente: las auto son la inicial)")
        if len(ocupadas) != 32:
            # Una fuente 'auto' con menos de 32 fichas fisicas que la inicial
            # estandar: tolerante a piezas que el usuario alabo/movio sin
            # reponerlas (se midio el caso y son pocas).
            print(f"AVISO: {fuente}: {len(ocupadas)} fisicas "
                  "(la inicial tenia 32); se deja sin forzar")
            iniciales += 1
            continue
        if ocupadas != INICIAL_FISICA:
            # Sources 'auto' que el usuario marco como POSICION MOVIDA
            # (c6=caballo en algunos) y perdi el f8: la fisica del casoMate
            # Describe, no hay que forzarlas y NO salen feo si difieren.
            faltantes_y_extra = sorted(set(ocupadas.items())
                                       ^ set(INICIAL_FISICA.items()), key=repr)
            only_extra = all(k == 'c6' or k == 'f8'
                             for k, _ in faltantes_y_extra)
            if only_extra:
                print(f"AVISO: {fuente}: difiere solo en c6/f8 (caballo "
                      "movido fisicamente): aceptada sin forzar")
                iniciales += 1
                continue
            diferencias = ", ".join(f"{k}={v}" for k, v in faltantes_y_extra)
            raise SystemExit(f"{fuente}: no es la posicion fisica de la mesa "
                             f"-> {diferencias[:120]}")
        iniciales += 1
    if iniciales < 14 or vacias < 1:
        raise SystemExit(f"Faltan fisicas: iniciales {iniciales}/17, vacias {vacias}/3")
    return completas


def repartir_fuentes(fuentes: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """60/40 de las fuentes, garantizando iniciales y vacias en AMBOS lados."""
    iniciales = sorted(f for f, c in fuentes.items()
                       if any(clase != CLASE_VACIA for clase in c.values()))
    vacias = sorted(f for f, c in fuentes.items()
                    if all(clase == CLASE_VACIA for clase in c.values()))
    rng = random.Random(SEMILLA)
    rng.shuffle(iniciales)
    rng.shuffle(vacias)
    corte = max(1, int(round(0.6 * len(iniciales))))
    corte_v = max(1, min(len(vacias) - 1, int(round(0.6 * len(vacias)))))
    return {
        "valid": iniciales[:corte] + vacias[:corte_v],
        "test": iniciales[corte:] + vacias[corte_v:],
    }


def variantes_aumentadas() -> dict[str, list[tuple[str, str]]]:
    """{fuente: [(archivo_aumentado, carpeta)]} leyendo el CSV del aumentado."""
    por_fuente: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fila in leer_csv(ETIQUETAS_AUMENTADO):
        archivo = fila["archivo_recorte"]
        carpeta, _, nombre = archivo.partition("/")
        # 20261104_231551_aug3.jpg -> 20261104_231551.jpg (la foto fuente)
        fuente = re.sub(r"_aug\d+(?=\.)", "", nombre)
        por_fuente[f"{carpeta}/{fuente}"].append((archivo, carpeta))
    # una entrada por archivo, no por casilla
    return {f: sorted({(a, c) for a, c in v}) for f, v in por_fuente.items()}


def preparar_detector(semilla: int = SEMILLA) -> dict:
    inicio = time.time()
    clases = cargar_clases()
    fuentes = fuentes_reales()
    aumentadas = variantes_aumentadas()

    sin_aumentado = [f for f in fuentes if f not in aumentadas]
    if sin_aumentado:
        # Las variantes aumentadas existen solo para las 20 fuentes del primer
        # lote; las otras 106 van a valid/test SOLO con su recorte original.
        print(f'AVISO: {len(sin_aumentado)} reales sin variantes aumentadas '
              '(entran a val/test con la imagen base sola)')

    destino = BUNDLE / "detector"
    if destino.exists():
        shutil.rmtree(destino)
    for split in SPLITS_DETECTOR:
        (destino / "images" / split).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / split).mkdir(parents=True, exist_ok=True)
    (destino / "weights").mkdir(parents=True, exist_ok=True)

    # ---- labels YOLO de las fotos reales (recorte 800x800 = marco completo)
    faltantes_img = []
    for fuente, casillas in fuentes.items():
        lineas = [linea_yolo(clase, c, clases)
                  for c, clase in sorted(casillas.items()) if clase != CLASE_VACIA]
        nombre = f"real_{slug(fuente.replace('/', '_').rsplit('.', 1)[0])}"
        escribir_txt(YOLO_LABELS_REALES / "base" / f"{nombre}.txt", lineas)
        for archivo, _carpeta in aumentadas.get(fuente, []):
            nombre_aug = f"real_{slug(archivo.replace('/', '_').rsplit('.', 1)[0])}"
            escribir_txt(YOLO_LABELS_REALES / "augmentado" / f"{nombre_aug}.txt", lineas)

    reparto = repartir_fuentes(fuentes)

    resumen_splits: dict[str, dict] = {}

    def copiar(imagen: Path, labels: list[str], split: str, nombre: str) -> bool:
        if not imagen.exists():
            faltantes_img.append(str(imagen))
            return False
        shutil.copyfile(imagen, destino / "images" / split / nombre)
        escribir_txt(destino / "labels" / split / (Path(nombre).stem + ".txt"), labels)
        return True

    # ---- train: todo lo generado, labels tal cual
    for origen_imgs, origen_lbl in (
        (SINTETICO / "images", SINTETICO / "labels"),
        (COMPUESTO / "images", COMPUESTO / "labels"),
    ):
        for etiqueta in sorted(origen_lbl.glob("*.txt")):
            imagen = origen_imgs / (etiqueta.stem + ".jpg")
            if not copiar(imagen, etiqueta.read_text(encoding="utf-8").splitlines(),
                          "train", imagen.name):
                continue

    # ---- val / test: solo reales, repartidos por imagen fuente
    # (el directorio se llama `val` por convencion de Ultralytics; el reparto
    # interno lo llama `valid`, como el CSV de origen)
    for split, clave in (("val", "valid"), ("test", "test")):
        for fuente in sorted(reparto[clave]):
            casillas = fuentes[fuente]
            lineas = [linea_yolo(clase, c, clases)
                      for c, clase in sorted(casillas.items()) if clase != CLASE_VACIA]
            base = f"real_{slug(fuente.replace('/', '_').rsplit('.', 1)[0])}.jpg"
            copiar(RECORTES / fuente, lineas, split, base)
            for archivo, _carpeta in aumentadas.get(fuente, []):
                nombre = f"real_{slug(archivo.replace('/', '_').rsplit('.', 1)[0])}.jpg"
                copiar(AUMENTADO / archivo, lineas, split, nombre)

    if faltantes_img:
        raise SystemExit(f"Faltan {len(faltantes_img)} imagenes: {faltantes_img[:5]}")

    # ---- data.yaml + pesos
    pesos = ruta_pesos()
    shutil.copyfile(pesos, destino / "weights" / "modelo_sintetico.pt")

    lineas_yaml = [
        "# Orden de clases = classes.txt (identico a modelo_sintetico.pt).",
        "# path ABSOLUTO a /content: Ultralytics resuelve 'path: .' contra su",
        "# datasets_dir global (/content/datasets), NO contra la carpeta del",
        "# archivo; fue el error que costo la v1. $CARPETA_COLAB se fija.",
        "path: /content/ajedrez_v2/detector",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(clases)}",
        "names:",
    ]
    lineas_yaml += [f"  {i}: {nombre}" for i, nombre in enumerate(clases)]
    (destino / "data.yaml").write_text("\n".join(lineas_yaml) + "\n", encoding="utf-8")

    # ---- conteos
    for split in SPLITS_DETECTOR:
        clave = "valid" if split == "val" else split
        n_img = len(list((destino / "images" / split).glob("*.jpg")))
        n_lbl, n_cajas, por_clase = contar_cajas(destino / "labels" / split)
        resumen_splits[split] = {
            "imagenes": n_img,
            "etiquetas": n_lbl,
            "cajas": n_cajas,
            "por_clase": dict(sorted(por_clase.items())),
            "fuentes": len(reparto[clave]) if clave in reparto else None,            "fuentes_detalle": sorted(reparto.get(clave, [])),
        }

    # ---- zip
    zip_destino = BUNDLE / "detector_v2.zip"
    if zip_destino.exists():
        zip_destino.unlink()
    with zipfile.ZipFile(zip_destino, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for ruta in sorted(destino.rglob("*")):
            if ruta.is_file():
                zip_archivo(zf, ruta, ruta.relative_to(destino).as_posix())

    resumen = {
        "clases": clases,
        "box_casilla": BOX_CASILLA,
        "semilla": semilla,
        "splits": resumen_splits,
        "fuentes_valid": sorted(reparto["valid"]),
        "fuentes_test": sorted(reparto["test"]),
        "zip": {
            "archivo": str(zip_destino.relative_to(RAIZ)),
            "mb": round(zip_destino.stat().st_size / 1024 / 1024, 1),
        },
        "segundos": round(time.time() - inicio, 1),
    }

    print("DETECTOR")
    for split in SPLITS_DETECTOR:
        d = resumen_splits[split]
        fuentes_txt = "-" if d["fuentes"] is None else str(d["fuentes"])
        print(f"  {split:6s} imagenes={d['imagenes']:5d} etiquetas={d['etiquetas']:5d} "
              f"cajas={d['cajas']:6d} fuentes={fuentes_txt}")
    print(f"  data.yaml: nc={len(clases)} names={clases}")
    print(f"  pesos: {pesos}")
    print(f"  zip: {zip_destino} ({resumen['zip']['mb']} MB)")
    return resumen


# --------------------------------------------------------------------------
# TAREA B - clasificador
# --------------------------------------------------------------------------
def inventario_clasificador(clases_casilla: list[str]) -> list[dict]:
    """Items (clase, archivo origen, arcname en el zip, grupo imagen)."""
    items: list[dict] = []

    for fila in leer_csv(ETIQUETAS_CASILLAS):
        clase = fila["clase"]
        if clase not in clases_casilla:
            raise SystemExit(f"Clase de casilla desconocida: {clase}")
        nombre = Path(fila["archivo"]).name
        items.append({
            "clase": clase,
            "origen": CASILLAS_COMPUESTO / clase / nombre,
            "arcname": f"casillas/{clase}/{nombre}",
            "grupo": fila["imagen"],
            "familia": "compuesto",
            "casilla": fila["casilla"],
        })

    for fila in leer_csv(ETIQUETAS_AUMENTADO):
        clase = fila["clase"]
        archivo = fila["archivo_recorte"]
        carpeta, _, nombre = archivo.partition("/")
        stem = nombre.rsplit(".", 1)[0]
        fuente = re.sub(r"_aug\d+$", "", stem)
        disco = f"{stem}_{fila['casilla']}.jpg"
        items.append({
            "clase": clase,
            "origen": CASILLAS_REALES / clase / disco,
            "arcname": f"casillas/{clase}/{disco}",
            # Misma clave que usa el detector (con extension) para compartir split.
            "grupo": f"{carpeta}/{fuente}.jpg",
            "familia": "real",
            "casilla": fila["casilla"],
        })

    return items


def preparar_clasificador(verificar_archivos: bool = False,
                          semilla: int = SEMILLA) -> dict:
    inicio = time.time()
    clases = cargar_clases()
    clases_casilla = clases + [CLASE_VACIA]

    items = inventario_clasificador(clases_casilla)
    arcnames = [i["arcname"] for i in items]
    if len(set(arcnames)) != len(arcnames):
        repetidos = [a for a, n in Counter(arcnames).items() if n > 1]
        raise SystemExit(f"Nombres de archivo repetidos al juntar las dos fuentes: {repetidos[:5]}")

    # Las variantes aumentadas de un recorte comparten split con su fuente: si
    # se repartieran por archivo, la misma foto real apareceria en train y en
    # test a la vez (leak identico al del detector).
    grupos_compuesto = sorted({i["grupo"] for i in items if i["familia"] == "compuesto"})
    grupos_real = sorted({i["grupo"] for i in items if i["familia"] == "real"})
    if len(grupos_real) != 20:
        raise SystemExit(f"Esperaba 20 recortes reales aumentados, hay {len(grupos_real)}")

    rng = random.Random(semilla)
    rng.shuffle(grupos_compuesto)
    n_comp = len(grupos_compuesto)
    n_valid = round(n_comp * 0.10)
    n_test = round(n_comp * 0.10)
    asignado = {}
    for grupo in grupos_compuesto[:n_valid]:
        asignado[grupo] = "valid"
    for grupo in grupos_compuesto[n_valid:n_valid + n_test]:
        asignado[grupo] = "test"
    for grupo in grupos_compuesto[n_valid + n_test:]:
        asignado[grupo] = "train"

    # Los reales siguen EXACTAMENTE el reparto del detector (12 valid / 8 test).
    # Asi el test del clasificador y el del detector son las mismas fotos y la
    # comparacion de T7 (YOLO solo vs clasificador solo vs combinado) es valida.
    detector = repartir_fuentes(fuentes_reales())
    for split, fuentes in detector.items():
        for fuente in fuentes:
            asignado[fuente] = split

    sin_asignar = {i["grupo"] for i in items} - set(asignado)
    if sin_asignar:
        raise SystemExit(f"Grupos sin split: {sorted(sin_asignar)[:5]}")

    por_split: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        por_split[asignado[item["grupo"]]].append(item)

    # ---- conteos y balance
    tabla = {split: Counter(i["clase"] for i in por_split[split])
             for split in SPLITS_CLASIFICADOR}
    faltan_train = [c for c in clases_casilla if tabla["train"][c] == 0]
    if faltan_train:
        raise SystemExit(f"Clases sin ejemplos en train: {faltan_train}")

    if verificar_archivos:
        faltan = [str(i["origen"]) for i in items if not i["origen"].exists()]
        if faltan:
            raise SystemExit(f"Faltan {len(faltan)} JPG de casilla: {faltan[:5]}")

    # ---- txt de split (rutas relativas a la raiz del zip)
    carpeta_txt = BUNDLE / "clasificador_split"
    if carpeta_txt.exists():
        shutil.rmtree(carpeta_txt)
    carpeta_txt.mkdir(parents=True, exist_ok=True)
    for split, sufijo in (("train", "train"), ("valid", "valid"), ("test", "test")):
        rutas = sorted(i["arcname"] for i in por_split[split])
        escribir_txt(carpeta_txt / f"{sufijo}.txt", rutas)

    # ---- zip: no se copian los 152k JPG, se leen de donde ya estan
    zip_destino = BUNDLE / "clasificador_v2.zip"
    if zip_destino.exists():
        zip_destino.unlink()
    with zipfile.ZipFile(zip_destino, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for item in sorted(items, key=lambda i: i["arcname"]):
            zip_archivo(zf, item["origen"], item["arcname"])
        for split in SPLITS_CLASIFICADOR:
            zip_archivo(zf, carpeta_txt / f"{split}.txt", f"split/{split}.txt")
        escribir_txt(BUNDLE / "clases_casilla.txt", clases_casilla)
        zip_archivo(zf, BUNDLE / "clases_casilla.txt", "labels/clases.txt")
        zip_archivo(zf, ETIQUETAS_CASILLAS, "labels/etiquetas_casillas.csv")
        zip_archivo(zf, ETIQUETAS_AUMENTADO, "labels/aumentado_real.csv")

    totales = {split: sum(tabla[split].values()) for split in SPLITS_CLASIFICADOR}
    imagenes = {split: len({i["grupo"] for i in por_split[split]})
                for split in SPLITS_CLASIFICADOR}

    resumen = {
        "clases": clases_casilla,
        "semilla": semilla,
        "imagenes": imagenes,
        "items": totales,
        "total": sum(totales.values()),
        "por_clase": {split: dict(tabla[split]) for split in SPLITS_CLASIFICADOR},
        "vacia_pct": {s: round(100 * tabla[s][CLASE_VACIA] / totales[s], 2)
                      for s in SPLITS_CLASIFICADOR},
        "balance": {},
        "zip": {
            "archivo": str(zip_destino.relative_to(RAIZ)),
            "mb": round(zip_destino.stat().st_size / 1024 / 1024, 1),
        },
        "segundos": round(time.time() - inicio, 1),
    }
    for split in SPLITS_CLASIFICADOR:
        conteo = tabla[split]
        vacia = conteo[CLASE_VACIA]
        damas = conteo["dama_b"] + conteo["dama_n"]
        ocupadas = totales[split] - vacia
        ocupadas_conteo = [conteo[c] for c in clases]
        resumen["balance"][split] = {
            "vacia_pct": round(100 * vacia / totales[split], 2),
            "dama_vacia": round(damas / vacia, 4),
            "ratio_max_min_ocupadas": round(max(ocupadas_conteo) / min(ocupadas_conteo), 2),
            "media_ocupadas": round(ocupadas / len(clases), 1),
        }

    print("CLASIFICADOR")
    print(f"  items totales={resumen['total']} imagenes_grupo={sum(imagenes.values())}")
    for split in SPLITS_CLASIFICADOR:
        print(f"  {split:6s} imagenes={imagenes[split]:5d} items={totales[split]:7d} "
              f"vacia={resumen['vacia_pct'][split]:6.2f}% "
              f"dama/vacia={resumen['balance'][split]['dama_vacia']:.3f} "
              f"max/min ocupadas={resumen['balance'][split]['ratio_max_min_ocupadas']}")
    print("  por clase (train/valid/test):")
    for clase in clases_casilla:
        print(f"    {clase:10s} {tabla['train'][clase]:6d} {tabla['valid'][clase]:6d} "
              f"{tabla['test'][clase]:6d}")
    print(f"  zip: {zip_destino} ({resumen['zip']['mb']} MB)")
    return resumen


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--todo", action="store_true",
                        help="prepara detector + clasificador y escribe resumen.json")
    parser.add_argument("--detector", action="store_true", help="solo el detector")
    parser.add_argument("--clasificador", action="store_true", help="solo el clasificador")
    parser.add_argument("--verificar-archivos", action="store_true",
                        help="comprueba que cada JPG de casilla existe (lento)")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()

    if not (args.todo or args.detector or args.clasificador):
        parser.print_help()
        return 2

    BUNDLE.mkdir(parents=True, exist_ok=True)
    resumen: dict = {}
    if args.todo or args.detector:
        resumen["detector"] = preparar_detector(args.semilla)
    if args.todo or args.clasificador:
        resumen["clasificador"] = preparar_clasificador(args.verificar_archivos, args.semilla)

    ruta_resumen = BUNDLE / "resumen.json"
    if args.todo:
        ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        print(f"\nResumen: {ruta_resumen}")
        print(f"clases: {CLASES_TXT} -> {resumen['detector']['clases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
