"""Empaqueta los dos modelos del dataset LIMPIO para entrenarlos en Colab.

Salidas dentro de ``dataset/colab_bundle/``:

  detector_v2_clean/
    images/{train,val,test}/    train = sintetico 1544 + compuesto_limpio 2500
                                + variantes aumentadas de 02/03/04 (train)
    labels/{train,val,test}/    val/test = recortes REALES (02/03/04 + parte de 01)
    data.yaml                   nc: 12 en el orden de classes.txt
  detector_v2_clean.zip
  clasificador_v2_clean/
    split/{train,valid,test}.txt   rutas RELATIVAS a la raiz del zip
    labels/clases.txt              13 clases (12 + vacia)
    MANIFEST.txt                   donde viven los JPG
  clasificador_v2_clean.zip
  entrega_v2.zip                  el paquete para el equipo de juego (visor + pesos)
  resumen.json

Regla de oro heredada de v1: **val y test SOLO fotos reales**. El train es todo
lo generado (sintetico de plantilla + compuestos con sprites limpios + las
variantes aumentadas). Medir el modelo sobre lo mismo con lo que entreno no mide
nada.

Ampliacion T10 (``dataset/augmentado_new``)
-------------------------------------------
``amplificar_declaradas.py`` genera 4 variantes de cada recorte declarado de
02/03/04 y las reparte por POSICION en train/valid/test. Este empaquetado lee
sus manifiestos (nunca lista carpetas a ciegas) y aplica:

* ``politica = exclusivo`` (por defecto): las variantes de una posicion de train
  entran al train del detector y las fotos CRUDAS de esa misma posicion NO
  entran a val/test. El test sigue siendo de posiciones nunca vistas.
* ``politica = replicar``: todas las fotos crudas se quedan en val/test y todas
  las variantes entran al train. El test deja de ser limpio (documentado).

El clasificador suma las casillas 128x128 de las variantes de train, con el
control de inversion (espejo) de las clases raras que ya trae el manifiesto. Las
variantes de valid/test NO se meten al clasificador: serian casi-duplicados de
la misma foto dentro del propio split de evaluacion y no aportan informacion.

De donde sale la verdad de las fotos reales
-------------------------------------------
De la FEN DECLARADA en el manifiesto de cada carpeta del crudo (ver
``rectificar_crudo_v2.py``). No se usa ningun modelo para etiquetar las fotos
reales: por eso este paquete no puede heredar el cruce caballo/alfil del lote
en cuarentena.

Reparto de los reales
---------------------
* ``02`` y ``03``: por POSICION declarada. Todas las fotos de una misma FEN
  caen en el mismo split: las variantes de luz de una posicion no se reparten.
* ``01`` y ``04``: tienen UNA sola posicion, asi que se reparte por IMAGEN
  (no hay fuga de posicion posible: la posicion es la misma en las dos).
* ``04`` (7 fotos) va completo a test: es la unica carpeta de finales y se
  prefiere que el test tenga ese dominio antes que repartirlo.

El clasificador se reparte por IMAGEN FUENTE (nunca por casilla): las 64
casillas de un tablero caen siempre en el mismo split. Los reales siguen
EXACTAMENTE el reparto del detector, asi el test de los dos modelos son las
mismas fotos (la comparacion de T7 se mide sobre lo mismo).

Uso
---
    python preparar_colab_v2_clean.py --todo
    python preparar_colab_v2_clean.py --detector
    python preparar_colab_v2_clean.py --clasificador
    python preparar_colab_v2_clean.py --entrega
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
from rectificar_crudo_v2 import CLASE_VACIA, ocupacion_de_fen

SINTETICO = DATASET / "sintetico"
COMPUESTO = DATASET / "compuesto_limpio"
RECORTES = DATASET / "recortes_800"
CASILLAS_REALES = DATASET / "casillas" / "clasificador"
CASILLAS_COMPUESTO = COMPUESTO / "casillas" / "clasificador"
AUMENTADO = DATASET / "augmentado_new"
CASILLAS_AUMENTADO = AUMENTADO / "casillas" / "clasificador"
BUNDLE = DATASET / "colab_bundle"

DETECTOR_DIR = BUNDLE / "detector_v2_clean"
DETECTOR_ZIP = BUNDLE / "detector_v2_clean.zip"
CLASIF_DIR = BUNDLE / "clasificador_v2_clean"
CLASIF_ZIP = BUNDLE / "clasificador_v2_clean.zip"
ENTREGA_ZIP = BUNDLE / "entrega_v2.zip"

SEMILLA = 42
SPLITS_DETECTOR = ("train", "val", "test")
SPLITS_CLASIFICADOR = ("train", "valid", "test")
CARPETAS_TEST = ("01_inicial_intacta", "02_inicial_con_jugadas",
                 "03_medio_juego", "04_finales", "05_tablero_vacio")
PROPORCION_VALID = 0.10
PROPORCION_TEST = 0.10

# Archivos del paquete de entrega al equipo de juego.
ENTREGA_SCRIPTS = (
    "comun.py", "esquinas_aruco.py", "visor_jugadas.py",
    "prueba_en_vivo_v2.py", "diagnostico_esp32.py", "captura_revision.py",
    "evaluar_en_vivo.py", "rectificar_crudo_v2.py",
)
ENTREGA_DOCS = ("ENTREGA.md", "d5-notas.md", "t4d-notas.md", "t9-notas.md",
                "t10-notas.md")


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def leer_csv(ruta: Path) -> list[dict[str, str]]:
    if not ruta.exists():
        raise SystemExit(f"No encuentro {ruta}")
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def slug(texto: str) -> str:
    limpio = re.sub(r"[^0-9A-Za-z]+", "_", texto).strip("_").lower()
    return limpio or "x"


def escribir_txt(ruta: Path, lineas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("".join(linea + "\n" for linea in lineas), encoding="utf-8")


def contar_cajas(carpeta: Path) -> tuple[int, int, Counter]:
    cajas: Counter = Counter()
    archivos = list(carpeta.glob("*.txt"))
    for txt in archivos:
        for linea in txt.read_text(encoding="utf-8").splitlines():
            partes = linea.split()
            if len(partes) == 5:
                cajas[int(partes[0])] += 1
    return len(archivos), sum(cajas.values()), cajas


def zip_archivo(zf: zipfile.ZipFile, origen: Path, arcname: str) -> None:
    """JPG/PNG y binarios ya vienen comprimidos: guardar sin re-comprimir.

    Los .exe y .pt se guardan tal cual: comprimirlos ahorra ~40 % pero rompe la
    firma/SHA de un binario firmado y hace la extraccion mas lenta.
    """
    compresion = (zipfile.ZIP_STORED
                  if origen.suffix.lower() in {".jpg", ".jpeg", ".png",
                                               ".exe", ".pt", ".pth"}
                  else zipfile.ZIP_DEFLATED)
    zf.write(origen, arcname, compress_type=compresion)


# --------------------------------------------------------------------------
# ampliacion T10 (variantes de las posiciones declaradas)
# --------------------------------------------------------------------------
def cargar_aumentado() -> dict | None:
    """Lee lo que produjo ``amplificar_declaradas.py`` (o None si no existe).

    Nunca lista las carpetas de casillas para armar el inventario: los datos
    salen de los manifiestos, que llevan la clase y el split explicitos. Es la
    misma regla que dejo T9 tras los archivos huerfanos de la primera corrida.
    """
    ruta = AUMENTADO / "manifiesto_ampliado.csv"
    if not ruta.exists():
        return None
    politica = "exclusivo"
    resumen = AUMENTADO / "resumen.json"
    if resumen.exists():
        politica = json.loads(resumen.read_text(encoding="utf-8")).get(
            "politica", "exclusivo")
    manifiesto = leer_csv(ruta)
    if not manifiesto:
        return None
    posiciones: dict[tuple[str, str], str] = {}
    ruta_splits = AUMENTADO / "splits_posiciones.csv"
    if ruta_splits.exists():
        for fila in leer_csv(ruta_splits):
            posiciones[(fila["carpeta"], fila["fen"])] = fila["split"]
    casillas = leer_csv(AUMENTADO / "casillas_manifesto.csv")
    return {
        "politica": politica,
        "manifiesto": manifiesto,
        "posiciones": posiciones,
        "casillas": casillas,
    }



# --------------------------------------------------------------------------
# reparto de las fotos reales
# --------------------------------------------------------------------------
def clave_real(carpeta: str, archivo: str) -> str:
    """Clave unica de un recorte: el nombre solo NO basta.

    ``capture (N).jpg`` existe en varias carpetas del crudo con el mismo
    nombre; indexar por nombre perderia fotos y cruzaria splits.
    """
    return f"{carpeta}/{archivo}"


def repartir_reales(aumentado: dict | None = None
                    ) -> tuple[dict[str, str], list[dict[str, str]]]:
    """{clave_real: 'val'|'test'} y las filas del manifiesto de recortes.

    Con la ampliacion T10 en politica ``exclusivo`` las fotos crudas de una
    posicion que quedo en train NO entran a val/test: esa posicion se representa
    en el train con sus 4 variantes aumentadas. Asi el test sigue midiendo
    posiciones que el modelo nunca vio, ni siquiera aumentadas.
    """
    filas = [f for f in leer_csv(RECORTES / "manifesto_recortes.csv")
             if f.get("archivo")
             and not (f.get("fallo") or "").strip()
             and (RECORTES / f["carpeta_origen"] / f["archivo"]).exists()]
    if not filas:
        raise SystemExit(
            "No hay recortes reales. Corre antes rectificar_crudo_v2.py"
        )

    por_carpeta: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for fila in filas:
        if fila["carpeta_origen"] not in CARPETAS_TEST:
            continue
        # La POSICION es la FEN declarada: todas las fotos de la misma FEN son
        # la misma posicion fisica (variaciones de luz / angulo).
        por_carpeta[fila["carpeta_origen"]][fila["fen"]].append(fila["archivo"])

    rng = random.Random(SEMILLA)
    asignado: dict[str, str] = {}
    for carpeta in sorted(por_carpeta):
        posiciones = sorted(por_carpeta[carpeta])
        if carpeta == "04_finales":
            for posicion in posiciones:
                for archivo in sorted(por_carpeta[carpeta][posicion]):
                    asignado[clave_real(carpeta, archivo)] = "test"
            continue
        if len(posiciones) == 1:
            imagenes = sorted(por_carpeta[carpeta][posiciones[0]])
            rng.shuffle(imagenes)
            mitad = len(imagenes) // 2
            for archivo in imagenes[:mitad]:
                asignado[clave_real(carpeta, archivo)] = "val"
            for archivo in imagenes[mitad:]:
                asignado[clave_real(carpeta, archivo)] = "test"
            continue
        rng.shuffle(posiciones)
        for indice, posicion in enumerate(posiciones):
            destino = "val" if indice % 2 == 0 else "test"
            for archivo in sorted(por_carpeta[carpeta][posicion]):
                asignado[clave_real(carpeta, archivo)] = destino

    # La posicion declarada que quedo en train se reserva para el aumento: sus
    # fotos crudas no se reparten entre val/test (ver docstring).
    if aumentado and aumentado["politica"] == "exclusivo":
        en_train = {clave for clave, split in aumentado["posiciones"].items()
                    if split == "train"}
        excluidas = {clave_real(f["carpeta_origen"], f["archivo"])
                     for f in filas
                     if (f["carpeta_origen"], f["fen"]) in en_train}
        if excluidas:
            print(f"aumentado (exclusivo): {len(excluidas)} recortes crudos de "
                  f"posiciones de train quedan fuera de val/test")
            filas = [f for f in filas
                     if clave_real(f["carpeta_origen"], f["archivo"])
                     not in excluidas]
            asignado = {clave: split for clave, split in asignado.items()
                        if clave not in excluidas}

    sin_split = [clave_real(f["carpeta_origen"], f["archivo"]) for f in filas
                 if clave_real(f["carpeta_origen"], f["archivo"]) not in asignado]
    if sin_split:
        raise SystemExit(f"{len(sin_split)} recortes sin split (carpeta desconocida): "
                         f"{sin_split[:3]}")
    return asignado, filas


def nombre_real(carpeta: str, archivo: str) -> str:
    return f"real_{slug(carpeta)}_{slug(Path(archivo).stem)}"


def nombre_aug(carpeta: str, archivo: str) -> str:
    return f"aug_{slug(carpeta)}_{slug(Path(archivo).stem)}"


# --------------------------------------------------------------------------
# TAREA A - detector
# --------------------------------------------------------------------------
def preparar_detector(aumentado: dict | None = None,
                      asignado: dict[str, str] | None = None,
                      filas: list[dict[str, str]] | None = None) -> dict:
    inicio = time.time()
    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    if asignado is None or filas is None:
        asignado, filas = repartir_reales(aumentado)
    destino = DETECTOR_DIR
    if destino.exists():
        shutil.rmtree(destino)
    for split in SPLITS_DETECTOR:
        (destino / "images" / split).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / split).mkdir(parents=True, exist_ok=True)

    faltantes: list[str] = []

    def copiar(imagen: Path, etiqueta: Path | None, label_texto: str | None,
               split: str, nombre: str) -> None:
        if not imagen.exists():
            faltantes.append(str(imagen))
            return
        shutil.copyfile(imagen, destino / "images" / split / nombre)
        if label_texto is None:
            if etiqueta is None or not etiqueta.exists():
                faltantes.append(str(etiqueta))
                return
            label_texto = etiqueta.read_text(encoding="utf-8")
        (destino / "labels" / split / (Path(nombre).stem + ".txt")).write_text(
            label_texto, encoding="utf-8")

    # ---- train: todo lo generado
    for imagenes_dir, labels_dir in ((SINTETICO / "images", SINTETICO / "labels"),
                                     (COMPUESTO / "images", COMPUESTO / "labels")):
        if not imagenes_dir.is_dir():
            raise SystemExit(f"No encuentro {imagenes_dir}")
        for etiqueta in sorted(labels_dir.glob("*.txt")):
            imagen = imagenes_dir / (etiqueta.stem + ".jpg")
            copiar(imagen, etiqueta, None, "train", imagen.name)

    # ---- train: variantes aumentadas de las posiciones declaradas de train
    if aumentado:
        agregadas = 0
        for fila in aumentado["manifiesto"]:
            if fila["split"] != "train":
                continue
            imagen = AUMENTADO / "images" / fila["imagen"]
            etiqueta = AUMENTADO / "labels" / Path(fila["imagen"]).with_suffix(".txt")
            nombre = (nombre_aug(fila["carpeta"], fila["imagen"]) + ".jpg")
            copiar(imagen, etiqueta, None, "train", nombre)
            agregadas += 1
        print(f"  aumentado: {agregadas} imagenes de train "
              f"(politica {aumentado['politica']})")

    # ---- val / test: recortes reales
    for fila in filas:
        archivo = fila["archivo"]
        carpeta = fila["carpeta_origen"]
        split = asignado[clave_real(carpeta, archivo)]
        nombre = nombre_real(carpeta, archivo) + ".jpg"
        copiar(RECORTES / carpeta / archivo,
               RECORTES / carpeta / "labels" / (Path(archivo).stem + ".txt"),
               None, split, nombre)

    if faltantes:
        raise SystemExit(f"Faltan {len(faltantes)} archivos: {faltantes[:5]}")

    # ---- data.yaml
    lineas_yaml = [
        "# Orden de clases = classes.txt (identico al dataset limpio).",
        "# path se reescribe en la celda 3b del notebook con la carpeta de",
        "# descompresion: Ultralytics resuelve 'path: .' contra su datasets_dir.",
        "path: /content/ajedrez_v2/detector",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(clases)}",
        "names:",
    ]
    lineas_yaml += [f"  {i}: {nombre}" for i, nombre in enumerate(clases)]
    (destino / "data.yaml").write_text("\n".join(lineas_yaml) + "\n",
                                       encoding="utf-8")

    # ---- conteos
    resumen_splits: dict[str, dict] = {}
    for split in SPLITS_DETECTOR:
        n_img = len(list((destino / "images" / split).glob("*.jpg")))
        n_lbl, n_cajas, por_clase = contar_cajas(destino / "labels" / split)
        fuentes = sorted({nombre_real(f["carpeta_origen"], f["archivo"])
                          for f in filas
                          if asignado[clave_real(f["carpeta_origen"],
                                                 f["archivo"])] == split})
        por_source: Counter = Counter()
        for imagen in (destino / "images" / split).glob("*.jpg"):
            if imagen.name.startswith("sintetico_"):
                por_source["sintetico"] += 1
            elif imagen.name.startswith("compuesto_"):
                por_source["compuesto"] += 1
            elif imagen.name.startswith("aug_"):
                por_source["aumentado"] += 1
            else:
                por_source["real"] += 1
        resumen_splits[split] = {
            "imagenes": n_img,
            "etiquetas": n_lbl,
            "cajas": n_cajas,
            "por_clase": dict(sorted(por_clase.items())),
            "por_source": dict(sorted(por_source.items())),
            "fuentes_reales": len(fuentes),
            "fuentes_detalle": fuentes,
        }

    # ---- zip
    if DETECTOR_ZIP.exists():
        DETECTOR_ZIP.unlink()
    with zipfile.ZipFile(DETECTOR_ZIP, "w", zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for ruta in sorted(destino.rglob("*")):
            if ruta.is_file():
                zip_archivo(zf, ruta, ruta.relative_to(destino).as_posix())

    mb = round(DETECTOR_ZIP.stat().st_size / 1024 / 1024, 1)
    print("DETECTOR")
    for split in SPLITS_DETECTOR:
        d = resumen_splits[split]
        print(f"  {split:6s} imagenes={d['imagenes']:5d} etiquetas={d['etiquetas']:5d} "
              f"cajas={d['cajas']:6d} fuentes_reales={d['fuentes_reales']}")
        print(f"         por source: {d['por_source']}")
    print(f"  data.yaml: nc={len(clases)} names={clases}")
    print(f"  zip: {DETECTOR_ZIP} ({mb} MB)")
    return {
        "clases": clases,
        "splits": resumen_splits,
        "zip": {"archivo": str(DETECTOR_ZIP.relative_to(RAIZ)), "mb": mb},
        "segundos": round(time.time() - inicio, 1),
    }


# --------------------------------------------------------------------------
# TAREA B - clasificador
# --------------------------------------------------------------------------
def inventario_casillas(compuesto: list[str], reales: list[str],
                        aumentado: dict | None = None) -> list[dict]:
    """Items (clase, origen, arcname, grupo, familia) de las tres fuentes."""
    items: list[dict] = []
    for fila in leer_csv(COMPUESTO / "etiquetas_casillas.csv"):
        imagen = fila["imagen"]
        if imagen not in compuesto:
            continue
        nombre = f"{Path(imagen).stem}_{fila['casilla']}.jpg"
        items.append({
            "clase": fila["clase"],
            "origen": CASILLAS_COMPUESTO / fila["clase"] / nombre,
            "arcname": f"casillas/{fila['clase']}/{nombre}",
            "grupo": imagen,
            "familia": "compuesto",
        })

    for fila in leer_csv(RECORTES / "manifesto_recortes.csv"):
        archivo = fila.get("archivo") or ""
        clave = clave_real(fila.get("carpeta_origen") or "", archivo)
        if clave not in reales:
            continue
        try:
            ocupacion = ocupacion_de_fen(fila["fen"])
        except ValueError:
            continue
        stem = Path(archivo).stem
        carpeta_origen = fila.get("carpeta_origen") or ""
        for casilla, clase in ocupacion.items():
            nombre = f"{carpeta_origen}_{stem}_{casilla}.jpg"
            items.append({
                "clase": clase,
                "origen": CASILLAS_REALES / clase / nombre,
                "arcname": f"casillas/{clase}/{nombre}",
                "grupo": clave,
                "familia": "real",
            })

    # T10: casillas 128x128 de las variantes aumentadas. Solo las de TRAIN: las
    # de valid/test serian casi-duplicados de la misma foto dentro de su propio
    # split de evaluacion (no aportan y sesgan la validacion).
    if aumentado:
        for fila in aumentado["casillas"]:
            if fila["split"] != "train":
                continue
            clase = fila["clase"]
            nombre = (f"{Path(fila['imagen']).stem}_{fila['casilla']}"
                      f"{'_espejo' if fila['tipo'] == 'espejo' else ''}.jpg")
            items.append({
                "clase": clase,
                "origen": CASILLAS_AUMENTADO / clase / nombre,
                "arcname": f"casillas/{clase}/{nombre}",
                "grupo": f"aug:{fila['imagen']}",
                "familia": "aumentado",
                "split": "train",
            })
    return items


def preparar_clasificador(asignado_reales: dict[str, str] | None = None,
                           aumentado: dict | None = None) -> dict:
    inicio = time.time()
    clases = cargar_clases()
    clases_casilla = clases + [CLASE_VACIA]

    if asignado_reales is None:
        asignado_reales, _filas = repartir_reales(aumentado)

    compuesto = sorted({f["imagen"] for f in leer_csv(COMPUESTO / "etiquetas_casillas.csv")})
    reales = sorted(asignado_reales)
    total_imagenes = len(compuesto) + len(reales)
    objetivo_valid = int(round(PROPORCION_VALID * total_imagenes))
    objetivo_test = int(round(PROPORCION_TEST * total_imagenes))

    real_valid = [a for a in reales if asignado_reales[a] == "val"]
    real_test = [a for a in reales if asignado_reales[a] == "test"]
    n_valid = max(0, objetivo_valid - len(real_valid))
    n_test = max(0, objetivo_test - len(real_test))

    rng = random.Random(SEMILLA)
    rng.shuffle(compuesto)
    compuesto_valid = compuesto[:n_valid]
    compuesto_test = compuesto[n_valid:n_valid + n_test]
    compuesto_train = compuesto[n_valid + n_test:]

    asignado: dict[str, str] = {}
    for nombre in compuesto_train:
        asignado[nombre] = "train"
    for nombre in compuesto_valid:
        asignado[nombre] = "valid"
    for nombre in compuesto_test:
        asignado[nombre] = "test"
    for clave in real_valid:
        asignado[clave] = "valid"
    for clave in real_test:
        asignado[clave] = "test"

    items = inventario_casillas(set(compuesto), set(reales), aumentado)
    for item in items:
        if "split" not in item:
            item["split"] = asignado[item["grupo"]]

    faltan = [str(i["origen"]) for i in items if not i["origen"].exists()]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} JPG de casilla: {faltan[:5]}")

    por_split: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        por_split[item["split"]].append(item)

    tabla = {split: Counter(i["clase"] for i in por_split[split])
             for split in SPLITS_CLASIFICADOR}
    for split in SPLITS_CLASIFICADOR:
        faltan_clases = [c for c in clases_casilla if tabla[split][c] == 0]
        if faltan_clases:
            raise SystemExit(f"{split}: clases sin ejemplos {faltan_clases}")

    # ---- carpeta de splits (rutas relativas a la raiz del zip)
    if CLASIF_DIR.exists():
        shutil.rmtree(CLASIF_DIR)
    (CLASIF_DIR / "labels").mkdir(parents=True, exist_ok=True)
    for split in SPLITS_CLASIFICADOR:
        escribir_txt(CLASIF_DIR / "split" / f"{split}.txt",
                     sorted(i["arcname"] for i in por_split[split]))
    escribir_txt(CLASIF_DIR / "labels" / "clases.txt", clases_casilla)
    por_familia = Counter(i["familia"] for i in items)
    tamano = sorted({128 if i["familia"] == "aumentado" else 100 for i in items})
    (CLASIF_DIR / "MANIFEST.txt").write_text(
        "Clasificador limpio v2 (T10: con variantes aumentadas).\n\n"
        "split/{train,valid,test}.txt  rutas relativas a la raiz de "
        "clasificador_v2_clean.zip\n"
        f"casillas/<clase>/<archivo>.jpg  {len(items)} recortes (13 clases), "
        f"lado {tamano} px\n"
        "  * compuesto_limpio: dataset/compuesto_limpio/casillas/clasificador "
        f"({por_familia['compuesto']} de 100x100)\n"
        "  * reales:           dataset/casillas/clasificador "
        f"({por_familia['real']} de 100x100)\n"
        "  * aumentadas T10:   dataset/augmentado_new/casillas/clasificador "
        f"({por_familia['aumentado']} de 128x128, solo train)\n"
        "labels/clases.txt  13 clases en orden (12 de classes.txt + vacia)\n\n"
        "El reparto es POR IMAGEN FUENTE: las 64 casillas de un tablero caen\n"
        "siempre en el mismo split. Los recortes reales siguen el reparto del\n"
        "detector, asi el test de los dos modelos son las mismas fotos.\n\n"
        "Ampliacion T10: las variantes aumentadas de las posiciones declaradas\n"
        "de 02/03/04 entran SOLO al train (4 variantes por recorte, rotacion\n"
        "+-4 grados y fotometria de aumentar_reales.py). Las clases raras\n"
        "(<25 % de la mas frecuente: dama_b/dama_n/rey_b/rey_n) llevan ademas\n"
        "el control de inversion (espejo horizontal, sufijo _espejo). Las\n"
        "variantes de valid/test NO se incluyen: serian casi-duplicados de la\n"
        "misma foto dentro de su propio split de evaluacion.\n",
        encoding="utf-8",
    )

    # ---- zip
    if CLASIF_ZIP.exists():
        CLASIF_ZIP.unlink()
    with zipfile.ZipFile(CLASIF_ZIP, "w", zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for item in sorted(items, key=lambda i: i["arcname"]):
            zip_archivo(zf, item["origen"], item["arcname"])
        for split in SPLITS_CLASIFICADOR:
            zip_archivo(zf, CLASIF_DIR / "split" / f"{split}.txt",
                        f"split/{split}.txt")
        zip_archivo(zf, CLASIF_DIR / "labels" / "clases.txt", "labels/clases.txt")
        zip_archivo(zf, CLASIF_DIR / "MANIFEST.txt", "MANIFEST.txt")

    totales = {split: sum(tabla[split].values()) for split in SPLITS_CLASIFICADOR}
    imagenes = {split: len({i["grupo"] for i in por_split[split]})
                for split in SPLITS_CLASIFICADOR}
    familias = {split: dict(sorted(Counter(
        i["familia"] for i in por_split[split]).items()))
        for split in SPLITS_CLASIFICADOR}
    mb = round(CLASIF_ZIP.stat().st_size / 1024 / 1024, 1)
    print("CLASIFICADOR")
    print(f"  items={sum(totales.values())} imagenes={sum(imagenes.values())} "
          f"({dict(sorted(por_familia.items()))})")
    for split in SPLITS_CLASIFICADOR:
        vacia = tabla[split][CLASE_VACIA]
        print(f"  {split:6s} imagenes={imagenes[split]:5d} items={totales[split]:7d} "
              f"vacia={100 * vacia / totales[split]:5.2f}%")
        print(f"         por source: {familias[split]}")
    print("  por clase (train/valid/test):")
    for clase in clases_casilla:
        print(f"    {clase:10s} {tabla['train'][clase]:6d} "
              f"{tabla['valid'][clase]:6d} {tabla['test'][clase]:6d}")
    print(f"  zip: {CLASIF_ZIP} ({mb} MB)")
    return {
        "clases": clases_casilla,
        "imagenes": imagenes,
        "items": totales,
        "total": sum(totales.values()),
        "por_clase": {s: dict(tabla[s]) for s in SPLITS_CLASIFICADOR},
        "por_familia": familias,
        "vacia_pct": {s: round(100 * tabla[s][CLASE_VACIA] / totales[s], 2)
                      for s in SPLITS_CLASIFICADOR},
        "zip": {"archivo": str(CLASIF_ZIP.relative_to(RAIZ)), "mb": mb},
        "segundos": round(time.time() - inicio, 1),
    }


# --------------------------------------------------------------------------
# TAREA C - paquete de entrega (juego en tiempo real)
# --------------------------------------------------------------------------
def preparar_entrega() -> dict:
    """Rehace ``entrega_v2.zip``: codigo + visor + Stockfish + muestras limpias.

    NO lleva el dataset de entrenamiento (va en los otros dos zips): aqui solo
    lo que el equipo de juego necesita para correr el visor y el conector, mas
    una muestra de los casos limpios para que vean de que data salen los pesos.
    """
    inicio = time.time()
    if ENTREGA_ZIP.exists():
        ENTREGA_ZIP.unlink()

    muestras: list[tuple[Path, str]] = []
    vistas: set[str] = set()
    for carpeta in CARPETAS_TEST:
        archivos = sorted((RECORTES / carpeta).glob("*.jpg"))[:2]
        for archivo in archivos:
            if archivo.name in vistas:
                continue
            vistas.add(archivo.name)
            muestras.append((archivo, f"muestras_limpias/recortes/{archivo.name}"))
    for ruta in sorted((COMPUESTO / "previews").glob("*.jpg"))[:4]:
        muestras.append((ruta, f"muestras_limpias/previews/{ruta.name}"))
    if (DATASET / "sprites_limpio_preview.jpg").exists():
        muestras.append((DATASET / "sprites_limpio_preview.jpg",
                         "muestras_limpias/sprites_limpio_preview.jpg"))
    for ruta in sorted((COMPUESTO / "images").glob("*.jpg"))[:2]:
        muestras.append((ruta, f"muestras_limpias/compuestos/{ruta.name}"))
    for archivo in sorted((RECORTES / "muestras").glob("*.jpg"))[:2]:
        muestras.append((archivo, f"muestras_limpias/anotadas/{archivo.name}"))

    with zipfile.ZipFile(ENTREGA_ZIP, "w", zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        # codigo y pesos
        for carpeta in ("src", "plantilla"):
            for ruta in sorted((RAIZ / carpeta).rglob("*")):
                if ruta.is_file() and "__pycache__" not in ruta.parts:
                    zip_archivo(zf, ruta, ruta.relative_to(RAIZ).as_posix())
        for nombre in ENTREGA_SCRIPTS:
            ruta = RAIZ / "scripts" / nombre
            if ruta.exists():
                zip_archivo(zf, ruta, f"scripts/{nombre}")
        for nombre in ("classes.txt", "README.md", "requirements.txt"):
            ruta = RAIZ / nombre
            if ruta.exists():
                zip_archivo(zf, ruta, nombre)
        for ruta in sorted(RAIZ.glob("visor_*.bat")):
            zip_archivo(zf, ruta, ruta.name)
        for ruta in sorted((RAIZ / "colab").glob("*")):
            if ruta.is_file() and ruta.suffix in {".ipynb", ".md", ".py"}:
                zip_archivo(zf, ruta, f"colab/{ruta.name}")
        for nombre in ENTREGA_DOCS:
            ruta = RAIZ / "docs" / nombre
            if ruta.exists():
                zip_archivo(zf, ruta, f"docs/{nombre}")
        modelo = RAIZ / "modelos" / "modelo_sintetico.pt"
        if modelo.exists():
            zip_archivo(zf, modelo, "modelos/modelo_sintetico.pt")
        stockfish = RAIZ / "tools" / "stockfish" / "stockfish.exe"
        if stockfish.exists():
            zip_archivo(zf, stockfish, "tools/stockfish/stockfish.exe")
        for ruta, arcname in muestras:
            zip_archivo(zf, ruta, arcname)
        (BUNDLE / "_entrega_leeme.txt").write_text(
            "Paquete de ENTREGA ajedrez-v2 (dataset limpio).\n"
            "Contenido: src/, scripts/ (con visor_jugadas.py), modelos/,\n"
            "plantilla/, classes.txt, colab/ (notebooks), docs/, y\n"
            "tools/stockfish/stockfish.exe. muestras_limpias/ son fotos del\n"
            "dataset nuevo (recortes rectificados, compuestos y sprites).\n"
            "modelos/modelo_sintetico.pt es SOLO un respaldo historico del\n"
            "detector sintetico de v1: el reentreno limpio arranca de\n"
            "yolo11n.pt (COCO) con el notebook ..._desde_cero.\n"
            "Los pesos entrenados NO van aqui: salen de correr los notebooks\n"
            "de colab/ con detector_v2_clean.zip y clasificador_v2_clean.zip.\n",
            encoding="utf-8",
        )
        zip_archivo(zf, BUNDLE / "_entrega_leeme.txt", "LEEME.txt")

    mb = round(ENTREGA_ZIP.stat().st_size / 1024 / 1024, 1)
    print("ENTREGA")
    print(f"  archivos: {len(muestras)} muestras + codigo/pesos")
    print(f"  zip: {ENTREGA_ZIP} ({mb} MB)")
    return {"zip": {"archivo": str(ENTREGA_ZIP.relative_to(RAIZ)), "mb": mb},
            "muestras": len(muestras),
            "segundos": round(time.time() - inicio, 1)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--todo", action="store_true",
                        help="detector + clasificador + resumen.json")
    parser.add_argument("--detector", action="store_true")
    parser.add_argument("--clasificador", action="store_true")
    parser.add_argument("--entrega", action="store_true")
    args = parser.parse_args(argv)

    if not (args.todo or args.detector or args.clasificador or args.entrega):
        parser.print_help()
        return 2

    BUNDLE.mkdir(parents=True, exist_ok=True)
    aumentado = cargar_aumentado()
    if aumentado:
        print(f"AUMENTADO: {len(aumentado['manifiesto'])} variantes, "
              f"politica={aumentado['politica']}, "
              f"{len(aumentado['casillas'])} casillas en el manifiesto")
    else:
        print("AUMENTADO: no hay dataset/augmentado_new "
              "(corre amplificar_declaradas.py para incluirlo)")
    asignado, filas = repartir_reales(aumentado)
    resumen: dict = {}
    if aumentado:
        resumen["aumentado"] = {
            "politica": aumentado["politica"],
            "variantes": len(aumentado["manifiesto"]),
            "casillas": len(aumentado["casillas"]),
            "posiciones": dict(sorted(
                Counter(aumentado["posiciones"].values()).items())),
        }
    if args.todo or args.detector:
        resumen["detector"] = preparar_detector(aumentado, asignado, filas)
    if args.todo or args.clasificador:
        resumen["clasificador"] = preparar_clasificador(asignado, aumentado)
    if args.todo or args.entrega:
        resumen["entrega"] = preparar_entrega()

    if args.todo:
        ruta = BUNDLE / "resumen.json"
        ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\nResumen: {ruta}")
        print(f"clases: {CLASES_TXT} -> {resumen['detector']['clases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
