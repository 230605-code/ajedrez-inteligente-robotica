"""Amplifica las posiciones DECLARADAS (02/03/04) con variantes aumentadas.

Por que existe
--------------
Las fotos del ESP de 02/03/04 traen su FEN declarada por el usuario (una sola
verdad por posicion, sin modelos). Son 188 recortes 800x800 = 16 posiciones.
Amplificarlas multiplica el train real sin inventar etiquetas: la FEN viaja con
la geometria y cada una de las 64 casillas sale con la clase que dice la FEN.

De donde sale cada cosa
-----------------------
* Entrada: ``dataset/recortes_800/manifesto_recortes.csv`` (FEN declarada) y
  ``dataset/recortes_800/<carpeta>/<archivo>.jpg`` (rectificados 800x800).
* Aumento: el MISMO estilo de ``aumentar_reales.py`` (T4b), reusando sus
  funciones. Rotacion pequena +-4 grados (aqui, por pedido) y jitter de
  perspectiva de hasta 2 % del borde, compuestos en UNA homografia ``H``.
  La imagen completa se deforma con ``H`` y las 64 casillas se recortan con
  ``inv(H)``: la etiqueta no puede desalinearse porque la geometria viaja con
  ella. Fotometria: contraste 0.75..1.35, brillo +-25, gamma 0.7..1.4,
  saturacion 0.8..1.2, ruido gaussiano sigma 3..12 y desenfoque sigma 0..1.5.
* Semilla estable por (imagen, variante) con crc32: dos corridas producen los
  mismos archivos, asi el paquete de Colab es reproducible.

Que produce
-----------
``dataset/augmentado_new/images/<carpeta>/<archivo>_augK.jpg``   800x800
``dataset/augmentado_new/labels/<carpeta>/<archivo>_augK.txt``   YOLO, 12 clases
``dataset/augmentado_new/casillas/clasificador/<clase>/<...>.jpg``
    128x128 (el lado de entrada del clasificador), las 13 clases con ``vacia``
``dataset/augmentado_new/manifiesto_ampliado.csv``
    imagen, carpeta, posicion, fen, variante, split, n_fichas, semilla
``dataset/augmentado_new/casillas_manifesto.csv``
    imagen, casilla, clase, split, tipo (normal | espejo)
``dataset/augmentado_new/splits_posiciones.csv``
    carpeta, posicion, fen, split  (la particion por POSICION, no por foto)
``dataset/augmentado_new/conteo_por_clase.csv``
    clase, total, train, valid, test, espejo
``dataset/augmentado_new/resumen.json``   politica, numeros, rareza
``dataset/augmentado_new/previews/``      hojas de contacto

Reparto (por POSICION declarada, nunca por foto)
------------------------------------------------
Una posicion es una FEN: todas sus fotos (luz/angulo) y todas sus variantes
caen en el MISMO split. 50 % train / 25 % valid / 25 % test por carpeta con
minimo 1 valid y 1 test; ``04_finales`` (una sola posicion) queda en test.

Politicas (``--politica-split``)
--------------------------------
* ``exclusivo`` (por defecto, sin fuga): las variantes de una posicion de train
  van al train del detector, y las fotos CRUDAS de esa misma posicion NO entran
  a val/test. Asi el test sigue siendo de posiciones que el modelo nunca vio,
  ni siquiera aumentadas.
* ``replicar``: todas las fotos crudas se quedan en val/test (como hoy) y todas
  las variantes entran al train. Documentado por si el usuario prefiere el
  volumen, pero EL TEST DEJA DE SER LIMPIO: el modelo entrena con copias
  aumentadas de las mismas fotos que evalua.

Controles de inversion (clases raras)
-------------------------------------
Las clases cuyo conteo en train queda por debajo del 25 % de la clase mas
frecuente (aqui dama_b/dama_n/rey_b/rey_n: 1 por tablero) reciben ademas un
control de INVERSION horizontal: la MISMA casilla espejada (``_espejo.jpg``).
Es una etiqueta-preservante (un rey espejado sigue siendo rey) que equilibra el
desbalance sin inventar posiciones en el tablero. Solo se escribe para train.

Uso
---
    python amplificar_declaradas.py
    python amplificar_declaradas.py --verificar        (solo comprueba el disco)
    python amplificar_declaradas.py --limite 3 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from aumentar_reales import (
    LADO,
    _semilla,
    fotometria,
    hoja_contacto,
    recortar_casilla,
)
from comun import CRUDO, DATASET, cargar_clases
from rectificar_crudo_v2 import (
    CLASE_VACIA,
    fila_columna_a_casilla,
    linea_yolo,
    ocupacion_de_fen,
)

RECORTES = DATASET / "recortes_800"
SALIDA = DATASET / "augmentado_new"
CASILLAS = SALIDA / "casillas" / "clasificador"

CARPETAS_DECLARADAS = (
    "02_inicial_con_jugadas",
    "03_medio_juego",
    "04_finales",
)

VARIANTES = 4               # pedido: 4 variantes por recorte declarado
ANGULO = 4.0                # grados (+-), pedido: rotacion pequena
JITTER = 0.02               # fraccion del borde (mismo valor que T4b)
SEMILLA = 20261107
TAMANO_CASILLA = 128        # lado de entrada del clasificador (colab: TAMANO=128)
CALIDAD_IMAGEN = 92
CALIDAD_CASILLA = 88
PROPORCION_TRAIN = 0.50
PROPORCION_VALID = 0.25
CONTROL_RARA = 0.25         # < 25 % de la clase mas frecuente => rara

CAMPOS_MANIFIESTO = ["imagen", "carpeta", "posicion", "fen", "variante",
                     "split", "n_fichas", "semilla"]
CAMPOS_CASILLAS = ["imagen", "casilla", "clase", "split", "tipo"]
CAMPOS_SPLITS = ["carpeta", "posicion", "fen", "split"]
CAMPOS_CONTEO = ["clase", "total", "train", "valid", "test", "espejo"]


# ---------------------------------------------------------------------------
# Lectura de la verdad declarada
# ---------------------------------------------------------------------------
def leer_posiciones_crudo(carpeta: str) -> dict[str, str]:
    """{archivo: posicion} del manifiesto de la carpeta del crudo.

    El manifiesto de ``dataset/recortes_800`` no trae la columna ``posicion``;
    se lee la del crudo (que si la declara) y si falta se usa un indice estable
    por orden de FEN. La identidad real es la FEN: la posicion solo la rotula.
    """
    ruta = CRUDO / carpeta / "manifesto.csv"
    if not ruta.exists():
        return {}
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        return {
            (fila.get("archivo") or "").strip(): (fila.get("posicion") or "").strip()
            for fila in csv.DictReader(archivo)
            if (fila.get("archivo") or "").strip()
        }


def leer_declaradas() -> list[dict]:
    """Recortes 800x800 de 02/03/04 con FEN declarada y 64 casillas."""
    ruta = RECORTES / "manifesto_recortes.csv"
    if not ruta.exists():
        raise SystemExit(
            f"No encuentro {ruta}. Corre antes rectificar_crudo_v2.py"
        )
    ocupaciones: dict[str, dict[str, str]] = {}
    posiciones_crudas = {carpeta: leer_posiciones_crudo(carpeta)
                         for carpeta in CARPETAS_DECLARADAS}
    indice_posicion: dict[tuple[str, str], str] = {}
    filas: list[dict] = []
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            carpeta = fila.get("carpeta_origen") or ""
            if carpeta not in CARPETAS_DECLARADAS:
                continue
            if (fila.get("fallo") or "").strip():
                continue
            if int(fila.get("casilla_ok") or 0) != 64:
                continue
            fen = (fila.get("fen") or "").strip()
            if not fen or fen == "(SIN DECLARAR)":
                continue
            if fen not in ocupaciones:
                try:
                    ocupaciones[fen] = ocupacion_de_fen(fen)
                except ValueError as error:
                    print(f"  AVISO: FEN invalida en {fila.get('archivo')}: {error}")
                    continue
            archivo_foto = fila.get("archivo") or ""
            posicion = posiciones_crudas[carpeta].get(archivo_foto) or ""
            if not posicion:
                clave = (carpeta, fen)
                if clave not in indice_posicion:
                    indice_posicion[clave] = str(len(
                        [c for c in indice_posicion if c[0] == carpeta]) + 1)
                posicion = indice_posicion[clave]
            imagen = RECORTES / carpeta / archivo_foto
            if not imagen.exists():
                print(f"  AVISO: falta el recorte {imagen}")
                continue
            filas.append({
                "carpeta": carpeta,
                "archivo": archivo_foto,
                "stem": Path(archivo_foto).stem,
                "fen": fen,
                "posicion": posicion,
                "ocupacion": ocupaciones[fen],
                "n_fichas": sum(1 for c in ocupaciones[fen].values()
                                if c != CLASE_VACIA),
            })
    if not filas:
        raise SystemExit(
            f"No hay recortes declarados en {CARPETAS_DECLARADAS} ({ruta})"
        )
    return filas


def repartir_posiciones(filas: list[dict]) -> dict[tuple[str, str], str]:
    """{(carpeta, fen): split} por POSICION, estratificado por carpeta.

    Una posicion es una FEN; sus fotos y sus variantes nunca cruzan de split.
    """
    posiciones: dict[str, list[str]] = defaultdict(list)
    for fila in filas:
        clave = (fila["carpeta"], fila["fen"])
        if fila["fen"] not in posiciones[fila["carpeta"]]:
            posiciones[fila["carpeta"]].append(fila["fen"])

    reparto: dict[tuple[str, str], str] = {}
    for carpeta in sorted(posiciones):
        claves = sorted(posiciones[carpeta])
        rng = np.random.default_rng(
            SEMILLA + sum(ord(c) for c in carpeta)
        )
        ordenadas = list(claves)
        rng.shuffle(ordenadas)
        n = len(ordenadas)
        if n == 1:
            n_train, n_valid = 0, 0
        else:
            n_train = max(1, int(n * PROPORCION_TRAIN))
            n_valid = max(1, int(round(n * PROPORCION_VALID)))
            if n_train + n_valid >= n:
                n_train = max(1, n - n_valid - 1)
        for indice, fen in enumerate(ordenadas):
            if indice < n_train:
                split = "train"
            elif indice < n_train + n_valid:
                split = "valid"
            else:
                split = "test"
            reparto[(carpeta, fen)] = split
    return reparto


# ---------------------------------------------------------------------------
# Aumento geometrico (mismo estilo que aumentar_reales.py, angulo +-4)
# ---------------------------------------------------------------------------
def homografia_variante(rng: np.random.Generator,
                        angulo_max: float = ANGULO) -> np.ndarray:
    """Rotacion pequena + jitter de perspectiva de hasta 2 % del borde."""
    angulo = float(rng.uniform(-angulo_max, angulo_max))
    centro = (LADO / 2.0, LADO / 2.0)
    rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    limite = JITTER * LADO
    origen = np.array(
        [[0, 0], [LADO - 1, 0], [LADO - 1, LADO - 1], [0, LADO - 1]],
        dtype="float32",
    )
    destino = origen + rng.uniform(-limite, limite, size=(4, 2)).astype("float32")
    jitter = cv2.getPerspectiveTransform(origen, destino.astype("float32"))
    return jitter @ np.vstack([rotacion, [0.0, 0.0, 1.0]])


def preparar_manifiestos(filas: list[dict]) -> None:
    """Crea las carpetas de salida."""
    for carpeta in sorted({f["carpeta"] for f in filas}):
        (SALIDA / "images" / carpeta).mkdir(parents=True, exist_ok=True)
        (SALIDA / "labels" / carpeta).mkdir(parents=True, exist_ok=True)
    (SALIDA / "previews").mkdir(parents=True, exist_ok=True)


def volcar_csv(ruta: Path, campos: list[str], filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def contar_por_clase(filas: list[dict],
                     reparto: dict[tuple[str, str], str]) -> dict[str, Counter]:
    """Conteo de casillas por clase y split, sin escribir imagen alguna."""
    total: Counter = Counter()
    for split in ("train", "valid", "test"):
        total[split] = Counter()
    for fila in filas:
        split = reparto[(fila["carpeta"], fila["fen"])]
        for clase in fila["ocupacion"].values():
            total[split][clase] += 1
    return total


def clases_raras(conteo_train: Counter, clases: list[str]) -> list[str]:
    """Clases de pieza por debajo del 25 % de la clase mas frecuente."""
    piezas = {c: conteo_train[c] for c in clases}
    if not piezas:
        return []
    maximo = max(piezas.values())
    if maximo <= 0:
        return []
    return sorted(c for c, n in piezas.items() if n < CONTROL_RARA * maximo)


# ---------------------------------------------------------------------------
# Generacion
# ---------------------------------------------------------------------------
def generar(variantes: int, semilla: int, angulo: float, limite: int,
            previews: int, politica: str) -> dict:
    ruta_manifiesto = SALIDA / "manifiesto_ampliado.csv"
    filas = leer_declaradas()
    if limite > 0:
        filas = filas[:limite]
    reparto = repartir_posiciones(filas)
    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    preparar_manifiestos(filas)
    for clase in clases + [CLASE_VACIA]:
        (CASILLAS / clase).mkdir(parents=True, exist_ok=True)

    # El reparto efectivo por posicion: en ``replicar`` todas las variantes van
    # al train; en ``exclusivo`` las de valid/test se quedan en su split (solo
    # las usa el clasificador, el detector no las mete a train).
    split_variante = {clave: ("train" if politica == "replicar" else split)
                      for clave, split in reparto.items()}

    conteo_base = contar_por_clase(filas, reparto)
    raras = clases_raras(conteo_base["train"], clases)
    print(f"posiciones: {len(reparto)} -> "
          f"{Counter(reparto.values())}")
    print(f"clases raras (control de inversion): {raras}")

    filas_manifiesto: list[dict] = []
    filas_casillas: list[dict] = []
    por_clase: Counter = Counter()
    espejos: Counter = Counter()
    nombres_imagen: list[str] = []
    nombres_casilla: list[str] = []
    muestras: list[tuple[str, list[np.ndarray]]] = []
    faltantes: list[str] = []
    discrepancias_label: list[str] = []
    inicio = time.time()

    for indice, fila in enumerate(filas, start=1):
        carpeta = fila["carpeta"]
        stem = fila["stem"]
        ocupacion = fila["ocupacion"]
        split = split_variante[(carpeta, fila["fen"])]
        origen = RECORTES / carpeta / fila["archivo"]
        imagen = cv2.imread(str(origen), cv2.IMREAD_COLOR
                            | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            faltantes.append(str(origen))
            continue
        if imagen.shape[0] != LADO or imagen.shape[1] != LADO:
            imagen = cv2.resize(imagen, (LADO, LADO),
                                interpolation=cv2.INTER_AREA)

        # Etiqueta YOLO de la FEN: se escribe desde la verdad, no se copia.
        lineas = [linea_yolo(ocupacion[fila_columna_a_casilla(f, c)], 
                             fila_columna_a_casilla(f, c), clases)
                  for f in range(8) for c in range(8)
                  if ocupacion[fila_columna_a_casilla(f, c)] != CLASE_VACIA]
        original = RECORTES / carpeta / "labels" / f"{stem}.txt"
        if original.exists():
            esperado = sorted(original.read_text(encoding="utf-8").splitlines())
            if esperado != sorted(lineas):
                discrepancias_label.append(str(original))

        lote: list[np.ndarray] = []
        for k in range(variantes):
            rng = np.random.default_rng(_semilla(f"{carpeta}/{stem}", k) + semilla)
            homografia = homografia_variante(rng, angulo)
            deformada = cv2.warpPerspective(
                imagen, homografia, (LADO, LADO), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            variante = fotometria(deformada, rng)
            lote.append(variante)

            nombre = f"{stem}_aug{k}"
            cv2.imwrite(str(SALIDA / "images" / carpeta / f"{nombre}.jpg"),
                        variante, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_IMAGEN])
            with open(SALIDA / "labels" / carpeta / f"{nombre}.txt", "w",
                      encoding="utf-8") as archivo:
                for linea in lineas:
                    archivo.write(linea + "\n")
            nombres_imagen.append(f"{carpeta}/{nombre}.jpg")
            filas_manifiesto.append({
                "imagen": f"{carpeta}/{nombre}.jpg",
                "carpeta": carpeta,
                "posicion": fila["posicion"],
                "fen": fila["fen"],
                "variante": k,
                "split": split,
                "n_fichas": fila["n_fichas"],
                "semilla": int(_semilla(f"{carpeta}/{stem}", k) + semilla),
            })

            inversa = np.linalg.inv(homografia)
            for indice_casilla in range(64):
                f, c = divmod(indice_casilla, 8)
                casilla = fila_columna_a_casilla(f, c)
                clase = ocupacion[casilla]
                recorte = recortar_casilla(variante, inversa, f, c)
                if recorte.shape[0] != TAMANO_CASILLA:
                    recorte = cv2.resize(
                        recorte, (TAMANO_CASILLA, TAMANO_CASILLA),
                        interpolation=cv2.INTER_CUBIC,
                    )
                archivo_casilla = f"{nombre}_{casilla}.jpg"
                cv2.imwrite(str(CASILLAS / clase / archivo_casilla), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_CASILLA])
                nombres_casilla.append(f"{clase}/{archivo_casilla}")
                por_clase[clase] += 1
                filas_casillas.append({
                    "imagen": f"{carpeta}/{nombre}.jpg",
                    "casilla": casilla,
                    "clase": clase,
                    "split": split,
                    "tipo": "normal",
                })
                # Control de inversion SOLO en train y SOLO clases raras.
                if split == "train" and clase in raras and clase != CLASE_VACIA:
                    espejo = cv2.flip(recorte, 1)
                    archivo_espejo = f"{nombre}_{casilla}_espejo.jpg"
                    cv2.imwrite(str(CASILLAS / clase / archivo_espejo), espejo,
                                [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_CASILLA])
                    nombres_casilla.append(f"{clase}/{archivo_espejo}")
                    espejos[clase] += 1
                    filas_casillas.append({
                        "imagen": f"{carpeta}/{nombre}.jpg",
                        "casilla": casilla,
                        "clase": clase,
                        "split": split,
                        "tipo": "espejo",
                    })

        if len(muestras) < previews:
            muestras.append((f"{carpeta}/{fila['archivo']}", lote))
        if indice % 20 == 0 or indice == len(filas):
            print(f"  {indice}/{len(filas)} recortes", flush=True)

    # ---- CSV y resumen
    volcar_csv(SALIDA / "manifiesto_ampliado.csv", CAMPOS_MANIFIESTO,
               filas_manifiesto)
    volcar_csv(SALIDA / "casillas_manifesto.csv", CAMPOS_CASILLAS,
               filas_casillas)
    posiciones_vistas: dict[tuple[str, str], dict] = {}
    for fila in filas:
        clave = (fila["carpeta"], fila["fen"])
        if clave not in posiciones_vistas:
            posiciones_vistas[clave] = {
                "carpeta": fila["carpeta"],
                "posicion": fila["posicion"],
                "fen": fila["fen"],
                "split": split_variante[clave],
            }
    volcar_csv(SALIDA / "splits_posiciones.csv", CAMPOS_SPLITS,
               sorted(posiciones_vistas.values(),
                      key=lambda f: (f["carpeta"], int(f["posicion"]))))

    tabla = {clase: Counter() for clase in clases + [CLASE_VACIA]}
    for fila in filas_casillas:
        if fila["tipo"] == "espejo":
            # El control de inversion NO es una casilla nueva del tablero: se
            # cuenta aparte para que "total" siga cuadrando con 64 x imagenes.
            tabla[fila["clase"]]["espejo"] += 1
            continue
        tabla[fila["clase"]][fila["split"]] += 1
    conteo_filas = [{
        "clase": clase,
        "total": sum(tabla[clase][s] for s in ("train", "valid", "test")),
        "train": tabla[clase]["train"],
        "valid": tabla[clase]["valid"],
        "test": tabla[clase]["test"],
        "espejo": tabla[clase]["espejo"],
    } for clase in clases + [CLASE_VACIA]]
    volcar_csv(SALIDA / "conteo_por_clase.csv", CAMPOS_CONTEO, conteo_filas)

    # ---- previews
    rutas_preview: list[str] = []
    for indice, (nombre, variantes_hoja) in enumerate(muestras, start=1):
        if not variantes_hoja:
            continue
        destino = SALIDA / "previews" / f"ampliado_muestra_{indice}.jpg"
        cv2.imwrite(str(destino), hoja_contacto(variantes_hoja, nombre),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        rutas_preview.append(str(destino))

    # ---- verificacion del disco
    problemas = verificar(conteo_filas, filas_manifiesto)
    if faltantes:
        problemas.append(f"{len(faltantes)} recortes ilegibles: {faltantes[:3]}")
    if discrepancias_label:
        problemas.append(f"{len(discrepancias_label)} labels no coinciden con la "
                         f"FEN: {discrepancias_label[:3]}")

    resumen = {
        "politica": politica,
        "recortes": len(filas),
        "posiciones": len(posiciones_vistas),
        "variantes": variantes,
        "angulo_max": angulo,
        "semilla": semilla,
        "split_posiciones": dict(sorted(
            Counter(f["split"] for f in posiciones_vistas.values()).items())),
        "split_imagenes": dict(sorted(
            Counter(f["split"] for f in filas_manifiesto).items())),
        "casillas": sum(conteo_filas[i]["total"] for i in range(len(conteo_filas))),
        "espejos": dict(sorted(espejos.items())),
        "clases_raras": raras,
        "conteo_por_clase": conteo_filas,
        "problemas": problemas,
        "segundos": round(time.time() - inicio, 1),
    }
    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    segundos = time.time() - inicio
    print()
    print(f"listo: {len(filas)} recortes x {variantes} variantes en {segundos:.1f} s")
    print(f"  imagenes: {len(nombres_imagen)} -> {SALIDA / 'images'}")
    print(f"  casillas {TAMANO_CASILLA}x{TAMANO_CASILLA}: {len(nombres_casilla)} "
          f"-> {CASILLAS}")
    print(f"  por split (imagenes): {resumen['split_imagenes']}")
    print("  por clase (total / train / valid / test / espejo):")
    for fila in conteo_filas:
        print(f"    {fila['clase']:10s} {fila['total']:6d} {fila['train']:6d} "
              f"{fila['valid']:6d} {fila['test']:6d} {fila['espejo']:5d}")
    print(f"  controles de inversion (espejo): {dict(sorted(espejos.items()))}")
    print(f"  previews: {[Path(p).name for p in rutas_preview]}")
    if problemas:
        print(f"  PROBLEMAS: {problemas}")
        print("RESULTADO: revisar")
        return resumen | {"correcto": False}
    print("RESULTADO: ok")
    return resumen | {"correcto": True}


# ---------------------------------------------------------------------------
# Verificacion
# ---------------------------------------------------------------------------
def verificar(conteo_filas: list[dict] | None = None,
              filas_manifiesto: list[dict] | None = None) -> list[str]:
    """Comprueba el disco contra los manifiestos. Devuelve la lista de fallos."""
    problemas: list[str] = []
    ruta_manifiesto = SALIDA / "manifiesto_ampliado.csv"
    ruta_casillas = SALIDA / "casillas_manifesto.csv"
    if not ruta_manifiesto.exists() or not ruta_casillas.exists():
        return [f"faltan manifiestos en {SALIDA}"]
    if filas_manifiesto is None:
        with ruta_manifiesto.open("r", encoding="utf-8-sig", newline="") as archivo:
            filas_manifiesto = list(csv.DictReader(archivo))

    imagenes = list((SALIDA / "images").rglob("*.jpg"))
    etiquetas = list((SALIDA / "labels").rglob("*.txt"))
    if len(imagenes) != len(filas_manifiesto):
        problemas.append(f"imagenes {len(imagenes)} != manifiesto "
                         f"{len(filas_manifiesto)}")
    if len(etiquetas) != len(filas_manifiesto):
        problemas.append(f"etiquetas {len(etiquetas)} != manifiesto "
                         f"{len(filas_manifiesto)}")

    faltan_img, faltan_lbl = 0, 0
    for fila in filas_manifiesto:
        if not (SALIDA / "images" / fila["imagen"]).exists():
            faltan_img += 1
        if not (SALIDA / "labels" / Path(fila["imagen"]).with_suffix(".txt")).exists():
            faltan_lbl += 1
    if faltan_img:
        problemas.append(f"{faltan_img} imagenes del manifiesto no existen")
    if faltan_lbl:
        problemas.append(f"{faltan_lbl} etiquetas del manifiesto no existen")

    # Una muestra de imagenes: lado y YOLO bien formado.
    for fila in filas_manifiesto[:3]:
        ruta = SALIDA / "images" / fila["imagen"]
        imagen = cv2.imread(str(ruta), cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None or imagen.shape[:2] != (LADO, LADO):
            problemas.append(f"{ruta}: no es {LADO}x{LADO}")
            continue
        texto = (SALIDA / "labels" /
                 Path(fila["imagen"]).with_suffix(".txt")).read_text(encoding="utf-8")
        lineas = [linea for linea in texto.splitlines() if linea.strip()]
        if len(lineas) != int(fila["n_fichas"]):
            problemas.append(f"{fila['imagen']}: {len(lineas)} cajas != "
                             f"{fila['n_fichas']} fichas del FEN")
        for linea in lineas:
            partes = linea.split()
            if len(partes) != 5 or not (0 <= int(partes[0]) < 12):
                problemas.append(f"{fila['imagen']}: linea YOLO invalida {linea!r}")
                break
            if not all(0.0 <= float(v) <= 1.0 for v in partes[1:]):
                problemas.append(f"{fila['imagen']}: coords fuera de rango {linea!r}")
                break

    with ruta_casillas.open("r", encoding="utf-8-sig", newline="") as archivo:
        filas_casillas = list(csv.DictReader(archivo))
    faltan_celdas = 0
    for fila in filas_casillas:
        if not (CASILLAS / fila["clase"] /
                f"{Path(fila['imagen']).stem}_{fila['casilla']}"
                f"{'_espejo' if fila['tipo'] == 'espejo' else ''}.jpg").exists():
            faltan_celdas += 1
    if faltan_celdas:
        problemas.append(f"{faltan_celdas} casillas del manifiesto no existen")

    # Tamano de las 13 clases (una muestra de cada carpeta).
    clases = cargar_clases() + [CLASE_VACIA]
    for clase in clases:
        ejemplos = sorted((CASILLAS / clase).glob("*.jpg"))
        if not ejemplos:
            problemas.append(f"clase {clase} sin casillas")
            continue
        recorte = cv2.imread(str(ejemplos[0]), cv2.IMREAD_COLOR)
        if recorte is None or recorte.shape[:2] != (TAMANO_CASILLA,
                                                    TAMANO_CASILLA):
            problemas.append(f"{clase}: casilla no es "
                             f"{TAMANO_CASILLA}x{TAMANO_CASILLA}")

    # Conteos por clase contra el CSV.
    if conteo_filas is None:
        with (SALIDA / "conteo_por_clase.csv").open(
                "r", encoding="utf-8-sig", newline="") as archivo:
            conteo_filas = list(csv.DictReader(archivo))
    for fila in conteo_filas:
        clase = fila["clase"]
        normales = sum(1 for c in filas_casillas
                       if c["clase"] == clase and c["tipo"] == "normal")
        espejo = sum(1 for c in filas_casillas
                     if c["clase"] == clase and c["tipo"] == "espejo")
        if normales != int(fila["total"]):
            problemas.append(f"{clase}: manifiesto {normales} != total "
                             f"{fila['total']}")
        if espejo != int(fila["espejo"]):
            problemas.append(f"{clase}: espejos {espejo} != {fila['espejo']}")
        if not (CASILLAS / clase).is_dir():
            problemas.append(f"falta la carpeta de clase {clase}")
            continue
        if len(list((CASILLAS / clase).glob("*.jpg"))) != normales + espejo:
            problemas.append(f"{clase}: archivos en disco != manifiesto "
                             f"({normales + espejo})")
    return problemas


def solo_verificar() -> int:
    problemas = verificar()
    print(f"verificacion de {SALIDA}")
    if problemas:
        for problema in problemas:
            print(f"  FALLO: {problema}")
        print("RESULTADO: revisar")
        return 1
    print("  imagenes == etiquetas == manifiesto")
    print("  casillas: manifiesto == disco, 13 clases, "
          f"{TAMANO_CASILLA}x{TAMANO_CASILLA}")
    print("RESULTADO: ok")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Amplifica 02/03/04 (FEN declarada) con variantes aumentadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--variantes", type=int, default=VARIANTES)
    analizador.add_argument("--semilla", type=int, default=0,
                            help="se suma a la semilla estable por imagen")
    analizador.add_argument("--angulo", type=float, default=ANGULO,
                            help="rotacion maxima en grados")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo N recortes (0 = todos)")
    analizador.add_argument("--previews", type=int, default=4)
    analizador.add_argument("--politica-split",
                            choices=("exclusivo", "replicar"),
                            default="exclusivo",
                            help="exclusivo = sin fuga (variantes de train no "
                                 "tocan las fotos crudas de val/test)")
    analizador.add_argument("--verificar", action="store_true",
                            help="no genera nada: solo comprueba el disco")
    analizador.add_argument("--salida", default=None,
                            help="carpeta de salida (para pruebas aisladas)")
    argumentos = analizador.parse_args(argv)

    global SALIDA, CASILLAS
    if argumentos.salida:
        SALIDA = Path(argumentos.salida)
        CASILLAS = SALIDA / "casillas" / "clasificador"

    if argumentos.verificar:
        return solo_verificar()
    if argumentos.variantes <= 0:
        print("nada que generar: --variantes debe ser mayor que 0")
        return 2
    resumen = generar(argumentos.variantes, argumentos.semilla,
                      argumentos.angulo, argumentos.limite,
                      argumentos.previews, argumentos.politica_split)
    return 0 if resumen.get("correcto") else 1


if __name__ == "__main__":
    raise SystemExit(main())
