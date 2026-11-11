# T10 — Amplificación de las posiciones declaradas + bundles de Colab al día

Fecha: 2026-11-07. Contexto: el usuario pidió «ampliar inicial con jugadas y medio
juego y finales usando las fotos que ya se tiene del ESP». Este documento registra
la amplificación, el reparto que la acompaña y los paquetes de Colab regenerados.
Todo sale de la FEN DECLARADA (nada de modelos etiquetando fotos reales).

## 1. Qué se amplificó (`scripts/amplificar_declaradas.py`)

Entrada: los 188 recortes 800x800 de 02/03/04 con FEN declarada en
`dataset/recortes_800/manifesto_recortes.csv`.

| Carpeta | Recortes | Posiciones (FEN) |
|---|---|---|
| 02_inicial_con_jugadas | 134 | 10 |
| 03_medio_juego | 47 | 5 |
| 04_finales | 7 | 1 |
| **Total** | **188** | **16** |

Por recorte se generan **4 variantes** con el MISMO estilo de `aumentar_reales.py`
(T4b), reusando `fotometria`, `recortar_casilla` y `_semilla` de ese módulo:

* geometría: rotación ±4° + jitter de perspectiva de ≤2 % del borde, compuestos en
  UNA homografía `H`; la imagen se deforma con `H` y las 64 casillas se recortan
  con `inv(H)` (la etiqueta viaja con la geometría, no puede desalinearse);
* fotometría: contraste 0.75–1.35, brillo ±25, gamma 0.7–1.4, saturación 0.8–1.2,
  ruido gaussiano σ 3–12 y desenfoque σ 0–1.5;
* semilla estable por (imagen, variante) con `crc32`: dos corridas producen los
  mismos archivos.

Salidas (verificadas contra los manifiestos, no listando carpetas):

| Producto | Cantidad |
|---|---|
| `images/` + `labels/` YOLO (800x800, 12 clases) | 752 + 752 |
| casillas `casillas/clasificador/<clase>/` (**128x128**, 13 clases) | 48,128 |
| controles de inversión (espejo de clases raras) | 1,312 |
| `manifiesto_ampliado.csv` (imagen, carpeta, posicion, fen, …) | 752 filas |
| `casillas_manifesto.csv` / `splits_posiciones.csv` / `conteo_por_clase.csv` | 49,440 / 16 / 13 filas |

### Reparto (por POSICIÓN, nunca por foto)

Una posición es una FEN: sus fotos y sus variantes caen siempre en el mismo split.
50 % train / 25 % valid / 25 % test por carpeta, mínimo 1 valid y 1 test;
`04_finales` (una sola posición) queda en test.

| Split | Posiciones | Imágenes aumentadas |
|---|---|---|
| train | 7 | 328 |
| valid | 3 | 156 |
| test | 6 | 268 |

### Conteo por clase (total / train / valid / test / espejo)

| clase | total | train | valid | test | espejo |
|---|---|---|---|---|---|
| peon_b | 5532 | 2492 | 1192 | 1848 | 0 |
| torre_b | 1448 | 656 | 312 | 480 | 0 |
| caballo_b | 1368 | 656 | 272 | 440 | 0 |
| alfil_b | 1448 | 656 | 312 | 480 | 0 |
| dama_b | 724 | 328 | 156 | 240 | 328 |
| rey_b | 752 | 328 | 156 | 268 | 328 |
| peon_n | 5648 | 2492 | 1248 | 1908 | 0 |
| torre_n | 1448 | 656 | 312 | 480 | 0 |
| caballo_n | 1448 | 656 | 312 | 480 | 0 |
| alfil_n | 1336 | 656 | 272 | 408 | 0 |
| dama_n | 724 | 328 | 156 | 240 | 328 |
| rey_n | 752 | 328 | 156 | 268 | 328 |
| vacia | 25500 | 10760 | 5128 | 9612 | 0 |

`vacia` = 51.6 % del total ampliado (menos que el 55.9 % del compuesto, porque las
posiciones declaradas reales tienen más piezas por tablero).

### Controles de inversión (clases raras)

La clase rara se decide con los datos, no a mano: **< 25 % de la clase más
frecuente en train**. Con este lote salen exactamente `dama_b`, `dama_n`, `rey_b`
y `rey_n` (1 por tablero). Para ellas se escribe además la MISMA casilla espejada
horizontalmente (`_espejo.jpg`), que es etiqueta-preservante (un rey espejado
sigue siendo rey) y equilibra el desbalance sin inventar posiciones en el
tablero. Solo en train: 4 × 328 = 1,312 recortes.

## 2. Qué cambió en los bundles (`scripts/preparar_colab_v2_clean.py`)

`preparar_colab_v2_clean.py --todo` lee los manifiestos de `dataset/augmentado_new`
(si existen) y aplica la política de reparto.

### Política `exclusivo` (por defecto, sin fuga)

* Las variantes de una posición de **train** entran al train del detector.
* Las fotos CRUDAS de esa misma posición **no entran a val/test**: la posición se
  representa en el train con sus 4 variantes. 82 recortes crudos salen de val/test.
* Las variantes de valid/test NO se empaquetan para el clasificador: serían
  casi-duplicados de la misma foto dentro de su propio split de evaluación.

Motivo: si se metieran las variantes de una posición de train y a la vez la foto
cruda de esa posición en test, el test dejaría de medir generalización (el modelo
habría entrenado con copias ±4° de la misma foto). La regla de oro del paquete
(«val y test SOLO fotos reales») se mantiene y además queda sin fuga de posición.

Alternativa implementada: `amplificar_declaradas.py --politica-split replicar`
deja las 219 fotos crudas en val/test y manda las 752 variantes al train. Está
documentada por si el usuario prefiere el volumen, pero ese test NO es limpio.

### Números medidos del paquete

| Paquete | train | val/valid | test | Tamaño |
|---|---|---|---|---|
| `detector_v2_clean.zip` | 4,372 (1,544 sintético + 2,500 compuesto + **328 aumentado**) | 73 | 64 | 321.9 MB |
| `clasificador_v2_clean.zip` | 157,280 (134,976 compuesto + **22,304 aumentado**) | 16,896 (12,224 compuesto + 4,672 real) | 16,896 (12,800 compuesto + 4,096 real) | 527.2 MB |
| `entrega_v2.zip` | — | — | — | 83.9 MB |

Detector: `train` 107,875 cajas; `val` 2,221 cajas reales; `test` 1,774 cajas
reales. `data.yaml` sigue con `nc=12` en el orden de `classes.txt`.

Clasificador: 191,072 casillas, 2,965 imágenes fuente, 13 clases, 0 grupos
compartidos entre splits; `vacia` = 54.81 / 55.13 / 56.00 % (train/valid/test),
medido y reportado por el script. Las casillas compuesto/reales siguen en 100x100
y las aumentadas van en 128x128 (lado de entrada del clasificador, `TAMANO=128`);
el `Dataset` del notebook hace `Resize((128,128))`, así que el tamaño mezclado no
cambia nada y no se reescala la data vieja.

## 3. Verificación

1. `python preparar_colab_v2_clean.py --todo` → resumen por split, por clase y por
   fuente impreso; `resumen.json` con `aumentado`, `detector` y `clasificador`.
2. `python amplificar_declaradas.py --verificar` → imágenes == etiquetas ==
   manifiesto (752), casillas del manifiesto == disco (49,440), 13 clases,
   128x128.
3. Inventario del zip del clasificador: 191,072 casillas, 22,304 con `_aug`,
   1,312 `_espejo`, 13 carpetas de clase, `157280/16896/16896` rutas y **todas**
   presentes en el zip.
4. `python -m py_compile` de los tres scripts tocados.
5. Diferencial sin ampliación (monkeypatch de `AUMENTADO` a una ruta inexistente):
   `repartir_reales(None)` → 219 recortes (109 val / 110 test), inventario
   174,016, detector 4,044 / 109 / 110, zip 286.7 MB. Es exactamente T9: la
   ampliación es aditiva y no altera el paquete viejo.

## 4. Límites honestos

1. **No hay posiciones nuevas.** Las 16 posiciones son las mismas; solo cambia la
   iluminación/rotación con la que se ven. El clasificador gana robustez, no
   cobertura de posiciones.
2. **128x128 sale de un recorte de 100x100** (remuestreo con INTER_CUBIC), no de
   más píxeles reales: se guarda al lado de entrada del modelo para que el
   clasificador no tenga que reescalar y el paquete sea homogéneo en `TAMANO`.
3. **Escala del detector.** El train ahora mezcla frames (ficha ~48 px) con
   recortes 800 (ficha ~85 px, la escala del test). Reduce el desajuste que
   documentó T9, pero también hace que el test deje de ser «dominio no visto»:
   lo que el test mide ahora es generalización a POSICIONES no vistas.
4. **82 recortes crudos no se usan en el detector** (los de posiciones de train):
   sus 4 variantes los representan. Es el precio de no filtrar el test.
5. **El espejo de las clases raras es un control de balance**, no una prueba de
   métrica: no se midió su efecto aislado.

## 5. Specs cubiertas

- **S2** (fichas y cámara nuevas): la ampliación sale solo de las capturas nuevas.
- **S3** (YOLO + clasificador por casilla): ambos paquetes quedan listos con la
  ampliación incluida; la comparación v1 vs v2 es T7 y no se ejecuta aquí.
- **S5** (protocolo de captura): no se recorta, renombra ni corrige el crudo; lo
  único que se rectifica/aumenta es a partir de los recortes ya rectificados.

No cubre: T0/T1 (cámara en vivo), T6 (pipeline `src/`), T7 (comparación) ni la
ejecución real del entrenamiento en Colab (la hace el usuario).
