# Entrenamiento en Google Colab — dataset LIMPIO (v2 desde cero)

Pasos para entrenar **el detector YOLO** y **el clasificador por casilla** en
Colab con GPU, usando el dataset reconstruido desde la posición inicial física
correcta (sin el cruce caballo/alfil del lote anterior).

Los dos notebooks son independientes: puedes correr sólo uno.

| Modelo | Notebook | Paquete | Qué produce |
|---|---|---|---|
| (a) Detector YOLO | `ajedrez_v2_detector_desde_cero.ipynb` | `detector_v2_clean.zip` (~287 MB) | `best.pt` |
| (b) Clasificador 13 clases | `ajedrez_v2_clasificador_desde_cero.ipynb` | `clasificador_v2_clean.zip` (~474 MB) | `best_clasificador.pt` |
| (c) Paquete de juego | — | `entrega_v2.zip` (~88 MB) | visor + Stockfish + muestras |

Los notebooks anteriores (`ajedrez_v2_detector.ipynb`, `ajedrez_v2_clasificador.ipynb`)
se conservan como referencia histórica del lote en cuarentena. **No los uses con
los zips nuevos**: esperan otro reparto (`train=3802`, `train=115584`) y el
detector viejo arranca desde `modelo_sintetico.pt`, que es justo el peso que se
entrenó con la convención contaminada.

## 1. Generar los paquetes (en tu máquina, Windows)

```powershell
cd F:\TAVO\Projects\ajedrez-v2
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\rectificar_crudo_v2.py
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\extraer_sprites_limpio.py
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\generar_compuesto_limpio.py --n 2500
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\preparar_colab_v2_clean.py --todo
```

Produce, dentro de `dataset\colab_bundle\`:

```
detector_v2_clean.zip            286.7 MB  -> sube este
clasificador_v2_clean.zip        473.9 MB  -> sube este
entrega_v2.zip                    87.9 MB  -> este es para el equipo de juego
detector_v2_clean\images\{train,val,test}
detector_v2_clean\labels\{train,val,test}
detector_v2_clean\data.yaml       nc: 12 en el orden de classes.txt
clasificador_v2_clean\split\{train,valid,test}.txt
clasificador_v2_clean\labels\clases.txt
resumen.json                      conteos exactos de los dos empaques
```

Reparto del **detector** (la regla de oro: `val` y `test` SOLO fotos reales):

| Split | Imágenes | Cajas | De dónde |
|---|---|---|---|
| train | 4,044 | 97,643 | 1,544 sintéticas + 2,500 compuestos limpios |
| val | 109 | 3,323 | recortes reales: 14 de 01, 66 de 02, 28 de 03, 1 de 05 |
| test | 110 | 3,230 | recortes reales: 14 de 01, 68 de 02, 19 de 03, 7 de 04, 2 de 05 |

El reparto de los reales es por **posición declarada** (todas las fotos de la
misma FEN caen en el mismo split); 01 y 04 tienen una sola posición y se
reparten por imagen. `04_finales` va completo a test.

Reparto del **clasificador** (por tablero, nunca por casilla):

| Split | Tableros | Casillas | % `vacia` |
|---|---|---|---|
| train | 2,175 | 139,200 | 55.90% |
| valid | 272 | 17,408 | 54.57% |
| test | 272 | 17,408 | 55.11% |

Los recortes reales de `valid`/`test` son **los mismos** que usa el detector
(mismo reparto por imagen fuente), así la comparación de T7 se mide sobre las
mismas fotos.

## 2. Subir los ZIP a Drive

Crea la carpeta **`Mi unidad/ajedrez_v2/`** y sube ahí:

```
Mi unidad/ajedrez_v2/detector_v2_clean.zip
Mi unidad/ajedrez_v2/clasificador_v2_clean.zip
```

Esas son las rutas que usan los notebooks (`ZIP_EN_DRIVE`). Si las cambias, edita
la constante en la celda 3.

`files.upload()` de Colab no aguanta bien archivos de cientos de MB: para estos
tamaños usa Drive.

## 3. Abrir el notebook y ejecutar

Sube el `.ipynb` a Colab (`File > Upload notebook`) o ábrelo desde Drive.
**Antes de ejecutar:** `Entorno de ejecución > Cambiar tipo de entorno > T4 GPU`.

Detector (`ajedrez_v2_detector_desde_cero.ipynb`):

| Celda | Qué hace | Qué revisar |
|---|---|---|
| 1 | Verifica GPU | `GPU disponible: True` |
| 2 | Comprueba `ultralytics` | No reinstala `torch` |
| 3 | Monta Drive y descomprime | `train=4372 val=73 test=64`, todo `OK` |
| 3b | Reescribe `path` del `data.yaml` | 1 cambio |
| 4 | Audita el dataset | Termina en `Dataset OK. Puedes entrenar.` |
| 5 | Configuración | `Modo: desde_cero`, `Base: yolo11n.pt`, 120 épocas |
| 6 | Entrena | Baja `box_loss`, sube `mAP50` |
| 7 | Evalúa en `test` | mAP50 por clase |
| 8 | Confusiones | Vigila `caballo_n <-> alfil_n` |
| 9 | Guarda en Drive | `Mi unidad/ajedrez_v2/detector/entrega/best.pt` |
| 10 | Predicción visual | Cajas sobre una foto real de test |

Clasificador (`ajedrez_v2_clasificador_desde_cero.ipynb`):

| Celda | Qué hace | Qué revisar |
|---|---|---|
| 1 | Verifica GPU | `GPU disponible: True` |
| 2 | Comprueba `torchvision` | No reinstala `torch` |
| 3 | Monta Drive y descomprime | `train=157280 valid=16896 test=16896` |
| 4 | Dataset + pesos de clase | `vacia = 55.90%` y su peso < 1 |
| 5 | Modelo | `mobilenet_v3_small`, 13 salidas, todos los parámetros entrenables |
| 6 | Entrena 20 épocas con AMP | Sube `valid`; guarda el mejor en Drive |
| 7 | Matriz por clase | `recall` de piezas alto, no sólo `vacia` |
| 8 | Copia a `entrega/` | Recarga el `.pt` para comprobar que sirve |

Tiempo estimado en T4: detector **45-75 min** (120 épocas, 4,372 imágenes a 640)
y clasificador **38-65 min** (20 épocas, 157 k casillas; la primera vez suma 5-7
min de descomprimir 191 mil archivos).

## 4. Bajar los modelos

Quedan en tu Drive:

```
Mi unidad/ajedrez_v2/detector/entrega/best.pt
Mi unidad/ajedrez_v2/clasificador/entrega/best_clasificador.pt
Mi unidad/ajedrez_v2/clasificador/entrega/clases.txt
```

Cópialos a:

```
F:\TAVO\Projects\ajedrez-v2\modelos\best.pt                  <- detector
F:\TAVO\Projects\ajedrez-v2\modelos\best_clasificador.pt     <- clasificador
```

`last.pt` (también en `entrega/`) sirve para reanudar un entrenamiento cortado.

## 5. Cambios declarados respecto a los zips anteriores

| | Zip anterior (cuarentena) | Zip limpio (este) |
|---|---|---|
| Sprites de ficha | `reales_compuesto/sprites` con el par caballo/alfil cruzado | `sprites_limpio` cortados de la posición inicial física (832 sprites, 26 fotos) |
| Etiqueta de las fotos reales | modelo/autoetiquetado con premisa de FEN | **FEN declarada** en el manifiesto de cada carpeta |
| Detector: base | `modelo_sintetico.pt` (convención vieja) | `yolo11n.pt` COCO, `MODO="desde_cero"` |
| Detector: épocas / lr | 60 / 0.001 | 120 / 0.01 |
| Train del detector | 3,802 (1,544 + 2,258) | 4,372 (1,544 sintético + 2,500 compuesto + **328 variantes aumentadas**) |
| val / test del detector | 84 / 56 (12 y 8 fuentes × variantes) | 73 / 64 recortes reales (posiciones nunca vistas en train) |
| Clasificador | MobileNetV3-Small, 12 épocas | MobileNetV3-Small ImageNet **sin congelar**, 20 épocas, **AMP**, pesos por frecuencia inversa, `RandomRotation(12)` |
| Casillas del clasificador | 152,192 | 191,072 (160,000 compuesto + 8,768 reales + **22,304 aumentadas T10**) |
| Reparto del clasificador | por tablero, reales reutilizados del detector | igual, con 80/10/10 exacto y reales alineados con el detector |
| Ampliación T10 | — | 4 variantes por recorte declarado de 02/03/04 (±4°, fotometría T4b), split por posición; espejo de control en dama/rey |

**T10 (ampliación)**: las variantes aumentadas de las posiciones de train entran
al train del detector y del clasificador; las fotos crudas de esas posiciones NO
entran a val/test, para que el test siga siendo de posiciones nunca vistas. Las
variantes de valid/test no se empaquetan (serían casi-duplicados de la misma
foto dentro de su propio split).

La **línea base** del detector tampoco es comparable: el `mAP50 0.546` del modelo
anterior se midió sobre otro test y con las etiquetas cruzadas. La referencia
honesta de esta corrida es su propia época 1 (`results.csv`).

## 6. Verificar en local (CPU, sin GPU)

```powershell
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe -c "import zipfile,yaml; z=zipfile.ZipFile('dataset/colab_bundle/detector_v2_clean.zip'); print(len(z.namelist()),'entradas'); print(yaml.safe_load(z.read('data.yaml'))['nc'])"
```

Debe imprimir `9019 entradas` y `12`.

## Archivos relacionados

- `scripts/rectificar_crudo_v2.py` — rectifica el crudo y deriva labels + casillas.
- `scripts/extraer_sprites_limpio.py` — sprites con etiqueta por layout físico.
- `scripts/generar_compuesto_limpio.py` — compuestos con esos sprites.
- `scripts/preparar_colab_v2_clean.py` — arma los dos bundles y los tres ZIP.
- `colab/generar_notebooks_v2_clean.py` — genera los dos `.ipynb` (no editar el JSON a mano).
- `docs/t9-notas.md` — decisiones y números medidos de esta reconstrucción.
