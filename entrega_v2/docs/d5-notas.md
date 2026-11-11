# T5 — Empaquetado para entrenar los dos modelos v2 en Colab

Notas del empaquetado: que se decidio, con que evidencia y que queda abierto.
Los artefactos los genera `scripts/preparar_colab_v2.py`; los notebooks
`colab/generar_notebooks_v2.py`.

## Que se genero

```
dataset/colab_bundle/
  detector/images/{train,val,test}        3,942 imagenes
  detector/labels/{train,val,test}        3,942 etiquetas
  detector/data.yaml                      nc: 12, orden de classes.txt
  detector/weights/modelo_sintetico.pt
  detector_v2.zip                         272.6 MB (2 coords)
  clasificador_split/{train,valid,test}.txt
  clasificador_v2.zip                     406.7 MB (152,192 JPG + 3 .txt + 3 CSV)
  resumen.json                            conteos exactos de los dos empaques
dataset/reales/yolo_labels/{base,augmentado}/   140 labels YOLO nuevos
```

## Decision 1 — tamano de la caja YOLO de las fotos reales: 0.85 de casilla

El enunciado proponia `w = h = 0.44/8 = 0.055` ("una ficha ocupa ~45% de la
casilla"). Se midio antes de aceptarlo, porque las cajas de `val`/`test` tienen
que estar en la misma escala que las de `train`:

| Fuente | Medida | Valor |
|---|---|---|
| sintetico (labels existentes, 445 imagenes) | ancho de caja / paso de casilla | mediana **0.891** (p10 0.428, p90 0.984) |
| compuesto (`rotaciones.csv`, 8,364 fichas) | diametro del disco / casilla | mediana **0.851** (rango 0.755-0.973) |

El compositor de T4d escribe la caja como el cuadrado que inscribe el **disco de
la base** de la ficha (`compilar_compuestos.py`, `radio = diametro/2`), y ese
disco mide ~0.85 de la casilla en las 12 clases por igual (medianas por clase
0.844-0.856). Con 0.44 las cajas reales serian la mitad que todas las de
entrenamiento: el mAP caeria por desajuste de escala, no por el modelo.

**Decision: `BOX_CASILLA = 0.85`** -> `w = h = 0.85/8 = 0.10625` en coordenadas
del recorte 800x800. Consecuencia: cajas de casillas vecinas no se solapan (paso
0.125 > lado 0.106), que es lo que NMS espera.

Verificacion geometrica del centrado (120 comparaciones, 20 fuentes, casillas de
esquina y del centro): el recorte de mi rejilla contra el recorte de referencia
del clasificador da una diferencia media de **17.7**; contra la casilla volteada
verticalmente **63.7** y horizontalmente **49.1**. La rejilla esta en el sentido
correcto (ratio 0.28).

## Decision 2 — reparto de las 20 fuentes reales: 12 valid / 8 test

Hay 20 recortes con las 64 casillas etiquetadas (17 posicion inicial + 3 tablero
vacio; las otras 108 fotos del CSV no tienen casillas, solo `desconocida`/`manual`).
Se reparte por CATEGORIA, con `random.Random(42)`:

- iniciales (17): 10 a valid, 7 a test
- vacios (3): 2 a valid, 1 a test

Resultado: 12 fuentes en val (84 imagenes, 2,240 cajas) y 8 en test (56 imagenes,
1,568 cajas). Garantiza lo que pedia la regla: ambos splits llevan posicion
inicial **y** tablero vacio, y las variantes aumentadas de un recorte nunca caen
en el split contrario al de su fuente (cada fuente aporta 1 recorte + 6
variantes).

Las 17 fuentes de posicion inicial se validaron contra la posicion estandar en el
orden de `classes.txt` (las 17 dan exactamente el mismo mapa casilla->clase, 32
fichas, `fila == 8 - rango` en las 1,280 filas). Si el etiquetado estuviera girado
o cruzado, el script aborta en vez de empaquetar basura.

## Decision 3 — clasificador: split por tablero, reales en val/test

- Compuestos: 2,258 grupos -> 1,806 train / 226 valid / 226 test (80/10/10 semilla 42).
- Reales: los 120 aumentados se agrupan por **recorte fuente** (20 grupos) y se
  asignan con el **mismo reparto que el detector** (12 valid / 8 test). Dos
  motivos: no hay fuga entre variantes de la misma foto, y el `test` real es
  literalmente el mismo que el del detector, que es lo que exige S3 para comparar
  YOLO solo vs clasificador solo vs combinado.
- No hay fotos reales en `train`: solo hay 20 fuentes y hay que reservarlas para
  medir. El clasificador se entrena con casillas compuestas (sprites reales sobre
  fondo real) y generaliza a las fotos reales.

Conteos por split (items = casillas):

| Split | Tableros | Items | % vacia | dama/vacia | ratio max/min (clases ocupadas) |
|---|---|---|---|---|---|
| train | 1,806 | 115,584 | 63.78% | 0.093 | 3.01 |
| valid | 238 | 19,072 | 62.44% | 0.089 | 3.12 |
| test | 234 | 17,536 | 61.96% | 0.084 | 3.37 |

Por clase (train/valid/test): peon_b 6792/1262/1263, torre_b 2465/408/406,
caballo_b 2422/429/394, alfil_b 2262/409/376, dama_b 3488/532/458,
rey_b 3527/539/444, peon_n 6801/1259/1257, torre_n 2461/419/397,
caballo_n 2408/404/378, alfil_n 2335/428/375, dama_n 3396/524/457,
rey_n 3504/551/465, vacia 73723/11908/10866.

Las 13 clases existen en train (el script aborta si falta alguna). Los pesos de
clase los calcula el notebook como frecuencia inversa normalizada a media 1.

## Empaque del clasificador: sin recopiar 152 mil JPG

El zip se construye leyendo los JPG desde donde ya estan
(`dataset/reales_compuesto/casillas/clasificador/<clase>/` y
`dataset/reales/clasificador/casillas/<clase>/`) con `arcname = casillas/<clase>/<archivo>.jpg`.
Los nombres no colisionan (compuestos empiezan por `compuesto_`, los reales por
el timestamp o `capture`). La lista de cada split es un `.txt` de rutas relativas
a la raiz del zip, dentro del zip en `split/{train,valid,test}.txt`.

## Verificacion local (CPU)

`scripts/verificar_detector_v2.py --bundle dataset/colab_bundle/detector_v2.zip`

- orden de clases del `data.yaml` y del `.pt` identico a `classes.txt` (12/12),
- smoke de 3 imagenes reales del test del bundle a `imgsz=320`, `batch=2`: 16 s,
  `mAP50 = 0.137`, `recall = 0.594` (3 imagenes: solo dice "corre").
- test completo (56 imagenes, `imgsz=640`, `batch=4`): 17 s,
  `mAP50 = 0.040`, `mAP50-95 = 0.011`, `precision = 0.088`, `recall = 0.128`.
  Esa es la linea base honesta del `modelo_sintetico.pt` sobre este test: el
  0.21 de v1 no es comparable (otras fotos, otro etiquetado).

Integridad de los paquetes (comprobada contra los ZIP finales):

- las 152,192 rutas de los tres `.txt` existen dentro de `clasificador_v2.zip`;
  cero solapamiento train/valid, train/test y valid/test; 13 clases en cada split;
- `detector_v2.zip`: 3,802/84/56 imagenes con etiqueta 1:1 y sin huerfanas;
- el codigo de las celdas 4 y 5 del notebook del clasificador se ejecuto tal
  cual sobre una muestra real del zip en local (torch CPU): 52 items, 13 clases,
  `entrada (8, 3, 128, 128)`, `salida (8, 13)`, `loss` y `backward` OK.

Tamanos finales: `detector_v2.zip` 272.6 MiB (285.8 MB decimales) y
`clasificador_v2.zip` 406.7 MiB (426.5 MB decimales).

## Riesgos abiertos (no bloquean el entrenamiento, si condicionan la lectura de T7)

1. **Fuga de apariencia train -> test.** Los 503 sprites de los compuestos
   (T4d) se extrajeron de los 17 recortes de posicion inicial, que son
   exactamente las fuentes que hoy caen en `val`/`test`. El modelo ve en train la
   textura de las fichas de las fotos con las que se mide. El `test` no es
   "dominio nuevo", es "combinacion nueva de piezas sobre texturas conocidas".
   Se sigue? Si la metrica de T7 sale sospechosamente alta, este es el primer
   sospechoso. Arreglarlo exige capturar fotos de tableros con fichas que no
   esten en el set de sprites.
2. **`test` solo tiene posicion inicial y tablero vacio** (7 + 1 fuentes). Medio
   juego y finales no estan etiquetados: 66 recortes esperan al detector nuevo
   (ver `colab/README.md`, "Siguiente paso"). Hasta entonces, el mAP de test mide
   las 2 situaciones mas faciles del problema.
3. **Los 3 tableros vacios son tambien el fondo** de los compuestos
   (`manifiesto.csv`, columna `fondo`). El fondo de test aparece en train; es
   inevitable con los datos actuales y afecta sobre todo a la precision (falsos
   positivos), que es justo la metrica que se quiere subir.
4. **El clasificador no tiene reales en train.** Es intencional (solo hay 20
   fuentes) pero deja el riesgo de que el dominio camara no este cubierto. Si el
   `recall` por clase en `valid` real sale bajo, la solucion es etiquetar mas
   fotos, no tocar hiperparametros.
5. **Tiempos de Colab estimados, no medidos.** 25-40 min el detector y 30-50 min
   el clasificador en T4 son estimaciones a partir del tamano del dataset; no se
   ejecutaron en GPU desde esta maquina (solo hay CPU local).

## Como regenerar

```powershell
cd F:\TAVO\Projects\ajedrez-v2
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\preparar_colab_v2.py --todo
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe colab\generar_notebooks_v2.py
```

`--todo` borra y rehace `dataset/colab_bundle/` completo. Es determinista con
semilla 42: dos corridas dan los mismos splits y los mismos conteos.
