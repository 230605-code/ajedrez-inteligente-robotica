# T9 — Reconstrucción COMPLETA del dataset limpio (ajedrez-v2)

Fecha: 2026-11-07. Contexto: el dataset anterior se movió a cuarentena
(`F:\TAVO\ajedrezv2_descartado\`) porque los sprites de `caballo_n` salieron de
una casilla de alfil: el par reversible quedó cruzado en la premisa de
etiquetado. Este documento registra la reconstrucción **desde cero** con la
verdad física correcta y los números medidos de cada paso.

**Regla que gobierna todo:** las etiquetas de las fotos reales salen de la FEN
DECLARADA en el manifiesto de cada carpeta del crudo, nunca de un modelo. Los
sprites salen de la POSICIÓN INICIAL FÍSICA. Ningún paso adivina una clase.

Layout físico confirmado por el usuario: **posición inicial estándar**
(b8=caballo_n, c8=alfil_n, f8=alfil_n, g8=caballo_n). No hay cruce.

## 1. Rectificación + etiquetas (`scripts/rectificar_crudo_v2.py`)

252 fotos del crudo nuevo → 219 recortes 800x800 (**86.9 %**, objetivo > 85 %).

| Carpeta | Fotos | Recortes | Fichas por recorte |
|---|---|---|---|
| 01_inicial_intacta | 44 | 28 | 32 |
| 02_inicial_con_jugadas | 146 | 134 | FEN declarada |
| 03_medio_juego | 52 | 47 | FEN declarada |
| 04_finales | 7 | 7 | 4 |
| 05_tablero_vacio | 3 | 3 | 0 |

Métodos: ArUco multi-escala 193, cuadriculado 6, rejilla de piezas 20.

33 fallos (13 %), TODOS por geometría rechazada, ninguno por etiqueta:
ArUco insuficiente (0-1 marcadores), damero con concordancia de color < 0.80,
o rejilla de piezas descentrada/compartida. Están en
`dataset/recortes_800/fallos.csv` con el motivo exacto. Los 16 fallos de
`01_inicial_intacta` son las fotos oscuras (`capture*.jpg` y varias `p01_*`).

Productos por recorte: label YOLO (caja 0.85 de casilla = 0.10625 en coords
normalizadas del 800, la misma medida del lote v2) y **64 recortes de 100x100**
para el clasificador, con la clase de la FEN. Controles: imágenes == labels ==
219, cajas == fichas del FEN en cada recorte, casillas == 64 x 219 = 14,016.

El nombre de cada casilla incluye la CARPETA (`<carpeta>_<stem>_<casilla>.jpg`):
`capture (N).jpg` existe en varias carpetas con el mismo nombre y sin el prefijo
se sobreescribían casillas de fotos distintas.

## 2. Sprites limpios (`scripts/extraer_sprites_limpio.py`)

832 sprites RGBA de 26 de los 28 recortes de `01_inicial_intacta`. Cada uno con
la clase que dice el layout físico de la posición inicial, en la MISMA foto:

| clase | n | clase | n |
|---|---|---|---|
| peon_b / peon_n | 208 / 208 | dama_b / dama_n | 26 / 26 |
| torre_b / torre_n | 52 / 52 | rey_b / rey_n | 26 / 26 |
| caballo_b / caballo_n | 52 / 52 | alfil_b / alfil_n | 52 / 52 |

**Prueba anti-cruce (falla si hay cruce):** `caballo_n` sólo se cortó de b8/g8,
`alfil_n` sólo de c8/f8, `caballo_b` de b1/g1, `alfil_b` de c1/f1. El control
corre en cada ejecución y aborta con `CRUCE` si una clase usa una casilla que no
le toca. **0 descartes** por el filtro de luminancia.

Dos fotos se excluyeron con motivo medido: corrimiento global de rejilla
(-10,-2) y (-12,+7) px, por encima del umbral de 8 px en una casilla de 100 px.
En esas fotos la homografía ArUco venía corrida y el sprite quedaba descentrado;
se descarta la FOTO, nunca la ficha.

Método: el mismo de `extraer_sprites.py` (modelo de fondo cuadrático por paridad
de casilla + Hough para el radio) con el centrado endurecido: en este lote el
centroide del mapa de diferencia se corría hasta 16 px (la base clara de las
fichas negras se confunde con la casilla gris). Ahora el centro maximiza
`0.65 * media del disco + 0.35 * respuesta del anillo` en dos pasadas: la
segunda corrige el corrimiento GLOBAL de la foto (mediana de los 32 centros),
como hace `pipeline_fen.relinear_por_fichas`.

**Límite honesto:** la métrica de fuga por saturación del anillo no sirve para
estas fichas (la base crema de las negras satura ~40-45 y la base negra de las
blancas da saturación inestable a brillo bajo); se guarda como columna de
auditoría, no como criterio. El criterio real es el control de origen por clase
y el preview `dataset/sprites_limpio_preview.jpg`, donde caballo y alfil se
distinguen a simple vista.

## 3. Compuestos (`scripts/generar_compuesto_limpio.py`)

2,500 tableros nuevos sobre el tablero real vacío (3 recortes de
`05_tablero_vacio`), 70,556 cajas, **0 violaciones** de superposición
(distancia mínima entre centros / suma de radios = 1.01).

| Perfil | Tableros | Resolución |
|---|---|---|
| esp | 1,750 (70 %) | 640x480 |
| celular | 375 (15 %) | 1024x768 |
| celular_alto | 375 (15 %) | 1280x1024 |

Modos: 1,250 legales (`python-chess`, partida desde la inicial) y 1,250
aleatorias con damas/reyes sobrerrepresentados (dama y rey ~6,000 cajas cada
uno contra ~4,000 de las menores).

Cada tablero guarda su **FEN en el manifiesto** y su homografía; la verificación
mapea cada caja de vuelta a la placa 100x100 y compara casilla+clase contra la
FEN: **2,500/2,500 exactos**, 70,556 cajas, 0 coordenadas fuera de rango.

Cambio respecto al compositor anterior: el "mantel" del marco sale ahora de
`04_finales` + `05_tablero_vacio` (esquinas sin fichas) y de un parche de
10-18 % del lado menor. Antes salía de las fotos de 32 fichas y del 42-62 %:
el fondo del compuesto traía FICHAS reales sin etiqueta (basura para el
detector) o era el propio damero ampliado.

Casillas del clasificador: 160,000 recortes 100x100 (2,500 x 64), 55.90 % vacías.

## 4. Paquetes de Colab (`scripts/preparar_colab_v2_clean.py`)

| Paquete | Imágenes/casillas | Tamaño |
|---|---|---|
| `detector_v2_clean.zip` | train 4,044 / val 109 / test 110 | 286.7 MB |
| `clasificador_v2_clean.zip` | train 139,200 / valid 17,408 / test 17,408 (2,175/272/272 tableros) | 473.9 MB |
| `entrega_v2.zip` | visor + Stockfish + muestras limpias | 87.9 MB |

`data.yaml`: `nc=12`, names en el orden de `classes.txt`.
Clasificador: 13 clases en `labels/clases.txt` (`vacia` al final), rutas
RELATIVAS, 0 grupos compartidos entre splits (comprobado sobre las 174,016
rutas), 174,016/174,016 rutas presentes en el zip.

Reparto de los reales por POSICIÓN declarada (todas las fotos de la misma FEN
en el mismo split); 01 y 04 tienen una sola posición y se reparten por imagen;
`04_finales` va completo a test (7 fotos). Los recortes reales de `valid`/`test`
del clasificador son los mismos que los del detector.

## 5. Notebooks (`colab/generar_notebooks_v2_clean.py`)

- `colab/ajedrez_v2_detector_desde_cero.ipynb`: `MODO="desde_cero"`, base
  `yolo11n.pt` (COCO), 120 épocas, batch 16, imgsz 640, lr0 0.01, proyecto en
  Drive. **No** arranca de `modelo_sintetico.pt`: ese peso viene de la
  convención contaminada.
- `colab/ajedrez_v2_clasificador_desde_cero.ipynb`: MobileNetV3-Small
  `weights=IMAGENET` sin congelar, 20 épocas, AMP (`torch.amp.autocast` +
  `GradScaler`), pesos por frecuencia inversa, `RandomRotation(12)`.
- 19 celdas de código, 0 magics, `ast.parse` OK en todas, `json.tool` OK en los
  dos archivos.
- `colab/README.md` documenta los cambios declarados respecto a los zips
  anteriores (tabla comparativa) y los repartos exactos.

## 6. Límites conocidos (leer antes de interpretar métricas)

1. **Escala entre train y test del detector.** El test son recortes 800x800 (la
   ficha mide ~85 px); el train son frames donde el tablero completo entra en el
   encuadre (ficha ~48 px en ESP 640x480, ~80 px en celular). Es la geometría
   real del protocolo de captura, no un error del empaquetado: el rectificador
   agranda la ficha al pasar a 800. Un mAP más bajo en test que en valid es
   esperable por esto.
2. **Rectificación imperfecta en fotos sueltas.** 33 fotos no pasaron la
   validación de color/geometría y 2 más se excluyeron de los sprites por
   corrimiento de rejilla. Se pierde cobertura, no se inventan etiquetas.
3. **Borde de tablero en algunos sprites.** En las fotos donde el disco de la
   ficha no llena el círculo medido, el sprite conserva 1-3 px de damero en el
   anillo; el alpha lo atenúa y el compuesto lo vuelve a escalar al 83-88 % de la
   casilla. El preview es la comprobación visual.
4. **Archivos de casilla obsoletos (no usados, no borrados).**
   `dataset/casillas/clasificador/` tiene 14,016 JPG vigentes (0 faltantes) y
   13,632 HUÉRFANOS (29.1 MB) con el nombre viejo `<stem>_<casilla>.jpg`, de la
   primera corrida de rectificación anterior al prefijo de carpeta. En
   `dataset/compuesto_limpio/casillas/clasificador/` hay 160,349 archivos: los
   160,000 vigentes + 349 con nombre repetido en OTRA carpeta de clase (residuo
   de una corrida previa con otro sorteo para el mismo número de tablero).
   Ninguno entra al paquete: el inventario se construye desde
   `manifiesto.csv` / `etiquetas_casillas.csv` con la clase explícita (verificado:
   174,016/174,016 rutas del zip existen y 0 duplicadas). No se borraron para no
   ejecutar acciones destructivas fuera del alcance del worker.
5. **Sin línea base del detector.** El `mAP50 0.546` del modelo anterior se
   midió con etiquetas cruzadas y sobre otro test: no es comparable. La
   referencia es la época 1 de esta corrida.

## 7. Specs cubiertas

- **S2** (fichas y cámara nuevas): todo el dataset sale de las capturas con la
  cámara nueva y las fichas nuevas; la posición inicial se usa como patrón de
  verdad para los sprites.
- **S3** (arquitectura YOLO + clasificador por casilla): (a) y (b) quedan con
  dataset limpio y paquetes listos para entrenar. La comparación de métricas
  YOLO solo vs clasificador solo vs combinado es **T7** y NO se ejecuta aquí.
- **S5** (protocolo de captura): se respetó el crudo tal cual (sin recortar, sin
  renombrar, sin corregir brillo); lo único que se rectifica es el área jugable
  para etiquetar.

No cubre: T0/T1 (cámara en vivo), T6 (pipeline `src/`, ya existente), T7
(comparación), ni la ejecución real del entrenamiento en Colab (la hace el
usuario).
