# Recortes al tablero (sin fondo) para el clasificador por casilla

Notas de la tarea T4c. Cubren **S3** (dataset para el clasificador por casilla)
y **S5** (dataset base sin retoques: el recorte es un derivado aparte, no una
edicion de las fotos originales).

## 1. Que es y por que

El proyecto necesita DOS datasets que salen de las mismas imagenes:

| Dataset | Que conserva | Para que |
|---|---|---|
| Base (`dataset/sintetico/images`, `dataset/crudo`) | tablero + marco impreso + mesa | detector YOLO: el contexto (papel, textos, mesa) es senal util |
| Derivado (`*/recortes/800x800`) | SOLO el area jugable, cuadrada y de frente | clasificador por casilla: 64 casillas sin ruido de fondo |

El recorte no reemplaza al original: lo complementa. Los originales no se
modifican ni se renombran.

La vista rectificada mide **800x800** (8 casillas de 100 px), el mismo lado que
usa `comun.py` para validar la rejilla en v1/v2. Se endereza con una homografia
de 4 puntos, no con un recorte rectangular.

## 2. Geometria

El area jugable se mide una sola vez en `plantilla/tablero.png`:

| Dato | Valor |
|---|---|
| Caja inclusiva de pixeles | `x 58..549`, `y 51..542` |
| Esquinas (TL, TR, BR, BL) | `(58,51) (550,51) (550,543) (58,543)` |
| Lado rectificado | 800 px |
| Casilla | 100 x 100 px |

Las cajas YOLO del recorte se normalizan dividiendo por 800, igual que el
generador normaliza por el ancho del frame. El recorte es SOLO el tablero: no
lleva margen de papel ni de mesa. Las 64 casillas se separan en un paso
posterior (el clasificador consume casillas de 100x100).

## 3. Sinteticas

```powershell
# lote masivo: continua desde sintetico_00024.jpg
python scripts/generar_sinteticos.py --n 1520 --semilla 7 --perfil tomo --posiciones mezcla
# recorte al tablero
python scripts/recortar_sinteticos.py
```

`generar_sinteticos.py` se extendio asi:

- `--inicio` (opcional). Sin el, el script continua desde la ultima
  `sintetico_NNNNN.jpg` de `images/`; los indices no se pisan.
- `--posiciones` acepta `finales` (4-8 fichas sueltas, con al menos una que no
  sea dama/rey para que la categoria sea `03_finales`).
- `mezcla` reparte por cuotas exactas de `MEZCLA_OBJETIVO`: 610 jugada, 380
  aleatoria, 380 finales, 120 dificil, 30 vacio. Se calcula antes del bucle con
  la semilla, asi el reparto no depende del orden.
- `manifesto.csv` ahora tiene `homografia_recorte`, `ancho`, `alto`. La matriz
  se guarda con 17 digitos significativos (ida y vuelta exacta). Las filas
  anteriores se conservan; solo se agregan o reemplazan las de la corrida.
- `--sin-previews` evita escribir las 3 previews (util en pruebas).

`homografia_recorte` es la composicion `area_jugable -> 800x800` con la inversa
de la homografia que uso el render para proyectar la hoja del tablero. Por eso
las cajas del recorte no dependen de volver a detectar nada: es la MISMA
geometria con la que se generaron las etiquetas originales.

Para las 24 imagenes viejas (filas sin la columna nueva) el recorte se resuelve
con `esquinas_aruco.detectar`: 22 salieron 4/4, y 2 (`sintetico_00009` y
`sintetico_00018`) tenian 3 y 2 marcadores. Para esas dos, y solo para
sinteticas, `recortar_sinteticos.py` completa los marcadores faltantes con
`_marcadores_por_plantilla` (template matching de la propia plantilla). En las
sinteticas el template coincide con los ArUco reales a menos de 1 px.

Resultado de la corrida:

| Dato | Valor |
|---|---|
| Imagenes sinteticas | **1544** (`sintetico_00000` .. `sintetico_01543`) |
| Recortes 800x800 | **1544** |
| Labels del recorte | **1544** |
| Cajas transformadas | **27087** (las mismas 27087 del dataset base) |
| Homografia | 1520 del manifiesto, 22 ArUco, 2 ArUco+plantilla |
| Tiempo del lote | 275 s (~4.6 min) |
| Tiempo del recorte | 23 s |
| Tamano imagenes base | 112.7 MB |
| Tamano recortes | 222.7 MB |

Reparto del lote nuevo (1520): por perfil `esp=740`, `celular=780`; por modo
`jugada=610`, `aleatoria=380`, `finales=380`, `dificil=120`, `vacio=30`. Por
categoria (nombre de carpeta del protocolo): `01_inicial=717`,
`02_medio_juego=108`, `03_finales=540`, `04_dificiles=125`,
`05_tablero_vacio=30`.

## 4. Reales

```powershell
python scripts/recortar_reales.py
```

Lee `dataset/crudo` en modo **solo lectura**, detecta con
`esquinas_aruco.detectar` y escribe:

```
dataset/reales/recortes/800x800/01_inicial/<nombre>.jpg
dataset/reales/recortes/800x800/02_medio_juego/<nombre>.jpg
...
dataset/auditoria/recortes_reales.csv
```

Detalles que importan:

- **Giro**: no se corrige a ojo. El emparejado por ID asigna TL=0, TR=1, BR=2,
  BL=3 aunque la foto venga girada, asi que la vista siempre sale con las
  blancas abajo.
- **EXIF**: los originales de celular tienen EXIF=6 espurio (los pixeles ya
  estan derechos, ver `docs/auditoria-dataset.md`). Se leen con
  `IMREAD_IGNORE_ORIENTATION`; si OpenCV aplicara el EXIF giraria una foto que
  ya viene bien.
- **Sin marcadores no hay recorte**: no se inventa una homografia. La foto
  queda en el CSV con el motivo exacto.

Resultado exacto de esta corrida (134 fotos):

| Carpeta | Fotos | Recortadas | Sin recorte |
|---|---|---|---|
| 01_inicial | 89 | 18 | 71 |
| 02_medio_juego | 37 | 4 | 33 |
| 03_finales | 5 | 0 | 5 |
| 05_tablero_vacio | 3 | 3 | 0 |
| **Total** | **134** | **25 (18.7%)** | **109 (81.3%)** |

Motivos de las 109 sin recorte:

| Motivo | Fotos |
|---|---|
| 2/4 marcadores (`ValueError` del emparejado) | 49 |
| 3/4 marcadores (`ValueError` del emparejado) | 39 |
| 1/4 marcadores (`ValueError` del emparejado) | 18 |
| 0/4 marcadores | 2 |
| 4 marcadores de otro diccionario con ids falsos (`DICT_4X4_100`, ids 1,2,3,85) | 1 |

Por resolucion: `4080x2296` (celular) 19/48 recortadas; `640x480` (ESP) 4/72;
`800x600` (ESP) 2/14. El detector ArUco actual pierde los marcadores de las
fotos ESP con mas frecuencia; es la brecha que ataca **T4b** (multi-escala y
respaldo por rejilla de piezas). No se oculto: el CSV es la evidencia.

Tiempo: 43 s. Recortes reales: 25 jpg (5.2 MB). Las reales no tienen labels: no
hay etiquetado fino de las fotos crudas todavia.

## 5. Controles de calidad de los recortes

Los imprime `recortar_sinteticos.py` al final:

| Control | Resultado |
|---|---|
| recortes == imagenes | 1544 == 1544 |
| cajas en memoria == cajas en disco | 27087 == 27087 |
| fichas del recorte == `n_fichas` del manifiesto | 0 discrepancias |
| pixeles fuera de la imagen original (desborde) | 0 imagenes, 0.0000% |
| perdida de ida y vuelta (10 imagenes al azar) | **1.895e-16** (< 0.001) |
| delta contra homografia recalculada con ArUco | 0.000715 (9/10 detectadas) |

La perdida se mide esquina contra esquina, en coordenadas normalizadas: es la
prueba de que la transformacion no pierde informacion. Un bbox de un
cuadrilatero proyectado cambia de tamano por la perspectiva, pero las esquinas
vuelven al mismo pixel.

Muestras guardadas en `assets/previews` (6):

- `recorte_muestra_sintetico_01.jpg` .. `_03.jpg`
- `recorte_muestra_real_01.jpg` .. `_03.jpg`

Cada muestra lleva la rejilla 8x8, las cajas de las fichas (solo sinteticas) y
un pie con nombre, perfil/carpeta, numero de fichas y % de borde. En las 3
reales el borde es 0.00%: no hay banda negra inventada.

## 6. Que hacer si el ArUco falla en una foto real

1. Buscar la foto en `dataset/auditoria/recortes_reales.csv`; la columna
   `motivo` dice cuantos marcadores se vieron y por que no hubo homografia.
2. NO inventar esquinas a mano para esa foto dentro del dataset: una homografia
   mala desplaza media casilla todas las piezas y envenena el clasificador.
3. Opciones, en orden:
   - repetir la captura con los 4 marcadores visibles y sin reflejo;
   - esperar la deteccion multi-escala de **T4b**;
   - si se corrige a mano, hacerlo en un archivo aparte y documentar la foto,
     nunca sobrescribiendo el CSV de auditoria.
4. Las fotos sin recorte no se borran: siguen en `crudo` y en el detector YOLO
   (que no necesita homografia).

Nota tecnica para T4b: `esquinas_aruco.detectar()` levanta `ValueError` cuando
ve 2 o 3 marcadores (el emparejado geometrico asume 4). Ademas, el bucle de
residuales de `detectar` (alrededor de la linea 492) recorre `pares` como
`(id_plantilla, id_detectado)` cuando `_emparejar` devuelve `(detectado,
plantilla)`; con los ids 0..3 iguales no se nota, pero con ids distintos lanza
`KeyError`. `recortar_reales.py` lo captura y lo reporta como falso positivo de
otro diccionario en vez de cortar la corrida.

## 7. Regenerar

```powershell
# 1) sinteticas nuevas (continua el indice solo)
python scripts/generar_sinteticos.py --n 1520 --semilla 7 --perfil tomo --posiciones mezcla
# 2) recorte sintetico + controles
python scripts/recortar_sinteticos.py
# 3) recorte real + CSV de auditoria
python scripts/recortar_reales.py
```

Los scripts son idempotentes: reescriben los recortes y sus labels, actualizan
la fila de cada imagen en el manifiesto y no tocan `crudo` ni `plantilla`.

## 8. Limites conocidos

- Solo el 18.7% de las fotos reales se pudo rectificar con el detector actual.
  `03_finales` quedo 0/5, asi que las previews reales salen de `01_inicial` y
  `02_medio_juego`.
- Los recortes reales no tienen etiquetas YOLO (falta el etiquetado de las fotos
  reales).
- El recorte sintetico hereda el estilo de ficha de la hoja de stickers; sirve
  para el clasificador por casilla, no para medir el dominio real.
- El split de 64 casillas y el balanceo de clases son de una tarea posterior.
