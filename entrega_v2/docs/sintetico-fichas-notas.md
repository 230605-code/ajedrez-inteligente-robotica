# Sintetico de fichas: esquinas y generador de dataset

Notas de los dos modulos nuevos de la tarea T4a. Cubren S3 (arquitectura v2:
linea base YOLO sobre dataset sintetico) y S5 (dataset objetivo 140 fotos, 12
piezas + `vacia`, sin retoques).

## 1. `scripts/esquinas_aruco.py`

Detecta las 4 esquinas del **area jugable** a partir de los ArUco que estan
fuera del tablero. No codifica a mano ni los IDs ni su ubicacion: los aprende de
`plantilla/tablero.png` en cada corrida.

### Que mide de la plantilla

| Dato | Valor actual | Como se obtiene |
|---|---|---|
| Diccionario | `DICT_4X4_50` | primer diccionario que devuelve 4 marcadores |
| IDs por lado | `TL=0, TR=1, BR=2, BL=3` | centro de cada marcador vs. centro del conjunto |
| Caja del area jugable | `x 58..549`, `y 51..542` | mayor componente de color plano del tablero |
| Esquinas (continuas) | `(58,51) (550,51) (550,543) (58,543)` | borde exterior de la caja |

El tablero mide 492x492 px = 8 casillas de 61.5 px. Las casillas azul y gris son
planas (`BGR 167,131,94` y `188,184,180`); el blanco de papel (255) y la tinta
negra de textos/marcadores quedan fuera del rango medio, asi que la caja sale
sola. Se descarta el antialias recuperando los dos colores dominantes.

### Como deriva las esquinas en una foto

La plantilla y la foto son dos proyecciones del mismo plano fisico, por lo que
existe una homografia entre ambas. Con los 4 centros de los marcadores como
correspondencias se ajusta esa homografia y se aplica a las esquinas medidas en
la plantilla. Asi las esquinas salen aunque el tablero este en perspectiva y los
marcadores queden fuera del area.

- Si los 4 IDs de la foto coinciden con la plantilla, se emparejan por ID.
- Si los IDs no coinciden (otra impresion del mismo layout), se emparejan por
  posicion geometrica TL/TR/BR/BL.
- Si faltan marcadores no se inventa una homografia: se devuelve confianza baja.

### Confianza

`confianza = cobertura * exp(-(rms_relativo / 0.25)^2)`, donde la cobertura es
`marcadores_emparejados / 4` y `rms_relativo` es el error de reproyeccion de las
4 esquinas de cada marcador, dividido por el lado medio del marcador. La
homografia se ajusta con 4 centros (8 grados de libertad) y cada marcador aporta
4 esquinas (16 ecuaciones), asi que el residual es una senal real. En la
plantilla da `rms 0.00 px` y `confianza 1.000`.

### API

```python
import esquinas_aruco

plantilla = esquinas_aruco.analizar_plantilla()   # rutas o ndarray aceptados
resultado = esquinas_aruco.detectar("foto.jpg", plantilla)
resultado.esquinas      # 4x2 float32, TL, TR, BR, BL del area jugable
resultado.ids           # ids en el mismo orden TL, TR, BR, BL
resultado.confianza     # 0..1
resultado.marcadores    # dict id -> 4x2
resultado.metodo        # aruco | aruco-parcial | fallback-plantilla
esquinas_aruco.dibujar_debug("foto.jpg", resultado, plantilla)  # BGR anotada
```

### Fallback sin `cv2.aruco`

Si el build de OpenCV no trae el modulo `aruco`, se usa `template matching` de
los 4 parches de la plantilla a varias escalas y giros de 90 grados. Es una red
de seguridad: no cubre giros arbitrarios (por ejemplo 45 grados). En este
entorno `cv2.aruco` SI existe (OpenCV 5.0.0) y la ruta principal es la que corre.

### CLI

```
python scripts/esquinas_aruco.py
python scripts/esquinas_aruco.py --imagen foto.jpg --debug-salida debug.jpg
python scripts/esquinas_aruco.py --fallback      # prueba el detector por plantilla
```

## 2. `scripts/generar_sinteticos.py`

Compone fotos sinteticas con el tablero impreso y las fichas nuevas, y escribe
etiquetas YOLO listas para entrenar.

### Recorte de las fichas

`plantilla/fichas.png` trae 4 filas de 8 discos. La deteccion es por
`HoughCircles`, agrupando en filas y columnas por saltos de 1D. Luego:

1. El brillo del disco separa los dos disenos: **disco negro = ficha blanca
   (`_b`)** y **disco blanco = ficha negra (`_n`)** (coincide con el diseno
   pedido: blanca = disco negro con simbolo blanco).
2. Dentro de cada color, la fila con menor variacion interna es la de peones (8
   simbolos identicos); la otra es la fila de piezas mayores, en el orden
   `torre, caballo, alfil, dama, rey, alfil, caballo, torre`.
3. El radio del disco se mide **una sola vez** sobre los discos negros, donde el
   borde es un cruce limpio de oscuro a claro, y se reutiliza en los 32 recortes.
   Medirlo en los discos blancos falla porque su simbolo negro tapa el perfil.
4. Cada sprite se guarda RGBA con mascara circular y se dibuja con rotacion libre
   de 0 a 360 grados (`--` no hay angulo fijo: es la variacion que pidio el
   usuario).

Clases y orden (identico a `classes.txt`; no se reordena):

| id | clase | id | clase |
|---|---|---|---|
| 0 | peon_b | 6 | peon_n |
| 1 | torre_b | 7 | torre_n |
| 2 | caballo_b | 8 | caballo_n |
| 3 | alfil_b | 9 | alfil_n |
| 4 | dama_b | 10 | dama_n |
| 5 | rey_b | 11 | rey_n |

### Perfiles de camara

| Perfil | Frame | Relleno de la hoja | Perspectiva | Postproceso |
|---|---|---|---|---|
| `esp` | 640x480 | 0.90-1.00 del encuadre | 0.012 | blur leve, +contraste, JPEG 65-82 |
| `celular` | 1024x768 | 0.64-0.84 | 0.05 | menos blur, mas libertad de luz, JPEG 80-93 |
| `tomo` | — | elige `esp` o `celular` por imagen | — | — |

Se renderiza directo a la resolucion del perfil (no se genera chico y se
agranda). La hoja se dibuja a resolucion nativa y se deforma una sola vez con
`warpPerspective`, asi que las fichas nunca se amplian mas de ~5%.

### Posiciones

- `jugada` (alias `juego`): `python-chess`, desde la posicion inicial, 4 a 40
  jugadas legales al azar. Blancas abajo (fila 7 = rank 1), negras arriba,
  coherente con los textos "LADO DE LAS BLANCAS" / "LADO DE LAS NEGRAS".
- `aleatoria`: fichas sueltas en casillas al azar, con casillas vacias. La
  cantidad se reparte entre finales, medio juego e inicial para que el dataset
  no quede lleno siempre.
- `mezcla` (default): 8% tablero vacio, 14% dificil (2-5 reinas/reyes, mitad en
  bordes), 38% jugada y 40% aleatoria. Da ~50/50 entre los dos modos principales
  y cubre `04_dificiles` y `05_tablero_vacio`.

### Salida

```
dataset/sintetico/
├── images/sintetico_00000.jpg
├── labels/sintetico_00000.txt     # clase xc yc w h normalizados
└── manifesto.csv
assets/previews/sintetico_00000_preview.jpg   # 3 previews con cajas
```

`manifesto.csv` trae `imagen, n_fichas, perfil, posiciones, categoria, semilla`.
`categoria` usa los nombres de carpeta del protocolo (`01_inicial`,
`02_medio_juego`, `03_finales`, `04_dificiles`, `05_tablero_vacio`) para saber a
que bloque del dataset real se parece cada sintetica.

### CLI

```
python scripts/generar_sinteticos.py --n 24 --semilla 42 --perfil tomo
python scripts/generar_sinteticos.py --n 120 --perfil tomo --posiciones mezcla
python scripts/generar_sinteticos.py --n 30 --posiciones jugada --semilla 7
python scripts/generar_sinteticos.py --n 10 --perfil esp --perspectiva 0.0
```

Opciones: `--n`, `--plantilla-fichas`, `--plantilla-tablero`, `--perfil`
(`esp|celular|celulo|tomo`), `--semilla`, `--salida`, `--posiciones`
(`aleatoria|jugada|juego|mezcla`), `--perspectiva`.

No instala dependencias: usa `cv2`, `numpy` y `python-chess` del venv de v1
(`F:\TAVO\Projects\ajedrez-yolo\.venv`). No se uso pip ni PIL: OpenCV cubre
resize, rotacion, warp y JPEG.

### Limites conocidos

- Las fichas son sprites de una hoja de 533x370 (discos de ~51 px). A 1024x768
  se dibujan casi a escala 1:1; no se inventa detalle que la hoja no tiene.
- La perspectiva es chica a proposito: la perspectiva real se aprende con fotos
  reales, no con sinteticas exageradas.
- Una prediccion con el `best.pt` de v1 sobre estas imagenes es solo un chequeo
  de sanidad: el estilo de ficha es nuevo, asi que los aciertos bajos no indican
  un error del generador.
