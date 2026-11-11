# Auditoría del dataset crudo

Fecha: 2026-11-05. Alcance: las 134 fotos de `dataset/crudo/` (T2a). No se movió,
renombró ni modificó ninguna foto original: todo el trabajo es de lectura y copia.

Generado por `scripts/auditar_dataset.py`; artefactos en `dataset/auditoria/`:

- `manifesto.csv`: una fila por foto (134), separador `;`, UTF-8 con BOM.
- `duplicados/`: copias de duplicados exactos (vacía: no hubo).
- `contactos/hoja_01..06.jpg`: hojas de contacto usadas para la revisión visual.

Reproducir:

```powershell
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe F:\TAVO\Projects\ajedrez-v2\scripts\auditar_dataset.py
```

## Resumen ejecutivo

| Indicador | Valor |
|---|---|
| Fotos procesadas | **134** |
| Únicas por SHA-256 (bytes) | **134** |
| Duplicadas exactas | **0** |
| Ilegibles para OpenCV | **0** |
| Tablero completo (revisión visual) | **134** |
| Tablero cortado / ausente (revisión visual) | **0 / 0** |
| Tablero según heurística de contornos | 9 completo, 0 cortado, 125 ausente |

El dato importante: **las 134 fotos tienen el tablero completo en el encuadre**. La
heurística de contornos de `diagnostico_esp32.py` marca 125 como "ausente", pero es
un falso negativo sistemático de esa heurística con este tablero impreso (ver
"Hallazgos"). No usar la columna `tablero` del CSV como filtro automático.

## Conteos por carpeta

Objetivo del README (140): 40 / 40 / 30 / 20 / 10.

| Carpeta | Fotos | Objetivo | Faltan | ESP32 | Celular |
|---|---:|---:|---:|---:|---:|
| 01_inicial | 89 | 40 | -49 (sobran) | 61 | 28 |
| 02_medio_juego | 37 | 40 | 3 | 20 | 17 |
| 03_finales | 5 | 30 | **25** | 5 | 0 |
| 04_dificiles | **0** | 20 | **20** | 0 | 0 |
| 05_tablero_vacio | 3 | 10 | **7** | 0 | 3 |
| **Total** | **134** | **140** | **6 netas** | **86** | **48** |

Las 6 netas engañan: 01_inicial está 49 por encima del objetivo mientras 04_dificiles
está vacío y 03_finales/05_tablero_vacio muy por debajo. El dataset no está
balanceado por tipo de posición.

## Conteos por fuente

| Fuente | Fotos | Resolución | Brillo medio (min/mediana/máx) | Nitidez (min/mediana/máx) |
|---|---:|---|---|---|
| ESP32-CAM | 86 | 640x480 (72), 800x600 (14) | 49.2 / 95.8 / 136.4 | 20 / 78 / 185 |
| Celular | 48 | 2296x4080 | 98.3 / 128.1 / 153.8 | 63 / 156 / 382 |

La fuente se deduce del nombre: `capture*.jpg` y `cap*.jpg` = ESP32; marca de tiempo
`YYYYMMDD_HHMMSS.jpg` = celular. No hubo nombres del pipeline (`01_inicial_001.jpg`)
ni fuentes desconocidas. Ninguna foto está fuera de rango de exposición
(ninguna con brillo < 40 ni > 190).

## Resoluciones y EXIF

| Resolución | Fotos | Fuente |
|---|---:|---|
| 2296x4080 (vertical) | 48 | Celular, EXIF orientación 6 |
| 800x600 | 14 | ESP32 (lote 02_medio_juego) |
| 640x480 | 72 | ESP32 |

- 86 fotos sin EXIF de orientación; 48 con `Orientation=6`.
- **El flag 6 de las 48 fotos de celular es espurio**: al leer los píxeles tal cual
  (OpenCV, sin EXIF) la imagen ya está derecha; aplicar `exif_transpose` la gira 90°
  y deja el tablero de costado. No rotar por EXIF: normalizar el giro con los
  marcadores ArUco del tablero o con el texto "LADO DE LAS NEGRAS/BLANCAS".
- La orientación del tablero varía a propósito (fotos desde el lado de las blancas y
  desde el lado de las negras), así que el giro hay que resolverlo por foto.

## Duplicados

Cero. Los 134 SHA-256 de bytes son distintos, incluso entre fotos de la misma
cámara y posición: cada captura es un archivo único. `dataset/auditoria/duplicados/`
quedó vacío. No se copió ni movió nada.

La clave única del manifiesto es (`carpeta_origen`, `archivo`): el nombre
`capture.jpg` aparece en 01, 02 y 03 con contenido distinto, así que no es un
duplicado. Ninguna fila tiene `duplicado_de`.

## Tablero: completo / cortado / ausente

Revisión visual de las hojas de contacto (`dataset/auditoria/contactos/`):

- **Completo: 134.** Todas muestran las 4 esquinas del tablero dentro del encuadre,
  la mayoría con margen; ninguna pierde un marcador ArUco de esquina por recorte.
- Cortado: 0. Ausente: 0.

Heurística de `diagnostico_esp32.py` (contornos + franjas), tal como pide el
manifiesto:

| Estado | Fotos |
|---|---:|
| completo | 9 |
| cortado | 0 |
| ausente | 125 |
| ilegible | 0 |

La heurística exige un cuadrilátero cerrado de área > 20% de la imagen. El borde del
tablero es una hoja blanca: contra fondos claros y con la espiral del cuaderno
cortando el contorno, `findContours` no cierra el cuadrilátero. En 115 de
las 125 filas "ausente" la corroboración por rejilla (casillas cuadradas de 4 lados)
sí detecta el tablero y el CSV lo anota en `nota`; las otras 10 son fotos donde la
rejilla tiene poco contraste. Cuatro verificaciones a resolución completa
(`01_inicial/capture.jpg`, `02_medio_juego/20261105_051528.jpg`,
`03_finales/capture (3).jpg`, `02_medio_juego/capture.jpg`) confirman tablero
completo con sus 4 marcadores.

## Nitidez y exposición a revisar

Las 8 fotos con menor nitidez (varianza del Laplaciano) son todas ESP32 de 640x480:

| Foto | Nitidez | Brillo |
|---|---:|---:|
| 01_inicial/capture (47).jpg | 19.8 | 76.8 |
| 01_inicial/capture (35).jpg | 22.9 | 74.7 |
| 01_inicial/capture (48).jpg | 22.9 | 76.3 |
| 01_inicial/capture (32).jpg | 23.7 | 73.2 |
| 01_inicial/capture (34).jpg | 24.5 | 72.3 |
| 02_medio_juego/capture.jpg | 28.1 | 74.0 |
| 01_inicial/capture (33).jpg | 30.7 | 72.4 |
| 02_medio_juego/capture (10).jpg | 33.5 | 52.6 |

Las 3 más oscuras son `02_medio_juego/capture (9).jpg` (49.2),
`capture (11).jpg` (49.5) y `capture (10).jpg` (52.6). Ninguna es ilegible; quedan
marcadas para revisar en T3, no descartadas. No se declaró ninguna "descartada":
no hay evidencia de foto inutilizable, y el "ausente" de la heurística no es motivo.

## Hallazgos

1. **La heurística de tablero no sirve para este dataset.** Es un falso negativo
   sistemático (125/134). El tablero sí está en todas. Para filtrar de verdad hace
   falta otra cosa: ArUco (probado y también falla a 640x480 con marcadores de
   ~20 px) o un detector de rejilla entrenado.
2. **EXIF engaña.** 48 fotos con `Orientation=6` cuyos píxeles ya están derechos.
   Cualquier `exif_transpose` las estropea. Es un riesgo directo para T4 (recorte de
   casillas).
3. **Dos cámaras, dos dominios.** 86 fotos ESP32 (640x480/800x600) y 48 de celular
   (2296x4080). Sin separar por fuente, el modelo aprende resolución y no piezas.
   Mezclar ambas en train y validar por fuente, o al menos estratificar.
4. **01_inicial se pasó de largo** (89 vs 40) y **04_dificiles está vacío**.
   El desbalance es de tipos de posición, no de cantidad total.
5. **Cero duplicados exactos**: no hay fotos repetidas que inflen el conteo.

## Recomendación de capturas restantes

Mantener el protocolo del README: cámara fija, tablero completo con margen, dos
sesiones de luz (~60% día / 40% noche), sin recortes ni retoques, y variar 1-3
piezas entre fotos.

### 04_dificiles (0 -> 20; mínimo 5 clases)

| # | Clase de captura | Fotos sugeridas |
|---|---|---:|
| 1 | Reyes solos, enfrentados y cerca del borde | 3 |
| 2 | Dama + rey vs rey (pieza de mucho valor aislada) | 3 |
| 3 | Piezas en el borde y en las esquinas del tablero | 3 |
| 4 | Pieza sobre casilla de su mismo color (blanca en blanca / negra en negra) | 3 |
| 5 | Contraste difícil: sombra dura, contraluz, luz lateral fuerte | 3 |
| 6 | Piezas que se tocan o se superponen parcialmente | 3 |
| 7 | Tablero visto desde el lado de las negras, con piezas cerca del marco | 2 |

Repartir 60/40 entre las dos luces. Estas son las fotos que más aportan al
clasificador porque son las que hoy no existen en el dataset.

### 03_finales (5 -> 30; faltan 25)

| Clase de captura | Fotos sugeridas |
|---|---:|
| Finales de peones (4-8 peones, sin piezas mayores) | 5 |
| Rey + peón vs rey | 4 |
| Torre + peón vs rey / torre vs torre | 4 |
| Alfil o caballo + peones | 4 |
| Reyes + una pieza menor, casillas vacías en ambos flancos | 4 |
| Posición mixta de 4-8 piezas con reyes lejos | 4 |

Las 5 actuales son todas ESP32 de 640x480; conviene que al menos la mitad de las 25
nuevas sean de celular para cubrir la resolución alta.

### Otras carpetas

- **05_tablero_vacio**: faltan 7 para llegar a 10. Alternar tablero limpio y con
  sombras/objetos fuera del tablero (control de falsos positivos).
- **02_medio_juego**: faltan 3 para 40; completar en la sesión de luz que esté
  subrepresentada.
- **01_inicial**: no capturar más; 89 ya cubre de sobra las 40 objetivo (49 de
  exceso). Si se quiere recortar, priorizar las de celular por resolución.

## Limitaciones de esta auditoría

- La revisión de tablero se hizo sobre hojas de contacto (6 hojas, 25 miniaturas
  cada una), no píxel a píxel; las dudas puntuales se verificaron a resolución
  completa. No se abrió cada foto individualmente.
- No se cargó ningún modelo YOLO; el estado del tablero no depende de detecciones.
- `04_dificiles` vacío es conocido y esperado (el usuario lo sabía).
- La nitidez (varianza del Laplaciano) depende de la resolución y del contenido:
  sirve para comparar dentro de la misma fuente, no entre ESP32 y celular.
