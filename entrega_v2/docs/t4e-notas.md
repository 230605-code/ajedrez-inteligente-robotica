# T4e — Derivado completo de recortes del dataset compuesto

Cubre **S3** (datos para el clasificador por casilla de la arquitectura v2) y
**S5** (dataset objetivo: el compuesto sigue siendo el mismo material, aquí
solo se deriva lo que el clasificador consume). No toca las imágenes del
compuesto: agrega el recorte al tablero y las 64 casillas de cada una.

Antes de esta tarea el dataset compuesto tenía las 1560 imágenes tal como las ve
el detector YOLO (con fondo, marco de mesa y perspectiva) y solo **128 PNG de
muestra** (las 64 casillas de la primera imagen de cada perfil). Ahora tiene el
derivado completo: 1560 recortes y 99 840 casillas.

## Qué genera

```powershell
$p = "F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe"
& $p scripts\recortar_compuestos.py --verificar 300      # lote completo
```

| Salida | Cantidad | Tamaño |
|---|---|---|
| `dataset/reales_compuesto/recortes/800x800/<compuesto>.jpg` | **1560** | 188,3 MB |
| `dataset/reales_compuesto/casillas/clasificador/<clase>/<compuesto>_<casilla>.jpg` | **99 840** | 269,3 MB (271,3 MB con los 128 PNG de muestra) |
| `dataset/reales_compuesto/etiquetas_casillas.csv` | **99 840** filas | 10,0 MB |
| `assets/previews/casilla_muestra_01..03.jpg` | 3 hojas 8x8 | 0,4 MB |

Total del derivado: **≈ 458 MB** (468 MB con el CSV). El dataset compuesto completo queda en
**584,5 MB** y en `F:` quedaban **11,44 GiB** libres al terminar. Presupuesto
del pedido: ≤ 3 GB → se usó ~15 %.

## Cómo se genera la geometría (sin detectar nada)

La vista rectificada sale de `homografia_recorte` de `manifiesto.csv`, la misma
matriz con la que `compilar_compuestos.py` proyectó el plano del tablero al
frame. Por eso las casillas caen exactamente donde el compositor pegó cada
ficha: no hay una segunda detección que pueda introducir corrimientos.

- `warpPerspective` a 800x800 con `INTER_CUBIC` y `BORDER_REPLICATE`.
- Las 64 casillas se obtienen cortando el recorte en una rejilla 8x8 de 100 px
  (fila 0 = rank 8, columna 0 = columna a). Es la misma geometría que el
  compositor usaba para sus recortes por celda, y es lo que verá el
  clasificador en producción. Además es 64 veces más barato que deformar cada
  celda por separado.
- La clase de cada celda sale de `rotaciones.csv` (una fila por ficha con
  `imagen`, `casilla` y `clase`). Todo lo que no aparece ahí es `vacia`. No se
  infiere de la imagen: la etiqueta viaja con la geometría.
- Formatos: JPG calidad 92 (pedido) para recortes y casillas. Los 128 PNG de
  muestra del compositor quedan intactos: el derivado usa el mismo nombre base
  con extensión `.jpg`, así que no colisiona ni los borra.

## Reparto por clase

| Clase | Casillas | Clase | Casillas |
|---|---:|---|---:|
| peon_b | 6 888 | peon_n | 6 939 |
| torre_b | 2 556 | torre_n | 2 557 |
| caballo_b | 2 540 | caballo_n | 2 512 |
| alfil_b | 2 521 | alfil_n | 2 592 |
| dama_b | 3 777 | dama_n | 3 729 |
| rey_b | 3 743 | rey_n | 3 771 |
| **vacia** | **55 715** | **Total** | **99 840** |

Ocupadas: **44 125**, exactamente las filas de `rotaciones.csv` y la suma de
`n_fichas` del manifiesto. `vacia` es el 55,8 % del dataset, como corresponde a
posiciones de ajedrez (32 fichas como máximo en 64 casillas).

## Verificación observada

| Control | Resultado |
|---|---|
| Recortes 800x800 | 1560 archivos, todos `800x800x3` al releerlos |
| Casillas 100x100 | 99 840 archivos, todos `100x100x3` al releerlos (lectura completa, no muestreo) |
| CSV | 99 840 filas; 0 inválidas (clase, casilla, fila/columna y archivo existen) |
| Carpetas | 13 (12 piezas + `vacia`); el conteo por carpeta coincide con el CSV |
| Clases vs `rotaciones.csv` | 0 diferencias por clase; 0 pares imagen+casilla duplicados |
| Borde inventado en el recorte | máx. 0,0000 % (ninguna imagen con píxeles fuera del original) |
| Homografías faltantes/inválidas | 0 de 1560 |
| `vacia` sin ficha (varianza) | desv. 33,09 vs 47,24 de las ocupadas → OK |

Los PNG de muestra (128) siguen en su carpeta y ya no aparecen en el CSV: todas
las filas apuntan al JPG derivado.

Brillo (escala de grises, media y desviación pooled por clase):

| Clase | Media | Desv | Clase | Media | Desv |
|---|---:|---:|---|---:|---:|
| peon_b | 131,39 | 47,75 | peon_n | 158,32 | 45,41 |
| torre_b | 124,52 | 47,21 | torre_n | 155,60 | 45,62 |
| caballo_b | 132,08 | 44,30 | caballo_n | 158,05 | 43,56 |
| alfil_b | 123,64 | 46,77 | alfil_n | 159,97 | 45,60 |
| dama_b | 132,02 | 44,78 | dama_n | 159,54 | 44,18 |
| rey_b | 137,92 | 42,87 | rey_n | 153,68 | 42,16 |
| vacia | 153,34 | **33,09** | | | |

La clase `vacia` tiene la menor desviación: el damero liso varía menos que una
celda con ficha. Es el control barato de que no quedó una ficha etiquetada como
vacía.

Previews: `casilla_muestra_01..03.jpg` son hojas 8x8 de tres compuestos al azar
del lote con la clase escrita sobre cada casilla. Sirven para revisar a ojo que
la rejilla y las etiquetas coinciden.

## Tiempo

| Paso | Tiempo |
|---|---|
| Lote completo (1560 recortes + 99 840 casillas + CSV + verificación) | **104,0 s** (~67 ms/compuesto; 127,8 s en la primera corrida con caché fría) |
| Relectura independiente de las 99 840 casillas | 269,9 s (solo verificación) |

El pedido ponía 10 min como techo para el lote: se cumplió con margen. El
tiempo lo domina la escritura de 99 840 JPG, no la geometría.

## Cómo re-ejecutar (y lotes)

```powershell
$p = "F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe"
& $p scripts\recortar_compuestos.py                                   # todo
& $p scripts\recortar_compuestos.py --inicio 1 --fin 200              # lote
& $p scripts\recortar_compuestos.py --inicio 201 --fin 400 --sin-previews
& $p scripts\recortar_compuestos.py --verificar 0                     # sin relectura
```

`--inicio` y `--fin` son 1-based e inclusivos sobre el orden de
`manifiesto.csv`. El script es idempotente: reescribe los recortes y casillas
del lote. El CSV conserva las filas de las imágenes **fuera** del lote
(apuntando al JPG o PNG que exista) y reemplaza completas las del lote, así que
un lote parcial no deja rutas rotas; la corrida completa escribe las 99 840
filas.

Salida esperada de una corrida correcta:

```
RESULTADO: ok
```

`--verificar N` relee N casillas al azar para confirmar que decodifican
`100x100x3`; el script además valida todas las filas del CSV y cuenta el borde
inventado de cada recorte.

## Limitaciones conocidas

- Las casillas del perfil ESP vienen de un recorte 800x800 construido desde
  640x480: son celdas de ~55 px ampliadas a 100. Es exactamente lo que verá el
  clasificador con esa cámara, pero no agrega detalle que la fuente no tenga.
- El compuesto ya es JPG: el derivado vuelve a comprimir (calidad 92). Para
  clasificación liviana es irrelevante; no usar estos recortes para medir
  nitidez de la cámara.
- Los discos de las fichas de las esquinas del tablero están cortados en el
  compuesto original (el compositor rellena replicando el borde). Aparecen
  cortados también en las casillas de borde; es fiel al material de origen.
- El reparto de clases es el del compuesto: `peon_*` domina porque hay 16
  peones por posición legal. El balanceo (pesos o submuestreo) es del
  entrenamiento, no de este derivado.
- Los 128 PNG de muestra siguen ocupando ~2,0 MB y no se referencian desde el
  CSV. Si se quiere un único formato, se pueden borrar sin afectar el dataset;
  se conservan por pedido.
