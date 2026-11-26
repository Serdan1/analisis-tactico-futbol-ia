# Formatos entre fases

Estos dos archivos conectan el trabajo de cada uno. **No se cambian sin avisar a todos**: si cambia una columna o una clave, el código de otra persona deja de funcionar.

## `posiciones.csv` (Fase 1 → Fase 2)

Lo produce Serdan1 y lo usa Alex. Una fila por fotograma del jugador analizado.

Posiciones en **metros** sobre un campo de **105 × 68** (por defecto). Si el estadio mide otra cosa, se indica con
`main.py --campo LARGOxANCHO` (p. ej. `--campo 100x64`); entonces x va de 0 a LARGO e y de 0 a ANCHO, y las medidas
usadas se guardan en `outputs/<clip>/campo.json` (`{"largo_m": 100, "ancho_m": 64}`) y en `resumen.json`
(`campo_m`). Las métricas y figuras deberían leerlas de ahí en vez de suponer 105 × 68.

- **x** va de 0 a 105 a lo largo del campo, de izquierda a derecha tal como lo ve la cámara.
- **y** va de 0 (banda del lado de la cámara) a 68 (banda lejana).
- Es decir, el origen (0, 0) es la esquina izquierda del lado de la cámara, y el campo se dibuja visto desde arriba con la cámara abajo (como `mplsoccer` con `pitch_type="custom"`).

En `posiciones.csv` el campo se gira si el equipo del jugador ataca hacia la izquierda de la cámara (`x → 105 − x`, `y → 68 − y`), para que **siempre ataque hacia x = 105**. El sentido de ataque se calcula automáticamente en la tarea #10 (cada equipo defiende la portería donde está su portero) y se guarda en `outputs/<clip>/equipos.json`.

El nombre de cada equipo lo pone la #10 según el color real de la camiseta: `blanco`, `azul`, `verde`, `rojo`… (si los dos equipos salen del mismo color, `azul claro` / `azul oscuro`).

| Columna | Tipo | Significado |
|---|---|---|
| `frame` | entero | Número de fotograma del clip (empieza en 0) |
| `t_s` | decimal | Tiempo en segundos (`frame / fps`) |
| `track_id` | entero | Identificador del jugador que da el tracker |
| `equipo` | texto | color de la camiseta: `blanco`, `azul`, `verde`… (tarea #10) |
| `x_m` | decimal | Posición x en metros. Vacío si en ese frame no se ve |
| `y_m` | decimal | Posición y en metros. Vacío si en ese frame no se ve |
| `valido` | 0 / 1 | 1 si la posición es fiable (jugador detectado y homografía correcta) |

```csv
frame,t_s,track_id,equipo,x_m,y_m,valido
0,0.00,9,blanco,71.4,30.2,1
1,0.04,9,blanco,71.6,30.3,1
2,0.08,9,blanco,,,0
```

## `metricas.json` (Fase 2 → Fase 3)

Lo produce Alex y lo usa Pablo. Hay un ejemplo completo en [`examples/metricas_ejemplo.json`](../examples/metricas_ejemplo.json).

| Clave | Tipo | Significado |
|---|---|---|
| `clip` | texto | Nombre del vídeo |
| `jugador` | objeto | `id`, `equipo`, `rol` |
| `tiempo_visible_s` | decimal | Segundos con posición válida |
| `frames_validos_pct` | decimal | % de frames con `valido = 1` |
| `distancia_m` | decimal | Distancia recorrida (trayectoria suavizada) |
| `vel_media_kmh`, `vel_max_kmh` | decimal | Velocidades |
| `sprints` | lista | Cada uno con `inicio_s`, `dur_s`, `vel_max_kmh` |
| `posicion_media_m` | [x, y] | Centroide de las posiciones |
| `zonas_pct` | objeto | % de tiempo por zona (`ultimo_tercio`, `area_rival`, `carril_central`, `banda_izq`, `banda_der`) |
| `entradas_area` | entero | Veces que entra en el área rival |

## Parámetros acordados

| Parámetro | Valor |
|---|---|
| Umbral de sprint | 25 km/h durante al menos 1 s |
| Velocidad máxima creíble | 36 km/h (por encima se descarta como error) |
| Zonas | 18 (6 a lo largo × 3 a lo ancho) |
