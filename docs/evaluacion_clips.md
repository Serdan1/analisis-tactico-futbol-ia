# Evaluación en varios clips

Todo se desarrolló y ajustó con `2e57b9_0` (τ = 0,80, radio de conflicto 2 m, umbrales de la homografía…).
Los demás clips se procesan **con los mismos parámetros, sin tocar nada**: son el test de verdad.

| Clip | Partido | Papel | Homografía válida | Puntos por fotograma | Error reproyección (mediana) | Jugador | % localizado | Cambios de número | Equipos | Sentido de ataque |
|---|---|---|---|---|---|---|---|---|---|---|
| `2e57b9_0` | blancos vs azules | **desarrollo** | 100 % | 8-17 (media 12,6) | 0,25 m | 7, delantero blanco | 99,2 % | 1 (fotograma 285, bien corregido) | 99,9 % en test (verdad apuntada a mano) | correcto (comprobado a ojo) |
| `08fd33_0` | blancos vs verde lima | test | 100 % | 6-12 (media 8,4) | 0,25 m | 9, atacante blanco | 99,6 % | 0 | correctos a ojo; **nombres mal** con la regla de brillo (corregido) | correcto (comprobado a ojo) |

## Qué ha enseñado el segundo clip

1. **Nombres de los equipos.** La regla "claro / oscuro" por brillo llamó "claro" al verde lima y "oscuro" al blanco:
   V de HSV es max(R, G, B), no la claridad. Ahora cada equipo se nombra por su color real (`blanco`, `verde`…).
2. **Ruido fotograma a fotograma.** En los dos clips, la posición del jugador salta de media 0,15 m entre fotogramas
   (25 por segundo), con picos de 2 m: velocidades instantáneas de más de 150 km/h. Viene de la caja de YOLO (los pies
   "bailan" unos píxeles) y de la homografía, que se recalcula en cada fotograma. Con una ventana de suavizado de
   11 fotogramas siguen quedando picos (la velocidad máxima sale justo en el límite de 36 km/h); con 25 fotogramas (1 s)
   desaparecen. Hay que medirlo con los fotogramas marcados a mano (#4) antes de dar velocidades.
3. **Portero adelantado.** El portero de los blancos está a ~29 m de su portería (su equipo ataca); aun así se asigna
   bien porque el centro de su equipo es el más cercano.

## Pendiente

- Error en metros frente a los fotogramas marcados a mano por Pablo (#4): es la única medida independiente.
- Un clip con equipaciones parecidas (dos colores oscuros) para poner a prueba la separación de equipos.


## SoccerNet GSR: 10 clips de test con la verdad (tarea #25)

Mismos parámetros que en el desarrollo. "Afín" = error si a nuestras posiciones de cada fotograma se les permite una
única corrección afín (diagnóstico). "Puntos perfectos" = error con los 32 puntos del campo sacados de la verdad (techo).

| Clip | Error mediano | Afín | Puntos perfectos | Detectadas | Cambios de número | Equipos | Sentido | Seguimiento re-ID / tracker |
|---|---|---|---|---|---|---|---|---|
| SNGS-116 | 5.04 m | 0.86 m | 0.35 m | 62.2 % | 163 | 99.6 % | sí | 27.0 % / 24.4 % |
| SNGS-117 | 8.13 m | 0.76 m | 2.13 m | 77.6 % | 95 | 99.1 % | no | 21.3 % / 35.1 % |
| SNGS-118 | 4.83 m | 0.96 m | 0.5 m | 81.3 % | 229 | 99.6 % | sí | 23.4 % / 46.0 % |
| SNGS-119 | 16.91 m | 0.67 m | 1.71 m | 75.7 % | 177 | 99.8 % | sí | 41.6 % / 63.2 % |
| SNGS-120 | 6.53 m | 0.69 m | 0.29 m | 86.2 % | 172 | 99.5 % | sí | 20.0 % / 28.6 % |
| SNGS-121 | 10.96 m | 1.32 m | 0.39 m | 84.0 % | 206 | 99.7 % | sí | 14.9 % / 29.1 % |
| SNGS-122 | 7.34 m | 0.96 m | 0.79 m | 70.0 % | 161 | 99.5 % | sí | 14.2 % / 20.1 % |
| SNGS-123 | 6.65 m | 0.66 m | 0.44 m | 85.2 % | 236 | 99.6 % | sí | 9.3 % / 24.7 % |
| SNGS-124 | 8.6 m | 0.57 m | 0.3 m | 78.5 % | 141 | 99.7 % | sí | 35.1 % / 37.8 % |
| SNGS-125 | 7.89 m | 1.36 m | 0.28 m | 66.2 % | 10 | 100.0 % | sí | 34.3 % / 52.6 % |
| **Global** | **7,39 m** (1,6 % < 1 m) | **0,82 m** | **0,44 m** (71 % < 1 m) | 77 % | 159 | 99,1-100 % | 9/10 | 24 % / 36 % |

**Conclusión:** los equipos generalizan; la posición no. La causa es el modelo de puntos del campo de Roboflow con la
cámara cerrada (coloca puntos donde no hay líneas, de forma coherente, y RANSAC no lo detecta). Se está reentrenando
con SoccerNet (tarea #17).
