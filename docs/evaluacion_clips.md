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

- Procesar `0bfacc_0`, `121364_0` y `573e61_0` (los descarga el script de Roboflow).
- Error en metros frente a los fotogramas marcados a mano por Pablo (#4): es la única medida independiente.
- Un clip con equipaciones parecidas (dos colores oscuros) para poner a prueba la separación de equipos.
