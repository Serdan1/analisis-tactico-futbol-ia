# Memoria · Evaluación del sistema y comparación con el estado del arte

> Sección para la memoria final (parte de Dani). Las cifras marcadas con **[pendiente]** se completan con
> `tools/evaluar_soccernet.py --saltar 10` (test final) y `tools/gs_hota.py` (métrica oficial).

## 1. Cómo se ha evaluado

El sistema se desarrolló con un único clip (`2e57b9_0`, Bundesliga, plano general). Un resultado medido con los
datos de desarrollo no demuestra que funcione en general, así que se evaluó en tres niveles, **sin tocar
parámetros entre ellos**:

| Nivel | Datos | Verdad | Qué mide |
|---|---|---|---|
| Desarrollo | `2e57b9_0` | corrección a mano del cambio de número | que el método funciona |
| Segundo clip | `08fd33_0` | revisión a ojo | que no está ajustado a un solo partido |
| SoccerNet GSR | 49 clips de test (Bundesliga, cámara de TV) | posición real en metros, rol, equipo y dorsal de cada persona en cada fotograma | error real, comparable con la literatura |

Los parámetros finales (modelo de puntos reentrenado, seguidor prudente) se eligieron mirando los clips SNGS-116 a
125, que por tanto hacen de **validación**. El número final se mide en los **39 clips restantes**, que no se han
mirado para tomar ninguna decisión.

## 2. Resultados

### 2.1 Posición en el campo (homografía)

| Modelo de puntos del campo | Error mediano | < 1 m | < 2 m | < 5 m |
|---|---|---|---|---|
| Roboflow (original) | 7,80 m | 1,5 % | 6,0 % | 28,4 % |
| Reentrenado con SoccerNet, backbone congelado | **1,14 m** | **44,6 %** | **69,9 %** | 88,5 % |
| Reentrenado, fine-tuning completo | 1,26 m | 39,5 % | 67,1 % | 89,4 % |
| Techo: puntos del campo perfectos | 0,44 m | 71 % | 81 % | — |

(10 clips de validación, todas las parejas jugador real ↔ detección.) El fallo principal del sistema original era el
modelo de puntos del campo, entrenado sobre todo con planos generales: con la cámara cerrada colocaba los puntos donde
"suelen estar". Reentrenarlo con etiquetas generadas automáticamente a partir de SoccerNet redujo el error **7 veces**.
El backbone congelado generaliza mejor que el fine-tuning completo aunque este tenga más mAP50 en validación
(0,995 frente a 0,967): con 3.365 imágenes, reentrenar toda la red se ajusta demasiado a los clips de entrenamiento.

### 2.2 Equipos y sentido de ataque

Acierto de equipos del **99,1 al 100 %** en 10 partidos con camisetas de colores distintos; sentido de ataque
correcto en **10 de 10** clips con el modelo reentrenado. Es la parte del sistema que mejor generaliza.

### 2.3 Seguir a un jugador concreto

| Seguidor (10 clips) | Posición correcta | Posición de otro jugador | Precisión |
|---|---|---|---|
| Solo el número del tracker | 35,9 % | 2,1 % | 94 % |
| Re-identificación | 39,1 % | 37,6 % | 51 % |
| Re-identificación prudente (final) | 36,0 % | 4,3 % | 89 % |

Con la cámara cerrada, el tracker pierde a cada jugador muchas veces (hasta 10 números distintos en 30 s) y las
huellas de apariencia (ResNet-18 de ImageNet + color del torso) **distinguen equipos, pero no a dos compañeros con la
misma camiseta** (distancia a la galería: compañeros 0,57; el propio jugador 0,58-0,63; rivales 0,90). El modo
prudente prefiere no dar dato antes que dar el de otro jugador.

### 2.4 Test final (39 clips no usados) **[pendiente]**

| Medida | Resultado |
|---|---|
| Error mediano de posición | [pendiente] |
| Jugadores a < 1 m | [pendiente] |
| Personas detectadas | [pendiente] |
| Equipos | [pendiente] |
| Sentido de ataque | [pendiente] |
| Seguimiento: correcta / de otro jugador | [pendiente] |

### 2.5 Métrica oficial GS-HOTA **[pendiente]**

GS-HOTA es la métrica oficial de SoccerNet GSR. Para cada persona real exige una predicción a menos de 5 m en el
campo, con el mismo número a lo largo del tiempo y con **rol, equipo y dorsal correctos**. Combina detección (DetA) y
asociación en el tiempo (AssA): GS-HOTA = √(DetA · AssA). Se calcula con el evaluador oficial (`sn-trackeval`).

| Sistema | GS-HOTA oficial (con dorsal) | Sin dorsal | Solo localización |
|---|---|---|---|
| Nuestro, modelo de Roboflow | [pendiente] | [pendiente] | [pendiente] |
| Nuestro, modelo reentrenado | [pendiente] | [pendiente] | [pendiente] |
| Sistema base de SoccerNet (2024) | 22,3 (test) | — | — |
| Mejor equipo del reto 2024 | 63,8 (challenge) | — | — |

Nuestro sistema **no lee dorsales**, así que su GS-HOTA oficial es necesariamente muy bajo: cada jugador con dorsal
anotado cuenta como fallo de identidad. La columna "sin dorsal" equivale a la ablación de la tabla 1 del artículo de
SoccerNet y es la comparación justa de lo que sí hacemos.

## 3. ¿Es un sistema comercial? No. ¿Es malo? Tampoco.

### Por qué no es comercial

- **Un error de ~1 m no basta para velocidades y sprints fiables.** Sí basta para mapas de calor, zonas (tercios de
  35 m) y posición media.
- **Seguir a un jugador concreto es la parte débil.** Cuando sale de plano o lo tapan, no se puede recuperar con
  seguridad entre sus compañeros: a esta resolución, dos jugadores con la misma camiseta son indistinguibles sin el
  dorsal.
- **No lee dorsales**, que es lo que exige la métrica oficial para identificar a cada jugador.
- **Detecta ~77-85 % de las personas**: el detector se entrenó con planos generales y pierde jugadores grandes,
  tapados o cortados por el borde.
- **No es tiempo real**: ~4,5 min por clip de 30 s en una GPU de portátil (RTX 3050 Ti, 4 GB).
- Los sistemas profesionales de datos de seguimiento no trabajan con una sola cámara de televisión: usan cámaras
  fijas dedicadas en el estadio, sensores en los jugadores o revisión humana.

### Por qué tampoco es tan malo

- **Es un problema de investigación abierto.** Reconstruir posiciones e identidades desde una sola cámara de
  televisión que se mueve es exactamente la tarea de SoccerNet GSR, presentada en CVPR 2024. Sus autores la describen
  como una tarea difícil que abre el campo a la investigación, y su sistema base obtiene un GS-HOTA del 22,3 % en test.
  El mejor equipo del reto de 2024 llegó a 63,8 en el conjunto challenge.
- **Lo que sí hace, lo hace bien y está medido contra una verdad independiente**: equipos al 99,7 %, sentido de ataque
  10/10 y posición con error mediano de ~1,1 m.
- **La mejora es real y explicada**: de 7,8 m a 1,1 m, localizando la causa con tres pruebas (orientación de los ejes,
  corrección afín, puntos perfectos) y corrigiéndola con un reentrenamiento (Tema 5).
- **Los límites están identificados y cuantificados**, no escondidos: sabemos qué parte falla, cuánto y por qué.

### Qué haría falta para acercarse a un producto

1. **Reconocimiento de dorsales** (SoccerNet tiene un reto y datos para ello): la única señal fiable para separar
   compañeros.
2. **Un modelo de re-identificación entrenado con jugadores** en lugar de ResNet de ImageNet + color.
3. **Calibración de cámara más robusta**: usar también las líneas del campo, no solo 32 puntos, y suavizar la
   homografía en el tiempo.
4. **Reentrenar el detector de jugadores con SoccerNet**, como hicimos con el de puntos del campo.
5. **Optimizar para tiempo real**: modelos más pequeños o TensorRT.

## Fuentes

- Somers et al. *SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap*.
  CVPR Workshops 2024. https://arxiv.org/abs/2404.11335
- *SoccerNet 2024 Challenges Results*. https://arxiv.org/abs/2409.10587
- *From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet Game State Reconstruction*.
  https://arxiv.org/abs/2504.06357
- Evaluador oficial: https://github.com/SoccerNet/sn-trackeval
