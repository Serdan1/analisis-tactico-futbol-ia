# Análisis táctico de fútbol con Visión Artificial e IA Generativa

Proyecto final de la asignatura de Inteligencia Artificial.

A partir de un **clip de vídeo** de un partido, el sistema detecta y sigue a un **jugador concreto**, calcula sus métricas de posicionamiento sobre el campo y genera un **informe táctico escrito por un LLM**, basado solo en datos medidos.

```
Vídeo ──► Fase 1: detección y seguimiento ──► posiciones.csv
                 (YOLO + ByteTrack/BoT-SORT + re-ID)
      ──► Fase 2: homografía y métricas   ──► metricas.json + figuras
                 (píxeles → metros, distancia, sprints, zonas)
      ──► Fase 3: informe con LLM          ──► informe (texto verificado)
```

## Equipo y responsabilidades

| Persona | Parte | Entrada → salida |
|---|---|---|
| [@Serdan1](https://github.com/Serdan1) | Fase 1, homografía, re-ID, integración (`main.py`) y experimentos | vídeo → `posiciones.csv` |
| Alex ([@alex6036](https://github.com/alex6036)) | Métricas, sprints y visualizaciones | `posiciones.csv` → `metricas.json` + figuras |
| Pablo ([@pablousky4](https://github.com/pablousky4)) | Clips, frames marcados a mano y Fase 3 (LLM + verificador) | `metricas.json` → informe |

El plan completo, con fechas y el porqué de cada tarea, está en [`docs/Plan_de_trabajo_equipo.pdf`](docs/Plan_de_trabajo_equipo.pdf). La explicación técnica de cada pieza está en [`docs/Guia_estudio_proyecto_IA_futbol.pdf`](docs/Guia_estudio_proyecto_IA_futbol.pdf).

## Hitos

| Fecha | Hito |
|---|---|
| Mar 1 dic | **Hito 1:** un clip real convertido en `posiciones.csv`; métricas y gráficas funcionando |
| Mar 8 dic | **Hito 2:** primer informe de principio a fin (vídeo → métricas → texto) |
| Lun 14 dic | **Hito 3:** experimentos hechos, figuras finales, sistema estable |
| Jue 17 dic | **Presentación** |

## Cómo seguimos las tareas

Cada tarea es un **issue** con su responsable, su fecha y su hito. El tablero del proyecto (pestaña *Projects*) tiene tres columnas:

- **Por hacer**: todavía no se ha empezado.
- **En proceso**: alguien está trabajando en ella. Muévela aquí cuando empieces.
- **Hecho**: se cierra el issue. Si haces un commit con `Closes #12` en el mensaje, se cierra solo.

Reglas:
1. Cada uno trabaja en su archivo de `src/` y hace commits pequeños y frecuentes.
2. **Los formatos de [`docs/formatos.md`](docs/formatos.md) no se cambian sin avisar a todos.**
3. Si algo se atasca más de un día, se comenta en su issue y se avisa por el grupo.
4. Nunca se suben vídeos, datos de SoccerNet ni pesos de modelos (ya están en `.gitignore`).

## Estructura

```
├── main.py                  # ejecuta todo: python main.py --video data/clips/clip_01.mp4
├── src/
│   ├── fase1_tracking.py    # detección, tracking, equipos, selección y re-ID   (Serdan1)
│   ├── fase2_homografia.py  # keypoints del campo y píxeles → metros           (Serdan1)
│   ├── fase2_metricas.py    # distancia, velocidad, sprints, zonas             (Alex)
│   ├── visualizaciones.py   # mapa de calor, zonas, trayectoria                (Alex)
│   ├── fase3_informe.py     # prompt, LLM y verificador de cifras              (Pablo)
│   └── utils.py             # carga de vídeo, rutas, semillas
├── examples/                # archivos de ejemplo para trabajar en paralelo
├── docs/                    # plan, guía y formatos
├── notebooks/               # pruebas
├── data/                    # clips y datasets (NO se suben)
├── models/                  # pesos .pt (NO se suben)
└── outputs/                 # resultados (NO se suben)
```

## Instalación

Python 3.10+ y GPU NVIDIA con CUDA (recomendado).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/roboflow/sports.git
```

Instala primero PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/ y después el resto.

## Uso

```bash
python main.py --video data/clips/clip_01.mp4
```

Al arrancar se abre el primer fotograma: haz clic sobre el jugador que quieres analizar. Los resultados se guardan en `outputs/<nombre_del_clip>/`.
