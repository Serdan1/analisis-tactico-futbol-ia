"""Ejecuta el sistema completo sobre un clip.

    python main.py --video data/clips/clip_01.mp4

Fases:
    1. Vídeo -> posiciones.csv            (src/fase1_tracking.py + src/fase2_homografia.py)
    2. posiciones.csv -> metricas.json   (src/fase2_metricas.py + src/visualizaciones.py)
    3. metricas.json -> informe          (src/fase3_informe.py)

Cada fase guarda su resultado en outputs/<clip>/, así se puede repetir
una fase sin volver a ejecutar las anteriores.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Análisis táctico de un jugador a partir de vídeo")
    parser.add_argument("--video", required=True, help="Ruta al clip .mp4")
    parser.add_argument("--desde", choices=["fase1", "fase2", "fase3"], default="fase1",
                        help="Empezar desde una fase reutilizando los resultados ya guardados")
    args = parser.parse_args()

    video = Path(args.video)
    salida = Path("outputs") / video.stem
    salida.mkdir(parents=True, exist_ok=True)

    # TODO (Serdan1): conectar las tres fases cuando estén listas.
    raise NotImplementedError("Pipeline todavía no conectado: ver issues del hito 2")


if __name__ == "__main__":
    main()
