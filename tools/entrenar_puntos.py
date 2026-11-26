"""Reentrenar el modelo de puntos del campo con las etiquetas de SoccerNet (tarea #17).

    python tools/entrenar_puntos.py --congelar      # backbone congelado (solo se entrena el cuello y la cabeza)
    python tools/entrenar_puntos.py                 # fine-tuning completo

Parte de los pesos de Roboflow (football-pitch-detection.pt, YOLOv8x-pose), que
ya saben qué son los 32 puntos: no se empieza de cero (transfer learning, Tema 5).
El mejor modelo se copia a models/puntos_<nombre>.pt.

Memoria: con la RTX 3050 Ti (4 GB), el congelado cabe con --batch 8 (3,1 GB), pero el
completo pide 6,6 GB con 8: usar --batch 4 o 2. Si la columna GPU_mem pasa de ~3,8 GB,
Windows usa la RAM del sistema y cada paso tarda 10 veces más.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congelar", action="store_true", help="congelar el backbone (las 10 primeras capas de YOLOv8)")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640, help="el modelo original se entrenó a 640")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--datos", default="data/puntos_campo/puntos_campo.yaml")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    nombre = "congelado" if a.congelar else "completo"
    modelo = YOLO("models/football-pitch-detection.pt")
    modelo.train(
        data=a.datos, epochs=a.epocas, imgsz=a.imgsz, batch=a.batch, device=a.device,
        freeze=10 if a.congelar else None,
        fliplr=0.0,        # voltear la imagen cambiaría qué punto es cuál (izquierda <-> derecha)
        mosaic=0.0,        # el mosaico parte el campo en trozos: los puntos dejarían de tener sentido
        degrees=0.0, patience=8, workers=2,
        # ruta absoluta: con una relativa, Ultralytics la mete dentro de su propia
        # carpeta de "runs" (que puede estar en otro sitio del disco)
        project=str(Path("runs/puntos").resolve()), name=nombre, exist_ok=True, plots=True,
    )
    carpeta = Path(modelo.trainer.save_dir)           # donde lo ha guardado de verdad
    mejor = carpeta / "weights" / "best.pt"
    destino = Path("models") / f"puntos_{nombre}.pt"
    shutil.copy(mejor, destino)
    print(f"\n✓ Modelo guardado en {destino}")
    print("  Gráficas de entrenamiento en", carpeta)
    print("  Siguiente: python tools/evaluar_homografia.py --modelos", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
