"""Reentrenar el modelo de puntos del campo con las etiquetas de SoccerNet (tarea #17).

    python tools/entrenar_puntos.py --congelar      # backbone congelado (solo se entrena el cuello y la cabeza)
    python tools/entrenar_puntos.py                 # fine-tuning completo

Parte de los pesos de Roboflow (football-pitch-detection.pt, YOLOv8x-pose), que
ya saben qué son los 32 puntos: no se empieza de cero (transfer learning, Tema 5).
El mejor modelo se copia a models/puntos_<nombre>.pt.

Si la GPU se queda sin memoria, baja --batch (8, 4...).
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
        project="runs/puntos", name=nombre, exist_ok=True, plots=True,
    )
    mejor = Path("runs/puntos") / nombre / "weights" / "best.pt"
    destino = Path("models") / f"puntos_{nombre}.pt"
    shutil.copy(mejor, destino)
    print(f"\n✓ Modelo guardado en {destino}")
    print("  Gráficas de entrenamiento en", Path("runs/puntos") / nombre)
    print("  Siguiente: python tools/evaluar_homografia.py --modelos", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
