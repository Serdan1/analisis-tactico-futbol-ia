"""Utilidades comunes (reutilizadas de las prácticas de clase)."""
import random

import numpy as np


def fijar_semilla(semilla=42):
    """Reproducibilidad, como en las prácticas."""
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch
        torch.manual_seed(semilla)
    except ImportError:
        pass


def leer_frames(ruta_video):
    """Recorre el vídeo frame a frame y devuelve (indice, frame_rgb, fps).
    OpenCV lee en BGR: se convierte a RGB aquí para no olvidarlo después."""
    import cv2
    cap = cv2.VideoCapture(str(ruta_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps
        i += 1
    cap.release()
