"""Fase 2a: de píxeles a metros sobre el campo.

Responsable: Serdan1
Salida: posiciones.csv con el formato de docs/formatos.md

Piezas:
- Detección de los 32 puntos clave del campo (modelo de roboflow/sports).
- cv2.findHomography con RANSAC, recalculada en cada frame.
- Proyección de los pies del jugador (centro inferior de la caja).
"""


def calcular_homografia(keypoints_imagen, keypoints_campo):
    """Devuelve la matriz H (3x3) o None si hay menos de 4 puntos fiables."""
    raise NotImplementedError


def pixel_a_metros(punto_px, H):
    """Convierte un punto (u, v) en píxeles a (x, y) en metros."""
    raise NotImplementedError


def exportar_posiciones(filas, ruta_csv):
    """Guarda posiciones.csv con las columnas frame, t_s, track_id, equipo, x_m, y_m, valido."""
    raise NotImplementedError
