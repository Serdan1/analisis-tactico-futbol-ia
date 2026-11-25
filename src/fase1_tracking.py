"""Fase 1: detección, seguimiento, equipos y re-identificación del jugador.

Responsable: Serdan1
Salida: detecciones por frame con track_id, caja y equipo.

Piezas:
- Detección: YOLO con los pesos de fútbol de roboflow/sports.
- Tracking: ByteTrack o BoT-SORT (librería `trackers` o el tracker de Ultralytics).
- Equipos: embeddings de ResNet-18 + KNN con unos pocos recortes etiquetados.
- Jugador objetivo: clic en el primer frame + galería y umbral τ
  (mismo método que la práctica de Face Recognition).
"""


def detectar_y_seguir(ruta_video):
    """Devuelve, por cada frame, las detecciones con track_id, caja y clase."""
    raise NotImplementedError


def asignar_equipos(recortes, etiquetas_ejemplo):
    """Clasifica cada jugador en equipo A o B a partir de sus embeddings."""
    raise NotImplementedError


def seleccionar_jugador(primer_frame):
    """Abre el primer frame y devuelve el track_id del jugador en el que se hace clic."""
    raise NotImplementedError


def reidentificar(galeria, candidatos, tau):
    """Devuelve el track_id del candidato más parecido a la galería si su distancia es <= tau, o None."""
    raise NotImplementedError
