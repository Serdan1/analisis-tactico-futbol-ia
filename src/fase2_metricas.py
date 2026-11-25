"""Fase 2b: métricas del jugador.

Responsable: Alex
Entrada: posiciones.csv  ·  Salida: metricas.json (formatos en docs/formatos.md)

Mientras no haya datos reales, trabajar con un posiciones.csv inventado
con el mismo formato.
"""

UMBRAL_SPRINT_KMH = 25
DURACION_MIN_SPRINT_S = 1.0
VEL_MAX_CREIBLE_KMH = 36


def cargar_posiciones(ruta_csv):
    """Lee posiciones.csv en un DataFrame."""
    raise NotImplementedError


def suavizar_trayectoria(df):
    """Suaviza x_m e y_m (Savitzky-Golay o media móvil) antes de calcular nada."""
    raise NotImplementedError


def distancia_y_velocidad(df):
    """Distancia total (m), velocidad media y máxima (km/h)."""
    raise NotImplementedError


def detectar_sprints(df):
    """Tramos con velocidad > UMBRAL_SPRINT_KMH durante al menos DURACION_MIN_SPRINT_S.
    Mismo algoritmo que la detección de habla por energía de la práctica de voz."""
    raise NotImplementedError


def ocupacion_zonas(df):
    """% de tiempo en cada una de las 18 zonas y en tercios y carriles."""
    raise NotImplementedError


def exportar_metricas(metricas, ruta_json):
    """Guarda metricas.json con el formato acordado."""
    raise NotImplementedError


def metricas_completas(ruta_csv, jugador: dict, clip: str = "") -> dict:
    """Une todo lo anterior y devuelve el diccionario de metricas.json.

    La llama main.py:  metricas = metricas_completas("outputs/<clip>/posiciones.csv",
                                                     {"id": 7, "equipo": "claro", "rol": "delantero"},
                                                     clip="2e57b9_0.mp4")
    Mientras lance NotImplementedError, main.py se salta esta fase y lo avisa.
    """
    raise NotImplementedError
