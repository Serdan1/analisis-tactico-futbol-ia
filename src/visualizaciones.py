"""Gráficas del jugador sobre el campo (mplsoccer).

Responsable: Alex
Entrada: posiciones.csv  ·  Salida: PNG en outputs/<clip>/
"""


def mapa_calor(df, ruta_png):
    raise NotImplementedError


def mapa_zonas(df, ruta_png):
    raise NotImplementedError


def trayectoria_por_velocidad(df, ruta_png):
    raise NotImplementedError


def grafica_sprints(df, ruta_png):
    """Velocidad en el tiempo con el umbral y los sprints marcados."""
    raise NotImplementedError


def figuras_completas(ruta_csv, carpeta) -> list:
    """Genera todas las figuras del jugador en `carpeta` y devuelve sus rutas.
    La llama main.py; mientras lance NotImplementedError, se salta."""
    raise NotImplementedError
