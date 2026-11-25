"""Fase 2b: métricas del jugador.

Responsable: Alex
Entrada: posiciones.csv  ·  Salida: metricas.json (formatos en docs/formatos.md)

Mientras no haya datos reales, trabajar con un posiciones.csv inventado
con el mismo formato.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import json
from pathlib import Path

UMBRAL_SPRINT_KMH = 25
DURACION_MIN_SPRINT_S = 1.0
VEL_MAX_CREIBLE_KMH = 36


def cargar_posiciones(ruta_csv):
    # 1. Leer el archivo CSV
    df = pd.read_csv(ruta_csv)

    # 2. Definir las columnas obligatorias
    columnas_requeridas = [
        "frame",
        "t_s",
        "track_id",
        "equipo",
        "x_m",
        "y_m",
        "valido"
    ]

    # 3. Comprobar si falta alguna columna
    columnas_faltantes = [
        columna for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    # 4. Ordenar los datos por fotograma
    df = df.sort_values("frame").reset_index(drop=True)

    # 5. Devolver el DataFrame
    return df



def suavizar_trayectoria(df):
    """Limpia y suaviza las coordenadas del jugador."""

    # 1. Crear una copia del DataFrame
    df = df.copy()

    # 2. Convertir las coordenadas a valores numericos
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")

    # 3. Comprobar que las posiciones sean validas
    mascara = (
        (df["valido"] == 1)
        & df["x_m"].between(0, 105)
        & df["y_m"].between(0, 68)
    )

    df["valido"] = mascara.astype(int)

    # 4. Crear las columnas suavizadas
    df["x_suave"] = np.nan
    df["y_suave"] = np.nan

    # 5. Separar los tramos consecutivos
    grupos = (
        mascara.ne(mascara.shift())
        | df["frame"].diff().ne(1)
    ).cumsum()

    # 6. Suavizar cada tramo valido
    for _, tramo in df[mascara].groupby(grupos[mascara]):

        n = len(tramo)

        # Ventana impar de hasta 11 fotogramas
        ventana = min(11, n if n % 2 == 1 else n - 1)

        if ventana >= 3:
            df.loc[tramo.index, "x_suave"] = savgol_filter(
                tramo["x_m"], ventana, 2
            )

            df.loc[tramo.index, "y_suave"] = savgol_filter(
                tramo["y_m"], ventana, 2
            )

        else:
            df.loc[tramo.index, "x_suave"] = tramo["x_m"]
            df.loc[tramo.index, "y_suave"] = tramo["y_m"]

    # 7. Devolver el DataFrame
    return df



def distancia_y_velocidad(df):
    """Calcula distancias y velocidades entre fotogramas."""

    # 1. Crear una copia del DataFrame
    df = df.copy()

    # 2. Calcular las diferencias de coordenadas
    df["dx"] = df["x_suave"].diff()
    df["dy"] = df["y_suave"].diff()

    # 3. Calcular la diferencia de tiempo
    df["dt"] = df["t_s"].diff()

    # 4. Calcular la distancia euclidiana
    df["distancia_m"] = np.sqrt(
        df["dx"] ** 2 + df["dy"] ** 2
    )

    # 5. Comprobar que los fotogramas sean consecutivos
    mascara = (
        (df["valido"] == 1)
        & (df["valido"].shift(1) == 1)
        & (df["frame"].diff() == 1)
        & (df["dt"] > 0)
    )

    # 6. Descartar desplazamientos no validos
    df.loc[~mascara, "distancia_m"] = np.nan

    # 7. Calcular la velocidad en km/h
    df["vel_kmh"] = (
        df["distancia_m"] / df["dt"]
    ) * 3.6

    # 8. Identificar velocidades no creibles
    df["vel_no_creible"] = (
        df["vel_kmh"] > VEL_MAX_CREIBLE_KMH
    )

    # 9. Descartar velocidades y distancias no creibles
    df.loc[
        df["vel_no_creible"],
        ["vel_kmh", "distancia_m"]
    ] = np.nan

    # 10. Devolver el DataFrame
    return df



def resumen_distancia_velocidad(df):
    """Devuelve un resumen de las metricas deportivas."""

    distancia_total = df["distancia_m"].sum(min_count=1)

    tiempo_aceptado = df.loc[
        df["vel_kmh"].notna(), "dt"
    ].sum()

    vel_media = (
        distancia_total / tiempo_aceptado * 3.6
        if tiempo_aceptado > 0 else np.nan
    )

    vel_maxima = df["vel_kmh"].max()

    return {
        "distancia_m": round(float(distancia_total), 2),
        "vel_media_kmh": round(float(vel_media), 2),
        "vel_max_kmh": round(float(vel_maxima), 2)
    }



def detectar_sprints(df):
    """Detecta sprints superiores a 25 km/h durante al menos 1 segundo."""

    # 1. Crear una lista para guardar los sprints
    sprints = []

    # 2. Identificar velocidades superiores al umbral
    mascara = df["vel_kmh"] > UMBRAL_SPRINT_KMH

    # 3. Separar los tramos consecutivos
    grupos = (
        mascara.ne(mascara.shift())
        | df["frame"].diff().ne(1)
    ).cumsum()

    # 4. Recorrer los tramos de velocidad alta
    for _, tramo in df[mascara].groupby(grupos[mascara]):

        # 5. Calcular la duracion del tramo
        duracion = tramo["dt"].sum()

        # 6. Comprobar la duracion minima
        if duracion >= DURACION_MIN_SPRINT_S - 1e-9:

            # 7. Calcular el inicio del sprint
            inicio = (
                tramo["t_s"].iloc[0]
                - tramo["dt"].iloc[0]
            )

            # 8. Calcular la velocidad maxima
            vel_max = tramo["vel_kmh"].max()

            # 9. Guardar el sprint
            sprints.append({
                "inicio_s": round(float(inicio), 2),
                "dur_s": round(float(duracion), 2),
                "vel_max_kmh": round(float(vel_max), 2)
            })

    # 10. Devolver todos los sprints detectados
    return sprints



def ocupacion_zonas(df):
    """Calcula la ocupacion de las zonas del campo."""

    # 1. Seleccionar las posiciones validas
    datos = df.loc[
        (df["valido"] == 1)
        & df["x_suave"].between(0, 105)
        & df["y_suave"].between(0, 68)
    ].copy()

    # 2. Comprobar que existen datos validos
    if datos.empty:
        raise ValueError("No hay posiciones validas")

    # 3. Calcular las dimensiones de cada zona
    ancho_zona = 105 / 6
    alto_zona = 68 / 3

    # 4. Calcular la columna y la fila
    datos["columna"] = (
        datos["x_suave"] / ancho_zona
    ).astype(int).clip(0, 5)

    datos["fila"] = (
        datos["y_suave"] / alto_zona
    ).astype(int).clip(0, 2)

    # 5. Calcular el numero de zona
    datos["zona"] = (
        datos["fila"] * 6
        + datos["columna"] + 1
    )

    # 6. Calcular el porcentaje de las 18 zonas
    zonas_18 = {}

    for zona in range(1, 19):
        porcentaje = (
            (datos["zona"] == zona).mean() * 100
        )

        zonas_18[f"Z{zona}"] = round(
            float(porcentaje), 2
        )

    # 7. Calcular los porcentajes de los tercios
    tercios = {
        "primer_tercio": round(float(
            (datos["x_suave"] < 35).mean() * 100
        ), 2),

        "tercio_medio": round(float(
            (
                (datos["x_suave"] >= 35)
                & (datos["x_suave"] < 70)
            ).mean() * 100
        ), 2),

        "ultimo_tercio": round(float(
            (datos["x_suave"] >= 70).mean() * 100
        ), 2)
    }

    # 8. Calcular los porcentajes de los carriles
    carriles = {
        "banda_izq": round(float(
            (datos["fila"] == 2).mean() * 100
        ), 2),

        "carril_central": round(float(
            (datos["fila"] == 1).mean() * 100
        ), 2),

        "banda_der": round(float(
            (datos["fila"] == 0).mean() * 100
        ), 2)
    }

    # 9. Devolver los resultados
    return {
        "zonas_18": zonas_18,
        "tercios": tercios,
        "carriles": carriles
    }



def posicion_media(df):
    """Calcula la posicion media del jugador."""

    # 1. Seleccionar posiciones validas
    datos = df.loc[
        (df["valido"] == 1)
        & df["x_suave"].between(0, 105)
        & df["y_suave"].between(0, 68)
    ]

    # 2. Comprobar que existen datos
    if datos.empty:
        raise ValueError("No hay posiciones validas")

    # 3. Calcular las medias
    x_media = datos["x_suave"].mean()
    y_media = datos["y_suave"].mean()

    # 4. Devolver las coordenadas
    return [
        round(float(x_media), 2),
        round(float(y_media), 2)
    ]



def analizar_area_rival(df):
    """Calcula la ocupacion y las entradas al area rival."""

    # 1. Identificar posiciones validas
    validos = (
        (df["valido"] == 1)
        & df["x_suave"].between(0, 105)
        & df["y_suave"].between(0, 68)
    )

    # 2. Comprobar que existen posiciones validas
    if validos.sum() == 0:
        raise ValueError("No hay posiciones validas")

    # 3. Identificar posiciones dentro del area
    dentro_area = (
        validos
        & df["x_suave"].between(88.5, 105)
        & df["y_suave"].between(13.84, 54.16)
    )

    # 4. Calcular el porcentaje de ocupacion
    porcentaje_area = (
        dentro_area.sum() / validos.sum()
    ) * 100

    # 5. Detectar entradas al area
    entradas = (
        dentro_area
        & ~dentro_area.shift(1, fill_value=False)
        & validos.shift(1, fill_value=False)
        & df["frame"].diff().eq(1)
    )

    # 6. Contar las entradas
    numero_entradas = int(entradas.sum())

    # 7. Devolver los resultados
    return {
        "area_rival": round(float(porcentaje_area), 2),
        "entradas_area": numero_entradas
    }



def calcular_visibilidad(df):
    """Calcula el tiempo visible y el porcentaje de frames validos."""

    # 1. Obtener el numero total de fotogramas
    total_frames = len(df)

    # 2. Comprobar que existen datos
    if total_frames == 0:
        raise ValueError("El DataFrame esta vacio")

    # 3. Contar los fotogramas validos
    frames_validos = int((df["valido"] == 1).sum())

    # 4. Calcular la duracion de cada fotograma
    intervalos = df["t_s"].diff()
    dt = intervalos[intervalos > 0].median()

    if not np.isfinite(dt):
        raise ValueError("No se puede calcular el intervalo temporal")

    # 5. Calcular el tiempo visible
    tiempo_visible = frames_validos * dt

    # 6. Calcular el porcentaje de fotogramas validos
    porcentaje_validos = (
        frames_validos / total_frames
    ) * 100

    # 7. Devolver las metricas
    return {
        "tiempo_visible_s": round(float(tiempo_visible), 2),
        "frames_validos_pct": round(float(porcentaje_validos), 2)
    }




def exportar_metricas(metricas, ruta_json):
    """Guarda metricas.json con el formato acordado."""

    # 1. Convertir la ruta en un objeto Path
    ruta = Path(ruta_json)

    # 2. Crear la carpeta si no existe
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 3. Convertir las metricas a JSON
    contenido = json.dumps(
        metricas,
        indent=4,
        ensure_ascii=False,
        allow_nan=False
    )

    # 4. Guardar el archivo JSON
    with ruta.open("w", encoding="utf-8") as archivo:
        archivo.write(contenido)




def metricas_completas(ruta_csv, jugador: dict, clip: str = "") -> dict:
    """Calcula todas las metricas del jugador."""

    # 1. Cargar las posiciones
    df = cargar_posiciones(ruta_csv)

    # 2. Calcular la visibilidad
    visibilidad = calcular_visibilidad(df)

    # 3. Suavizar la trayectoria
    df = suavizar_trayectoria(df)

    # 4. Calcular distancias y velocidades
    df = distancia_y_velocidad(df)

    # 5. Obtener las metricas fisicas
    resumen = resumen_distancia_velocidad(df)

    # 6. Detectar sprints
    sprints = detectar_sprints(df)

    # 7. Calcular la ocupacion de zonas
    zonas = ocupacion_zonas(df)

    # 8. Calcular la posicion media
    centroide = posicion_media(df)

    # 9. Analizar el area rival
    area = analizar_area_rival(df)

    # 10. Reunir todas las metricas
    metricas = {
        "clip": clip,
        "jugador": jugador,

        "tiempo_visible_s": visibilidad["tiempo_visible_s"],
        "frames_validos_pct": visibilidad["frames_validos_pct"],

        "distancia_m": resumen["distancia_m"],
        "vel_media_kmh": resumen["vel_media_kmh"],
        "vel_max_kmh": resumen["vel_max_kmh"],

        "sprints": sprints,

        "posicion_media_m": centroide,

        "zonas_pct": {
            "ultimo_tercio": zonas["tercios"]["ultimo_tercio"],
            "area_rival": area["area_rival"],
            "carril_central": zonas["carriles"]["carril_central"],
            "banda_izq": zonas["carriles"]["banda_izq"],
            "banda_der": zonas["carriles"]["banda_der"]
        },

        "entradas_area": area["entradas_area"]
    }

    # 11. Devolver todas las metricas
    return metricas

