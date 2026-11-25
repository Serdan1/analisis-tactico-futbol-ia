"""Gráficas del jugador sobre el campo (mplsoccer).

Responsable: Alex
Entrada: posiciones.csv  ·  Salida: PNG en outputs/<clip>/
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from scipy.ndimage import gaussian_filter
from src.fase2_metricas import (
    cargar_posiciones,
    suavizar_trayectoria,
    distancia_y_velocidad,
    resumen_distancia_velocidad,
    ocupacion_zonas,
    detectar_sprints
)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

def dibujar_campo(ax):
    """Dibuja un campo de futbol de 105 x 68 metros."""

    # 1. Configurar los limites del campo
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")

    # 2. Configurar el color del cesped
    ax.set_facecolor("#287449")

    # 3. Dibujar los limites exteriores
    ax.add_patch(Rectangle(
        (0, 0), 105, 68,
        fill=False,
        edgecolor="white",
        linewidth=2
    ))

    # 4. Dibujar la linea central
    ax.plot(
        [52.5, 52.5], [0, 68],
        color="white",
        linewidth=1.5
    )

    # 5. Dibujar el circulo central
    ax.add_patch(Circle(
        (52.5, 34), 9.15,
        fill=False,
        edgecolor="white",
        linewidth=1.5
    ))

    # 6. Dibujar el punto central
    ax.plot(52.5, 34, "wo", markersize=3)

    # 7. Dibujar las areas grandes
    for x in [0, 88.5]:
        ax.add_patch(Rectangle(
            (x, 13.84), 16.5, 40.32,
            fill=False,
            edgecolor="white",
            linewidth=1.5
        ))

    # 8. Dibujar las areas pequenas
    for x in [0, 99.5]:
        ax.add_patch(Rectangle(
            (x, 24.84), 5.5, 18.32,
            fill=False,
            edgecolor="white",
            linewidth=1.5
        ))

    # 9. Dibujar los puntos de penalti
    ax.plot([11, 94], [34, 34],
            "wo", markersize=3)

    # 10. Ocultar los ejes
    ax.set_xticks([])
    ax.set_yticks([])




def mapa_calor(df, ruta_png):
    """Genera un mapa de calor de las posiciones del jugador."""

    # 1. Seleccionar posiciones validas
    datos = df.loc[
        (df["valido"] == 1)
        & df["x_suave"].between(0, 105)
        & df["y_suave"].between(0, 68)
    ]

    # 2. Comprobar que existen posiciones
    if datos.empty:
        raise ValueError("No hay posiciones validas")

    # 3. Obtener las coordenadas
    x = datos["x_suave"].to_numpy()
    y = datos["y_suave"].to_numpy()

    # 4. Crear el histograma bidimensional
    histograma, _, _ = np.histogram2d(
        x,
        y,
        bins=[105, 68],
        range=[[0, 105], [0, 68]]
    )

    # 5. Suavizar el histograma
    calor = gaussian_filter(histograma, sigma=2)

    # 6. Normalizar la intensidad entre 0 y 1
    calor = calor / calor.max()

    # 7. Crear la figura y dibujar el campo
    fig, ax = plt.subplots(figsize=(12, 8))
    dibujar_campo(ax)

    # 8. Representar el mapa de calor
    imagen = ax.imshow(
        calor.T,
        extent=[0, 105, 0, 68],
        origin="lower",
        cmap="hot",
        alpha=calor.T * 0.9,
        interpolation="bilinear",
        vmin=0,
        vmax=1,
        zorder=0
    )

    # 9. Añadir titulo y barra de intensidad
    ax.set_title("Mapa de calor del jugador", fontsize=16)

    barra = fig.colorbar(
        imagen,
        ax=ax,
        fraction=0.035,
        pad=0.03
    )

    barra.set_label("Intensidad relativa")

    # 10. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 11. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    # 12. Cerrar la figura
    plt.close(fig)



def trayectoria(df, ruta_png):
    """Representa la trayectoria del jugador sobre el campo."""

    # 1. Ordenar las posiciones temporalmente
    datos = df.sort_values("frame").copy()

    # 2. Identificar las posiciones validas
    mascara = (
        (datos["valido"] == 1)
        & datos["x_suave"].between(0, 105)
        & datos["y_suave"].between(0, 68)
    )

    validos = datos.loc[mascara]

    # 3. Comprobar que existen posiciones validas
    if validos.empty:
        raise ValueError("No hay posiciones validas")

    # 4. Obtener las coordenadas
    x = datos["x_suave"].where(mascara)
    y = datos["y_suave"].where(mascara)

    # 5. Calcular la posicion media
    x_media = validos["x_suave"].mean()
    y_media = validos["y_suave"].mean()

    # 6. Crear la figura y dibujar el campo
    fig, ax = plt.subplots(figsize=(12, 8))
    dibujar_campo(ax)

    # 7. Dibujar la trayectoria
    ax.plot(
        x, y,
        color="deepskyblue",
        linewidth=1.5,
        alpha=0.85,
        label="Trayectoria",
        zorder=3
    )

    # 8. Marcar la posicion inicial
    ax.scatter(
        validos["x_suave"].iloc[0],
        validos["y_suave"].iloc[0],
        color="lime",
        edgecolor="white",
        s=100,
        label="Inicio",
        zorder=4
    )

    # 9. Marcar la posicion final
    ax.scatter(
        validos["x_suave"].iloc[-1],
        validos["y_suave"].iloc[-1],
        color="red",
        edgecolor="white",
        s=100,
        label="Final",
        zorder=4
    )

    # 10. Marcar la posicion media
    ax.scatter(
        x_media,
        y_media,
        color="gold",
        edgecolor="black",
        marker="*",
        s=250,
        label="Posicion media",
        zorder=5
    )

    # 11. Añadir titulo y leyenda
    ax.set_title(
        "Trayectoria del jugador",
        fontsize=16
    )

    ax.legend(loc="upper left")

    # 12. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 13. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    # 14. Cerrar la figura
    plt.close(fig)



def grafica_velocidad(df, ruta_png):
    """Representa la velocidad del jugador en funcion del tiempo."""

    # 1. Ordenar los datos temporalmente
    datos = df.sort_values("frame").copy()

    # 2. Identificar las velocidades validas
    mascara = (
        np.isfinite(datos["vel_kmh"])
        & (datos["vel_kmh"] >= 0)
        & ~datos["vel_no_creible"]
    )

    validos = datos.loc[mascara]

    # 3. Comprobar que existen velocidades validas
    if validos.empty:
        raise ValueError("No hay velocidades validas")

    # 4. Obtener tiempo y velocidad
    tiempo = datos["t_s"]
    velocidad = datos["vel_kmh"].where(mascara)

    # 5. Obtener las metricas de velocidad
    resumen = resumen_distancia_velocidad(datos)

    velocidad_media = resumen["vel_media_kmh"]
    velocidad_maxima = resumen["vel_max_kmh"]

    # 6. Localizar el instante de velocidad maxima
    indice_max = validos["vel_kmh"].idxmax()
    tiempo_max = validos.loc[indice_max, "t_s"]

    # 7. Crear la figura
    fig, ax = plt.subplots(figsize=(12, 5))

    # 8. Dibujar la velocidad
    ax.plot(
        tiempo,
        velocidad,
        color="royalblue",
        linewidth=1.5,
        label="Velocidad"
    )

    # 9. Dibujar la velocidad media
    ax.axhline(
        y=velocidad_media,
        color="green",
        linestyle="--",
        label=f"Media: {velocidad_media:.2f} km/h"
    )

    # 10. Dibujar el umbral de sprint
    ax.axhline(
        y=25,
        color="red",
        linestyle="--",
        label="Umbral de sprint: 25 km/h"
    )

    # 11. Marcar la velocidad maxima
    ax.scatter(
        tiempo_max,
        velocidad_maxima,
        color="orange",
        edgecolor="black",
        s=80,
        zorder=5,
        label=f"Maxima: {velocidad_maxima:.2f} km/h"
    )

    # 12. Configurar la grafica
    ax.set_title("Velocidad del jugador", fontsize=16)
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Velocidad (km/h)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # 13. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 14. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    # 15. Cerrar la figura
    plt.close(fig)



def mapa_zonas(df, ruta_png):
    """Representa la ocupacion del jugador en las 18 zonas."""
    
    # 1. Obtener los porcentajes de ocupacion
    resultado = ocupacion_zonas(df)
    zonas = resultado["zonas_18"]
    
    # 2. Definir las dimensiones de cada zona
    ancho_zona = 105 / 6
    alto_zona = 68 / 3
    
    # 3. Crear la figura y dibujar el campo
    fig, ax = plt.subplots(figsize=(12, 8))
    dibujar_campo(ax)
    
    # 4. Configurar la escala de colores
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=0, vmax=100)
    
    # 5. Recorrer las 18 zonas
    for fila in range(3):
        for columna in range(6):
    
            # Calcular el numero de zona
            numero = fila * 6 + columna + 1
            nombre = f"Z{numero}"
    
            # Obtener el porcentaje
            porcentaje = zonas[nombre]
    
            # Calcular las coordenadas de la zona
            x = columna * ancho_zona
            y = fila * alto_zona
    
            # Seleccionar el color
            if porcentaje == 0:
                color = "#287449"
            else:
                color = cmap(norm(porcentaje))
    
            # Dibujar el rectangulo
            ax.add_patch(Rectangle(
                (x, y),
                ancho_zona,
                alto_zona,
                facecolor=color,
                edgecolor="white",
                linestyle="--",
                linewidth=0.8,
                alpha=0.85,
                zorder=0.5
            ))
    
            # Escribir el nombre y el porcentaje
            ax.text(
                x + ancho_zona / 2,
                y + alto_zona / 2,
                f"{nombre}\n{porcentaje:.2f}%",
                ha="center",
                va="center",
                color="white" if porcentaje == 0 else "black",
                fontsize=10,
                fontweight="bold",
                zorder=4
            )
    
    # 6. Crear la barra de colores
    escala = ScalarMappable(
        norm=norm,
        cmap=cmap
    )
    
    escala.set_array([])
    
    barra = fig.colorbar(
        escala,
        ax=ax,
        fraction=0.035,
        pad=0.03
    )
    
    barra.set_label("Ocupacion (%)")
    
    # 7. Añadir el titulo
    ax.set_title(
        "Ocupacion de zonas del jugador",
        fontsize=16
    )
    
    # 8. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    
    # 9. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )
    
    # 10. Cerrar la figura
    plt.close(fig)




def trayectoria_por_velocidad(df, ruta_png):
    """Representa la trayectoria coloreada segun la velocidad."""

    # 1. Ordenar los datos temporalmente
    datos = df.sort_values("frame").reset_index(drop=True)

    # 2. Obtener las coordenadas y velocidades
    x = datos["x_suave"].to_numpy()
    y = datos["y_suave"].to_numpy()
    velocidades = datos["vel_kmh"].to_numpy()

    # 3. Identificar las posiciones validas
    posiciones_validas = (
        (datos["valido"] == 1)
        & datos["x_suave"].between(0, 105)
        & datos["y_suave"].between(0, 68)
    ).to_numpy()

    # 4. Crear las listas de segmentos y velocidades
    segmentos = []
    colores_velocidad = []

    # 5. Recorrer los desplazamientos del jugador
    for i in range(1, len(datos)):

        if not posiciones_validas[i - 1]:
            continue

        if not posiciones_validas[i]:
            continue

        if datos["frame"].iloc[i] != datos["frame"].iloc[i - 1] + 1:
            continue

        if not np.isfinite(velocidades[i]):
            continue

        if velocidades[i] < 0:
            continue

        if datos["vel_no_creible"].iloc[i]:
            continue

        # Crear el segmento entre dos posiciones
        segmento = [
            (x[i - 1], y[i - 1]),
            (x[i], y[i])
        ]

        segmentos.append(segmento)
        colores_velocidad.append(velocidades[i])

    # 6. Comprobar que existen segmentos validos
    if not segmentos:
        raise ValueError("No hay desplazamientos validos")

    # 7. Crear la figura y dibujar el campo
    fig, ax = plt.subplots(figsize=(12, 8))
    dibujar_campo(ax)

    # 8. Configurar la escala de colores
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=35)

    # 9. Crear la trayectoria coloreada
    lineas = LineCollection(
        segmentos,
        cmap=cmap,
        norm=norm,
        linewidths=2.5,
        zorder=3
    )

    lineas.set_array(np.array(colores_velocidad))

    # 10. Añadir la trayectoria al campo
    ax.add_collection(lineas)

    # 11. Marcar el inicio
    ax.scatter(
        segmentos[0][0][0],
        segmentos[0][0][1],
        color="lime",
        edgecolor="white",
        s=100,
        label="Inicio",
        zorder=4
    )

    # 12. Marcar el final
    ax.scatter(
        segmentos[-1][1][0],
        segmentos[-1][1][1],
        color="red",
        edgecolor="white",
        s=100,
        label="Final",
        zorder=4
    )

    # 13. Crear la barra de velocidades
    barra = fig.colorbar(
        lineas,
        ax=ax,
        fraction=0.035,
        pad=0.03
    )

    barra.set_label("Velocidad (km/h)")

    # 14. Añadir titulo y leyenda
    ax.set_title(
        "Trayectoria del jugador por velocidad",
        fontsize=16
    )

    ax.legend(loc="upper left")

    # 15. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 16. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    # 17. Cerrar la figura
    plt.close(fig)




def grafica_sprints(df, ruta_png):
    """Velocidad en el tiempo con el umbral y los sprints marcados."""

    # 1. Ordenar los datos temporalmente
    datos = df.sort_values("frame").reset_index(drop=True)

    # 2. Identificar las velocidades validas
    mascara = (
        np.isfinite(datos["vel_kmh"])
        & (datos["vel_kmh"] >= 0)
        & ~datos["vel_no_creible"]
    )

    # 3. Comprobar que existen velocidades validas
    if not mascara.any():
        raise ValueError("No hay velocidades validas")

    # 4. Obtener el tiempo y la velocidad
    tiempo = datos["t_s"]
    velocidad = datos["vel_kmh"].where(mascara)

    # 5. Detectar los sprints
    datos["vel_kmh"] = velocidad
    sprints = detectar_sprints(datos)

    # 6. Crear la figura
    fig, ax = plt.subplots(figsize=(12, 5))

    # 7. Dibujar la velocidad
    ax.plot(
        tiempo,
        velocidad,
        color="royalblue",
        linewidth=1.5,
        label="Velocidad"
    )

    # 8. Dibujar el umbral de sprint
    ax.axhline(
        y=25,
        color="red",
        linestyle="--",
        label="Umbral de sprint: 25 km/h"
    )

    # 9. Marcar cada sprint detectado
    for i, sprint in enumerate(sprints):

        inicio = sprint["inicio_s"]
        duracion = sprint["dur_s"]
        final = inicio + duracion

        ax.axvspan(
            inicio,
            final,
            color="orange",
            alpha=0.3,
            label="Sprint detectado" if i == 0 else None
        )

        ax.text(
            (inicio + final) / 2,
            0.95,
            f"Sprint {i + 1}\n{duracion:.2f} s",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9
        )

    # 10. Configurar la grafica
    ax.set_title(
        f"Sprints del jugador - Total: {len(sprints)}",
        fontsize=16
    )

    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Velocidad (km/h)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # 11. Mostrar un mensaje si no hay sprints
    if not sprints:
        ax.text(
            0.5,
            0.95,
            "No se han detectado sprints",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.9
            )
        )

    # 12. Preparar la ruta de salida
    ruta = Path(ruta_png)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # 13. Guardar la imagen
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    # 14. Cerrar la figura
    plt.close(fig)




def figuras_completas(ruta_csv, carpeta) -> list:
    """Genera todas las figuras del jugador y devuelve sus rutas."""

    # 1. Crear la carpeta de salida
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    # 2. Cargar las posiciones del jugador
    df = cargar_posiciones(ruta_csv)

    # 3. Suavizar la trayectoria
    df = suavizar_trayectoria(df)

    # 4. Calcular distancia y velocidad
    df = distancia_y_velocidad(df)

    # 5. Definir las visualizaciones
    figuras = [
        (mapa_calor, "heatmap.png"),
        (trayectoria, "trayectoria.png"),
        (grafica_velocidad, "velocidad.png"),
        (mapa_zonas, "zonas.png"),
        (trayectoria_por_velocidad, "trayectoria_velocidad.png"),
        (grafica_sprints, "sprints.png")
    ]

    # 6. Lista para almacenar las rutas
    rutas = []

    # 7. Generar todas las visualizaciones
    for funcion, nombre in figuras:

        ruta = carpeta / nombre

        funcion(df, ruta)

        rutas.append(str(ruta))

    # 8. Devolver las rutas generadas
    return rutas

