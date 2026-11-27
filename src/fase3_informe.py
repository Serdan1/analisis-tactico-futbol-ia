"""Fase 3: generación y verificación del informe táctico.

Entrada: diccionario creado por src.fase2_metricas.metricas_completas() o
         leído desde outputs/<clip>/metricas.json.
Salida: informe Markdown y lista de cifras no verificadas.

Configuración opcional del LLM mediante variables de entorno:
  LLM_PROVIDER=ollama (por defecto) u openai
  OLLAMA_URL=http://localhost:11434/api/generate
  OLLAMA_MODEL=llama3.1
  OPENAI_API_KEY=... y OPENAI_MODEL=gpt-4o-mini

No requiere dependencias adicionales: utiliza la biblioteca estándar.
"""
"Pablo Castro"
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def construir_prompt(metricas: dict) -> str:
    """Construye el prompt a partir del JSON generado por la Fase 2."""
    if not isinstance(metricas, dict):
        raise TypeError("metricas debe ser un diccionario")

    obligatorias = (
        "jugador", "tiempo_visible_s", "frames_validos_pct", "distancia_m",
        "vel_media_kmh", "vel_max_kmh", "sprints", "posicion_media_m",
        "zonas_pct", "entradas_area",
    )
    faltan = [clave for clave in obligatorias if clave not in metricas]
    if faltan:
        raise ValueError(f"Faltan claves en metricas.json: {faltan}")

    pct = metricas.get("frames_validos_pct")
    aviso = ""
    if isinstance(pct, (int, float)) and pct < 80:
        aviso = (
            "AVISO OBLIGATORIO: menos del 80 % de fotogramas válidos. "
            "Indica que las métricas son parciales y deben interpretarse con cautela."
        )

    datos = json.dumps(metricas, ensure_ascii=False, indent=2, allow_nan=False)
    return f"""Eres analista de rendimiento y táctica de fútbol. Redacta en español un
informe individual, claro y profesional, en Markdown, usando exclusivamente
el JSON que aparece abajo.

REGLAS:
1. No inventes pases, tiros, goles, ocasiones, posesión, eventos ni estadísticas.
2. Acompaña las afirmaciones cuantitativas con su cifra y unidad.
3. Distingue los datos medidos de las interpretaciones y recomendaciones.
4. No extrapoles el clip a un partido completo.
5. Explica las limitaciones según frames_validos_pct y tiempo_visible_s.
6. Si faltan datos para una sección, omítela o indica que no están disponibles.
7. Usa exactamente los datos de cada elemento de la lista sprints.
8. zonas_pct representa ocupación espacial, no posesión ni porcentaje de acciones.
9. posicion_media_m es [x, y] en metros; no deduzcas una posición nominal solo
   a partir de esas coordenadas.
10. No afirmes mejoras o empeoramientos: no hay comparación entre partidos.
11. No añadas cifras externas ni estadísticas que no estén en el JSON.

ESTRUCTURA SUGERIDA:
- Título y ficha del jugador.
- Resumen del periodo analizado.
- Indicadores físicos: distancia, velocidades y sprints.
- Ocupación espacial y entradas al área.
- Interpretación táctica prudente.
- Calidad de los datos, limitaciones y conclusión.

{aviso}

DATOS (fuente única):
{datos}

Devuelve únicamente el informe en Markdown.""".strip()


def generar_informe(prompt: str) -> str:
    """Llama a Ollama local (por defecto) o a la API de OpenAI."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("El prompt está vacío")

    proveedor = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    timeout = int(os.getenv("LLM_TIMEOUT_S", "180"))

    if proveedor == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        headers = {"Content-Type": "application/json"}
    elif proveedor == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY para usar el proveedor OpenAI")
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": "Redacta informes tácticos basados solo en datos medidos. No inventes cifras."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    else:
        raise ValueError("LLM_PROVIDER debe ser 'ollama' u 'openai'")

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            resultado = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detalle = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Error HTTP del proveedor {proveedor} ({exc.code}): {detalle}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"No se pudo conectar con {proveedor}: {exc.reason}. "
            "Comprueba que Ollama esté iniciado o que tengas conexión."
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo interpretar la respuesta de {proveedor}: {exc}") from exc

    if proveedor == "ollama":
        texto = resultado.get("response", "")
    else:
        choices = resultado.get("choices") or []
        texto = choices[0].get("message", {}).get("content", "") if choices else ""
    if not isinstance(texto, str) or not texto.strip():
        raise RuntimeError(f"{proveedor} no devolvió texto: {resultado}")
    return texto.strip()


# Cardinales españoles frecuentes para detectar cifras escritas con palabras.
_UNIDADES = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3,
    "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintiun": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
    "noventa": 90, "cien": 100, "ciento": 100, "doscientos": 200,
    "trescientos": 300, "cuatrocientos": 400, "quinientos": 500,
    "seiscientos": 600, "setecientos": 700, "ochocientos": 800,
    "novecientos": 900,
}
_ESCALAS = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}
_PALABRAS_NUMERO = set(_UNIDADES) | set(_ESCALAS) | {"y"}
_DIGITOS_RE = re.compile(r"(?<![\w/])[-+]?(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:[,.]\d+)?\s*%?(?!\w)")


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def _parsear_cardinal(tokens: list[str]) -> float | None:
    total = 0
    grupo = 0
    visto = False
    for palabra in tokens:
        if palabra == "y":
            continue
        if palabra in _ESCALAS:
            grupo = max(grupo, 1)
            total += grupo * _ESCALAS[palabra]
            grupo = 0
            visto = True
        elif palabra in _UNIDADES:
            grupo += _UNIDADES[palabra]
            visto = True
        else:
            return None
    return float(total + grupo) if visto else None


def _parsear_cifra(token: str) -> float | None:
    token = token.strip().replace("%", "").replace(" ", "")
    if not token:
        return None
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif token.count(".") > 1:
        token = token.replace(".", "")
    else:
        partes = token.split(".")
        if len(partes) == 2 and len(partes[1]) == 3 and len(partes[0]) <= 3:
            token = "".join(partes)
    try:
        numero = float(token)
    except ValueError:
        return None
    return numero if math.isfinite(numero) else None


def normalizar_numeros(texto: str) -> list[float]:
    """Extrae números como floats: 512,0; 29.8; 87,4 %; quinientos doce."""
    if not isinstance(texto, str):
        raise TypeError("texto debe ser una cadena")

    numeros = []
    for match in _DIGITOS_RE.finditer(texto):
        valor = _parsear_cifra(match.group())
        if valor is not None:
            numeros.append(valor)

    # También detecta cardinales en palabras, como "quinientos doce".
    texto_limpio = _sin_tildes(texto)
    tokens = list(re.finditer(r"[a-z]+", texto_limpio))
    i = 0
    while i < len(tokens):
        if tokens[i].group() not in _PALABRAS_NUMERO:
            i += 1
            continue
        secuencia = []
        while i < len(tokens) and tokens[i].group() in _PALABRAS_NUMERO:
            secuencia.append(tokens[i].group())
            i += 1
        valor = _parsear_cardinal(secuencia)
        if valor is not None:
            numeros.append(valor)
    return numeros


def _numeros_en_metricas(valor: Any) -> list[float]:
    """Recorre diccionarios y listas y recoge sus valores numéricos."""
    if isinstance(valor, bool) or valor is None:
        return []
    if isinstance(valor, (int, float)):
        return [float(valor)] if math.isfinite(float(valor)) else []
    if isinstance(valor, dict):
        return [n for v in valor.values() for n in _numeros_en_metricas(v)]
    if isinstance(valor, (list, tuple)):
        return [n for v in valor for n in _numeros_en_metricas(v)]
    return []


def verificar_cifras(texto: str, metricas: dict, tolerancia: float = 0.05) -> list[float]:
    """Devuelve las cifras del informe no respaldadas por el JSON.

    La tolerancia es relativa (5 % por defecto), con un margen mínimo de 0,05
    para diferencias de redondeo. Las cifras repetidas se devuelven una sola vez.
    """
    if not isinstance(metricas, dict):
        raise TypeError("metricas debe ser un diccionario")
    if tolerancia < 0:
        raise ValueError("tolerancia no puede ser negativa")

    # Elimina numeración Markdown de títulos y listas ("## 1. Resumen", "1. Punto").
    lineas = []
    for linea in texto.splitlines():
        linea = re.sub(r"^\s{0,3}#{1,6}\s*\d+[.)]?\s*", "", linea)
        linea = re.sub(r"^\s*\d+[.)]\s+", "", linea)
        lineas.append(linea)

    cifras_texto = normalizar_numeros("\n".join(lineas))
    cifras_json = _numeros_en_metricas(metricas)
    no_verificadas = []
    for cifra in cifras_texto:
        respaldada = any(
            abs(cifra - dato) <= max(0.05, abs(dato) * tolerancia)
            for dato in cifras_json
        )
        if not respaldada and not any(abs(cifra - previa) <= 1e-9 for previa in no_verificadas):
            no_verificadas.append(round(cifra, 6))
    return no_verificadas


def informe_completo(metricas: dict) -> tuple[str, list]:
    """Genera el informe y devuelve (Markdown, cifras_no_verificadas).

    Si el LLM falla, propaga el error para que main.py guarde prompt.txt y
    conserve metricas.json.
    """
    prompt = construir_prompt(metricas)
    texto = generar_informe(prompt)
    return texto, verificar_cifras(texto, metricas)
