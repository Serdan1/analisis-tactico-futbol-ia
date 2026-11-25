"""Fase 3: informe táctico con un LLM.

Responsable: Pablo
Entrada: metricas.json (ejemplo en examples/metricas_ejemplo.json)
Salida: informe en Markdown o PDF

Reglas del prompt:
- Usar SOLO los datos del JSON; no inventar eventos.
- Acompañar cada afirmación de la cifra en la que se basa.
- Si frames_validos_pct < 80, avisar de que las métricas son parciales.
"""


def construir_prompt(metricas):
    """Devuelve el prompt (instrucciones + JSON) para el LLM."""
    raise NotImplementedError


def generar_informe(prompt):
    """Llama al LLM (API o local con Ollama) y devuelve el texto."""
    raise NotImplementedError


def normalizar_numeros(texto):
    """Convierte '512,0', '512 m' o 'quinientos doce' en 512 antes de comparar.
    Mismo problema que el '14' frente a 'catorce' de la práctica de voz."""
    raise NotImplementedError


def verificar_cifras(texto, metricas, tolerancia=0.05):
    """Devuelve las cifras del informe que no aparecen en el JSON."""
    raise NotImplementedError
