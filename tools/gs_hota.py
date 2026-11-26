"""Métrica oficial GS-HOTA de SoccerNet Game State Reconstruction (tarea #25).

Instalar una vez el evaluador oficial (fork de TrackEval hecho por SoccerNet):
    pip install git+https://github.com/SoccerNet/sn-trackeval.git

    python tools/gs_hota.py --base outputs --rango 116 125 --etiqueta roboflow
    python tools/gs_hota.py --base outputs/puntos_congelado_mi4 --rango 116 125 --etiqueta congelado
    python tools/gs_hota.py --base outputs --rango 126 200 --etiqueta test_final

Exporta cada clip al formato oficial (src/exportar_gsr.py) y pasa el evaluador
tres veces:
    oficial      rol + equipo + dorsal   (es el número del leaderboard; nosotros no
                                          leemos dorsales, así que sale muy bajo)
    sin_dorsal   rol + equipo            (como la ablación de la tabla 1 del artículo)
    solo_campo   sin atributos           (localización en el campo + identidad del track)
GS-HOTA = sqrt(DetA · AssA): DetA mide encontrar a cada persona a < 5 m con los
atributos correctos; AssA, mantener el mismo número a lo largo del tiempo.
Resultados en outputs/gsr/<etiqueta>_gs_hota.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.exportar_gsr import exportar_clip  # noqa: E402

ORIGEN = Path("data/soccernet/SN-GSR-2025")
VARIANTES = {"oficial": dict(USE_ROLES=True, USE_TEAMS=True, USE_JERSEY_NUMBERS=True),
             "sin_dorsal": dict(USE_ROLES=True, USE_TEAMS=True, USE_JERSEY_NUMBERS=False),
             "solo_campo": dict(USE_ROLES=False, USE_TEAMS=False, USE_JERSEY_NUMBERS=False)}


def evaluar(carpeta_trackers, etiqueta, clips, opciones):
    import trackeval
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False,
                    OUTPUT_DETAILED=False, PLOT_CURVES=False, DISPLAY_LESS_PROGRESS=True)
    data_cfg = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    data_cfg.update(GT_FOLDER=str(ORIGEN), TRACKERS_FOLDER=str(carpeta_trackers), TRACKERS_TO_EVAL=[etiqueta],
                    TRACKER_SUB_FOLDER="", SPLIT_TO_EVAL="test", SKIP_SPLIT_FOL=True,
                    SEQ_INFO={c: None for c in clips}, PRINT_CONFIG=False, **opciones)
    evaluador = trackeval.Evaluator(eval_cfg)
    datos = [trackeval.datasets.SoccerNetGS(data_cfg)]
    res, _ = evaluador.evaluate(datos, [trackeval.metrics.HOTA({"THRESHOLD": 0.5})])
    combinado = res["SoccerNetGS"][etiqueta]["COMBINED_SEQ"]
    h = (combinado["person"] if "person" in combinado else next(iter(combinado.values())))["HOTA"]
    return {k: round(100 * float(np.mean(h[k])), 2) for k in ["HOTA", "DetA", "AssA"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs", help="carpeta con outputs/<clip>/ (posiciones_todos, equipos)")
    ap.add_argument("--etiqueta", required=True, help="nombre del sistema en la tabla")
    ap.add_argument("--rango", type=int, nargs=2, metavar=("DESDE", "HASTA"), help="SNGS-DESDE a SNGS-HASTA")
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()

    clips = a.clips or sorted(p.name for p in Path(a.base).glob("SNGS-*") if (p / "equipos.json").exists())
    if a.rango:
        clips = [c for c in clips if a.rango[0] <= int(c.split("-")[1]) <= a.rango[1]]
    clips = [c for c in clips if (ORIGEN / c / "Labels-GameState.json").exists()]
    if not clips:
        print("✗ No hay clips con resultados y verdad"); return 1
    try:
        import trackeval  # noqa: F401
    except ImportError:
        print("✗ Falta el evaluador oficial: pip install git+https://github.com/SoccerNet/sn-trackeval.git"); return 1

    carpeta = Path("outputs/gsr")
    for c in clips:
        exportar_clip(Path(a.base) / c, ORIGEN / c / "Labels-GameState.json", carpeta / a.etiqueta / f"{c}.json")
    print(f"Exportados {len(clips)} clips ({clips[0]} … {clips[-1]}) a {carpeta / a.etiqueta}")

    filas = []
    for nombre, opciones in VARIANTES.items():
        r = evaluar(carpeta, a.etiqueta, clips, opciones)
        filas.append(dict(sistema=a.etiqueta, variante=nombre, clips=len(clips), **r))
        print(f"  {nombre:10s}  GS-HOTA {r['HOTA']:6.2f}   DetA {r['DetA']:6.2f}   AssA {r['AssA']:6.2f}", flush=True)
    pd.DataFrame(filas).to_csv(carpeta / f"{a.etiqueta}_gs_hota.csv", index=False)
    print(f"\nReferencias (test / challenge, con dorsal): sistema base de SoccerNet 22-23; mejor del reto 2024: 63,8")
    print(f"Guardado en {carpeta / (a.etiqueta + '_gs_hota.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
