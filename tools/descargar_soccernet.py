"""Descarga y descomprime un split de SoccerNet Game State Reconstruction (tarea #25).

    python tools/descargar_soccernet.py              # test (8,8 GB): el que usamos para evaluar
    python tools/descargar_soccernet.py --split valid

Deja los clips en data/soccernet/SN-GSR-2025/SNGS-XXX/ (data/ no se sube a GitHub).
GSR no necesita la contraseña del NDA. Si algún zip la pidiera, el script la
pregunta por teclado (no se guarda en ningún sitio).
"""
import argparse
import getpass
import os
import shutil
import time
import zipfile
from pathlib import Path

DESTINO = Path("data/soccernet/SN-GSR-2025")
TAMANOS_GB = {"train": 9.8, "valid": 11.2, "test": 8.8, "challenge": 5.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=list(TAMANOS_GB),
                    help="test por defecto (challenge no trae las anotaciones)")
    ap.add_argument("--borrar_zip", action="store_true", help="borrar el .zip al terminar")
    ap.add_argument("--sin_xet", action="store_true",
                    help="descarga HTTP normal en vez del sistema por trozos de Hugging Face (si los cortes se repiten)")
    a = ap.parse_args()

    if a.sin_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"      # tiene que ir antes de importar huggingface_hub
    from huggingface_hub import hf_hub_download

    DESTINO.mkdir(parents=True, exist_ok=True)
    libre = shutil.disk_usage(DESTINO).free / 1e9
    necesario = TAMANOS_GB[a.split] * 2.2      # zip + descomprimido
    if libre < necesario:
        print(f"✗ Hacen falta unos {necesario:.0f} GB libres y hay {libre:.0f} GB.")
        return 1

    print(f"Descargando {a.split}.zip ({TAMANOS_GB[a.split]} GB). Si se corta, vuelve a ejecutarlo: continúa donde se quedó.")
    for intento in range(1, 9):
        try:
            zip_path = Path(hf_hub_download(repo_id="SoccerNet/SN-GSR-2025", repo_type="dataset",
                                            filename=f"{a.split}.zip", local_dir=DESTINO))
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:                       # cortes de red: se reintenta y sigue donde iba
            espera = min(60, 10 * intento)
            print(f"\n⚠ Corte en la descarga ({type(e).__name__}). Reintento {intento}/8 en {espera} s...", flush=True)
            time.sleep(espera)
    else:
        print("✗ No se ha podido descargar tras 8 intentos. Prueba con --sin_xet o más tarde.")
        return 1

    print(f"Descomprimiendo en {DESTINO} ...")
    with zipfile.ZipFile(zip_path) as z:
        protegido = any(i.flag_bits & 0x1 for i in z.infolist()[:20])
        pwd = getpass.getpass("Contraseña del zip (NDA): ").encode() if protegido else None
        nombres = z.namelist()
        for k, n in enumerate(nombres):
            z.extract(n, DESTINO, pwd=pwd)
            if k % 5000 == 0:
                print(f"  {k}/{len(nombres)} ficheros", flush=True)

    clips = sorted({p for n in nombres for p in Path(n).parts if p.startswith("SNGS-")})
    print(f"✓ {len(clips)} clips en {DESTINO}  (p. ej. {', '.join(clips[:3])})")
    if a.borrar_zip:
        zip_path.unlink()
        print("  zip borrado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
