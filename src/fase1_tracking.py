"""Fase 1: elegir al jugador y no perderlo (tarea #9).

Responsable: Serdan1

El tracker (ByteTrack) solo mira POSICIONES: si dos jugadores se cruzan y uno
tapa al otro, puede intercambiarles el número (lo vimos en 2e57b9_0.mp4 con
el delantero 7 y el defensa 21). Para evitarlo usamos el mismo método que la
práctica de Face Recognition, pero con el cuerpo entero en lugar de la cara
(en un plano general la cara ocupa unos pocos píxeles):

1. Huella de apariencia (embedding) de cada recorte de jugador:
   - ResNet-18 preentrenada en ImageNet como extractor fijo (Tema 5): sin la
     última capa, en eval() y con no_grad() -> 512 valores.
   - Histograma de color del torso (HSV): los jugadores miden ~20 x 50 píxeles
     y ResNet sola apenas distingue el blanco del oscuro; el color de la
     camiseta sí. Comprobado en el cruce del 7 y el 21: distancia a la
     galería 0,65 (delantero) frente a 0,66 (defensa) solo con ResNet, y
     0,64 frente a 0,93 con ResNet + color.
   Las dos partes se normalizan y se concatenan.
2. Galería: al elegir al jugador se guardan varios embeddings suyos.
3. En cada fotograma se confía en el tracker mientras no haya conflicto. La
   apariencia decide solo cuando otro jugador está pegado al nuestro (posible
   cambio de número) o cuando su número desaparece. A esa distancia y
   resolución dos compañeros con la misma camiseta son indistinguibles, pero
   un rival no. Si ningún candidato baja del umbral tau, la respuesta es
   "desconocido": mejor no dar posición que dar la de otro.
4. tau se elige como en la práctica: barriendo umbrales y minimizando la
   media de FAR (falsas aceptaciones) y FRR (falsos rechazos).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], np.float32)
ALTO, ANCHO = 224, 112          # recortes de jugador: más altos que anchos


# --------------------------------------------------------------- 1. extractor
class ExtractorApariencia:
    """ResNet-18 de ImageNet como extractor fijo de embeddings (Tema 5)."""

    def __init__(self, device="cuda", pesos=None):
        import torch
        import torchvision
        self.torch = torch
        self.device = device
        if pesos:   # ruta a un .pth local (por si no hay internet)
            modelo = torchvision.models.resnet18()
            modelo.load_state_dict(torch.load(pesos, map_location="cpu"))
        else:
            modelo = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        modelo.fc = torch.nn.Identity()   # quitar el clasificador -> vector de 512
        self.modelo = modelo.eval().to(device)

    def __call__(self, recortes_rgb: list[np.ndarray], lote: int = 64) -> np.ndarray:
        """Lista de recortes RGB (uint8) -> (N, 512) embeddings de norma 1."""
        torch = self.torch
        if len(recortes_rgb) == 0:
            return np.empty((0, 512), np.float32)
        salida = []
        for i in range(0, len(recortes_rgb), lote):
            x = np.stack([cv2.resize(r, (ANCHO, ALTO)) for r in recortes_rgb[i:i + lote]])
            x = (x.astype(np.float32) / 255.0 - MEDIA_IMAGENET) / STD_IMAGENET
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                e = self.modelo(x)
                e = torch.nn.functional.normalize(e, dim=1)
            salida.append(e.cpu().numpy())
        return np.concatenate(salida)


def recortar(frame_bgr: np.ndarray, x1, y1, x2, y2, margen: float = 0.0) -> np.ndarray:
    """Recorte RGB de un jugador (OpenCV lee en BGR: se convierte aquí)."""
    h, w = frame_bgr.shape[:2]
    dx, dy = (x2 - x1) * margen, (y2 - y1) * margen
    x1, x2 = int(max(0, x1 - dx)), int(min(w, x2 + dx))
    y1, y2 = int(max(0, y1 - dy)), int(min(h, y2 + dy))
    return cv2.cvtColor(frame_bgr[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)], cv2.COLOR_BGR2RGB)


def color_torso(frame_bgr: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Histograma HSV (8 x 4 x 4 = 128 valores) de la zona del torso, normalizado."""
    h, w = y2 - y1, x2 - x1
    zona = frame_bgr[y1 + int(.15 * h):y1 + int(.55 * h), x1 + int(.15 * w):max(x2 - int(.15 * w), x1 + int(.15 * w) + 1)]
    if zona.size == 0:
        return np.zeros(128, np.float32)
    hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = np.sqrt(hist / max(hist.sum(), 1))          # raíz: compara mejor histogramas
    return (hist / (np.linalg.norm(hist) + 1e-9)).astype(np.float32)


def calcular_embeddings(video, todos: pd.DataFrame, extractor: ExtractorApariencia,
                        con_color: bool = True, progreso: bool = True) -> np.ndarray:
    """Huella de cada fila de posiciones_todos.csv (necesita las columnas x1, y1, x2, y2).

    Devuelve (len(todos), 640) = ResNet-18 (512) + color del torso (128), en el
    mismo orden que las filas, con norma 1. Con con_color=False, solo ResNet (512).
    Sirve para la re-identificación (#9) y para separar equipos (#10).
    """
    if not {"x1", "y1", "x2", "y2"} <= set(todos.columns):
        raise ValueError("posiciones_todos.csv no tiene las cajas: vuelve a ejecutar procesar_video")
    embs = np.zeros((len(todos), 640 if con_color else 512), np.float32)
    por_frame = todos.reset_index(drop=True).groupby("frame").indices
    cap = cv2.VideoCapture(str(video))
    frames = sorted(por_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    actual = frames[0]
    for k, f in enumerate(frames):
        while actual < f:          # avanzar sin decodificar de más
            cap.grab(); actual += 1
        ok, frame = cap.read(); actual += 1
        if not ok:
            break
        filas = por_frame[f]
        cajas = todos.iloc[filas][["x1", "y1", "x2", "y2"]].to_numpy()
        e = extractor([recortar(frame, *c) for c in cajas])
        if con_color:
            col = np.stack([color_torso(frame, *c) for c in cajas])
            e = np.hstack([e, col]) / np.sqrt(2)       # las dos partes pesan lo mismo
        embs[filas] = e
        if progreso and k % 100 == 0:
            print(f"  embeddings: fotograma {k}/{len(frames)}", flush=True)
    cap.release()
    return embs


# --------------------------------------------------------------- 2. elegir al jugador
def elegir_jugador_con_clic(video, todos: pd.DataFrame, frame: int | None = None) -> int:
    """Abre una ventana con el fotograma y devuelve el track_id del jugador en el que se hace clic.

    Clic izquierdo sobre el jugador y después cualquier tecla. Esc cancela.
    """
    frame = int(todos.frame.min() + 2) if frame is None else frame
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read(); cap.release()
    if not ok:
        raise ValueError(f"No se pudo leer el fotograma {frame}")
    filas = todos[todos.frame == frame]
    for _, r in filas.iterrows():
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (255, 255, 255), 2)
        cv2.putText(img, str(r.track_id), (r.x1, r.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elegido = {}

    def clic(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            dentro = filas[(filas.x1 <= x) & (x <= filas.x2) & (filas.y1 <= y) & (y <= filas.y2)]
            if len(dentro):
                r = dentro.iloc[0]
                elegido["id"] = int(r.track_id)
                cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (0, 255, 0), 3)
                cv2.imshow(titulo, img)

    titulo = "Haz clic sobre el jugador y pulsa una tecla"
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titulo, 1280, 720)
    cv2.setMouseCallback(titulo, clic)
    cv2.imshow(titulo, img)
    tecla = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if tecla == 27 or "id" not in elegido:
        raise ValueError("No se ha elegido ningún jugador")
    return elegido["id"]


# --------------------------------------------------------------- 3. seguirlo
def seguir_jugador(todos: pd.DataFrame, embs: np.ndarray, track_id: int, tau: float,
                   frame_inicio: int | None = None, n_galeria: int = 5, sep_galeria: int = 5,
                   vel_max_ms: float = 10.0, margen_m: float = 3.0, radio_conflicto_m: float = 2.0,
                   fps: float = 25.0) -> pd.DataFrame:
    """Sigue a UN jugador fotograma a fotograma usando su galería de apariencia.

    - Galería: huellas de los primeros `n_galeria` fotogramas del track elegido,
      separados `sep_galeria` fotogramas (posturas distintas).
    - Candidatos: jugadores a menos de vel_max * tiempo + margen metros de su
      última posición conocida (nadie corre a más de ~10 m/s).
    - Si su número sigue ahí, sin nadie a menos de `radio_conflicto_m` y con
      d <= tau: se mantiene (confiamos en el tracker).
    - Si hay otro jugador pegado (posible cambio de número) o su número ha
      desaparecido: gana el candidato más parecido a la galería, si d <= tau.
    - Si ninguno baja de tau: "desconocido" en ese fotograma.

    Devuelve una fila por fotograma: frame, t_s, track_id seguido (o -1),
    x_m, y_m, valido, distancia a la galería, si hubo cambio de número y si
    hubo conflicto.
    """
    todos = todos.reset_index(drop=True)
    propio = todos[todos.track_id == track_id]
    if frame_inicio is not None:
        propio = propio[propio.frame >= frame_inicio]
    idx_gal = propio.index[::sep_galeria][:n_galeria]
    if len(idx_gal) == 0:
        raise ValueError(f"El track {track_id} no aparece")
    galeria = embs[idx_gal]

    def dist_galeria(ix):
        return np.linalg.norm(embs[ix][:, None, :] - galeria[None, :, :], axis=2).min(axis=1)

    actual, ultima_pos, ultimo_frame = track_id, None, None
    filas = []
    for f, grupo in todos[todos.frame >= propio.frame.min()].groupby("frame"):
        t_s = grupo.t_s.iloc[0]
        cand = grupo[grupo.valido == 1]
        if ultima_pos is not None and len(cand):
            radio = vel_max_ms * (f - ultimo_frame) / fps + margen_m
            cand = cand[np.hypot(cand.x_m - ultima_pos[0], cand.y_m - ultima_pos[1]) <= radio]
        elegido, d_elegido, cambio, conflicto = None, np.nan, False, False
        if len(cand):
            d = dist_galeria(cand.index.to_numpy())
            ids = cand.track_id.to_numpy()
            mio = np.flatnonzero(ids == actual)
            if len(mio):
                m = mio[0]
                cerca = np.hypot(cand.x_m - cand.x_m.iloc[m], cand.y_m - cand.y_m.iloc[m]).to_numpy()
                grupo_conf = np.flatnonzero(cerca <= radio_conflicto_m)   # incluye al propio
                conflicto = len(grupo_conf) > 1
                k = grupo_conf[np.argmin(d[grupo_conf])] if conflicto else m
            else:
                k = int(np.argmin(d))
            if d[k] <= tau:
                elegido, d_elegido = cand.iloc[k], float(d[k])
                cambio = int(elegido.track_id) != actual
                actual = int(elegido.track_id)
            else:
                d_elegido = float(d[k])
        if elegido is not None:
            ultima_pos, ultimo_frame = (elegido.x_m, elegido.y_m), f
            filas.append(dict(frame=f, t_s=t_s, track_id=actual, x_m=elegido.x_m, y_m=elegido.y_m,
                              valido=1, distancia=round(d_elegido, 4), cambio_id=int(cambio),
                              conflicto=int(conflicto)))
        else:
            filas.append(dict(frame=f, t_s=t_s, track_id=-1, x_m=np.nan, y_m=np.nan, valido=0,
                              distancia=round(d_elegido, 4) if np.isfinite(d_elegido) else np.nan,
                              cambio_id=0, conflicto=int(conflicto)))
    return pd.DataFrame(filas)


# --------------------------------------------------------------- 4. elegir tau
def calibrar_umbral(d_mismo: np.ndarray, d_otro: np.ndarray, n: int = 400):
    """Como en la práctica de Face Recognition: barrer umbrales y elegir el que
    minimiza la media de FAR y FRR.

    d_mismo : distancias a la galería de recortes del MISMO jugador (deben aceptarse).
    d_otro  : distancias de OTROS jugadores (deben rechazarse).
    Devuelve (tau, tabla con FAR y FRR para cada umbral).
    """
    d_mismo, d_otro = np.asarray(d_mismo), np.asarray(d_otro)
    umbrales = np.linspace(0, max(d_mismo.max(), d_otro.max()), n)
    far = np.array([(d_otro <= u).mean() for u in umbrales])    # otros aceptados por error
    frr = np.array([(d_mismo > u).mean() for u in umbrales])    # él rechazado por error
    tabla = pd.DataFrame({"tau": umbrales, "FAR": far, "FRR": frr, "media": (far + frr) / 2})
    # Si varios umbrales empatan en el mínimo, el del centro: deja margen a ambos lados
    mejores = tabla.tau[np.isclose(tabla.media, tabla.media.min())]
    return float(mejores.iloc[len(mejores) // 2]), tabla
