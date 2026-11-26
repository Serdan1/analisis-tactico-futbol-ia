"""Fase 1: separar a los jugadores en dos equipos (tarea #10).

Responsable: Serdan1

Idea: igual que en la práctica de Face Recognition, una huella (embedding) por
recorte + un clasificador sencillo encima. Las huellas ya están calculadas en
la #9 (outputs/<clip>/embeddings.npy): ResNet-18 de ImageNet (512 valores) +
histograma de color del torso (128 valores).

Pasos:
1. Proponer etiquetas: K-Means con 2 grupos sobre los recortes de jugadores
   (aprendizaje NO supervisado). Sirve solo para elegir unos 20 recortes por
   equipo; una persona los revisa en un mosaico y corrige los que estén mal.
2. Entrenar KNN y regresión logística con esos ~40 recortes etiquetados
   (aprendizaje supervisado, Temas 2-3) y compararlos en validación.
3. Clasificar todas las detecciones y suavizar dentro de cada track con la
   mayoría de una ventana de fotogramas. La ventana es corta a propósito: si el
   tracker intercambia dos números (el 7 y el 21), el equipo cambia con él.
4. Porteros y árbitros: los marca el detector (su ropa es distinta). Cada
   portero se asigna al equipo cuyo centro está más cerca (como hace Roboflow).
   Limitación: si un equipo entero está atacando encima de la portería rival,
   su centro puede quedar más cerca del portero contrario; con clips de 30 s
   y el bloque defensivo del rival delante, en la práctica funciona.
5. Sentido de ataque: un equipo defiende la portería donde está su portero y
   ataca la contraria. Si no se ve ningún portero, el equipo con la media de x
   más baja se supone que defiende la izquierda (menos fiable: se avisa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARTES = {"resnet": slice(0, 512), "color": slice(512, 640), "ambos": slice(0, 640)}
LARGO_CAMPO = 105.0


# --------------------------------------------------------------- huellas
def seleccionar_partes(embs: np.ndarray, partes: str = "ambos") -> np.ndarray:
    """Quedarse con ResNet, color o ambos, y volver a dejar cada fila con norma 1."""
    x = embs[:, PARTES[partes]].astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def clase_de_track(todos: pd.DataFrame, min_portero: int = 25) -> pd.Series:
    """Clase mayoritaria del detector en cada track (jugador / portero / arbitro).

    Un track es portero si al menos el 30 % de sus detecciones lo son (el
    detector a veces confunde al portero con un jugador cuando se ve de lejos)
    y son al menos `min_portero` (1 s): en SNGS-116 un track de 3 detecciones,
    2 de ellas "portero", decidía el sentido de ataque (y mal).
    """
    def decidir(s):
        cuenta = s.value_counts()
        frac = cuenta / len(s)
        if frac.get("portero", 0) >= 0.3 and cuenta.get("portero", 0) >= min_portero:
            return "portero"
        resto = cuenta.drop("portero", errors="ignore")
        return resto.idxmax() if len(resto) else "jugador"
    return todos.groupby("track_id").clase.agg(decidir)


# --------------------------------------------------------------- 1. proponer etiquetas
def proponer_etiquetas(todos: pd.DataFrame, embs: np.ndarray, n_por_equipo: int = 20,
                       frame_max: int | None = None, partes: str = "color",
                       semilla: int = 0) -> pd.DataFrame:
    """K-Means (k=2) sobre los jugadores y n recortes de cada grupo para revisar.

    Los recortes salen de tracks distintos en lo posible (si no, 20 recortes
    del mismo jugador no enseñan nada) y de fotogramas < frame_max (para que
    el resto del vídeo quede para validar y probar). Devuelve las filas de
    `todos` elegidas con las columnas `indice` (fila en embeddings) y `grupo`.
    """
    from sklearn.cluster import KMeans
    todos = todos.reset_index(drop=True)
    tipo = todos.track_id.map(clase_de_track(todos))
    mask = (tipo == "jugador").to_numpy().copy()
    if frame_max is not None:
        mask &= (todos.frame < frame_max).to_numpy()
    idx = np.flatnonzero(mask)
    X = seleccionar_partes(embs[idx], partes)
    km = KMeans(2, n_init=10, random_state=semilla).fit(X)
    cand = todos.iloc[idx].assign(indice=idx, grupo=km.labels_,
                                  dist=np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1))
    # Cada track cuenta solo para el grupo donde cae la mayoría de sus recortes:
    # así no se eligen los pocos recortes raros (tapados, borrosos) de un track.
    cand = cand[cand.grupo == cand.track_id.map(cand.groupby("track_id").grupo.agg(lambda s: s.mode()[0]))]
    elegidas = []
    for g in (0, 1):
        c = cand[cand.grupo == g]
        c = c[c.dist <= c.dist.median()]              # la mitad más típica del grupo
        c = c.sample(frac=1, random_state=semilla)    # al azar: fotogramas variados
        c["ronda"] = c.groupby("track_id").cumcount() # uno de cada track, luego otro...
        elegidas.append(c.sort_values("ronda", kind="stable").head(n_por_equipo))
    return pd.concat(elegidas).drop(columns=["ronda", "dist"]).reset_index(drop=True)


# --------------------------------------------------------------- 2. clasificador
class ClasificadorEquipos:
    """KNN o regresión logística sobre las huellas (Temas 2-3).

    metodo: "knn" o "logreg"; partes: "resnet", "color" o "ambos".
    """

    def __init__(self, metodo: str = "logreg", partes: str = "ambos", k: int = 5, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        self.metodo, self.partes = metodo, partes
        if metodo == "knn":
            self.modelo = KNeighborsClassifier(n_neighbors=k)
        elif metodo == "logreg":
            self.modelo = LogisticRegression(C=C, max_iter=1000)
        else:
            raise ValueError("metodo debe ser 'knn' o 'logreg'")

    def fit(self, embs: np.ndarray, y) -> "ClasificadorEquipos":
        self.modelo.fit(seleccionar_partes(embs, self.partes), np.asarray(y))
        return self

    def predict(self, embs: np.ndarray) -> np.ndarray:
        return self.modelo.predict(seleccionar_partes(embs, self.partes))

    def predict_proba(self, embs: np.ndarray) -> np.ndarray:
        return self.modelo.predict_proba(seleccionar_partes(embs, self.partes))

    @property
    def clases(self):
        return self.modelo.classes_


# --------------------------------------------------------------- 3-4. asignar equipos
def suavizar_por_track(todos: pd.DataFrame, pred: pd.Series, ventana: int = 25) -> pd.Series:
    """Mayoría en una ventana centrada de `ventana` detecciones dentro de cada track."""
    codigos, nombres = pd.factorize(pred)
    s = pd.Series(codigos, index=todos.index)
    orden = todos.sort_values(["track_id", "frame"]).index
    salida = pd.Series(-1, index=todos.index)
    for _, filas in todos.loc[orden].groupby("track_id").groups.items():
        v = s.loc[filas]
        if (v < 0).all():
            continue
        unos = pd.get_dummies(v[v >= 0]).reindex(v.index, fill_value=0).astype(float)
        votos = unos.rolling(ventana, center=True, min_periods=1).sum()
        salida.loc[filas] = votos.idxmax(axis=1).to_numpy()
    return pd.Series(np.where(salida >= 0, np.asarray(nombres, dtype=object)[salida.clip(lower=0)], ""),
                     index=todos.index)


def asignar_equipos(todos: pd.DataFrame, embs: np.ndarray, clf: ClasificadorEquipos,
                    ventana: int = 25) -> pd.DataFrame:
    """Columna `equipo` para cada detección: nombre de equipo, "arbitro" u "otro".

    También añade `equipo_bruto` (sin suavizar) y `prob` (confianza del
    clasificador) para poder evaluar. Los porteros reciben el equipo con
    asignar_porteros().
    """
    out = todos.copy()
    tipo = out.track_id.map(clase_de_track(out))
    jug = (tipo == "jugador").to_numpy()
    bruto = pd.Series("", index=out.index, dtype=object)
    prob = np.full(len(out), np.nan)
    if jug.any():
        p = clf.predict_proba(embs[jug])
        bruto[jug] = clf.clases[p.argmax(1)]
        prob[jug] = p.max(1)
    out["equipo_bruto"] = bruto
    out["prob"] = prob
    eq = suavizar_por_track(out[jug], bruto[jug], ventana) if jug.any() else pd.Series(dtype=object)
    out["equipo"] = ""
    out.loc[jug, "equipo"] = eq
    out.loc[(tipo == "arbitro").to_numpy(), "equipo"] = "arbitro"
    out.loc[(tipo == "portero").to_numpy(), "equipo"] = "portero"
    return asignar_porteros(out)


def asignar_porteros(out: pd.DataFrame) -> pd.DataFrame:
    """Cada track de portero va al equipo cuyo centro (media de x en el campo,
    en los mismos fotogramas) está más cerca de él. Columna `es_portero`."""
    out = out.copy()
    out["es_portero"] = out.equipo == "portero"
    equipos = sorted(set(out.equipo) - {"", "arbitro", "portero", "otro"})
    validos = out[out.valido.astype(bool)]
    for tid, g in validos[validos.es_portero].groupby("track_id"):
        campo = validos[validos.frame.isin(g.frame) & validos.equipo.isin(equipos)]
        if campo.empty:
            out.loc[out.track_id == tid, "equipo"] = "otro"
            continue
        centro = campo.groupby("equipo").x_m.mean()
        out.loc[out.track_id == tid, "equipo"] = (centro - g.x_m.median()).abs().idxmin()
    return out


# --------------------------------------------------------------- 5. sentido de ataque
def sentido_de_ataque(out: pd.DataFrame) -> dict:
    """{equipo: {"ataca": "derecha"|"izquierda", "metodo": "portero"|"centro"}}.

    "derecha" = hacia x = 105 en las coordenadas de la cámara (docs/formatos.md).
    """
    equipos = sorted(set(out.equipo) - {"", "arbitro", "portero", "otro"})
    validos = out[out.valido.astype(bool)]
    res = {}
    for e in equipos:
        pt = validos[validos.es_portero & (validos.equipo == e)]
        if len(pt):
            defiende_izq = pt.x_m.median() < LARGO_CAMPO / 2
            res[e] = {"ataca": "derecha" if defiende_izq else "izquierda", "metodo": "portero",
                      "x_portero": round(float(pt.x_m.median()), 1)}
    if len(res) == 1 and len(equipos) == 2:        # el otro equipo, al revés
        a, b = equipos if equipos[0] in res else equipos[::-1]
        res[b] = {"ataca": "izquierda" if res[a]["ataca"] == "derecha" else "derecha",
                  "metodo": "portero del rival"}
    if not res and len(equipos) == 2:               # sin porteros: centro de cada equipo
        centro = validos[~validos.es_portero & validos.equipo.isin(equipos)].groupby("equipo").x_m.mean()
        izq = centro.idxmin()
        for e in equipos:
            res[e] = {"ataca": "derecha" if e == izq else "izquierda", "metodo": "centro (poco fiable)",
                      "x_centro": round(float(centro[e]), 1)}
    return res


def equipo_de_un_jugador(out: pd.DataFrame, frames_y_tracks: pd.DataFrame) -> str:
    """Equipo mayoritario de un jugador seguido (columnas frame y track_id)."""
    m = frames_y_tracks[["frame", "track_id"]].merge(out[["frame", "track_id", "equipo"]],
                                                     on=["frame", "track_id"])
    m = m[~m.equipo.isin(["", "arbitro", "otro"])]
    return m.equipo.mode().iloc[0] if len(m) else ""


# --------------------------------------------------------------- todo junto (main.py)
def nombre_de_color(h: float, s: float, v: float) -> str:
    """Nombre de un color a partir de su H, S, V en la escala de OpenCV (H 0-180)."""
    if v < 70:
        return "negro"
    if s < 45:
        return "blanco" if v >= 170 else "gris"
    for limite, nombre in [(8, "rojo"), (20, "naranja"), (33, "amarillo"), (85, "verde"),
                           (130, "azul"), (150, "morado"), (170, "rosa")]:
        if h < limite:
            return nombre
    return "rojo"


def colores_de_grupos(video, filas: pd.DataFrame) -> dict:
    """{grupo: nombre del color de la camiseta} a partir de los recortes propuestos.

    Se usa la mediana HSV del centro del pecho (sin brazos ni césped). Al
    principio se usaba el brillo ("claro" / "oscuro"), pero en 08fd33_0 el
    verde lima salía más "brillante" que el blanco: V de HSV es max(R, G, B),
    no la claridad. Si los dos grupos salen del mismo color se distinguen
    por la claridad L de Lab ("azul claro" / "azul oscuro").
    """
    import cv2
    cap = cv2.VideoCapture(str(video))
    pixeles = {}
    actual, f = -1, None
    for i in np.argsort(filas.frame.to_numpy(), kind="stable"):   # leer hacia delante: sin saltos lentos
        r = filas.iloc[i]
        while actual < r.frame:
            ok, f = cap.read(); actual += 1
            if not ok:
                break
        if f is None or actual != r.frame:
            continue
        h, w = int(r.y2) - int(r.y1), int(r.x2) - int(r.x1)
        zona = f[int(r.y1) + int(.2 * h):int(r.y1) + int(.5 * h), int(r.x1) + int(.25 * w):int(r.x2) - int(.25 * w)]
        if zona.size:
            pixeles.setdefault(int(r.grupo), []).append(zona.reshape(-1, 3))
    cap.release()
    hsv, luz = {}, {}
    for g, lista in pixeles.items():
        px = np.vstack(lista)[None]
        hsv[g] = np.median(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0], axis=0)
        luz[g] = float(np.median(cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][:, 0]))
    nombres = {g: nombre_de_color(*hsv[g]) for g in hsv}
    if len(set(nombres.values())) < len(nombres):
        claro = max(luz, key=luz.get)
        nombres = {g: f"{n} {'claro' if g == claro else 'oscuro'}" for g, n in nombres.items()}
    return nombres


def equipos_automaticos(video, todos: pd.DataFrame, embs: np.ndarray, frame_max: int | None = None,
                        n_por_equipo: int = 20, metodo: str = "knn", partes: str = "ambos",
                        ventana: int = 25):
    """Versión sin revisión humana del notebook 10 (para main.py).

    K-Means propone 20 recortes por grupo, se nombran por el color de la
    camiseta, se entrena el clasificador (por defecto el mejor en validación
    en 2e57b9_0: KNN k=5 con ResNet + color) y se asigna equipo a todo.
    frame_max por defecto: el primer tercio del clip.
    Devuelve (out, sentido, propuesta).
    """
    if frame_max is None:
        frame_max = int(todos.frame.min() + (todos.frame.max() - todos.frame.min()) / 3) + 1
    prop = proponer_etiquetas(todos, embs, n_por_equipo, frame_max=frame_max)
    if prop.grupo.nunique() < 2:
        raise ValueError("No se han encontrado dos equipos distintos en el clip")
    prop["equipo"] = prop.grupo.map(colores_de_grupos(video, prop))
    kw = {"k": 5} if metodo == "knn" else {}
    clf = ClasificadorEquipos(metodo, partes, **kw).fit(embs[prop.indice], prop.equipo)
    out = asignar_equipos(todos, embs, clf, ventana=ventana)
    return out, sentido_de_ataque(out), prop
