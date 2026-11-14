"""Pipeline de preprocesamiento: del Excel crudo a las variables del modelo.

Hace en un solo lugar, y con reglas generales, lo que se analizó y justificó en
notebooks/eda_1_1.ipynb y notebooks/preprocesamiento.ipynb:

    Excel  ->  lecturas horarias  ->  una fila por día (valor de las 23:59)
           ->  limpieza  ->  variables  ->  data/processed/dataset_final.csv

Lo usan los tres pasos del proyecto, para que el preprocesamiento sea siempre el mismo:
    - este script (arma el dataset de entrenamiento)
    - src/train_model.py (lee el dataset que se genera acá)
    - src/predict.py (llama a preparar_datos() sobre un archivo nuevo)

Uso:
    python src/preprocessing_pipeline.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from eda_1_1_auxiliar import a_snake_case, snake_case_unicas, limpiar_numericas
from integridad import huella, actualizar_checksums

SEMILLA = 42

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ / "data"
RUTA_DATASET_FINAL = CARPETA_DATOS / "processed" / "dataset_final.csv"
RUTA_LINAJE = CARPETA_DATOS / "processed" / "data_lineage.json"

# Archivos de entrenamiento, en orden de prioridad: si dos archivos tienen el mismo día,
# se usa el valor del que está primero (regla de desempate definida en el EDA).
ARCHIVOS_ENTRENAMIENTO = [
    "Totalizadores Planta - 2021_2023.xlsx",
    "Totalizadores Planta 2022_2023.xlsx",
    "Totalizadores Planta 2020_2022.xlsx",
]

# Todas las columnas que usa el modelo salen de una sola hoja.
# Los nombres están en snake_case: "Consolidado EE" -> "consolidado_ee", "Frio (Kw)" -> "frio_kw".
HOJA = "consolidado_ee"
COLUMNA_FRIO = "frio_kw"
COLUMNAS_BASE = ["frio_kw", "sala_maq_kw", "servicios_kw", "planta_kw"]
# Cómo se llaman en el Excel, para los mensajes de error
NOMBRE_EN_EXCEL = {
    "dia": "DIA", "hora": "HORA", "frio_kw": "Frio (Kw)", "sala_maq_kw": "Sala Maq (Kw)",
    "servicios_kw": "Servicios (Kw)", "planta_kw": "Planta (Kw)",
}

# Variables finales del modelo (selección justificada en notebooks/preprocesamiento.ipynb)
VARIABLES = [
    "frio_kw", "frio_media_3", "frio_media_7", "frio_media_14", "frio_min_7", "frio_max_7",
    "sala_maq_kw", "servicios_kw", "planta_kw", "servicios_kw_media_7", "planta_kw_media_7",
    "dia_semana", "mes", "fin_de_semana",
]
OBJETIVO = "y"

UMBRAL_IMPOSIBLE = 1e6        # un total diario de un millón o más es un error de registro
LIMITE_HUECO_CORTO = 3        # días seguidos sin dato que todavía se imputan
PROPORCION_TRAIN, PROPORCION_VAL = 0.70, 0.15

H_2359 = pd.Timedelta("23:59:00")
H_23 = pd.Timedelta("23:00:00")
H_20 = pd.Timedelta("20:00:00")


class ErrorDeDatos(ValueError):
    """El archivo de entrada no tiene lo que el pipeline necesita."""


# ------------------------------------------------------------------ 1. LECTURA

def leer_lecturas_horarias(ruta):
    """Lee la hoja de energía eléctrica y devuelve las lecturas horarias de las columnas base.
    Busca la hoja y las columnas por nombre, nunca por posición."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorDeDatos(f"No existe el archivo: {ruta}")

    try:
        libro = pd.ExcelFile(ruta, engine="calamine")
    except Exception as error:
        raise ErrorDeDatos(f"No se pudo abrir '{ruta.name}' como Excel: {error}")

    # nombre en snake_case -> nombre original de la hoja
    hojas = {a_snake_case(nombre): nombre for nombre in libro.sheet_names}
    if HOJA not in hojas:
        raise ErrorDeDatos(
            f"En '{ruta.name}' falta la hoja 'Consolidado EE'. Hojas encontradas: {libro.sheet_names}"
        )

    h = libro.parse(hojas[HOJA])
    columnas_originales = [str(c) for c in h.columns]
    h.columns = snake_case_unicas(columnas_originales)

    necesarias = ["dia", "hora"] + COLUMNAS_BASE
    faltan = [c for c in necesarias if c not in h.columns]
    if faltan:
        raise ErrorDeDatos(
            f"En la hoja '{hojas[HOJA]}' de '{ruta.name}' faltan las columnas: "
            f"{[NOMBRE_EN_EXCEL[c] for c in faltan]}. Columnas encontradas: {columnas_originales}"
        )

    h = h[necesarias]
    h, _ = limpiar_numericas(h)
    for c in COLUMNAS_BASE:
        h[c] = pd.to_numeric(h[c], errors="coerce")

    h["dia"] = pd.to_datetime(h["dia"], errors="coerce").dt.normalize()
    # Los segundos varían (05:00:01), por eso se redondea al minuto
    h["t"] = pd.to_timedelta(h["hora"].astype(str), errors="coerce").dt.floor("min")
    h = h.dropna(subset=["dia", "t"]).reset_index(drop=True)
    if h.empty:
        raise ErrorDeDatos(f"La hoja '{hojas[HOJA]}' de '{ruta.name}' no tiene filas con día y hora válidos.")
    return h


# ------------------------------------------------------------------ 2. DÍAS APILADOS

def reasignar_dias_apilados(h):
    """Corrige las fechas que tienen más de un día apilado bajo la misma etiqueta.

    En los archivos hay fechas con más de 25 registros: son varias jornadas cargadas con la
    misma fecha (ver "Días pisados" en el EDA). Cada vez que la hora vuelve hacia atrás empieza
    una jornada nueva. La primera es la de esa fecha; las demás corresponden a días que faltan
    en el calendario, y se asignan al día faltante anterior más cercano que siga libre."""
    h = h.copy()
    vuelve_atras = h.groupby("dia")["t"].diff() < pd.Timedelta(hours=-6)
    h["bloque"] = vuelve_atras.groupby(h["dia"]).cumsum()

    calendario = pd.date_range(h["dia"].min(), h["dia"].max(), freq="D")
    libres = sorted(calendario.difference(h["dia"].unique()))

    apilados = h.loc[h["bloque"] > 0, ["dia", "bloque"]].drop_duplicates().sort_values(["dia", "bloque"])
    for dia, bloque in apilados.itertuples(index=False):
        filas = (h["dia"] == dia) & (h["bloque"] == bloque)
        anteriores = [d for d in libres if d < dia]
        if anteriores:
            h.loc[filas, "dia"] = anteriores[-1]
            libres.remove(anteriores[-1])
        else:
            # No hay ningún día libre antes: no sabemos de qué día son esas lecturas
            h = h[~filas]
    return h.drop(columns="bloque").sort_values(["dia", "t"]).reset_index(drop=True)


# ------------------------------------------------------------------ 3. TOTAL DIARIO

def total_sin_salto(valores):
    """Total de un día cuyo totalizador tiene un salto imposible: se saca el salto y en su
    lugar se suma la mediana de los demás incrementos horarios."""
    incrementos = np.diff(valores)
    salto = np.nanargmax(np.abs(incrementos))
    return valores[-1] - incrementos[salto] + np.nanmedian(np.delete(incrementos, salto))


def calcular_total_diario(h):
    """Devuelve una fila por día con el total de las 23:59 de cada columna base.

    - Si el día no tiene registro de las 23:59 pero llega hasta las 23:00, se estima sumando
      59 minutos al ritmo de las últimas 3 horas (queda marcado en dia_estimado_23h).
    - Si el día no llega a las 23:00 queda vacío: el totalizador de un día incompleto no
      representa el consumo de la jornada.
    - Los totales imposibles y los negativos se vacían. El de frio_kw se intenta reconstruir
      sacando el salto (queda marcado en frio_reconstruido)."""
    def foto(hora):
        return h[h["t"] == hora].drop_duplicates("dia").set_index("dia")[COLUMNAS_BASE]

    calendario = pd.date_range(h["dia"].min(), h["dia"].max(), freq="D", name="dia")
    a_2359 = foto(H_2359).reindex(calendario)
    a_23 = foto(H_23).reindex(calendario)
    a_20 = foto(H_20).reindex(calendario)

    # Estimación de las 23:59 para los días que solo llegan hasta las 23:00
    ritmo = ((a_23 - a_20) / 3).clip(lower=0).fillna(0)
    estimado = a_23 + ritmo * 59 / 60

    diario = a_2359.copy()
    sin_2359 = a_2359[COLUMNA_FRIO].isna() & a_23[COLUMNA_FRIO].notna()
    diario[sin_2359] = estimado[sin_2359]
    diario["dia_estimado_23h"] = sin_2359.astype(int)

    # Totales imposibles: se intenta reconstruir frio_kw y el resto se vacía
    diario["frio_reconstruido"] = 0
    for dia in diario.index[diario[COLUMNA_FRIO].abs() >= UMBRAL_IMPOSIBLE]:
        lecturas = h.loc[h["dia"] == dia, COLUMNA_FRIO].to_numpy(dtype=float)
        if len(lecturas) >= 24 and not np.isnan(lecturas).any():
            reconstruido = total_sin_salto(lecturas)
            if 0 <= reconstruido < UMBRAL_IMPOSIBLE:
                diario.loc[dia, COLUMNA_FRIO] = reconstruido
                diario.loc[dia, "frio_reconstruido"] = 1

    valores = diario[COLUMNAS_BASE]
    diario[COLUMNAS_BASE] = valores.mask((valores.abs() >= UMBRAL_IMPOSIBLE) | (valores < 0))
    return diario


def excel_a_diario(ruta):
    """Excel crudo -> una fila por día con las columnas base."""
    lecturas = leer_lecturas_horarias(ruta)
    lecturas = reasignar_dias_apilados(lecturas)
    return calcular_total_diario(lecturas)


def unir_archivos(diarios):
    """Une los datos diarios de varios archivos. `diarios` va en orden de prioridad: para cada
    celda se usa el primer archivo que tenga dato. Los valores medidos a las 23:59 tienen
    prioridad sobre los estimados desde las 23:00."""
    marcas = ["dia_estimado_23h", "frio_reconstruido"]
    inicio = min(d.index.min() for d in diarios)
    fin = max(d.index.max() for d in diarios)
    calendario = pd.date_range(inicio, fin, freq="D", name="dia")

    medidos = [d.where(d["dia_estimado_23h"] == 0) for d in diarios]
    estimados = [d.where(d["dia_estimado_23h"] == 1) for d in diarios]

    unido = pd.DataFrame(index=calendario, columns=COLUMNAS_BASE + marcas, dtype=float)
    for d in medidos + estimados:
        unido = unido.combine_first(d.reindex(calendario))
    unido[marcas] = unido[marcas].fillna(0).astype(int)
    return unido[COLUMNAS_BASE + marcas]


# ------------------------------------------------------------------ 4. LIMPIEZA

def largo_del_hueco(columna):
    """Para cada día de una columna, cuántos días seguidos dura el hueco al que pertenece (0 si hay dato)."""
    vacia = columna.isna()
    tramo = (vacia != vacia.shift()).cumsum()
    return vacia.groupby(tramo).transform("sum") * vacia


def limpiar(diario):
    """Marca los días vacíos e imputa los huecos cortos de las columnas predictoras.

    Los huecos de hasta 3 días se rellenan con la mediana de los 7 días anteriores: solo mira
    hacia atrás, así que no usa información del futuro ni aprende nada del dataset.
    frio_kw no se imputa nunca, porque es lo que queremos predecir."""
    diario = diario.copy()
    diario["dia_vacio"] = diario[COLUMNAS_BASE].isna().all(axis=1).astype(int)

    predictoras = [c for c in COLUMNAS_BASE if c != COLUMNA_FRIO]
    largo = diario[predictoras].apply(largo_del_hueco)
    a_imputar = (largo > 0) & (largo <= LIMITE_HUECO_CORTO)
    a_imputar = a_imputar.mul(diario["dia_vacio"] == 0, axis=0)

    relleno = diario[predictoras].rolling(7, min_periods=1).median().shift(1)
    diario[predictoras] = diario[predictoras].where(~a_imputar, relleno)
    return diario


# ------------------------------------------------------------------ 5. VARIABLES

def crear_variables(diario):
    """Agrega las variables del modelo y el target.

    Cada fila es un día D y sirve para predecir el consumo de D+1. Los promedios móviles
    terminan en D (incluyen el día D y los anteriores), así que nunca miran el día a predecir.
    Las variables de calendario son las del día D+1, que se conocen de antemano."""
    datos = diario.copy()
    frio = datos[COLUMNA_FRIO]

    # min_periods: alcanza con la mitad de la ventana para calcular el valor
    datos["frio_media_3"] = frio.rolling(3, min_periods=2).mean()
    datos["frio_media_7"] = frio.rolling(7, min_periods=4).mean()
    datos["frio_media_14"] = frio.rolling(14, min_periods=7).mean()
    datos["frio_min_7"] = frio.rolling(7, min_periods=4).min()
    datos["frio_max_7"] = frio.rolling(7, min_periods=4).max()
    datos["servicios_kw_media_7"] = datos["servicios_kw"].rolling(7, min_periods=4).mean()
    datos["planta_kw_media_7"] = datos["planta_kw"].rolling(7, min_periods=4).mean()

    dia_a_predecir = datos.index + pd.Timedelta(days=1)
    datos["dia_semana"] = dia_a_predecir.dayofweek            # 0 = lunes, 6 = domingo
    datos["mes"] = dia_a_predecir.month
    datos["fin_de_semana"] = (dia_a_predecir.dayofweek >= 5).astype(int)

    # Target: consumo del día siguiente
    datos[OBJETIVO] = frio.shift(-1)
    # La fila sirve para entrenar si el día tiene datos propios y el día siguiente tiene frio_kw
    datos["fila_valida"] = ((datos["dia_vacio"] == 0) & datos[OBJETIVO].notna()).astype(int)
    return datos


def preparar_datos(ruta):
    """Pipeline completo sobre un archivo Excel nuevo. Devuelve una fila por día con las
    variables del modelo. Es lo que usa predict.py."""
    diario = excel_a_diario(ruta)
    diario = unir_archivos([diario])
    return crear_variables(limpiar(diario))


def preparar_datos_entrenamiento():
    """Pipeline completo sobre los archivos de entrenamiento."""
    rutas = [CARPETA_DATOS / nombre for nombre in ARCHIVOS_ENTRENAMIENTO]
    faltan = [r.name for r in rutas if not r.exists()]
    if faltan:
        raise ErrorDeDatos(f"Faltan archivos en {CARPETA_DATOS}: {faltan}. El README explica de dónde se obtienen.")

    diario = unir_archivos([excel_a_diario(r) for r in rutas])
    return crear_variables(limpiar(diario))


def dividir(datos):
    """División temporal: 70% más antiguo para train, 15% validation y 15% más reciente para test.
    Devuelve una Serie con el conjunto de cada fila."""
    n = len(datos)
    n_train = int(n * PROPORCION_TRAIN)
    n_val = int(n * PROPORCION_VAL)
    conjunto = np.array(["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val))
    return pd.Series(conjunto, index=datos.index, name="conjunto")


# ------------------------------------------------------------------ 6. PRINCIPAL

def verificar_archivos_crudos():
    """Compara los Excel con data/checksums.json y avisa si alguno es distinto."""
    guardados = json.load(open(CARPETA_DATOS / "checksums.json"))
    for nombre in ARCHIVOS_ENTRENAMIENTO:
        ruta = CARPETA_DATOS / nombre
        if ruta.exists() and nombre in guardados and huella(ruta) != guardados[nombre]:
            print(f"AVISO: '{nombre}' no coincide con el hash de checksums.json")


def main():
    verificar_archivos_crudos()

    datos = preparar_datos_entrenamiento()
    print("Días en el calendario:", len(datos), "| desde", datos.index.min().date(), "hasta", datos.index.max().date())

    # Los lags ya están calculados, así que ahora sí se pueden sacar las filas que no sirven
    final = datos.loc[datos["fila_valida"] == 1, VARIABLES + [OBJETIVO]].copy()
    final["conjunto"] = dividir(final)

    RUTA_DATASET_FINAL.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator fijo para que el archivo (y su hash) sea igual en Windows, Linux y Mac
    final.to_csv(RUTA_DATASET_FINAL, float_format="%.4f", lineterminator="\n")
    checksums = actualizar_checksums([RUTA_DATASET_FINAL], CARPETA_DATOS)

    for nombre in ["train", "validation", "test"]:
        parte = final[final["conjunto"] == nombre]
        print(f"X_{nombre}: {parte[VARIABLES].shape} | {parte.index.min().date()} a {parte.index.max().date()}")

    linaje = {
        "archivo": "processed/dataset_final.csv",
        "sha256": checksums["processed/dataset_final.csv"],
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "script": "src/preprocessing_pipeline.py",
        "semilla": SEMILLA,
        "origen": [
            {"archivo": nombre, "sha256": huella(CARPETA_DATOS / nombre), "prioridad": i + 1}
            for i, nombre in enumerate(ARCHIVOS_ENTRENAMIENTO)
        ],
        "hoja": "Consolidado EE",
        "columnas_base": COLUMNAS_BASE,
        "transformaciones": [
            "Lectura de la hoja Consolidado EE y conversión de textos a números",
            "Reasignación de los días apilados bajo una misma fecha al día faltante anterior más cercano",
            "Total diario = registro de las 23:59 de cada día",
            "Días sin 23:59 que llegan a las 23:00: estimación con el ritmo de las últimas 3 horas",
            f"Totales de {UMBRAL_IMPOSIBLE:.0f} o más y negativos: se vacían (frio_kw se reconstruye sacando el salto)",
            "Unión de los tres archivos por día, con prioridad 2021_2023 > 2022_2023 > 2020_2022",
            f"Huecos de hasta {LIMITE_HUECO_CORTO} días en las predictoras: mediana de los 7 días anteriores",
            "Variables: promedios móviles, mínimo y máximo de 7 días, y calendario del día a predecir",
            "Target y = frio_kw del día siguiente (shift(-1))",
            "Se eliminan las filas sin target o de días vacíos",
            "División temporal 70% train, 15% validation, 15% test. Datos sin escalar",
        ],
        "variables": VARIABLES,
        "objetivo": OBJETIVO,
        "filas": {nombre: int((final["conjunto"] == nombre).sum()) for nombre in ["train", "validation", "test"]},
    }
    with open(RUTA_LINAJE, "w", encoding="utf-8") as f:
        json.dump(linaje, f, indent=2, ensure_ascii=False)

    print("Dataset guardado en", RUTA_DATASET_FINAL.relative_to(RAIZ))
    print("Linaje guardado en", RUTA_LINAJE.relative_to(RAIZ))


if __name__ == "__main__":
    try:
        main()
    except ErrorDeDatos as error:
        sys.exit(f"ERROR: {error}")
