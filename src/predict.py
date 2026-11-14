"""Predicción del consumo de Frio (kW) del día siguiente sobre un archivo Excel nuevo.

Uso:
    python src/predict.py "ruta/al/archivo.xlsx"

Genera results/predicciones.csv con una fila por día:
    fecha               -> el día D (el día de los datos)
    prediccion_frio_kw  -> consumo predicho para el día D+1

El archivo tiene que tener la hoja "Consolidado EE" con las columnas DIA, HORA, Frio (Kw),
Sala Maq (Kw), Servicios (Kw) y Planta (Kw). Puede tener más hojas y más columnas: se buscan
por nombre. Si falta alguna, el script termina con un mensaje que dice cuál.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing_pipeline import RAIZ, ErrorDeDatos, preparar_datos

RUTA_REGISTRO = RAIZ / "models" / "model_registry.json"
RUTA_PREDICCIONES = RAIZ / "results" / "predicciones.csv"


def leer_registro():
    if not RUTA_REGISTRO.exists():
        raise ErrorDeDatos(f"No existe {RUTA_REGISTRO}. Entrená primero el modelo: python src/train_model.py")
    with open(RUTA_REGISTRO, "r", encoding="utf-8") as f:
        return json.load(f)


def elegir_version(registry, version="latest"):
    """Devuelve la entrada del registro pedida. 'latest' es la versión semántica más alta."""
    if version == "latest":
        # "1.10.0" es mayor que "1.9.0": se compara número por número, no como texto
        version = max(registry, key=lambda v: tuple(int(n) for n in v.split(".")))
    if version not in registry:
        raise ErrorDeDatos(f"La versión '{version}' no está en el registro. Versiones disponibles: {list(registry)}")
    return registry[version]


def load_model(version="latest"):
    """Carga el modelo y los preprocesadores desde el registro.
    'latest' carga el modelo con la versión más alta.
    El modelo es un Pipeline de scikit-learn: incluye la imputación y el escalado
    ajustados en el entrenamiento."""
    entrada = elegir_version(leer_registro(), version)
    model_path = RAIZ / entrada["path"]
    if not model_path.exists():
        raise ErrorDeDatos(f"No existe el archivo del modelo: {model_path}")
    return joblib.load(model_path)


def load_and_preprocess(filepath, version="latest"):
    """Carga el archivo Excel y aplica todo el preprocesamiento.
    Reusa el pipeline de preprocessing_pipeline.py: no lo reimplementa.
    Devuelve la matriz de características y las fechas correspondientes."""
    variables = elegir_version(leer_registro(), version)["variables"]

    datos = preparar_datos(filepath)
    faltan = [v for v in variables if v not in datos.columns]
    if faltan:
        raise ErrorDeDatos(f"El preprocesamiento no generó las variables que espera el modelo: {faltan}")

    X_processed = datos[variables]
    fechas = datos.index
    return X_processed, fechas


def predict_consumption(filepath):
    """Función principal de predicción."""
    model = load_model()  # carga desde el registry
    X, fechas = load_and_preprocess(filepath)
    predictions = model.predict(X)

    # La predicción es DIARIA: una fila por día, no por hora.
    # 'fecha' es el día D; la predicción corresponde al consumo del día D+1.
    return pd.DataFrame({
        "fecha": fechas.strftime("%Y-%m-%d"),
        "prediccion_frio_kw": predictions.round(1),
    })


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Uso: python src/predict.py "ruta/al/archivo.xlsx"')

    try:
        results = predict_consumption(sys.argv[1])
    except ErrorDeDatos as error:
        sys.exit(f"ERROR: {error}")
    except Exception as error:
        sys.exit(f"ERROR inesperado al predecir ({type(error).__name__}): {error}")

    RUTA_PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RUTA_PREDICCIONES, index=False, lineterminator="\n")
    print(f"Predicciones generadas en results/predicciones.csv ({len(results)} días)")
