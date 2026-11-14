"""Entrenamiento, comparación y registro de modelos.

Lee data/processed/dataset_final.csv (lo genera src/preprocessing_pipeline.py) y:
    1. Calcula los dos baselines: persistencia y media móvil de 7 días.
    2. Prueba Ridge, Lasso, Random Forest, XGBoost y LightGBM con varios hiperparámetros.
       Cada prueba se entrena con train, se evalúa en validation y queda anotada en
       results/experiment_logs.csv.
    3. Elige el modelo con menor MAE en validation.
    4. Lo reentrena con train + validation y recién ahí lo evalúa en test, una sola vez.
    5. Guarda el modelo en models/ y lo anota en models/model_registry.json.

Uso:
    python src/train_model.py
"""
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from integridad import huella
from preprocessing_pipeline import OBJETIVO, RAIZ, RUTA_DATASET_FINAL, SEMILLA, VARIABLES

VERSION = "1.0.0"

CARPETA_MODELOS = RAIZ / "models"
RUTA_REGISTRO = CARPETA_MODELOS / "model_registry.json"
RUTA_LOGS = RAIZ / "results" / "experiment_logs.csv"

# Espacio de búsqueda: para cada modelo, los valores que se prueban de cada hiperparámetro.
# Se prueban todas las combinaciones.
ESPACIO_DE_BUSQUEDA = {
    "Ridge": {"alpha": [0.1, 1, 10, 100, 1000]},
    "Lasso": {"alpha": [1, 10, 100, 500, 1000]},
    "RandomForest": {
        "n_estimators": [300],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [3, 5, 10, 20],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "learning_rate": [0.02, 0.05],
        "max_depth": [2, 3, 4],
    },
    "LightGBM": {
        "n_estimators": [200, 400],
        "learning_rate": [0.02, 0.05],
        "num_leaves": [4, 8, 15],
    },
}


def crear_modelo(nombre, parametros):
    """Arma el Pipeline de scikit-learn de un modelo.

    Todos empiezan imputando con la mediana los pocos faltantes que quedan (por ejemplo los
    promedios móviles de los primeros días). Los modelos lineales además se estandarizan.
    Como todo está dentro del Pipeline, la mediana, la media y el desvío se calculan solo
    con los datos con los que se llama a fit()."""
    pasos = [("imputar", SimpleImputer(strategy="median"))]

    if nombre == "Ridge":
        pasos += [("escalar", StandardScaler()), ("modelo", Ridge(**parametros))]
    elif nombre == "Lasso":
        pasos += [("escalar", StandardScaler()), ("modelo", Lasso(max_iter=50000, **parametros))]
    elif nombre == "RandomForest":
        pasos += [("modelo", RandomForestRegressor(random_state=SEMILLA, n_jobs=-1, **parametros))]
    elif nombre == "XGBoost":
        pasos += [("modelo", XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                                          random_state=SEMILLA, n_jobs=1, **parametros))]
    elif nombre == "LightGBM":
        pasos += [("modelo", LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           min_child_samples=20, random_state=SEMILLA, n_jobs=1,
                                           deterministic=True, force_col_wise=True, verbose=-1,
                                           **parametros))]
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return Pipeline(pasos)


def metricas(real, prediccion):
    return {
        "MAE": round(float(mean_absolute_error(real, prediccion)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(real, prediccion))), 2),
        "R2": round(float(r2_score(real, prediccion)), 4),
    }


def cargar_datos():
    """Devuelve el dataset final separado en train, validation y test."""
    if not RUTA_DATASET_FINAL.exists():
        sys.exit(f"ERROR: no existe {RUTA_DATASET_FINAL}. Corré primero: python src/preprocessing_pipeline.py")
    datos = pd.read_csv(RUTA_DATASET_FINAL, parse_dates=["dia"], index_col="dia")
    return {nombre: datos[datos["conjunto"] == nombre] for nombre in ["train", "validation", "test"]}


def baselines(parte):
    """Los dos baselines sobre un conjunto. El día D predice el consumo de D+1.
    - Persistencia: mañana consume lo mismo que hoy.
    - Media móvil: mañana consume el promedio de los últimos 7 días (de D-6 a D)."""
    return {
        "Persistencia": metricas(parte[OBJETIVO], parte["frio_kw"]),
        "Media móvil 7 días": metricas(parte[OBJETIVO], parte["frio_media_7"]),
    }


def commit_actual():
    """Hash del commit de Git sobre el que se está entrenando."""
    try:
        salida = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True, check=True)
        return salida.stdout.strip()
    except Exception:
        return "desconocido"


def buscar_mejor_modelo(train, validation):
    """Entrena todas las combinaciones con train, las evalúa en validation y las anota en el log."""
    corrida = datetime.now().isoformat(timespec="seconds")
    filas = []
    for nombre, espacio in ESPACIO_DE_BUSQUEDA.items():
        claves = list(espacio)
        for valores in itertools.product(*espacio.values()):
            parametros = dict(zip(claves, valores))
            modelo = crear_modelo(nombre, parametros)

            inicio = time.perf_counter()
            modelo.fit(train[VARIABLES], train[OBJETIVO])
            segundos = time.perf_counter() - inicio

            filas.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "corrida": corrida,
                "modelo": nombre,
                "hiperparametros": json.dumps(parametros),
                "conjunto_evaluado": "validation",
                **metricas(validation[OBJETIVO], modelo.predict(validation[VARIABLES])),
                "MAE_train": metricas(train[OBJETIVO], modelo.predict(train[VARIABLES]))["MAE"],
                "segundos_entrenamiento": round(segundos, 3),
                "n_variables": len(VARIABLES),
                "semilla": SEMILLA,
            })
        mejor = min((f for f in filas if f["modelo"] == nombre), key=lambda f: f["MAE"])
        print(f"  {nombre:13s} mejor MAE en validation: {mejor['MAE']:8.1f}   {mejor['hiperparametros']}")

    logs = pd.DataFrame(filas)
    RUTA_LOGS.parent.mkdir(parents=True, exist_ok=True)
    # Cada corrida se agrega al final del archivo, sin borrar las anteriores
    logs.to_csv(RUTA_LOGS, mode="a", header=not RUTA_LOGS.exists(), index=False, lineterminator="\n")
    return logs


def main():
    partes = cargar_datos()
    train, validation, test = partes["train"], partes["validation"], partes["test"]
    print("Filas -> train:", len(train), "| validation:", len(validation), "| test:", len(test))

    print("\nBaselines en validation:")
    baselines_validation = baselines(validation)
    for nombre, m in baselines_validation.items():
        print(f"  {nombre:20s} MAE {m['MAE']:8.1f}")

    print("\nBúsqueda de hiperparámetros (se entrena con train, se evalúa en validation):")
    logs = buscar_mejor_modelo(train, validation)

    # El modelo se elige mirando SOLO validation
    mejor = logs.loc[logs["MAE"].idxmin()]
    nombre, parametros = mejor["modelo"], json.loads(mejor["hiperparametros"])
    print(f"\nModelo elegido: {nombre} {parametros} (MAE en validation: {mejor['MAE']})")

    # Reentrenamiento con train + validation
    train_y_validation = pd.concat([train, validation])
    modelo_final = crear_modelo(nombre, parametros)
    modelo_final.fit(train_y_validation[VARIABLES], train_y_validation[OBJETIVO])

    # Qué tan estable es el MAE: validación cruzada temporal sobre train + validation
    mae_folds = []
    for indices_train, indices_test in TimeSeriesSplit(n_splits=5).split(train_y_validation):
        parte_train, parte_test = train_y_validation.iloc[indices_train], train_y_validation.iloc[indices_test]
        modelo_fold = crear_modelo(nombre, parametros).fit(parte_train[VARIABLES], parte_train[OBJETIVO])
        mae_folds.append(mean_absolute_error(parte_test[OBJETIVO], modelo_fold.predict(parte_test[VARIABLES])))

    # Test se toca una sola vez, acá, con el modelo ya elegido
    metricas_test = metricas(test[OBJETIVO], modelo_final.predict(test[VARIABLES]))
    baselines_test = baselines(test)
    mejor_baseline = min(m["MAE"] for m in baselines_test.values())
    mejora = (mejor_baseline - metricas_test["MAE"]) / mejor_baseline * 100

    print("\nResultados en test:")
    for nombre_baseline, m in baselines_test.items():
        print(f"  {nombre_baseline:20s} MAE {m['MAE']:8.1f}")
    print(f"  {'Modelo final':20s} MAE {metricas_test['MAE']:8.1f}   (mejora de {mejora:.1f}% sobre el mejor baseline)")

    # Modelo serializado con versión semántica
    CARPETA_MODELOS.mkdir(exist_ok=True)
    ruta_modelo = CARPETA_MODELOS / f"modelo_v{VERSION}.pkl"
    joblib.dump(modelo_final, ruta_modelo)

    # Registro de modelos: una entrada por versión
    registro = json.load(open(RUTA_REGISTRO, encoding="utf-8")) if RUTA_REGISTRO.exists() else {}
    registro[VERSION] = {
        "version": VERSION,
        "path": ruta_modelo.relative_to(RAIZ).as_posix(),
        "sha256": huella(ruta_modelo),
        "modelo": nombre,
        "hiperparametros": parametros,
        "variables": VARIABLES,
        "entrenado_con": "train + validation",
        "filas": {"train": len(train), "validation": len(validation), "test": len(test)},
        "metricas_validacion": {k: float(mejor[k]) for k in ["MAE", "RMSE", "R2"]},
        "baselines_validacion": baselines_validation,
        "metricas_test": metricas_test,
        "baselines_test": baselines_test,
        "mejora_sobre_mejor_baseline_test_pct": round(mejora, 2),
        "mae_validacion_cruzada": {
            "folds": [round(float(m), 2) for m in mae_folds],
            "media": round(float(np.mean(mae_folds)), 2),
            "desvio": round(float(np.std(mae_folds, ddof=1)), 2),
        },
        "semilla": SEMILLA,
        "commit_git": commit_actual(),
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "sha256_dataset": huella(RUTA_DATASET_FINAL),
    }
    with open(RUTA_REGISTRO, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    print("\nModelo guardado en", ruta_modelo.relative_to(RAIZ))
    print("Registro actualizado en", RUTA_REGISTRO.relative_to(RAIZ))
    print("Experimentos anotados en", RUTA_LOGS.relative_to(RAIZ))


if __name__ == "__main__":
    main()
