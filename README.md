# Cervecera - Trabajo Final

Trabajo final de Laboratorio de Datos II. Es un pipeline de machine learning completo y reproducible para predecir el consumo energético del sistema de refrigeración, Frio (kW), del día siguiente en una planta cervecera mexicana.

## Cómo clonar el repositorio

```powershell
git clone https://github.com/franbaudra/cervecera.git
cd cervecera
```

## Configurar el entorno

Se necesita Python 3.12. 

```powershell
python -m venv cervecera_env
cervecera_env\Scripts\Activate.ps1
pip install uv
uv pip install -r requirements.txt
```

En Linux o Mac el entorno se activa con `source cervecera_env/bin/activate`.

LightGBM necesita la librería OpenMP del sistema. En Windows ya viene incluida. En Linux, si falta, se instala con `sudo apt install libgomp1`, y en Mac con `brew install libomp`.

## Datos

Los archivos Excel no están en el repositorio (están en el `.gitignore`). Los proporciona la cátedra junto con la consigna. Hay que copiarlos en la carpeta `data/` con estos nombres:

- `Totalizadores Planta - 2021_2023.xlsx`
- `Totalizadores Planta 2020_2022.xlsx`
- `Totalizadores Planta 2022_2023.xlsx`

`data/checksums.json` tiene el hash SHA-256 de cada uno. `src/preprocessing_pipeline.py` los compara al empezar y avisa si algún archivo no es el mismo que usamos nosotros.

## Cómo ejecutar

Todos los comandos se corren desde la carpeta raíz del proyecto, con el entorno activado.

### 1. Preprocesamiento

```powershell
python src/preprocessing_pipeline.py
```

Lee los tres Excel y genera:

- `data/processed/dataset_final.csv`: el dataset limpio, con las variables del modelo, el target y la división en train, validation y test. Está sin escalar.
- `data/processed/data_lineage.json`: qué transformaciones se aplicaron, con qué script y cuándo.
- Agrega el hash del dataset a `data/checksums.json`.

### 2. Entrenamiento

```powershell
python src/train_model.py
```

Calcula los dos baselines, prueba Ridge, Lasso, Random Forest, XGBoost y LightGBM, y elige el de menor MAE en validation. Genera:

- `results/experiment_logs.csv`: una fila por cada prueba, con hiperparámetros, MAE, RMSE, R² y la fecha y hora.
- `models/modelo_v1.1.0.pkl`: el modelo final, reentrenado con train + validation. La versión anterior (`modelo_v1.0.0.pkl`) queda guardada.
- `models/model_registry.json`: versión, hiperparámetros, métricas, ruta del modelo y commit de Git.

### 3. Predicción

```powershell
python src/predict.py "data/Totalizadores Planta 2022_2023.xlsx"
```

Recibe un archivo Excel con la misma estructura que los de entrenamiento y genera `results/predicciones.csv`, con una fila por día:

- `fecha`: el día D, que es el día de los datos.
- `prediccion_frio_kw`: el consumo predicho para el día D+1.

El archivo tiene que tener la hoja `Consolidado EE` con las columnas `DIA`, `HORA`, `Frio (Kw)`, `Sala Maq (Kw)`, `Servicios (Kw)` y `Planta (Kw)`. Puede tener más hojas y más columnas, en cualquier orden. Si falta alguna de las necesarias, el script termina con un mensaje que dice cuál.

### Notebooks

Los notebooks explican y justifican las decisiones. No hace falta correrlos para usar los scripts.

| Notebook | Contenido |
|---|---|
| `notebooks/eda_1_1.ipynb` | Fase 1.1: lectura, unificación de los archivos, checksums y análisis descriptivo. Genera `dataset_unificado_0.csv`. |
| `notebooks/eda.ipynb` | Fase 1.2 a 1.4: análisis temporal, correlaciones, visualizaciones y conclusiones del EDA. |
| `notebooks/preprocesamiento.ipynb` | Fase 2: limpieza, feature engineering, selección de variables y dataset final. |
| `notebooks/modelado.ipynb` | Fase 3: baselines, comparación de modelos, residuos, importancia de variables y modelo final. |

`preprocesamiento.ipynb` y `modelado.ipynb` leen archivos que generan los scripts, así que hay que correr primero los pasos 1 y 2. `eda.ipynb` lee `data/processed/dataset_unificado_0.csv`, que ya está en el repositorio.

## Informe

Las decisiones, su justificación, las conclusiones y las limitaciones del modelo están en [`informe.md`](informe.md).

## Resultados

MAE sobre el conjunto de test (del 2023-04-24 al 2023-10-24), en kW:

| | MAE |
|---|---|
| Persistencia | 3.492 |
| Media móvil de 7 días | 3.067 |
| Modelo v1.0.0 (Lasso, 14 variables) | 2.868 |
| **Modelo final v1.1.0 (Lasso, 17 variables)** | **2.709** |

## Estructura del proyecto

```
├── data/
│   ├── Totalizadores Planta*.xlsx     (no se versionan)
│   ├── checksums.json
│   └── processed/
│       ├── dataset_final.csv
│       └── data_lineage.json
├── models/
│   ├── modelo_v1.0.0.pkl
│   ├── modelo_v1.1.0.pkl
│   └── model_registry.json
├── notebooks/
│   ├── eda_1_1.ipynb
│   ├── eda.ipynb
│   ├── preprocesamiento.ipynb
│   └── modelado.ipynb
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   ├── predict.py
│   ├── eda_1_1_auxiliar.py            (funciones auxiliares)
│   └── integridad.py                  (checksums)
├── results/
│   ├── predicciones.csv
│   └── experiment_logs.csv
├── consigna.pdf
├── informe.md
├── requirements.txt
└── README.md
```

## Reproducibilidad

Se usa `SEMILLA = 42` en todos los modelos. La división en train, validation y test es por orden temporal, así que no depende del azar. Dos ejecuciones seguidas de los scripts dan las mismas métricas.

## Integrantes

- Francisco Baudracco
- Camila Roncoroni
