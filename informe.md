# Informe: predicción del consumo de frío del día siguiente

Trabajo final de Laboratorio de Datos II.

## 1. Resumen

Armamos un pipeline reproducible que predice el consumo eléctrico diario del sistema de refrigeración (`Frio (kW)`) del día siguiente en una planta cervecera, a partir de los totalizadores horarios de la planta.

El modelo final es una regresión Lasso (versión 1.0.0). Sobre el conjunto de test, del 2023-04-24 al 2023-10-24:

| | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Persistencia | 3.492 | 4.725 | 0,31 |
| Media móvil de 7 días | 3.067 | 3.967 | 0,51 |
| **Modelo final** | **2.868** | **3.821** | **0,55** |

El modelo le gana a los dos baselines y queda por debajo de los 3.000 kW, pero la mejora sobre el mejor baseline es chica (6,5%). Lo discutimos en las conclusiones.

## 2. Decisiones tomadas

El detalle y la justificación de cada decisión están en los notebooks. Acá van las principales.

### Definición del total diario

Usamos el registro de las **23:59** de cada día, y no el de las 23:00. El totalizador acumula desde las 00:00, así que el de las 23:59 es el que tiene el consumo de toda la jornada. Entre uno y otro hay en promedio unos 975 kW de diferencia. La definición es la misma en todo el trabajo: cuando a un día le falta el registro de las 23:59 pero llega hasta las 23:00, no usamos el valor de las 23:00 tal cual sino que estimamos el de las 23:59.

### Unificación y limpieza

| Problema | Decisión | Justificación |
|---|---|---|
| Los tres archivos se solapan en el tiempo | Prioridad 2021_2023 > 2022_2023 > 2020_2022 | Regla de desempate definida en el EDA al comparar los solapes. |
| Fechas con más de 25 registros (varios días apilados) | Cada bloque extra se asigna al día faltante anterior más cercano | Sin esto el `shift(-1)` queda desalineado. La regla general reproduce las reasignaciones hechas a mano en el EDA. |
| 3 días que llegan solo hasta las 23:00 | Se estima el valor de las 23:59 con el ritmo de las últimas 3 horas | Error de unos 230 kW, contra 975 kW de usar las 23:00 sin corregir. |
| Totales imposibles (un millón o más) | Se vacían; el de `frio_kw` se reconstruye sacando el salto | Son saltos del totalizador, no consumos. Error de prueba de unos 195 kW. |
| Valores negativos | Se vacían (menos las temperaturas) | Un consumo no puede ser negativo. |
| Días sin reinicio del totalizador | No se corrige nada | El EDA no encontró casos con impacto. |
| Faltantes: bloques largos y KPI sin producción | No se imputan | No hay datos cercanos para estimarlos, o el dato directamente no existe ese día. |
| Faltantes: huecos de hasta 3 días | Mediana de los 7 días anteriores | Solo mira hacia atrás: no usa el futuro ni aprende del dataset. |
| `frio_kw` faltante | No se imputa nunca | Es el target. Esas filas no se usan para entrenar. |
| Outliers de `frio_kw` | No se tocan | Son días de calor y paradas reales de la planta. |

### Variables

Creamos promedios móviles, mínimos y máximos de `frio_kw`, promedios de otras áreas y variables de calendario del día a predecir. Todas se calculan con lo que se sabe al terminar el día D.

La selección se hizo **solo con el tramo de entrenamiento**, con dos métodos: importancia de variables con Random Forest y LASSO. Quedaron 14 variables, y todas salen de una sola hoja del Excel (Consolidado EE). Eso hace que el script de predicción dependa de muy poco del archivo nuevo.

### División de los datos

Por orden temporal: 70% más antiguo para train (843 días), 15% para validation (180) y 15% más reciente para test (182). El modelo y los hiperparámetros se eligieron mirando únicamente validation. Test se evaluó una sola vez, con el modelo ya elegido.

La imputación y el escalado están dentro de un `Pipeline` de scikit-learn, que se ajusta solo con los datos de entrenamiento. El dataset guardado está sin escalar.

### Modelos

Probamos Ridge, Lasso, Random Forest, XGBoost y LightGBM, 46 combinaciones de hiperparámetros en total. Todas quedaron registradas en `results/experiment_logs.csv`.

| Modelo | Mejor MAE en validation (kW) |
|---|---|
| Lasso | 2.294 |
| Ridge | 2.327 |
| XGBoost | 2.338 |
| LightGBM | 2.356 |
| Random Forest | 2.405 |
| *Media móvil de 7 días* | *2.583* |
| *Persistencia* | *2.864* |

Elegimos Lasso con `alpha = 500` por tener el menor MAE en validation. Después lo reentrenamos con train + validation.

## 3. Conclusiones

### 3.1 ¿El modelo agrega valor?

| Sobre test | MAE (kW) | Mejora del modelo |
|---|---|---|
| Persistencia | 3.492 | 17,9% |
| Media móvil de 7 días | 3.067 | 6,5% |
| Modelo final | 2.868 | |

Agrega valor, pero poco. La mejora sobre el mejor baseline es de 199 kW, un 6,5%, que es menos del 10%.

¿Qué estamos comprando con la complejidad que agregamos? Bastante poco, y por eso creemos que igual conviene: el modelo final es una regresión lineal con cuatro coeficientes distintos de cero (`frio_media_14`, `frio_kw`, `planta_kw` y `frio_media_7`). Se entrena en 14 milésimas de segundo y solo necesita scikit-learn. En la práctica es una versión afinada de los baselines: un promedio entre el consumo de hoy y el de las últimas dos semanas, corregido por el consumo total de la planta.

Lo que sí cuesta mantener no es el modelo sino el **pipeline de limpieza** (días apilados, totales imposibles, días incompletos). Pero ese pipeline hace falta igual para calcular los baselines, así que no es un costo del modelo.

La compraríamos, sabiendo que la ganancia es chica. Si hubiéramos terminado eligiendo XGBoost o Random Forest para conseguir esta misma mejora, diríamos que no: ahí sí se suman dependencias, tiempo y un modelo que no se puede interpretar.

### 3.2 ¿Qué variables se parecen demasiado al objetivo?

`sala_maq_kw` y `servicios_kw` tienen una correlación de Spearman de 0,98 y 0,95 con `frio_kw` **del mismo día**, porque el frío es la mayor parte de la sala de máquinas y de los servicios. `kw_gral_planta` y `planta_kw` tienen 0,999 entre sí: son el mismo consumo medido desde dos hojas.

Las dejamos entrar como candidatas. Nuestro caso es el legítimo: usamos el valor de **hoy** (día D) para predecir el consumo de **mañana** (D+1). Al terminar el día D ese valor ya se conoce.

Cómo nos aseguramos de no usar el valor de mañana:

- El target se construye con `shift(-1)` sobre `frio_kw`, y es el único dato del día D+1 que entra en cada fila.
- Todos los promedios móviles terminan en el día D.
- Las únicas variables que miran a D+1 son las de calendario (día de la semana y mes), que se conocen de antemano.
- `predict.py` arma las variables a partir de un archivo que no tiene el día siguiente, y sobre el tramo de test da exactamente el mismo MAE que el entrenamiento (2.868,2 kW). Si alguna variable usara el futuro, esos números no coincidirían.

El problema real de estas variables no es la fuga sino la redundancia. En el modelo final `sala_maq_kw` y `servicios_kw` quedaron con coeficiente cero: una vez que está `frio_kw`, no aportan nada.

### 3.3 Un resultado que contradice la intuición

La consigna afirma que hay un patrón semanal. Nuestra medición dice que no:

- El día de la semana explica el **0,5%** de la variación de `frio_kw`. El mes explica el 57%.
- Entre el día de mayor consumo promedio y el de menor hay 1.698 kW (7% de la media), pero dentro de cada día el desvío es de unos 7.500 kW.
- El test de Kruskal-Wallis da un p-valor de 0,47: no hay evidencia de diferencias entre días.

Qué pudo haberlo borrado: el frío mantiene tanques y cámaras a temperatura las 24 horas, todos los días, se produzca o no. Lo que cambia entre un día hábil y un domingo es la producción, pero la cerveza que está fermentando o guardada hay que seguir enfriándola. Lo vimos también dentro del día: entre la hora de mayor y la de menor consumo hay apenas un 15%.

Qué hicimos en consecuencia: creamos igual las variables de día de la semana y fin de semana, porque la consigna las sugiere, y dejamos que decidieran los números. El Random Forest no las eligió, LASSO les dio coeficiente cero, y en validation el MAE promedio de los cuatro tipos de modelo fue de 2.403 kW con calendario y 2.409 sin calendario. Quedaron en la lista de variables, pero en el modelo final su coeficiente es cero: no las usa.

### 3.4 Un umbral para reemplazar un modelo

Proponemos **230 kW**: no reemplazaríamos un modelo desplegado por otro que no baje el MAE al menos esa cantidad.

De dónde sale: comparamos el modelo y la media móvil día por día sobre los 182 días de test. La diferencia entre los errores absolutos de los dos tiene una media de 199 kW y un desvío de 1.529 kW. El error estándar de esa diferencia media es 1.529 / √182 ≈ 113 kW. Dos errores estándar son unos 230 kW: una diferencia menor que eso puede ser ruido del período en que se midió.

El umbral es exigente, y hay una razón: el error cambia mucho según el período. En la validación cruzada temporal el MAE del modelo fue de 2.246 a 4.676 kW según el fold, con un desvío de 925 kW.

Aplicado a nosotros mismos, el umbral da una respuesta incómoda: la mejora de 199 kW sobre la media móvil **no lo alcanza**. Sí lo supera con margen la mejora sobre la persistencia (624 kW). O sea que podemos afirmar que el modelo es mejor que la persistencia, y solo que probablemente es mejor que la media móvil.

### 3.5 Lo que no usamos

**PCA.** Después de la selección quedamos con 14 variables, así que no había un problema de dimensionalidad que resolver. Y PCA mezcla las variables en componentes: perderíamos poder decir que el modelo se apoya en el consumo de hoy y en el promedio de dos semanas. Como el problema de fondo era la redundancia, preferimos selección (quedarnos con pocas variables originales) antes que extracción.

**LOF.** Es un método multivariado de detección de atípicos basado en distancias. Lo descartamos por dos motivos. Primero, con casi 300 columnas las distancias dejan de ser informativas (maldición de la dimensionalidad). Segundo, no íbamos a hacer nada con lo que marcara: los atípicos de `frio_kw` son días de calor o paradas reales que el modelo tiene que ver, y no eliminamos filas porque cada una es un día y sacar una desalinea los lags. Los errores evidentes los detectamos con reglas simples (un millón o más, valores negativos).

También dejamos afuera la temperatura de Open-Meteo, porque haría que `predict.py` dependa de tener internet, y la imputación con k-NN, que daba menos error pero no se justificaba para 142 celdas.

## 4. Limitaciones

- **La mejora sobre la media móvil no es estadísticamente firme**, como mostramos en 3.4.
- **No anticipa paradas ni caídas bruscas de producción.** Son sus peores errores: el mayor en validation fue de casi 12.000 kW, un día en que el consumo cayó a menos de la mitad sin aviso.
- **No usa la temperatura ambiente**, que es lo que más influye en el frío. Solo la ve de forma indirecta, por el consumo reciente y el mes.
- **Pocos datos y un solo test.** Son unos 3 años, y el comportamiento cambió de un año a otro: 2021 fue mucho más variable que 2023. El MAE de un período no garantiza el de otro.
- **El mejor `alpha` de Ridge quedó en el borde del espacio de búsqueda.** Lo vimos después de evaluar en test, y no ampliamos la búsqueda para no elegir sabiendo ese resultado.
- **Todas las variables salen del consumo eléctrico.** Si esos medidores fallan, el modelo se queda sin información.
- **No calculamos la eficiencia contra la meta**, porque las columnas `Meta` se descartaron en el EDA.

## 5. Reproducibilidad

- `SEMILLA = 42` en todos los modelos. La división es temporal, así que no depende del azar.
- Dos ejecuciones seguidas de los tres scripts dan el mismo dataset, el mismo modelo byte a byte y las mismas predicciones.
- `data/checksums.json` tiene el hash SHA-256 de los archivos crudos y de los datasets procesados. Usamos SHA-256 y no MD5 porque en MD5 se pueden fabricar dos archivos distintos con el mismo hash.
- `data/processed/data_lineage.json` registra las transformaciones, el script y la fecha.
- El modelo está versionado como `models/modelo_v1.0.0.pkl`, registrado en `models/model_registry.json` y etiquetado en Git con `v1.0.0`.

Los pasos para correr todo están en el `README.md`.
