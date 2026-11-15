# Informe: predicción del consumo de frío del día siguiente

Trabajo final de Laboratorio de Datos II.

## 1. Resumen

Armamos un pipeline reproducible que predice el consumo eléctrico diario del sistema de refrigeración (`Frio (kW)`) del día siguiente en una planta cervecera, a partir de los totalizadores horarios de la planta.

El modelo final es una regresión Lasso (versión 1.1.0). Sobre el conjunto de test, del 2023-04-24 al 2023-10-24:

| | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Persistencia | 3.492 | 4.725 | 0,31 |
| Media móvil de 7 días | 3.067 | 3.967 | 0,51 |
| **Modelo final (v1.1.0)** | **2.709** | **3.714** | **0,57** |

El modelo le gana a los dos baselines, queda por debajo de los 3.000 kW y mejora un 11,7% al mejor baseline.

Hubo dos versiones. La 1.0.0 usaba solo los totales diarios y dio 2.868 kW en test. La 1.1.0 agrega el consumo de las últimas 6 horas del día y es la que queda como modelo final. El cambio se explica en la sección 2.

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

La selección se hizo **solo con el tramo de entrenamiento**, con dos métodos: importancia de variables con Random Forest y LASSO. Todas las variables elegidas salen de una sola hoja del Excel (Consolidado EE). Eso hace que el script de predicción dependa de muy poco del archivo nuevo.

### De la versión 1.0.0 a la 1.1.0

La primera versión usaba 14 variables, todas calculadas con el total diario. Dio 2.868 kW en test, apenas un 6,5% mejor que la media móvil y muy cerca del límite de 3.000 kW.

Buscando qué información no estábamos usando, vimos que el modelo solo conocía el total del día, cuando los archivos tienen las lecturas hora por hora. Agregamos el **consumo de las últimas 6 horas del día** (de las 18:00 al último registro) de frío, planta y servicios. Es el dato más reciente que existe antes del día a predecir, y sigue siendo información del día D.

La decisión se tomó sin mirar test:

- En validation el MAE bajó de 2.294 a 2.191 kW, y también mejoró en la validación cruzada temporal sobre train + validation.
- La mejora apareció en los cuatro tipos de modelo, no solo en el elegido.
- Los dos métodos de selección eligieron las variables nuevas: para el Random Forest `frio_ult_6h` es la cuarta en importancia, y para LASSO `planta_ult_6h` tiene el segundo coeficiente más grande.

Con las 3 variables nuevas quedaron 17. El espacio de búsqueda y el criterio para elegir modelo no se tocaron, para que lo único distinto entre las versiones sean las variables.

También probamos cosas que **no** mejoraron en validation y quedaron afuera: promediar las predicciones de Lasso con las de XGBoost o LightGBM, agregar la producción de mosto del día y agregar el ciclo anual con seno y coseno.

### División de los datos

Por orden temporal: 70% más antiguo para train (843 días), 15% para validation (180) y 15% más reciente para test (182). El modelo, los hiperparámetros y las variables se eligieron mirando únicamente train y validation.

Test se evaluó dos veces, una por versión del modelo, y cada vez con todo ya decidido. No se eligió nada mirando test. Igual es justo decirlo: el número de la versión 1.1.0 no es tan limpio como el de un test que se usa una única vez.

La imputación y el escalado están dentro de un `Pipeline` de scikit-learn, que se ajusta solo con los datos de entrenamiento. El dataset guardado está sin escalar.

### Modelos

Probamos Ridge, Lasso, Random Forest, XGBoost y LightGBM, 46 combinaciones de hiperparámetros por corrida. Todas quedaron registradas en `results/experiment_logs.csv`. Estos son los resultados de la versión 1.1.0:

| Modelo | Mejor MAE en validation (kW) |
|---|---|
| Lasso | 2.191 |
| Ridge | 2.246 |
| LightGBM | 2.340 |
| XGBoost | 2.352 |
| Random Forest | 2.379 |
| *Media móvil de 7 días* | *2.583* |
| *Persistencia* | *2.864* |

Elegimos Lasso con `alpha = 500` por tener el menor MAE en validation. Después lo reentrenamos con train + validation.

## 3. Conclusiones

### 3.1 ¿El modelo agrega valor?

| Sobre test | MAE (kW) | Mejora del modelo final |
|---|---|---|
| Persistencia | 3.492 | 22,4% |
| Media móvil de 7 días | 3.067 | 11,7% |
| Modelo final (v1.1.0) | 2.709 | |

Sí. La mejora sobre el mejor baseline es de 358 kW, un 11,7%.

No siempre fue así: la versión 1.0.0 mejoraba solo un 6,5%, y en ese momento la respuesta honesta era que el modelo agregaba poco sobre promediar la última semana. Lo que cambió el resultado no fue un modelo más complejo sino una variable mejor, el consumo de las últimas horas del día.

¿Qué estamos comprando con la complejidad que agregamos? El modelo final es una regresión lineal con cinco coeficientes distintos de cero (`frio_media_14`, `frio_kw`, `planta_ult_6h`, `frio_media_7` y `servicios_ult_6h`). Se entrena en centésimas de segundo y solo necesita scikit-learn. Se puede explicar en una frase: el consumo de mañana se parece a un promedio entre el de hoy y el de las últimas dos semanas, corregido por cuánto estaba trabajando la planta al terminar el día.

Lo que sí cuesta mantener no es el modelo sino el **pipeline de limpieza** (días apilados, totales imposibles, días incompletos). Pero ese pipeline hace falta igual para calcular los baselines, así que no es un costo del modelo.

La compraríamos. Si para conseguir esta mejora hubiéramos tenido que usar XGBoost o Random Forest, lo dudaríamos más: ahí sí se suman dependencias, tiempo y un modelo que no se puede interpretar. En validation, además, rindieron peor que Lasso.

### 3.2 ¿Qué variables se parecen demasiado al objetivo?

`sala_maq_kw` y `servicios_kw` tienen una correlación de Spearman de 0,98 y 0,95 con `frio_kw` **del mismo día**, porque el frío es la mayor parte de la sala de máquinas y de los servicios. `kw_gral_planta` y `planta_kw` tienen 0,999 entre sí: son el mismo consumo medido desde dos hojas.

Las dejamos entrar como candidatas. Nuestro caso es el legítimo: usamos el valor de **hoy** (día D) para predecir el consumo de **mañana** (D+1). Al terminar el día D ese valor ya se conoce.

Cómo nos aseguramos de no usar el valor de mañana:

- El target se construye con `shift(-1)` sobre `frio_kw`, y es el único dato del día D+1 que entra en cada fila.
- Todos los promedios móviles terminan en el día D.
- Las únicas variables que miran a D+1 son las de calendario (día de la semana y mes), que se conocen de antemano.
- `predict.py` arma las variables a partir de un archivo que no tiene el día siguiente, y sobre el tramo de test da exactamente el mismo MAE que el entrenamiento (2.708,8 kW). Si alguna variable usara el futuro, esos números no coincidirían.

El problema real de estas variables no es la fuga sino la redundancia. En el modelo final `sala_maq_kw` y `servicios_kw` quedaron con coeficiente cero: una vez que está `frio_kw`, no aportan nada.

### 3.3 Un resultado que contradice la intuición

La consigna afirma que hay un patrón semanal. Nuestra medición dice que no:

- El día de la semana explica el **0,5%** de la variación de `frio_kw`. El mes explica el 57%.
- Entre el día de mayor consumo promedio y el de menor hay 1.698 kW (7% de la media), pero dentro de cada día el desvío es de unos 7.500 kW.
- El test de Kruskal-Wallis da un p-valor de 0,47: no hay evidencia de diferencias entre días.

Qué pudo haberlo borrado: el frío mantiene tanques y cámaras a temperatura las 24 horas, todos los días, se produzca o no. Lo que cambia entre un día hábil y un domingo es la producción, pero la cerveza que está fermentando o guardada hay que seguir enfriándola. Lo vimos también dentro del día: entre la hora de mayor y la de menor consumo hay apenas un 15%.

Qué hicimos en consecuencia: creamos igual las variables de día de la semana y fin de semana, porque la consigna las sugiere, y dejamos que decidieran los números. El Random Forest no las eligió, LASSO les dio coeficiente cero, y en validation el MAE promedio de los cuatro tipos de modelo fue de 2.356 kW con calendario y 2.348 sin calendario: prácticamente lo mismo. Quedaron en la lista de variables, pero en el modelo final su coeficiente es cero: no las usa.

### 3.4 Un umbral para reemplazar un modelo

Proponemos **100 kW**: no reemplazaríamos un modelo desplegado por otro que no baje el MAE al menos esa cantidad.

De dónde sale: de comparar nuestras dos versiones día por día sobre los 182 días de test. Para cada día restamos el error absoluto de una versión menos el de la otra. Esa diferencia tiene un desvío de 689 kW, así que el error estándar de la diferencia media es 689 / √182 ≈ 51 kW. Dos errores estándar son unos 100 kW. Una diferencia de MAE más chica que eso puede ser ruido del período en que se midió, y no justifica el trabajo ni el riesgo de cambiar un modelo que ya está funcionando.

Lo aplicamos a nuestra propia decisión: la versión 1.1.0 baja el MAE 159 kW respecto de la 1.0.0, por encima del umbral. Por eso la reemplazamos.

Dos aclaraciones:

- El umbral vale para comparar modelos parecidos. Cuando los dos modelos son muy distintos, sus errores se parecen menos día a día y hace falta más diferencia para estar seguros. Contra la media móvil, por ejemplo, el error estándar es de 131 kW y el umbral equivalente sería de unos 260 kW. La versión 1.1.0 la supera (358 kW); la 1.0.0 no llegaba (199 kW).
- Una diferencia medida en un solo período no dice todo. En la validación cruzada temporal el MAE del modelo fue de 2.134 a 4.675 kW según el fold, con un desvío de 967 kW. Antes de reemplazar un modelo en producción convendría ver que la mejora se repite en más de un período.

### 3.5 Lo que no usamos

**PCA.** Después de la selección quedamos con 17 variables, así que no había un problema de dimensionalidad que resolver. Y PCA mezcla las variables en componentes: perderíamos poder decir que el modelo se apoya en el consumo de hoy y en el promedio de dos semanas. Como el problema de fondo era la redundancia, preferimos selección (quedarnos con pocas variables originales) antes que extracción.

**LOF.** Es un método multivariado de detección de atípicos basado en distancias. Lo descartamos por dos motivos. Primero, con casi 300 columnas las distancias dejan de ser informativas (maldición de la dimensionalidad). Segundo, no íbamos a hacer nada con lo que marcara: los atípicos de `frio_kw` son días de calor o paradas reales que el modelo tiene que ver, y no eliminamos filas porque cada una es un día y sacar una desalinea los lags. Los errores evidentes los detectamos con reglas simples (un millón o más, valores negativos).

También dejamos afuera la temperatura de Open-Meteo, porque haría que `predict.py` dependa de tener internet, y la imputación con k-NN, que daba menos error pero no se justificaba para 142 celdas.

## 4. Limitaciones

- **No anticipa paradas ni caídas bruscas de producción.** Son sus peores errores: el mayor en validation fue de más de 12.000 kW, un día en que el consumo cayó a menos de la mitad sin aviso.
- **No usa la temperatura ambiente**, que es lo que más influye en el frío. Solo la ve de forma indirecta, por el consumo reciente.
- **Pocos datos y un solo test.** Son unos 3 años, y el comportamiento cambió de un año a otro: 2021 fue mucho más variable que 2023. El MAE de un período no garantiza el de otro.
- **Test se evaluó dos veces**, una por versión. No se eligió nada mirándolo, pero la segunda medición es menos independiente que la primera.
- **El mejor `alpha` de Ridge quedó en el borde del espacio de búsqueda.** No lo ampliamos, para mantener el mismo espacio en las dos versiones.
- **Todas las variables salen del consumo eléctrico.** Si esos medidores fallan, el modelo se queda sin información.
- **No calculamos la eficiencia contra la meta**, porque las columnas `Meta` se descartaron en el EDA.

## 5. Reproducibilidad

- `SEMILLA = 42` en todos los modelos. La división es temporal, así que no depende del azar.
- Dos ejecuciones seguidas de los tres scripts dan el mismo dataset, el mismo modelo byte a byte y las mismas predicciones.
- `data/checksums.json` tiene el hash SHA-256 de los archivos crudos y de los datasets procesados. Usamos SHA-256 y no MD5 porque en MD5 se pueden fabricar dos archivos distintos con el mismo hash.
- `data/processed/data_lineage.json` registra las transformaciones, el script y la fecha.
- Cada versión del modelo está guardada (`models/modelo_v1.0.0.pkl` y `models/modelo_v1.1.0.pkl`), registrada en `models/model_registry.json` y etiquetada en Git (`v1.0.0` y `v1.1.0`). `predict.py` usa la más alta.

Los pasos para correr todo están en el `README.md`.
