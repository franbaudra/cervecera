import re
import unicodedata
import pandas as pd
import numpy as np
import itertools
from IPython.display import display
import copy


# -------------------------------- # LECTURA DE LOS ARCHIVOS # -------------------------------- #

# - Para evitar inconsistencias vamos a llevar todos los nombres de columnas a snake_case
def a_snake_case(texto):

    # Quitamos los espacios al principio y al final del texto 
    texto = str(texto).strip()
    # Saca tildes y ñ: "Elaboración" -> "Elaboracion"
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    # Pasamos a minúsculas
    texto = texto.lower()
    # Todo lo que no sea letra o número pasa a "_" y se colapsa
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    
    return texto or "sin_nombre"

def snake_case_unicas(columnas):
    # Si dos columnas terminan con el mismo nombre, agrega sufijo _1, _2...
    vistos, resultado = {}, []
    for c in map(a_snake_case, columnas):
        if c in vistos:
            vistos[c] += 1
            resultado.append(f"{c}_{vistos[c]}")
        else:
            vistos[c] = 0
            resultado.append(c)
    return resultado

def texto_a_numero(s):
    """Convierte textos numéricos con formato regional a float.
    '1.016.475' -> 1016475 | '26,5' -> 26.5 | '1.234,5' -> 1234.5"""
    s = s.astype(str).str.strip()
    con_coma = s.str.contains(",", regex=False)
    # Con coma: los puntos son miles y la coma es el decimal
    s = s.where(~con_coma,
                s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    # Sin coma pero con varios puntos: son separadores de miles
    varios_puntos = s.str.count(r"\.") > 1
    s = s.where(~varios_puntos, s.str.replace(".", "", regex=False))
    return pd.to_numeric(s, errors="coerce")


def limpiar_numericas(df, excluir=("dia", "hora", "unnamed_0"), umbral=0.5):
    df = df.copy()
    reporte = []
    for c in df.columns:
        if c in excluir or pd.api.types.is_numeric_dtype(df[c]):
            continue
        original = df[c]
        es_txt = original.map(lambda x: isinstance(x, str))

        convertida = pd.to_numeric(original, errors="coerce")
        convertida[es_txt] = texto_a_numero(original[es_txt])   # textos con formato regional

        con_contenido = original.notna().sum()
        if con_contenido == 0 or convertida.notna().sum() / con_contenido < umbral:
            continue

        perdidos = original.notna() & convertida.isna()
        if perdidos.any():
            reporte.append((c, int(perdidos.sum()),
                            original[perdidos].astype(str).unique()[:5].tolist()))
        df[c] = convertida
    return df, reporte


# -------------------------------- # LIMPIEZA DE COLUMNAS # -------------------------------- #

def comparar_columnas(datos, archivos, ref_i=None, mostrar=True, detalle=True):
    """Compara, hoja por hoja, las columnas entre archivos.
    Calcula las hojas comunes a partir de `datos`, así que sirve después de
    quitar hojas o columnas. ref_i: archivo de referencia (por defecto, el primero).
    Devuelve (resumen, diferencias)."""
    ref_i = min(datos) if ref_i is None else ref_i
    hojas_comunes = sorted(set.intersection(*[set(h) for h in datos.values()]))

    filas, diferencias = [], {}
    for hoja in hojas_comunes:
        cols_por_archivo = {i: list(datos[i][hoja].columns) for i in datos}
        ref = cols_por_archivo[ref_i]

        dif = {}
        for i, c in cols_por_archivo.items():
            faltan = [x for x in ref if x not in c]
            sobran = [x for x in c if x not in ref]
            if faltan or sobran:
                dif[i] = {"faltan": faltan, "sobran": sobran}
        if dif:
            diferencias[hoja] = dif

        filas.append({
            "Hoja": hoja,
            "Columnas por archivo": [len(c) for c in cols_por_archivo.values()],
            "Mismas columnas": all(set(c) == set(ref) for c in cols_por_archivo.values()),
            "Mismo orden": all(c == ref for c in cols_por_archivo.values()),
            "Archivos distintos": [archivos[i].name for i in dif],
        })

        if detalle and dif:
            print(f"\n[{hoja}] Diferencias respecto de {archivos[ref_i].name}:")
            for i, d in dif.items():
                print(f"  {archivos[i].name}:")
                print(f"     Faltan: {d['faltan']}")
                print(f"     Sobran: {d['sobran']}")

    resumen = pd.DataFrame(filas)
    if mostrar:
        display(resumen)
    return resumen, diferencias

def resumen_columnas_patron(datos, archivos, patron, min_faltantes=0, mostrar=True):
    """Tabla de columnas cuyo nombre cumple el patrón.
    min_faltantes: solo devuelve columnas con al menos ese % de faltantes (0-100).
    Orden: % de faltantes de mayor a menor."""
    filas = []
    for i, hojas in datos.items():
        for hoja, df in hojas.items():
            for c in [c for c in df.columns if re.search(patron, str(c))]:
                s = df[c]
                num = pd.to_numeric(s, errors="coerce")
                con_valor = s.notna().sum()
                es_numerica = con_valor > 0 and num.notna().sum() / con_valor >= 0.5
                filas.append({
                    "Archivo": archivos[i].name,
                    "Hoja": hoja,
                    "Columna": c,
                    "Tipo de dato": "numérico" if es_numerica else ("vacía" if con_valor == 0 else "texto / otro"),
                    "% datos faltantes": round(100 * s.isna().mean(), 2),
                    "Valores únicos": int(s.nunique(dropna=True)),
                    "Desvío estándar": num.std() if es_numerica else np.nan,
                })

    resumen = pd.DataFrame(filas)
    if resumen.empty:
        print("No hay columnas con ese patrón.")
        return resumen
    resumen = resumen[resumen["% datos faltantes"] >= min_faltantes]
    resumen = (resumen
               .sort_values(["% datos faltantes", "Archivo", "Hoja", "Columna"],
                            ascending=[False, True, True, True])
               .reset_index(drop=True))
    if mostrar:
        display(resumen)
    return resumen


def eliminar_columnas(datos, archivos, resumen, max_faltantes=90, min_desvio=10, min_unicos=None,
                      proteger=("dia", "hora"), aplicar=False, mostrar=True):
    """Elimina de `datos` las columnas de `resumen` que cumplan ALGUNO de estos criterios:
       - más de `max_faltantes` % de datos faltantes
       - desvío estándar menor que `min_desvio` (solo si se pudo calcular)
       - menos de `min_unicos` valores únicos (sin contar vacíos)
    Poné en None un criterio para desactivarlo.
    proteger: columnas que nunca se eliminan, aunque cumplan algún criterio.
    aplicar=False (por defecto) es un ensayo: muestra qué se borraría sin borrar nada.
    Devuelve un DataFrame con las columnas afectadas y el motivo."""
    if resumen is None or resumen.empty:
        print("Nada para evaluar.")
        return pd.DataFrame()

    idx_archivo = {a.name: i for i, a in enumerate(archivos)}
    registro = []
    for _, r in resumen.iterrows():
        if r["Columna"] in proteger:
            continue
        motivos = []
        if max_faltantes is not None and r["% datos faltantes"] > max_faltantes:
            motivos.append(f"faltantes > {max_faltantes}%")
        if min_desvio is not None and pd.notna(r["Desvío estándar"]) and r["Desvío estándar"] < min_desvio:
            motivos.append(f"desvío < {min_desvio}")
        if min_unicos is not None and r["Valores únicos"] < min_unicos:
            motivos.append(f"únicos < {min_unicos}")
        if not motivos:
            continue

        i = idx_archivo[r["Archivo"]]
        df = datos[i].get(r["Hoja"])
        if df is None or r["Columna"] not in df.columns:
            continue
        if aplicar:
            datos[i][r["Hoja"]] = df.drop(columns=r["Columna"])
        registro.append({
            "Archivo": r["Archivo"], "Hoja": r["Hoja"], "Columna": r["Columna"],
            "% datos faltantes": r["% datos faltantes"],
            "Valores únicos": r["Valores únicos"],
            "Desvío estándar": r["Desvío estándar"],
            "Motivo": " y ".join(motivos),
        })

    registro = pd.DataFrame(registro)
    print(f"Columnas {'eliminadas' if aplicar else 'que se eliminarían (ensayo)'}: {len(registro)}")
    if mostrar and not registro.empty:
        display(registro)
    return registro

def eliminar_por_nombre(datos, archivos, patron, aplicar=False, mostrar=True):
    """Elimina de `datos` las columnas cuyo nombre completo cumple `patron` (fullmatch).
    Ej: patron=r"id" borra 'id' pero no 'hidrologia'.
    aplicar=False (por defecto) es un ensayo: no borra nada."""
    registro = []
    for i, hojas in datos.items():
        for hoja, df in hojas.items():
            cols = [c for c in df.columns if re.fullmatch(patron, str(c))]
            if not cols:
                continue
            for c in cols:
                registro.append({"Archivo": archivos[i].name, "Hoja": hoja, "Columna": c})
            if aplicar:
                datos[i][hoja] = df.drop(columns=cols)

    registro = pd.DataFrame(registro)
    print(f"Columnas {'eliminadas' if aplicar else 'que se eliminarían (ensayo)'}: {len(registro)}")
    if mostrar and not registro.empty:
        display(registro)
    return registro

def acumuladores_sin_reiniciar(datos, archivos, hojas=None, columnas="todas",
                               col_dia="dia", col_hora="hora", solo_medianoche=True,
                               min_registros=2, tol=1e-9, mostrar=True):
    """Busca los días en que un acumulador diario no se reinició:
      1) el primer registro del día no es nulo (no vacío y distinto de 0), y
      2) los registros del día son estrictamente crecientes.
    Si no se cumple lo segundo, se descarta.
    Un bloque diario empieza cuando cambia el día o la hora baja.
    solo_medianoche=True: el primer registro tiene que ser de las 00:00 (antes de las 00:30).
    min_registros: mínimo de registros con valor que debe tener el día para evaluar si crece.
    hojas: lista de hojas, una sola, o None para todas las del archivo.
    Devuelve (resumen por archivo, hoja y columna, detalle de los días que cumplen la condición).
    En el detalle, 'Diferencia relativa (%)' = (último valor del día - último valor del día anterior)
    / |último valor del día anterior| * 100."""

    def a_timedelta(h):
        if pd.api.types.is_timedelta64_dtype(h):
            return h
        if pd.api.types.is_numeric_dtype(h):
            return pd.to_timedelta(h, unit="h")
        return pd.to_timedelta(h.astype(str).str.strip(), errors="coerce")

    if isinstance(hojas, str):
        hojas = [hojas]

    resumen, detalle = [], []
    for i, hojas_i in datos.items():
        for hoja in (hojas if hojas is not None else list(hojas_i)):
            if hoja not in hojas_i:
                continue
            df = hojas_i[hoja]
            if col_dia not in df.columns or col_hora not in df.columns:
                continue

            dia = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
            t = a_timedelta(df[col_hora])
            ok = (dia.notna() & t.notna()).to_numpy()
            d_ok, t_ok = dia[ok].reset_index(drop=True), t[ok].reset_index(drop=True)

            nuevo = (t_ok < t_ok.shift()) | (d_ok != d_ok.shift())
            b = nuevo.cumsum().to_numpy() - 1                       # número de bloque, desde 0
            n_bloques = int(b.max()) + 1 if len(b) else 0
            if n_bloques == 0:
                continue
            primero = np.flatnonzero(nuevo.to_numpy())              # posición del primer registro de cada bloque
            dias_bloque = d_ok.iloc[primero].dt.date.to_numpy()
            t_primero = t_ok.iloc[primero]
            a_medianoche = (t_primero < pd.Timedelta("00:30:00")).to_numpy()

            cols = ([c for c in df.columns if c not in (col_dia, col_hora)] if columnas == "todas"
                    else [c for c in columnas if c in df.columns])

            for c in cols:
                v = pd.to_numeric(df[c], errors="coerce")[ok].reset_index(drop=True)
                v0 = v.iloc[primero].to_numpy(float)                # valor del primer registro de cada bloque

                # Estrictamente creciente: todas las diferencias entre registros con valor son > 0
                con_valor = v.notna()
                sv, bb = v[con_valor], pd.Series(b)[con_valor.to_numpy()]
                n_reg = sv.groupby(bb).size().reindex(range(n_bloques), fill_value=0)
                malas = (sv.groupby(bb).diff() <= 0).groupby(bb).sum().reindex(range(n_bloques), fill_value=0)
                ultimo = sv.groupby(bb).last().reindex(range(n_bloques))
                estricto = ((n_reg >= min_registros) & (malas == 0)).to_numpy()

                primero_ok = ~np.isnan(v0) & (np.abs(np.nan_to_num(v0)) > tol)
                if solo_medianoche:
                    primero_ok &= a_medianoche
                flag = primero_ok & estricto

                resumen.append({
                    "Archivo": archivos[i].name, "Hoja": hoja, "Columna": c,
                    "Bloques": n_bloques,
                    "Primer registro no nulo": int(primero_ok.sum()),
                    "Sin reiniciar y estrictamente crecientes": int(flag.sum()),
                })
                if flag.any():
                    k = np.flatnonzero(flag)
                    ult = ultimo.to_numpy(float)
                    prev = ultimo.shift(1).to_numpy(float)          # último valor del bloque anterior
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dif_rel = np.where(np.abs(prev[k]) > tol, 100 * (ult[k] - prev[k]) / np.abs(prev[k]), np.nan)
                    detalle.append(pd.DataFrame({
                        "Archivo": archivos[i].name, "Hoja": hoja, "Columna": c,
                        "Día": dias_bloque[k],
                        "Hora del primer registro": t_primero.iloc[k].astype(str).str[-8:].to_numpy(),
                        "Primer valor": v0[k],
                        "Último valor del día": ult[k],
                        "Registros con valor": n_reg.to_numpy()[k],
                        "Último valor del día anterior": prev[k],
                        "Diferencia relativa (%)": np.round(dif_rel, 1),
                    }))

    resumen = pd.DataFrame(resumen)
    detalle = pd.concat(detalle, ignore_index=True) if detalle else pd.DataFrame()

    if mostrar:
        if resumen.empty:
            print("No se encontraron las hojas o columnas pedidas.")
        else:
            problemas = resumen[resumen["Sin reiniciar y estrictamente crecientes"] > 0]
            print(f"Columnas con días sin reiniciar: {len(problemas)} de {len(resumen)} evaluadas | "
                  f"días en total: {int(problemas['Sin reiniciar y estrictamente crecientes'].sum())}")
            if len(problemas):
                display(problemas.sort_values("Sin reiniciar y estrictamente crecientes", ascending=False)
                                 .reset_index(drop=True))
    return resumen, detalle

# -------------------------------- # UNIFICACIÓN DE HOJAS # -------------------------------- #

def filas_por_dia(datos, archivos, col_dia="dia", esperado=25, mostrar=True):
    """Cuenta cuántas filas hay por día en cada hoja de cada archivo, sin mostrar los días
    con menos de `esperado` filas.
    Devuelve (detalle, resumen): detalle por archivo-hoja-día y resumen con una fila por archivo."""
    partes = []
    for i, hojas in datos.items():
        for hoja, df in hojas.items():
            if col_dia not in df.columns:
                continue
            dia = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
            conteo = (dia.value_counts().sort_index()
                         .rename_axis("Día").reset_index(name="Cantidad de filas"))
            conteo.insert(0, "Hoja", hoja)
            conteo.insert(0, "Archivo", archivos[i].name)
            partes.append(conteo)

    if not partes:
        print("Ninguna hoja tiene la columna de fecha.")
        return pd.DataFrame(), pd.DataFrame()

    detalle = pd.concat(partes, ignore_index=True)
    detalle = detalle[detalle["Cantidad de filas"] >= esperado].reset_index(drop=True)   # fuera los días incompletos

    # Una fila por archivo: para cada día se toma el máximo entre sus hojas
    por_dia = detalle.groupby(["Archivo", "Día"])["Cantidad de filas"].max().reset_index()

    filas = []
    for archivo, g in por_dia.groupby("Archivo"):
        extra = g[g["Cantidad de filas"] > esperado]
        filas.append({
            "Archivo": archivo,
            f"Días con {esperado} filas": int((g["Cantidad de filas"] == esperado).sum()),
            f"Días con más de {esperado}": len(extra),
            "Máximo de filas por día": int(g["Cantidad de filas"].max()),
            f"Fechas con más de {esperado}": ", ".join(
                f"{d:%d/%m/%Y} ({n})" for d, n in zip(extra["Día"], extra["Cantidad de filas"])),
        })
    resumen = pd.DataFrame(filas)

    if mostrar:
        display(resumen)
    return detalle, resumen

def dias_con_muchas_filas(datos, archivos, hoja, umbral=25, col_dia="dia", mostrar=True):
    """Una fila por fecha con más de `umbral` filas en alguno de los archivos.
    Una columna por archivo con la cantidad de filas de esa fecha (vacío si no supera el umbral
    o la hoja no existe en ese archivo)."""
    series = {}
    for i, hojas in datos.items():
        if hoja not in hojas or col_dia not in hojas[hoja].columns:
            continue
        dia = pd.to_datetime(hojas[hoja][col_dia], errors="coerce").dt.normalize()
        conteo = dia.value_counts()
        series[archivos[i].stem] = conteo[conteo > umbral]

    if not series:
        print("Ningún archivo tiene esa hoja con la columna de fecha.")
        return pd.DataFrame()

    tabla = pd.DataFrame(series).sort_index()
    tabla.index.name = "Fecha"
    if tabla.empty:
        print(f"Ninguna fecha supera las {umbral} filas.")
        return tabla

    tabla = tabla.astype("Int64")        # enteros con vacíos (<NA>) donde no aplica
    if mostrar:
        display(tabla)
    return tabla

def fechas_faltantes(datos, archivos, col_dia="dia", mostrar=True):
    """Un solo DataFrame: una fila por fecha que falta en alguna hoja de algún archivo,
    una columna por archivo.
      Falta                   -> falta en todas las hojas que cubren esa fecha
      Falta en algunas hojas  -> falta en unas hojas y en otras no
      Presente                -> está en todas las hojas que la cubren
      —                       -> la fecha cae fuera del rango del archivo"""
    info = {}            # archivo -> {hoja: (desde, hasta, set de fechas faltantes)}
    for i, hojas in datos.items():
        for hoja, df in hojas.items():
            if col_dia not in df.columns:
                continue
            dias = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize().dropna()
            if dias.empty:
                continue
            falta = pd.date_range(dias.min(), dias.max(), freq="D").difference(dias.unique())
            info.setdefault(i, {})[hoja] = (dias.min(), dias.max(), set(falta))

    todas = sorted(set().union(*[f for hojas in info.values() for _, _, f in hojas.values()]))
    tabla = pd.DataFrame(index=pd.DatetimeIndex(todas, name="Fecha"))

    for i, hojas in info.items():
        valores = []
        for d in tabla.index:
            cubren = [f for lo, hi, f in hojas.values() if lo <= d <= hi]
            faltan = sum(d in f for f in cubren)
            if not cubren:
                valores.append("—")
            elif faltan == len(cubren):
                valores.append("Falta")
            elif faltan:
                valores.append("Falta en algunas hojas")
            else:
                valores.append("Presente")
        tabla[archivos[i].stem] = valores

    tabla["Estado"] = [
        "Falta en todos los que la cubren" if all(v in ("Falta", "—") for v in fila) and "Falta" in list(fila)
        else "Falta solo en algunos" for fila in tabla.drop(columns="Estado", errors="ignore").values
    ]
    tabla.index = tabla.index.strftime("%d/%m/%Y")

    if mostrar:
        display(tabla)
    return tabla

def mover_bloque(datos, archivos, i, fecha_origen, bloque, fecha_destino,
                 col_dia="dia", col_hora="hora", reordenar=True, aplicar=False):
    """Cambia la fecha de un bloque diario (un período de horas 00:00 → 23:59) de `fecha_origen`
    a `fecha_destino`, en TODAS las hojas del archivo `i`.
    bloque: número del bloque dentro de la fecha, contando desde 1 (2 = el segundo, 3 = el tercero...).
    Un bloque nuevo empieza cuando la hora vuelve a bajar (23:59 → 00:00).
    reordenar: ordena cada hoja por fecha (orden estable) para que el bloque quede en su lugar.
    aplicar=False (por defecto) es un ensayo: muestra qué se movería sin tocar nada."""
    origen = pd.Timestamp(fecha_origen).normalize()
    destino = pd.Timestamp(fecha_destino).normalize()
    registro = []

    for hoja, df in datos[i].items():
        if col_dia not in df.columns or col_hora not in df.columns:
            continue
        dia = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
        pos = np.flatnonzero((dia == origen).to_numpy())
        if len(pos) == 0:
            #registro.append({"Hoja": hoja, "Resultado": f"no hay filas con fecha {origen:%d/%m/%Y}"})
            continue

        hora = pd.to_timedelta(df[col_hora].iloc[pos].astype(str), errors="coerce")
        nro = hora.lt(hora.shift()).fillna(False).cumsum().to_numpy() + 1 # número de bloque, desde 1
        sel = pos[nro == bloque]
        if len(sel) == 0:
            registro.append({"Hoja": hoja, "Resultado": f"no existe el bloque {bloque} (hay {nro.max()})"})
            continue

        registro.append({
            "Hoja": hoja,
            "Bloques en la fecha": int(nro.max()),
            "Filas del bloque": len(sel),
            "Hora inicial": str(df[col_hora].iloc[sel[0]]),
            "Hora final": str(df[col_hora].iloc[sel[-1]]),
            "Filas que ya tiene el destino": int((dia == destino).sum()),
            "Resultado": "movido" if aplicar else "se movería (ensayo)",
        })

        if aplicar:
            df = df.copy()
            nuevo = destino if pd.api.types.is_datetime64_any_dtype(df[col_dia]) else destino.strftime("%Y-%m-%d")
            df.iloc[sel, df.columns.get_loc(col_dia)] = nuevo
            if reordenar:
                df = df.sort_values(col_dia, kind="stable").reset_index(drop=True)
            datos[i][hoja] = df

    registro = pd.DataFrame(registro)
    print(f"Archivo: {archivos[i].name} | bloque {bloque} de {origen:%d/%m/%Y} → {destino:%d/%m/%Y}")
    # display(registro)
    return registro


def frecuencia_ultima_hora(datos, archivos, hoja=None, col_dia="dia", col_hora="hora",
                           mostrar=True):
    """Para cada archivo, cuenta cuántos días terminan en cada hora (la hora más tardía
    registrada en cada día).
    hoja: nombre de una hoja para mirar solo esa; None = todas las hojas, por separado.
    Devuelve una tabla larga: Archivo - Hoja - Última hora - Frecuencia absoluta."""

    def hora_texto(t):
        s = int(t.total_seconds())
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    filas = []
    for i, hojas in datos.items():
        for h, df in hojas.items():
            if hoja is not None and h != hoja:
                continue
            if col_dia not in df.columns or col_hora not in df.columns:
                continue
            dia = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
            hr = df[col_hora]
            if pd.api.types.is_timedelta64_dtype(hr):
                t = hr
            elif pd.api.types.is_numeric_dtype(hr):
                t = pd.to_timedelta(hr, unit="h")
            else:
                t = pd.to_timedelta(hr.astype(str), errors="coerce")

            ok = dia.notna() & t.notna()
            ultima = t[ok].groupby(dia[ok]).max()            # última hora de cada día
            conteo = ultima.map(hora_texto).value_counts().sort_index()
            for hora, n in conteo.items():
                filas.append({"Archivo": archivos[i].name, "Hoja": h,
                              "Última hora": hora, "Frecuencia absoluta": int(n)})

    res = pd.DataFrame(filas)
    if res.empty:
        print("No se encontró ninguna hoja con día y hora.")
        return res
    res = res.sort_values(["Archivo", "Hoja", "Última hora"]).reset_index(drop=True)

    if mostrar:
        if hoja is not None:                                  # una hoja: hora x archivo
            display(res.pivot(index="Última hora", columns="Archivo",
                              values="Frecuencia absoluta").fillna(0).astype(int))
        else:
            display(res)
    return res

def _hora_a_timedelta(h):
    if pd.api.types.is_timedelta64_dtype(h):
        return h
    if pd.api.types.is_numeric_dtype(h):
        return pd.to_timedelta(h, unit="h")
    return pd.to_timedelta(h.astype(str).str.strip(), errors="coerce")


def filtrar_ultima_fila(datos, archivos, col_dia="dia", col_hora="hora", mostrar = True):
    """Devuelve una copia de `datos` con una sola fila por día, la de las 23:59:00:
      1) elimina las filas con hora 23:59:59
      2) lleva las horas con segundos a segundos 00 (04:00:01 -> 04:00:00)
      3) en los días que no tienen 23:59:00 crea esa fila, con todo en NaN
      4) se queda, para cada fecha, con la fila de hora == 23:59:00
    `datos` no se modifica. Devuelve (datos_filtrados, registro)."""
    datos_filtrados = copy.deepcopy(datos)
    ultima = pd.Timedelta("23:59:00")
    registro = []

    for i, hojas in datos_filtrados.items():
        for hoja, df in list(hojas.items()):
            if col_dia not in df.columns or col_hora not in df.columns:
                registro.append({"Archivo": archivos[i].name, "Hoja": hoja,
                                 "Observación": "sin columnas de día y hora: no se modificó"})
                continue

            df = df.copy()
            df[col_dia] = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
            t = _hora_a_timedelta(df[col_hora])

            # 1) Eliminar 23:59:59
            elim = (t == pd.Timedelta("23:59:59")).to_numpy()
            df, t = df[~elim].copy(), t[~elim]

            # 2) Segundos a 00 (se trunca al minuto); las horas ilegibles se dejan como están
            tf = t.dt.floor("min")
            cambiadas = int((t.notna() & (tf != t)).sum())
            seg = tf.dt.total_seconds()
            texto = ((seg // 3600).astype("Int64").astype(str).str.zfill(2) + ":"
                     + ((seg % 3600) // 60).astype("Int64").astype(str).str.zfill(2) + ":00")
            df[col_hora] = texto.where(tf.notna(), df[col_hora])

            # 3) Crear la fila 23:59:00 (todo NaN) en los días que no la tienen
            dias = df[col_dia].dropna().unique()
            con_ultima = df.loc[tf.to_numpy() == ultima, col_dia].dropna().unique()
            faltan = pd.DatetimeIndex(dias).difference(pd.DatetimeIndex(con_ultima))
            if len(faltan):
                nuevas = pd.DataFrame(index=range(len(faltan)), columns=df.columns, dtype="object")
                nuevas[col_dia] = faltan
                nuevas[col_hora] = "23:59:00"
                df = pd.concat([df, nuevas.astype(df.dtypes.to_dict(), errors="ignore")],
                               ignore_index=True)

            # 4) Una fila por fecha: la de las 23:59:00
            es_ultima = _hora_a_timedelta(df[col_hora]) == ultima
            sel = df[es_ultima & df[col_dia].notna()]
            repetidos = int(sel[col_dia].duplicated().sum())
            sel = (sel.drop_duplicates(col_dia, keep="first")
                      .sort_values(col_dia).reset_index(drop=True))

            hojas[hoja] = sel
            registro.append({
                "Archivo": archivos[i].name, "Hoja": hoja,
                "Filas eliminadas (23:59:59)": int(elim.sum()),
                "Horas con segundos llevadas a 00": cambiadas,
                "Días": len(dias),
                "Días sin 23:59:00 (fila creada con NaN)": len(faltan),
                "Días con 23:59:00 repetido": repetidos,
                "Filas finales": len(sel),
            })

    registro = pd.DataFrame(registro)
    if mostrar == True:
        display(registro)
    return datos_filtrados, registro

def eliminar_columnas_duplicadas(datos, archivos, col_dia="dia", col_hora="hora",
                                 prioridad_hojas=None, min_coincidencia=1.0, min_comunes=30,
                                 tol=1e-9, ignorar_constantes=True, aplicar=False, mostrar=True):
    """Dentro de cada archivo, busca columnas con los mismos valores (en cualquier hoja, incluso
    dentro de la misma) y elimina las copias, dejando una sola por grupo. No une los archivos.

    Las columnas se alinean por día (y hora, si existe). Dos columnas son duplicadas si, en las filas
    donde ambas tienen valor, coinciden en al menos `min_coincidencia` (1.0 = todas) y esas filas son
    al menos `min_comunes`.
    prioridad_hojas: lista de hojas, de la que más querés conservar a la que menos. Por defecto, las
    hojas 'consolidado_*' se conservan antes que las demás.
    ignorar_constantes: no compara columnas con un único valor (por ejemplo, todo 0).
    aplicar=False (por defecto) es un ensayo: muestra qué se borraría sin tocar `datos`.
    Solo compara columnas numéricas."""
    registro = []

    for i, hojas in datos.items():
        # 1) Una tabla ancha en memoria: filas = día (y hora), columnas = (hoja, columna)
        partes = {}
        for hoja, df in hojas.items():
            if col_dia not in df.columns:
                continue
            d = df.copy()
            dia = pd.to_datetime(d[col_dia], errors="coerce").dt.normalize()
            clave = dia.astype(str) + (" " + d[col_hora].astype(str) if col_hora in d.columns else "")
            d = d.drop(columns=[c for c in (col_dia, col_hora) if c in d.columns])
            d = d.apply(pd.to_numeric, errors="coerce")
            d.index = clave
            d = d[dia.notna().to_numpy() & ~d.index.duplicated(keep="first")]
            partes[hoja] = d.dropna(axis=1, how="all")
        if not partes:
            continue

        wide = pd.concat(partes, axis=1)
        cols = list(wide.columns)                 # (hoja, columna)
        M = wide.to_numpy(dtype=float)
        n = M.shape[1]
        if n < 2:
            continue

        if ignorar_constantes:
            cand = np.array([np.unique(M[~np.isnan(M[:, k]), k]).size > 1 for k in range(n)])
        else:
            cand = np.ones(n, dtype=bool)
        idx = np.flatnonzero(cand)

        # 2) Pares duplicados (con union-find para agruparlos)
        padre = list(range(n))
        def raiz(x):
            while padre[x] != x:
                padre[x] = padre[padre[x]]
                x = padre[x]
            return x

        for p, c in enumerate(idx):
            otros = idx[p + 1:]
            if otros.size == 0:
                break
            A, B = M[:, [c]], M[:, otros]
            ambos = ~np.isnan(A) & ~np.isnan(B)
            comunes = ambos.sum(axis=0)
            coinc = (np.isclose(A, B, rtol=tol, atol=tol) & ambos).sum(axis=0) / np.maximum(comunes, 1)
            for j in otros[(comunes >= min_comunes) & (coinc >= min_coincidencia)]:
                padre[raiz(j)] = raiz(c)

        grupos = {}
        for k in idx:
            grupos.setdefault(raiz(k), []).append(k)

        # 3) En cada grupo se conserva una columna según la prioridad de hojas
        orden = (sorted(partes, key=lambda h: not h.startswith("consolidado"))
                 if prioridad_hojas is None else list(prioridad_hojas))
        rango = {h: (orden.index(h) if h in orden else len(orden)) for h in partes}

        quitar = {}
        for g in grupos.values():
            if len(g) < 2:
                continue
            g = sorted(g, key=lambda k: (rango[cols[k][0]], k))
            conserva = g[0]
            for k in g[1:]:
                ambos = ~np.isnan(M[:, k]) & ~np.isnan(M[:, conserva])
                pct = np.isclose(M[ambos, k], M[ambos, conserva], rtol=tol, atol=tol).mean()
                registro.append({
                    "Archivo": archivos[i].name,
                    "Hoja": cols[k][0], "Columna": cols[k][1],
                    "Duplicada de (hoja)": cols[conserva][0],
                    "Duplicada de (columna)": cols[conserva][1],
                    "Filas comparadas": int(ambos.sum()),
                    "% coincidencia": round(100 * pct, 2),
                })
                quitar.setdefault(cols[k][0], []).append(cols[k][1])

        if aplicar:
            for hoja, cs in quitar.items():
                hojas[hoja] = hojas[hoja].drop(columns=cs)

    registro = pd.DataFrame(registro)
    print(f"Columnas {'eliminadas' if aplicar else 'que se eliminarían (ensayo)'}: {len(registro)}")
    if mostrar and not registro.empty:
        display(registro.sort_values(["Archivo", "Hoja", "Columna"]).reset_index(drop=True))
    return registro


def unificar_archivo(hojas, archivo_nombre="", col_dia="dia", col_hora="hora",
                     hoja_base=None, quitar_hora=True):
    """Une todas las hojas de un archivo en un único DataFrame, con left join por `dia`.
    hoja_base: hoja que define las filas. Si es None, la base es la unión de los días de todas las hojas
               (así el left join no pierde ningún día).
    quitar_hora: elimina la columna `hora` (en datos_filtrados vale siempre 23:59:00).
    Si una columna ya existe en el resultado, la de la hoja que llega se renombra a `columna_hoja`.
    Devuelve (DataFrame unificado, registro de columnas con nombre repetido)."""
    marcos = {}
    for hoja, df in hojas.items():
        if col_dia not in df.columns:
            continue
        d = df.copy()
        d[col_dia] = pd.to_datetime(d[col_dia], errors="coerce").dt.normalize()
        d = d.dropna(subset=[col_dia]).drop_duplicates(col_dia, keep="first")
        if quitar_hora and col_hora in d.columns:
            d = d.drop(columns=col_hora)
        marcos[hoja] = d

    if hoja_base is not None:
        res = marcos[hoja_base].copy()
        orden = [h for h in marcos if h != hoja_base]
        propietaria = {c: hoja_base for c in res.columns if c != col_dia}
    else:
        dias = sorted(set().union(*[set(d[col_dia]) for d in marcos.values()]))
        res = pd.DataFrame({col_dia: dias})
        orden = list(marcos)
        propietaria = {}

    registro = []
    for hoja in orden:
        d = marcos[hoja]
        cols = [c for c in d.columns if c != col_dia]
        renombres = {}
        for c in cols:
            if c in propietaria:
                a = res.set_index(col_dia)[c]
                b = d.set_index(col_dia)[c].reindex(a.index)
                iguales = bool((((a == b) | (a.isna() & b.isna())).all()))
                renombres[c] = f"{c}_{hoja}"
                registro.append({"Archivo": archivo_nombre, "Hoja": hoja, "Columna": c,
                                 "Ya existía por la hoja": propietaria[c],
                                 "Renombrada a": renombres[c],
                                 "Valores idénticos": iguales})
        d = d.rename(columns=renombres)
        for c in d.columns:
            if c != col_dia:
                propietaria[c] = hoja
        res = res.merge(d, on=col_dia, how="left")

    return res.sort_values(col_dia).reset_index(drop=True), pd.DataFrame(registro)


def unificar_archivos(datos_filtrados, archivos, **kwargs):
    unificados, registros = {}, []
    for i, hojas in datos_filtrados.items():
        df, reg = unificar_archivo(hojas, archivos[i].name, **kwargs)
        unificados[i] = df
        registros.append(reg)
        print(f"archivo_{i} ({archivos[i].name}): {df.shape[0]} filas × {df.shape[1]} columnas")
    registro = pd.concat([r for r in registros if not r.empty], ignore_index=True) if any(
        not r.empty for r in registros) else pd.DataFrame()
    return unificados, registro


def _indexar_por_dia(df, col_dia="dia"):
    """Indexa por día, sin filas repetidas ni filas sin fecha, y sin la columna de fecha."""
    dia = pd.to_datetime(df[col_dia], errors="coerce").dt.normalize()
    d = df.drop(columns=col_dia).copy()
    d.index = pd.Index(dia, name="dia")
    d = d[d.index.notna()]
    return d[~d.index.duplicated(keep="first")]


def _alinear_tipos(a, b):
    """Si una columna llegó como texto pero en realidad es numérica, las dos se pasan a número."""
    if a.dtype == object or b.dtype == object:
        na_, nb_ = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        if na_.notna().sum() == a.notna().sum() and nb_.notna().sum() == b.notna().sum():
            return na_, nb_
    return a, b


# ------------------------------------------------------------------ #
# 1) Rango de fechas de cada archivo y superposición con los demás
# ------------------------------------------------------------------ #
def rango_y_superposicion(unificados, archivos, col_dia="dia", mostrar=True):
    """Una fila por archivo: primer y último día, y cuántos días comparte con cada uno de los otros.
    unificados: dict {0: archivo_0, 1: archivo_1, 2: archivo_2} (un DataFrame por archivo)."""
    dias = {i: set(pd.to_datetime(df[col_dia], errors="coerce").dt.normalize().dropna())
            for i, df in unificados.items()}

    filas = []
    for i in dias:
        fila = {
            "Archivo": archivos[i].name,
            "Código": f"archivo_{i}",
            "Inicio": min(dias[i]).date() if dias[i] else None,
            "Final": max(dias[i]).date() if dias[i] else None,
            "Días distintos": len(dias[i]),
        }
        for j in dias:
            fila[f"Superposición con archivo_{j}"] = pd.NA if j == i else len(dias[i] & dias[j])
        filas.append(fila)

    resumen = pd.DataFrame(filas)
    sup = [c for c in resumen.columns if c.startswith("Superposición")]
    resumen[sup] = resumen[sup].astype("Int64")
    if mostrar:
        display(resumen)
    return resumen


# ------------------------------------------------------------------ #
# 2) Igualdad de valores en los días superpuestos
# ------------------------------------------------------------------ #
def comparar_solapes(unificados, archivos, col_dia="dia", mostrar=True):
    """Para cada par de archivos, compara las columnas comunes en los días que están en ambos.
    Comparación exacta, sin tolerancia.
    Devuelve (resumen por par de archivos, diferencias por columna, detalle celda por celda)."""
    ids = list(unificados)
    resumen, por_columna, celdas = [], [], []

    for a, b in itertools.combinations(ids, 2):
        A, B = _indexar_por_dia(unificados[a], col_dia), _indexar_por_dia(unificados[b], col_dia)
        comunes = A.index.intersection(B.index)
        if len(comunes) == 0:
            continue
        cols = [c for c in A.columns if c in B.columns]
        Aa, Bb = A.loc[comunes, cols], B.loc[comunes, cols]

        tot = dict(iguales=0, ambas=0, uno=0, dist=0)
        cols_mal = 0
        for c in cols:
            va, vb = _alinear_tipos(Aa[c], Bb[c])
            na, nb = va.isna(), vb.isna()
            con = ~na & ~nb
            igual = con & (va == vb)
            dist = con & ~igual
            uno = na ^ nb
            tot["iguales"] += int(igual.sum())
            tot["ambas"] += int((na & nb).sum())
            tot["uno"] += int(uno.sum())
            tot["dist"] += int(dist.sum())

            ne = (dist | uno).to_numpy()
            if not ne.any():
                continue
            cols_mal += 1
            dif = pd.to_numeric(va[ne], errors="coerce") - pd.to_numeric(vb[ne], errors="coerce")
            det = pd.DataFrame({
                "Archivo A": archivos[a].name, "Archivo B": archivos[b].name, "Columna": c,
                "Fecha": comunes[ne].date,
                "Tipo": np.where(dist[ne], "Valores distintos",
                                 np.where(na[ne], "Vacío en A", "Vacío en B")),
                "Valor en A": va[ne].to_numpy(), "Valor en B": vb[ne].to_numpy(),
                "Diferencia (A - B)": dif.to_numpy(),
            })
            celdas.append(det)
            por_columna.append({
                "Archivo A": archivos[a].name, "Archivo B": archivos[b].name, "Columna": c,
                "Fechas con valores distintos": int(dist.sum()),
                "Fechas con vacío en un solo archivo": int(uno.sum()),
                "Primera fecha con diferencia": det["Fecha"].iloc[0],
                "Valor en A": det["Valor en A"].iloc[0], "Valor en B": det["Valor en B"].iloc[0],
                "Máxima diferencia absoluta": dif.abs().max(),
            })

        n_cel = Aa.shape[0] * Aa.shape[1]
        resumen.append({
            "Archivo A": archivos[a].name, "Archivo B": archivos[b].name,
            "Fechas en común": len(comunes),
            "Columnas comunes": len(cols),
            "Columnas solo en A": len([c for c in A.columns if c not in B.columns]),
            "Columnas solo en B": len([c for c in B.columns if c not in A.columns]),
            "Celdas comparadas": n_cel,
            "Iguales (con valor)": tot["iguales"], "Ambas vacías": tot["ambas"],
            "Vacío en un solo archivo": tot["uno"], "Valores distintos": tot["dist"],
            "% celdas iguales": round(100 * (tot["iguales"] + tot["ambas"]) / n_cel, 2) if n_cel else np.nan,
            "Columnas con diferencias": cols_mal,
            "% columnas idénticas": round(100 * (len(cols) - cols_mal) / len(cols), 2) if cols else np.nan,
        })

    resumen = pd.DataFrame(resumen)
    por_columna = pd.DataFrame(por_columna)
    celdas = pd.concat(celdas, ignore_index=True) if celdas else pd.DataFrame()

    if resumen.empty:
        print("No hay fechas en común entre archivos.")
        return resumen, por_columna, celdas
    if not por_columna.empty:
        por_columna = por_columna.sort_values(
            ["Archivo A", "Archivo B", "Fechas con valores distintos", "Fechas con vacío en un solo archivo"],
            ascending=[True, True, False, False]).reset_index(drop=True)

    if mostrar:
        display(resumen)
        if por_columna.empty:
            print("Todas las celdas de los días superpuestos son iguales.")
        #else:
            #print(f"Columnas con diferencias: {len(por_columna)}")
            #display(por_columna)
    return resumen, por_columna, celdas

def valores_distintos(dif_celdas, nombre_a, nombre_b, mostrar=True, filas=30):
    """Celdas con valores distintos (ambas con valor) entre dos archivos, buscando por parte del nombre."""
    par = dif_celdas[
        dif_celdas["Archivo A"].str.contains(nombre_a)
        & dif_celdas["Archivo B"].str.contains(nombre_b)
        & (dif_celdas["Tipo"] == "Valores distintos")
    ].copy()
    a = pd.to_numeric(par["Valor en A"], errors="coerce").abs()
    b = pd.to_numeric(par["Valor en B"], errors="coerce").abs()
    par["Diferencia relativa %"] = (100 * par["Diferencia (A - B)"].abs() / np.maximum(a, b)).round(3)
    par = par.sort_values(["Fecha", "Columna"]).reset_index(drop=True)

    if mostrar:
        print(f"{nombre_a} vs {nombre_b}: {len(par)} celdas con valores distintos "
              f"| {par['Fecha'].nunique()} fechas | {par['Columna'].nunique()} columnas")
        if len(par):
            display(par[["Fecha", "Columna", "Valor en A", "Valor en B",
                         "Diferencia (A - B)", "Diferencia relativa %"]].head(filas))
    return par

def unir_archivos(unificados, prioridad):
    """Une los DataFrames de `unificados` en uno solo, por día, aplicando la regla por celda:
      - ambas vacías                -> queda vacía
      - una vacía y otra con valor  -> se usa el valor
      - ambas con valor y distintas -> gana el archivo que aparece primero en `prioridad`
    prioridad: lista de claves de `unificados`, de mayor a menor prioridad.
    Devuelve un DataFrame con una fila por día (la unión de todos los días)."""
    partes = [_indexar_por_dia(unificados[i]) for i in prioridad]

    res = partes[0]
    for p in partes[1:]:
        res = res.combine_first(p)          # toma el valor de `res` y, donde está vacío, el de `p`

    orden = []                              # mismo orden de columnas en ambas versiones
    for i in unificados:
        orden += [c for c in _indexar_por_dia(unificados[i]).columns if c not in orden]

    return res[orden].sort_index().reset_index()      # vuelve a tener la columna `dia`

def verificar_union(unido, unificados, prioridad):
    """Por cada archivo: cuántas de sus celdas con valor se mantienen en la unión, cuántas fueron
    reemplazadas por otro archivo (conflictos perdidos) y cuántas de sus celdas vacías se completaron."""
    u = _indexar_por_dia(unido)
    filas = []
    for i in prioridad:
        a = _indexar_por_dia(unificados[i])
        cols = [c for c in a.columns if c in u.columns]
        a, v = a[cols], u.loc[a.index, cols]
        con_valor = a.notna()
        cambiadas = int((con_valor & ~(a == v)).sum().sum())
        filas.append({
            "Archivo": f"archivo_{i}", "Prioridad": prioridad.index(i) + 1,
            "Celdas con valor": int(con_valor.sum().sum()),
            "Se mantienen": int(con_valor.sum().sum()) - cambiadas,
            "Reemplazadas por otro archivo": cambiadas,
            "Vacías completadas con otro archivo": int((a.isna() & v.notna()).sum().sum()),
        })
    return pd.DataFrame(filas)

def resumen_variables(df):
    """Una fila por variable: tipo, rango de valores y valores faltantes."""
    filas = []
    for c in df.columns:
        s = df[c]
        try:
            minimo, maximo = s.min(), s.max()
        except TypeError:                       # columna con tipos mezclados
            minimo, maximo = np.nan, np.nan
        filas.append({
            "Variable": c,
            "Tipo": str(s.dtype),
            "Mínimo": minimo,
            "Máximo": maximo,
            "Valores únicos": s.nunique(dropna=True),
            "Faltantes": int(s.isna().sum()),
            "% faltantes": round(100 * s.isna().mean(), 2),
        })
    return (pd.DataFrame(filas)
              .sort_values(["% faltantes", "Variable"], ascending=[False, True])
              .reset_index(drop=True))