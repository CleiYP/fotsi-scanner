import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import time
import pytz
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Escáner FOTSI", layout="wide")

PARES = [
    "EURUSD=X","EURGBP=X","EURCHF=X","EURJPY=X","EURAUD=X","EURCAD=X","EURNZD=X",
    "USDCHF=X","USDJPY=X","USDCAD=X",
    "GBPUSD=X","GBPCHF=X","GBPJPY=X","GBPAUD=X","GBPCAD=X","GBPNZD=X",
    "CHFJPY=X",
    "AUDUSD=X","AUDCHF=X","AUDJPY=X","AUDCAD=X","AUDNZD=X",
    "CADCHF=X","CADJPY=X",
    "NZDUSD=X","NZDCHF=X","NZDJPY=X","NZDCAD=X"
]
MONEDAS = ["EUR","USD","GBP","CHF","JPY","AUD","CAD","NZD"]
L2, L3 = 25, 15
ZONA = pytz.timezone("America/Cancun")

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def tsi(momentum, l2, l3):
    s1 = ema(momentum, l2)
    s2 = ema(s1, l3)
    s1a = ema(momentum.abs(), l2)
    s2a = ema(s1a, l3)
    return 100 * (s2 / s2a)

def descargar_robusto(pares, period="2y", interval="1h"):
    datos = yf.download(pares, period=period, interval=interval, group_by="ticker", progress=False)
    pares_disp = list(datos.columns.get_level_values(0).unique())
    pares_vacios = [p for p in pares_disp if datos[p]["Close"].notna().sum() < 100]

    if pares_vacios:
        dfs_individuales = {}
        for par in pares_vacios:
            time.sleep(3)
            df_ind = yf.download(par, period=period, interval=interval, progress=False)
            if len(df_ind) > 100:
                dfs_individuales[par] = df_ind

        columnas_buenas = {p: datos[p] for p in pares_disp if p not in dfs_individuales}
        for par, df_ind in dfs_individuales.items():
            if df_ind.columns.nlevels > 1:
                df_ind.columns = df_ind.columns.droplevel(1)
            columnas_buenas[par] = df_ind[["Open","High","Low","Close","Volume"]]
        datos = pd.concat(columnas_buenas, axis=1)

    return datos

def calcular_fotsi(datos):
    mom = {}
    for par in list(datos.columns.get_level_values(0).unique()):
        try:
            mom[par] = datos[par]["Close"] - datos[par]["Open"]
        except:
            continue
    for par in ["EURJPY=X","USDJPY=X","GBPJPY=X","CHFJPY=X","AUDJPY=X","CADJPY=X","NZDJPY=X"]:
        if par in mom:
            mom[par] = mom[par] / 100
    m = mom
    return pd.DataFrame({
        "EUR": tsi(m["EURUSD=X"]+m["EURGBP=X"]+m["EURCHF=X"]+m["EURJPY=X"]+m["EURAUD=X"]+m["EURCAD=X"]+m["EURNZD=X"], L2, L3),
        "USD": tsi(-m["EURUSD=X"]-m["GBPUSD=X"]+m["USDCHF=X"]+m["USDJPY=X"]-m["AUDUSD=X"]+m["USDCAD=X"]-m["NZDUSD=X"], L2, L3),
        "GBP": tsi(-m["EURGBP=X"]+m["GBPUSD=X"]+m["GBPCHF=X"]+m["GBPJPY=X"]+m["GBPAUD=X"]+m["GBPCAD=X"]+m["GBPNZD=X"], L2, L3),
        "CHF": tsi(-m["EURCHF=X"]-m["USDCHF=X"]-m["GBPCHF=X"]+m["CHFJPY=X"]-m["AUDCHF=X"]-m["CADCHF=X"]-m["NZDCHF=X"], L2, L3),
        "JPY": tsi(-m["EURJPY=X"]-m["USDJPY=X"]-m["GBPJPY=X"]-m["CHFJPY=X"]-m["AUDJPY=X"]-m["CADJPY=X"]-m["NZDJPY=X"], L2, L3),
        "AUD": tsi(-m["EURAUD=X"]+m["AUDUSD=X"]-m["GBPAUD=X"]+m["AUDCHF=X"]+m["AUDJPY=X"]+m["AUDCAD=X"]+m["AUDNZD=X"], L2, L3),
        "CAD": tsi(-m["EURCAD=X"]-m["USDCAD=X"]-m["GBPCAD=X"]+m["CADCHF=X"]+m["CADJPY=X"]-m["AUDCAD=X"]-m["NZDCAD=X"], L2, L3),
        "NZD": tsi(-m["EURNZD=X"]+m["NZDUSD=X"]-m["GBPNZD=X"]+m["NZDCHF=X"]+m["NZDJPY=X"]-m["AUDNZD=X"]+m["NZDCAD=X"], L2, L3),
    })

def detectar_macd(par, intervalo="4h"):
    try:
        df = yf.download(par, period="6mo", interval=intervalo, progress=False)
        close = df["Close"].squeeze().values
        macd, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[~np.isnan(macd)]
        if len(macd) < 20:
            return None
        val = macd[-1]
        p10 = np.percentile(macd, 10)
        p90 = np.percentile(macd, 90)
        percentil = (val - p10) / (p90 - p10) * 100
        direccion = "bajista" if percentil > 80 else ("alcista" if percentil < 20 else "neutral")
        return {"percentil": round(float(percentil), 1), "direccion": direccion}
    except:
        return None

@st.cache_resource(show_spinner="Entrenando modelos (solo la primera vez, ~10-15 min)...")
def entrenar_modelos():
    datos = descargar_robusto(PARES, period="2y", interval="1h")
    fotsi = calcular_fotsi(datos)

    distancias = pd.DataFrame(index=fotsi.index)
    for m1, m2 in combinations(MONEDAS, 2):
        distancias[f"{m1}_{m2}"] = fotsi[m1] - fotsi[m2]

    dataset = pd.DataFrame(index=fotsi.index)
    for moneda in MONEDAS:
        dataset[f"fotsi_{moneda}"] = fotsi[moneda]
    for col in distancias.columns:
        dataset[f"dist_{col}"]  = distancias[col]
        dataset[f"vel3_{col}"]  = distancias[col].diff(3)
        dataset[f"vel5_{col}"]  = distancias[col].diff(5)
        dataset[f"vel10_{col}"] = distancias[col].diff(10)
    for n in [1, 2, 3]:
        for col in distancias.columns:
            dist_futura = distancias[col].shift(-n)
            dataset[f"target_{n}h_{col}"] = (dist_futura.abs() > distancias[col].abs()).astype(int)

    dataset.dropna(inplace=True)
    feature_cols = [c for c in dataset.columns if not c.startswith("target_")]
    target_cols  = [c for c in dataset.columns if c.startswith("target_")]
    split = int(len(dataset) * 0.8)
    X_train = dataset[feature_cols].iloc[:split]

    modelos = {}
    for target in target_cols:
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_train, dataset[target].iloc[:split])
        modelos[target] = m

    return modelos, feature_cols

def generar_tabla(modelos, feature_cols):
    datos_live = descargar_robusto(PARES, period="5d", interval="1h")
    fotsi_live = calcular_fotsi(datos_live)
    dist_live = pd.DataFrame(index=fotsi_live.index)
    for m1, m2 in combinations(MONEDAS, 2):
        dist_live[f"{m1}_{m2}"] = fotsi_live[m1] - fotsi_live[m2]

    ultima = {}
    for moneda in MONEDAS:
        ultima[f"fotsi_{moneda}"] = fotsi_live[moneda].iloc[-1]
    for col in dist_live.columns:
        ultima[f"dist_{col}"]  = dist_live[col].iloc[-1]
        ultima[f"vel3_{col}"]  = dist_live[col].diff(3).iloc[-1]
        ultima[f"vel5_{col}"]  = dist_live[col].diff(5).iloc[-1]
        ultima[f"vel10_{col}"] = dist_live[col].diff(10).iloc[-1]

    X_actual = pd.DataFrame([ultima])[feature_cols]

    pares_unicos = sorted(set("_".join(t.replace("target_","").split("_")[1:]) for t in modelos.keys()))

    filas = []
    macd_cache = {}
    for par in pares_unicos:
        m1, m2 = par.split("_")
        fila = {"Par": f"{m1}/{m2}"}
        for h in ["1h", "2h", "3h"]:
            modelo = modelos[f"target_{h}_{par}"]
            prob = modelo.predict_proba(X_actual)[0][1]
            fila[f"FOTSI {h.upper()}"] = f"Div {prob*100:.0f}%" if prob > 0.5 else f"Conv {(1-prob)*100:.0f}%"

        ticker = f"{m1}{m2}=X"
        if ticker not in PARES:
            ticker = f"{m2}{m1}=X"
        if ticker in PARES:
            r4h = detectar_macd(ticker, "4h")
            r1h = detectar_macd(ticker, "1h")
            fila["MACD 4H"] = f"{r4h['percentil']}% {r4h['direccion']}" if r4h else "N/A"
            fila["MACD 1H"] = f"{r1h['percentil']}% {r1h['direccion']}" if r1h else "N/A"
        else:
            fila["MACD 4H"] = "N/A"
            fila["MACD 1H"] = "N/A"

        filas.append(fila)

    return pd.DataFrame(filas)

# ── Interfaz ─────────────────────────────────────────
st.title("Escáner FOTSI + MACD")

modelos, feature_cols = entrenar_modelos()

st.caption(f"Última actualización: {datetime.now(ZONA).strftime('%d/%m/%Y %H:%M')}")

if st.button("🔄 Actualizar tabla"):
    st.rerun()

with st.spinner("Calculando tabla..."):
    tabla = generar_tabla(modelos, feature_cols)

st.dataframe(tabla, use_container_width=True, hide_index=True)
