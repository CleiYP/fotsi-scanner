import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
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

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def tsi(momentum, l2, l3):
    s1 = ema(momentum, l2)
    s2 = ema(s1, l3)
    s1a = ema(momentum.abs(), l2)
    s2a = ema(s1a, l3)
    return 100 * (s2 / s2a)

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
        sobreextendido = percentil > 80 or percentil < 20
        direccion = "bajista" if percentil > 80 else ("alcista" if percentil < 20 else None)
        return {"sobreextendido": sobreextendido, "direccion": direccion, "percentil": round(float(percentil), 1)}
    except:
        return None

@st.cache_resource(show_spinner="Entrenando modelos (solo la primera vez)...")
def entrenar_modelos():
    datos = yf.download(PARES, period="2y", interval="1h", group_by="ticker")
    fotsi = calcular_fotsi(datos)
    distancias = pd.DataFrame(index=fotsi.index)
    for m1, m2 in combinations(MONEDAS, 2):
        distancias[f"{m1}_{m2}"] = fotsi[m1] - fotsi[m2]
    dataset = pd.DataFrame(index=fotsi.index)
    for moneda in MONEDAS:
        dataset[f"fotsi_{moneda}"] = fotsi[moneda]
    for col in distancias.columns:
        dataset[f"dist_{col}"] = distancias[col]
        dataset[f"vel3_{col}"] = distancias[col].diff(3)
        dataset[f"vel5_{col}"] = distancias[col].diff(5)
        dataset[f"vel10_{col}"] = distancias[col].diff(10)
    for col in distancias.columns:
        dataset[f"target_{col}"] = (distancias[col].shift(-3).abs() > distancias[col].abs()).astype(int)
    dataset.dropna(inplace=True)
    feature_cols = [c for c in dataset.columns if not c.startswith("target_")]
    target_cols  = [c for c in dataset.columns if c.startswith("target_")]
    split = int(len(dataset) * 0.8)
    X_train = dataset[feature_cols].iloc[:split]
    modelos = {}
    for target in target_cols:
        par = target.replace("target_", "")
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_train, dataset[target].iloc[:split])
        modelos[par] = m
    return modelos, feature_cols

# ── Título
st.title("Escáner FOTSI + MACD")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ── Entrenar modelos
modelos, feature_cols = entrenar_modelos()

# ── Botón actualizar
if st.button("🔄 Actualizar ranking"):
    st.rerun()

# ── Ranking FOTSI
with st.spinner("Calculando ranking FOTSI..."):
    datos_live = yf.download(PARES, period="5d", interval="1h", group_by="ticker", progress=False)
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
    X_live = pd.DataFrame([ultima])[feature_cols]
    res_live = {par: m.predict_proba(X_live)[0][1] for par, m in modelos.items()}
    ranking_live = pd.Series(res_live).sort_values(ascending=False)

# ── Capa 1
st.subheader("📊 Capa 1 — FOTSI (próximas 3 velas H1)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🔺 Divergencia**")
    for par in ranking_live.head(3).index:
        m1, m2 = par.split("_")
        st.metric(f"{m1}/{m2}", f"{ranking_live[par]*100:.1f}%")
with col2:
    st.markdown("**🔻 Convergencia**")
    for par in ranking_live.tail(3).index:
        m1, m2 = par.split("_")
        st.metric(f"{m1}/{m2}", f"{(1-ranking_live[par])*100:.1f}%")

# ── Capa 2 y 3
with st.spinner("Escaneando MACD..."):
    macd_res = {}
    for par in PARES:
        nombre = f"{par[:3]}/{par[3:6]}"
        macd_res[nombre] = {"4H": detectar_macd(par,"4h"), "1H": detectar_macd(par,"1h")}

st.subheader("📈 Capa 2 — MACD 4H")
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Alcista**")
    for p, r in macd_res.items():
        if r["4H"] and r["4H"]["direccion"] == "alcista":
            st.write(f"→ {p} ({r['4H']['percentil']}%)")
with col4:
    st.markdown("**Bajista**")
    for p, r in macd_res.items():
        if r["4H"] and r["4H"]["direccion"] == "bajista":
            st.write(f"→ {p} ({r['4H']['percentil']}%)")

st.subheader("📉 Capa 3 — MACD 1H")
col5, col6 = st.columns(2)
with col5:
    st.markdown("**Alcista**")
    for p, r in macd_res.items():
        if r["1H"] and r["1H"]["direccion"] == "alcista":
            st.write(f"→ {p} ({r['1H']['percentil']}%)")
with col6:
    st.markdown("**Bajista**")
    for p, r in macd_res.items():
        if r["1H"] and r["1H"]["direccion"] == "bajista":
            st.write(f"→ {p} ({r['1H']['percentil']}%)")

# ── Confluencias
st.subheader("⭐ Confluencias")
fotsi_top = [f"{p.split('_')[0]}/{p.split('_')[1]}" for p in ranking_live.head(5).index]
fotsi_bot = [f"{p.split('_')[0]}/{p.split('_')[1]}" for p in ranking_live.tail(5).index]
encontrado = False
for par, r in macd_res.items():
    capas = []
    if par in fotsi_top or par in fotsi_bot:
        capas.append("FOTSI")
    if r["4H"] and r["4H"]["sobreextendido"]:
        capas.append("MACD 4H")
    if r["1H"] and r["1H"]["sobreextendido"]:
        capas.append("MACD 1H")
    if len(capas) >= 2:
        st.success(f"**{par}** — {' + '.join(capas)}")
        encontrado = True
if not encontrado:
    st.info("No hay confluencias en este momento")