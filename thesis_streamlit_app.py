# rainfall_forecast_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from datetime import datetime, timedelta

st.set_page_config(page_title="Rainfall Forecasting - Malaysia", layout="wide")

# Title and Description
st.title("Forecasting Extreme Rainfall Events in Malaysia")
st.markdown("""
This application demonstrates statistical modeling of extreme rainfall events in Malaysia 
using the Generalized Extreme Value (GEV) distribution. The tool is designed to support flood risk management efforts.
""")

# Upload CSV or use sample data
uploaded_file = st.file_uploader("Upload Rainfall Data (CSV format)", type=["csv"])

@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2000-01-01", end="2020-12-31", freq="D")
    rainfall = np.random.gamma(shape=2, scale=10, size=len(dates))  # Synthetic daily rainfall
    data = pd.DataFrame({"Date": dates, "Rainfall_mm": rainfall})
    return data

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
else:
    st.info("Using sample rainfall dataset.")
    df = load_sample_data()

df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Visualize Rainfall
st.subheader("📈 Rainfall Time Series")
st.line_chart(df["Rainfall_mm"])

# Peak Over Threshold (Extreme Events)
threshold = st.slider("Set Rainfall Threshold for Extreme Events (mm)", 50, 200, 100)
extreme_events = df[df["Rainfall_mm"] > threshold]

st.subheader("🌧️ Extreme Rainfall Events")
st.write(f"Number of extreme events: {len(extreme_events)}")
st.dataframe(extreme_events.tail())

# GEV Fitting
st.subheader("📊 GEV Model Fit for Annual Maxima")
annual_max = df["Rainfall_mm"].resample("Y").max().dropna()
shape, loc, scale = gev.fit(annual_max)

x = np.linspace(annual_max.min(), annual_max.max(), 100)
pdf = gev.pdf(x, shape, loc, scale)

fig, ax = plt.subplots()
ax.hist(annual_max, bins=15, density=True, alpha=0.6, color='skyblue', label='Observed')
ax.plot(x, pdf, 'r-', label='GEV Fit')
ax.set_xlabel("Annual Maximum Rainfall (mm)")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# Return Level Estimation
st.subheader("🔮 Return Level Estimation")
return_period = st.slider("Select Return Period (Years)", 2, 100, 10)
return_level = gev.ppf(1 - 1/return_period, shape, loc, scale)
st.write(f"Estimated Rainfall for a {return_period}-year Event: **{return_level:.2f} mm**")

# Flood Risk Assessment
st.subheader("🚨 Flood Risk Indicator")
risk_level = "High" if return_level > threshold else "Moderate" if return_level > (threshold * 0.75) else "Low"
st.metric(label="Flood Risk Level", value=risk_level)

# Export Results
st.download_button("Download Extreme Events", extreme_events.to_csv(index=True), file_name="extreme_events.csv")

