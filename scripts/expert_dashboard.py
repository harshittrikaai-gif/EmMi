import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Emmit Nova MoE Monitor", layout="wide")

st.title("🚀 Emmit Nova Sunya: Expert Utilization Dashboard")

# Config
LOG_DIR = Path("outputs/monitor")
METRICS_FILE = LOG_DIR / "latest_metrics.json"
HISTORY_FILE = LOG_DIR / "history.json"

def load_metrics():
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

def load_history():
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

# Sidebar
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 10, 2)

# Main UI
metrics = load_metrics()
if metrics:
    cols = st.columns(4)
    cols[0].metric("Step", metrics["step"])
    cols[1].metric("Loss", f"{metrics['loss']:.4f}")
    cols[2].metric("LR", f"{metrics['lr']:.2e}")
    cols[3].metric("Status", metrics.get("status", "running"))

    # Expert Utilization Bar Chart
    if "expert_utilization" in metrics:
        st.subheader("Current Expert Load Distribution")
        util = metrics["expert_utilization"]
        df_util = pd.DataFrame({
            "Expert ID": range(len(util)),
            "Utilization": util
        })
        fig = px.bar(df_util, x="Expert ID", y="Utilization", color="Utilization",
                     color_continuous_scale="Viridis", title="Top-1 Expert Selection Frequency")
        st.plotly_chart(fig, use_container_width=True)

# Training History
history = load_history()
if history:
    st.subheader("Training History")
    df_history = pd.DataFrame(history)
    fig_loss = px.line(df_history, x="step", y="loss", title="Loss Curve")
    st.plotly_chart(fig_loss, use_container_width=True)

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
