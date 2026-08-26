"""
NEXUS-01 — Neural Autonomous Logistics System
Mission Control Interface
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
from config.settings import COLORS, SimulationConfig
from src.simulation.engine import SimulationEngine
from src.visualization.warehouse import create_warehouse_figure
from src.visualization.neural_network import create_neural_figure
from src.visualization.telemetry import robots_to_dataframe, format_events
from src.visualization.charts import create_sei_gauge, create_battery_chart, create_fleet_bars

st.set_page_config(
    page_title="NEXUS-01 | Neural Autonomous Logistics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
    .stApp {{ background-color: {COLORS['bg']}; }}
    header[data-testid="stHeader"] {{ background: {COLORS['bg']}; border-bottom: 1px solid {COLORS['border']}; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; max-width: 1600px; }}
    h1, h2, h3 {{ font-family: 'JetBrains Mono', monospace !important; letter-spacing: 0.04em; }}
    .nexus-header {{ display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 1rem; background: linear-gradient(90deg, {COLORS['panel']} 0%, #0d1219 100%); border: 1px solid {COLORS['border']}; border-radius: 4px; margin-bottom: 0.8rem; }}
    .nexus-title {{ font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: {COLORS['text']}; letter-spacing: 0.12em; }}
    .nexus-sub {{ font-size: 0.72rem; color: {COLORS['muted']}; letter-spacing: 0.08em; margin-top: 2px; }}
    .status-online {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {COLORS['success']}; border: 1px solid {COLORS['success']}; padding: 3px 10px; border-radius: 2px; letter-spacing: 0.1em; }}
    div[data-testid="stMetric"] {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 0.4rem 0.6rem; }}
    div[data-testid="stMetric"] label {{ color: {COLORS['muted']} !important; font-size: 0.7rem !important; font-family: 'JetBrains Mono', monospace !important; }}
    .stButton > button {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.06em; border-radius: 3px; border: 1px solid {COLORS['border']}; background: {COLORS['panel']}; color: {COLORS['text']}; }}
    .stButton > button:hover {{ border-color: {COLORS['accent']}; color: {COLORS['accent']}; }}
</style>
""", unsafe_allow_html=True)


def init_state():
    if "engine" not in st.session_state:
        seed = 42
        st.session_state.engine = SimulationEngine(seed=seed)
        st.session_state.running = False
        st.session_state.speed = 1.0
        st.session_state.seed = seed
        st.session_state.last_state = None
        st.session_state.tick = 0


init_state()

st.markdown("""
<div class="nexus-header">
    <div>
        <div class="nexus-title">NEXUS-01</div>
        <div class="nexus-sub">NEURAL AUTONOMOUS LOGISTICS SYSTEM</div>
    </div>
    <div class="status-online">● SYSTEM STATUS: ONLINE</div>
</div>
""", unsafe_allow_html=True)

ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6 = st.columns([1, 1, 1, 1, 1.5, 1.5])
with ctrl1:
    if st.button("▶ START", use_container_width=True):
        st.session_state.running = True
with ctrl2:
    if st.button("⏸ PAUSE", use_container_width=True):
        st.session_state.running = False
with ctrl3:
    if st.button("↺ RESET", use_container_width=True):
        st.session_state.engine.reset(seed=st.session_state.seed)
        st.session_state.last_state = None
        st.session_state.running = False
with ctrl4:
    if st.button("⟶ STEP", use_container_width=True):
        st.session_state.last_state = st.session_state.engine.step()
        st.session_state.tick += 1
with ctrl5:
    speed = st.select_slider("SPEED", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=st.session_state.speed)
    st.session_state.speed = speed
    st.session_state.engine.speed = speed
with ctrl6:
    new_seed = st.number_input("SEED", min_value=0, max_value=99999, value=st.session_state.seed, step=1)
    if new_seed != st.session_state.seed:
        st.session_state.seed = int(new_seed)
        st.session_state.engine.reset(seed=st.session_state.seed)
        st.session_state.last_state = None

if st.session_state.running:
    st.session_state.last_state = st.session_state.engine.step()
    st.session_state.tick += 1

state = st.session_state.last_state
if state is None:
    state = st.session_state.engine.step()
    st.session_state.last_state = state

col_map, col_neural = st.columns(2)
with col_map:
    st.plotly_chart(create_warehouse_figure(state), use_container_width=True, key=f"map_{st.session_state.tick}")
with col_neural:
    pulse = (st.session_state.tick * 0.13) % 1.0
    st.plotly_chart(create_neural_figure(state["neural"], pulse_phase=pulse), use_container_width=True, key=f"nn_{st.session_state.tick}")

m = state["metrics"]
k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("ACTIVE", m["fleet"]["active"])
k2.metric("CHARGING", m["fleet"]["charging"])
k3.metric("DEGRADED", m["fleet"]["degraded"] + m["fleet"]["alert"])
k4.metric("UTIL %", f"{m['operations']['utilization']}")
k5.metric("AVG BAT", f"{m['energy']['avg_battery']}%")
k6.metric("CONF", f"{m['intelligence']['decision_confidence']:.2f}")
k7.metric("INFER ms", f"{m['intelligence']['inference_ms']:.2f}")
k8.metric("SEI", f"{m['sei']}")

tab1, tab2, tab3, tab4 = st.tabs(["FLEET TELEMETRY", "SYSTEM EVENTS", "PERFORMANCE", "ANOMALIES"])
with tab1:
    df = robots_to_dataframe(state["robots"])
    st.dataframe(df, use_container_width=True, height=280, hide_index=True)
with tab2:
    st.markdown(format_events(state["events"]))
with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(create_sei_gauge(m["sei"]), use_container_width=True)
    with c2:
        st.plotly_chart(create_battery_chart(st.session_state.engine.history_metrics), use_container_width=True)
    with c3:
        st.plotly_chart(create_fleet_bars(state["fleet_stats"]), use_container_width=True)
    st.markdown(f"""
| Domain | Metric | Value |
|--------|--------|-------|
| Intelligence | Active Connections | {m['intelligence']['active_connections']} |
| Intelligence | Total Inferences | {m['intelligence']['total_inferences']} |
| Operations | Tasks Completed | {m['operations']['tasks_completed']} |
| Energy | Charging Demand | {m['energy']['charging_demand']}% |
| Reliability | System Health | {m['reliability']['system_health']}% |
| Reliability | Anomalies | {m['reliability']['anomalies']} |
| Environment | Demand | {m['demand']} |
| Environment | Event Intensity | {m['event_intensity']} |
""")
with tab4:
    anoms = state.get("anomalies", [])
    if anoms:
        for a in anoms:
            st.warning(f"**{a['robot_id']}** · score {a['score']} · {', '.join(a['reasons'])} · BAT {a['battery']}% · TEMP {a['temp']}°C · HP {a['health']}")
    else:
        st.success("No active anomalies detected.")

st.caption(f"TS {state['timestep']} · SIM {state['sim_time']:.1f}s · SEED {st.session_state.seed} · TICK {st.session_state.tick} · {'RUNNING' if st.session_state.running else 'PAUSED'}")

if st.session_state.running:
    time.sleep(0.12)
    st.rerun()
