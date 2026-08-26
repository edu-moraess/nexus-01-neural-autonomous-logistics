"""
Logistics / Tactical Map Visualization
NEXUS-01
"""

from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any
from config.settings import COLORS, ZONES


def create_warehouse_figure(state: Dict[str, Any]) -> go.Figure:
    robots = state.get("robots", [])
    stations = state.get("stations", {})
    congestion = state.get("congestion")
    anomalies = {a["robot_id"] for a in state.get("anomalies", [])}
    fig = go.Figure()
    if congestion is not None:
        fig.add_trace(go.Heatmap(
            z=congestion,
            colorscale=[[0, "rgba(10,14,20,0)"], [0.4, "rgba(245,165,36,0.15)"], [1, "rgba(243,18,96,0.35)"]],
            showscale=False, hoverinfo="skip", zsmooth="best",
        ))
    zone_colors = {
        "storage": "rgba(30,50,70,0.25)", "charging": "rgba(245,165,36,0.12)",
        "maintenance": "rgba(167,139,250,0.12)", "pickup": "rgba(61,139,253,0.12)",
        "delivery": "rgba(61,214,140,0.12)", "ops_center": "rgba(61,139,253,0.08)",
    }
    for stype, color, symbol in [
        ("charge", COLORS["station_charge"], "diamond"),
        ("pickup", COLORS["station_pick"], "square"),
        ("delivery", COLORS["station_deliver"], "square"),
        ("maintenance", COLORS["robot_maint"], "hexagon"),
    ]:
        pts = stations.get(stype, [])
        if not pts:
            continue
        xs, ys, ids, avails = zip(*[(p[0], p[1], p[2], p[3]) for p in pts])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=14, color=[color if a else "#555" for a in avails], symbol=symbol, line=dict(width=1, color="#1e2a38")),
            text=[i.split("-")[0] for i in ids], textposition="top center",
            textfont=dict(size=8, color=COLORS["muted"]), name=stype.upper(),
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ))
    state_color = {
        "IDLE": COLORS["robot_idle"], "MOVING": COLORS["robot_moving"], "PICKING": COLORS["robot_moving"],
        "TRANSPORTING": COLORS["robot_transport"], "CHARGING": COLORS["robot_charging"],
        "MAINTENANCE": COLORS["robot_maint"], "WAITING": COLORS["muted"],
        "ALERT": COLORS["robot_alert"], "DEGRADED": COLORS["warning"],
    }
    if robots:
        xs = [r["x"] for r in robots]
        ys = [r["y"] for r in robots]
        colors = [state_color.get(r["state"], COLORS["muted"]) for r in robots]
        sizes = [16 if r["id"] in anomalies else 12 for r in robots]
        texts = [r["id"] for r in robots]
        hover = [f"{r['id']}<br>State: {r['state']}<br>Bat: {r['battery']}%<br>Temp: {r['temp']}°C<br>Load: {r['load']}<br>Health: {r['health']}<br>Decision: {r['decision']}<br>Conf: {r['confidence']}" for r in robots]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=colors, symbol="circle",
                        line=dict(width=2, color=["#f31260" if r["id"] in anomalies else "#0a0e14" for r in robots]), opacity=0.92),
            text=texts, textposition="bottom center", textfont=dict(size=8, color=COLORS["text"]),
            name="ROBOTS", hovertext=hover, hoverinfo="text",
        ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"], size=11),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[0, 80], showgrid=True, gridcolor="#1a2332", zeroline=False, title=None, fixedrange=True),
        yaxis=dict(range=[0, 50], showgrid=True, gridcolor="#1a2332", zeroline=False, scaleanchor="x", scaleratio=1, title=None, fixedrange=True),
        showlegend=False, height=420,
        title=dict(text="TACTICAL LOGISTICS MAP", font=dict(size=12, color=COLORS["muted"]), x=0.02, y=0.98),
    )
    shapes = []
    for name, (x0, y0, x1, y1) in ZONES.items():
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           fillcolor=zone_colors.get(name, "rgba(30,40,50,0.15)"),
                           line=dict(color="#1e2a38", width=1), layer="below"))
    fig.update_layout(shapes=shapes)
    return fig
