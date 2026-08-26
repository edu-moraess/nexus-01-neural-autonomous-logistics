"""
Logistics / Tactical Map Visualization
NEXUS-01 — Persistent STATIC scene + DYNAMIC robot layer

Architecture:
  build_static_warehouse(stations)  → Figure with zones, stations, layout (once)
  update_warehouse_robots(fig, state) → mutates only the robot Scatter trace
"""

from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Optional
from config.settings import COLORS, ZONES


_STATE_COLOR = {
    "IDLE": COLORS["robot_idle"],
    "MOVING": COLORS["robot_moving"],
    "PICKING": COLORS["robot_moving"],
    "TRANSPORTING": COLORS["robot_transport"],
    "CHARGING": COLORS["robot_charging"],
    "MAINTENANCE": COLORS["robot_maint"],
    "WAITING": COLORS["muted"],
    "ALERT": COLORS["robot_alert"],
    "DEGRADED": COLORS["warning"],
}

_ZONE_COLORS = {
    "storage": "rgba(30,50,70,0.25)",
    "charging": "rgba(245,165,36,0.12)",
    "maintenance": "rgba(167,139,250,0.12)",
    "pickup": "rgba(61,139,253,0.12)",
    "delivery": "rgba(61,214,140,0.12)",
    "ops_center": "rgba(61,139,253,0.08)",
}

_STATION_SPEC = [
    ("charge", COLORS["station_charge"], "diamond"),
    ("pickup", COLORS["station_pick"], "square"),
    ("delivery", COLORS["station_deliver"], "square"),
    ("maintenance", COLORS["robot_maint"], "hexagon"),
]


def build_static_warehouse(stations: Dict[str, List] = None) -> go.Figure:
    """
    Build the STATIC scene once: zones, stations, axes, layout, shapes.
    Returns a Figure that should be stored and reused.
    Includes a placeholder robot trace (empty) at a fixed index.
    """
    fig = go.Figure()

    # 0 — empty congestion placeholder (kept for index stability; rarely updated)
    fig.add_trace(go.Heatmap(
        z=[[0]],
        colorscale=[[0, "rgba(10,14,20,0)"], [0.4, "rgba(245,165,36,0.15)"], [1, "rgba(243,18,96,0.35)"]],
        showscale=False,
        hoverinfo="skip",
        zsmooth="best",
        visible=False,
    ))

    # 1..4 — stations (static positions)
    stations = stations or {}
    for stype, color, symbol in _STATION_SPEC:
        pts = stations.get(stype, [])
        if pts:
            xs, ys, ids, avails = zip(*[(p[0], p[1], p[2], p[3]) for p in pts])
            colors = [color if a else "#555" for a in avails]
            texts = [i.split("-")[0] for i in ids]
        else:
            xs, ys, colors, texts = [], [], [], []
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys),
            mode="markers+text",
            marker=dict(size=14, color=list(colors), symbol=symbol,
                        line=dict(width=1, color="#1e2a38")),
            text=list(texts),
            textposition="top center",
            textfont=dict(size=8, color=COLORS["muted"]),
            name=stype.upper(),
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ))

    # 5 — robots (DYNAMIC placeholder; updated every frame)
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="markers+text",
        marker=dict(size=12, color=COLORS["robot_idle"], symbol="circle",
                    line=dict(width=2, color="#0a0e14"), opacity=0.92),
        text=[],
        textposition="bottom center",
        textfont=dict(size=8, color=COLORS["text"]),
        name="ROBOTS",
        hoverinfo="text",
        hovertext=[],
    ))

    # Layout — fixed, never changes
    shapes = []
    for name, (x0, y0, x1, y1) in ZONES.items():
        shapes.append(dict(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_ZONE_COLORS.get(name, "rgba(30,40,50,0.15)"),
            line=dict(color="#1e2a38", width=1),
            layer="below",
        ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"], size=11),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[0, 80], showgrid=True, gridcolor="#1a2332",
                   zeroline=False, title=None, fixedrange=True),
        yaxis=dict(range=[0, 50], showgrid=True, gridcolor="#1a2332",
                   zeroline=False, scaleanchor="x", scaleratio=1,
                   title=None, fixedrange=True),
        showlegend=False,
        height=420,
        title=dict(text="TACTICAL LOGISTICS MAP",
                   font=dict(size=12, color=COLORS["muted"]), x=0.02, y=0.98),
        shapes=shapes,
        # Critical: prevent Plotly from resetting view / recalculating layout
        uirevision="nexus-warehouse-static",
        transition={"duration": 0},
    )
    return fig


def update_warehouse_robots(fig: go.Figure, state: Dict[str, Any]) -> go.Figure:
    """
    Mutate ONLY the robot Scatter trace (last trace).
    Does NOT recreate Figure, layout, shapes, or station traces.
    Returns the same fig object for chaining.
    """
    robots = state.get("robots", [])
    anomalies = {a["robot_id"] for a in state.get("anomalies", [])}

    if not robots:
        return fig

    xs = [r["x"] for r in robots]
    ys = [r["y"] for r in robots]
    colors = [_STATE_COLOR.get(r["state"], COLORS["muted"]) for r in robots]
    sizes = [16 if r["id"] in anomalies else 12 for r in robots]
    texts = [r["id"] for r in robots]
    line_colors = ["#f31260" if r["id"] in anomalies else "#0a0e14" for r in robots]
    hover = [
        f"{r['id']}<br>State: {r['state']}<br>Bat: {r['battery']}%<br>Temp: {r['temp']}°C<br>"
        f"Load: {r['load']}<br>Health: {r['health']}<br>Decision: {r['decision']}<br>Conf: {r['confidence']}"
        for r in robots
    ]

    # Robot trace is always the last one (index 5)
    robot_idx = len(fig.data) - 1
    fig.data[robot_idx].x = xs
    fig.data[robot_idx].y = ys
    fig.data[robot_idx].marker.size = sizes
    fig.data[robot_idx].marker.color = colors
    fig.data[robot_idx].marker.line.color = line_colors
    fig.data[robot_idx].text = texts
    fig.data[robot_idx].hovertext = hover

    # Light congestion update (optional, low frequency — skip every frame cost)
    congestion = state.get("congestion")
    if congestion is not None and robot_idx > 0:
        try:
            fig.data[0].z = congestion.tolist() if hasattr(congestion, "tolist") else congestion
            fig.data[0].visible = True
        except Exception:
            pass

    return fig


def create_warehouse_figure(state: Dict[str, Any],
                            static_fig: Optional[go.Figure] = None) -> go.Figure:
    """
    Public API used by app.py.
    If static_fig is provided, update robots in-place.
    Otherwise build a full figure (first call / fallback).
    """
    if static_fig is not None and len(static_fig.data) >= 2:
        fig = go.Figure(static_fig)
        return update_warehouse_robots(fig, state)

    # Fallback: full build (first frame or no cache)
    stations = state.get("stations", {})
    fig = build_static_warehouse(stations)
    return update_warehouse_robots(fig, state)
