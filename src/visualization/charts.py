"""
Performance Charts
"""

from __future__ import annotations
import plotly.graph_objects as go
from typing import List, Dict, Any
from config.settings import COLORS


def create_sei_gauge(sei: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=sei,
        number=dict(suffix="", font=dict(size=28, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS["muted"]),
            bar=dict(color=COLORS["accent"]),
            bgcolor=COLORS["panel"], borderwidth=1, bordercolor=COLORS["border"],
            steps=[
                dict(range=[0, 40], color="rgba(243,18,96,0.25)"),
                dict(range=[40, 70], color="rgba(245,165,36,0.25)"),
                dict(range=[70, 100], color="rgba(61,214,140,0.25)"),
            ],
            threshold=dict(line=dict(color=COLORS["success"], width=2), value=80),
        ),
        title=dict(text="SYSTEM EFFICIENCY INDEX", font=dict(size=11, color=COLORS["muted"])),
    ))
    fig.update_layout(paper_bgcolor=COLORS["bg"], font=dict(family="JetBrains Mono, monospace", color=COLORS["text"]), height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def create_battery_chart(history: List[Dict]) -> go.Figure:
    if not history:
        history = [{"energy": {"avg_battery": 80}}]
    ys = [h.get("energy", {}).get("avg_battery", 0) for h in history]
    fig = go.Figure(go.Scatter(y=ys, mode="lines", line=dict(color=COLORS["accent"], width=2), fill="tozeroy", fillcolor="rgba(61,139,253,0.15)"))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, monospace", color=COLORS["text"], size=10),
        height=140, margin=dict(l=30, r=10, t=25, b=20),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(range=[0, 100], gridcolor="#1a2332", title="BAT %"),
        title=dict(text="AVG FLEET BATTERY", font=dict(size=10, color=COLORS["muted"])),
    )
    return fig


def create_fleet_bars(fleet_stats: Dict[str, int]) -> go.Figure:
    labels = list(fleet_stats.keys())
    values = list(fleet_stats.values())
    colors = [COLORS["accent"] if v > 0 else COLORS["muted"] for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside", textfont=dict(size=9)))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, monospace", color=COLORS["text"], size=9),
        height=160, margin=dict(l=20, r=10, t=25, b=30),
        xaxis=dict(tickangle=-35, gridcolor="#1a2332"),
        yaxis=dict(gridcolor="#1a2332"),
        title=dict(text="FLEET STATE DISTRIBUTION", font=dict(size=10, color=COLORS["muted"])),
        showlegend=False,
    )
    return fig
