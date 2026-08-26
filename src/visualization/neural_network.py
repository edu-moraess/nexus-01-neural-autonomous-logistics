"""
Live Neural Network Visualization
NEXUS-01 — Animated signal flow with pulses
"""

from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any
from config.settings import COLORS


def create_neural_figure(neural_data: Dict[str, Any], pulse_phase: float = 0.0) -> go.Figure:
    layers = neural_data.get("layers", [])
    edges = neural_data.get("edges", [])
    decision = neural_data.get("decision", "—")
    confidence = neural_data.get("confidence", 0.0)
    if not layers:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"], height=420, margin=dict(l=10, r=10, t=30, b=10))
        return fig
    n_layers = len(layers)
    positions = []
    max_neurons = max(l["size"] for l in layers)
    for li, layer in enumerate(layers):
        size = layer["size"]
        xs = [li * 1.8] * size
        span = max(1.0, (size - 1) * 0.55)
        ys = np.linspace(-span / 2, span / 2, size) if size > 1 else [0.0]
        positions.append(list(zip(xs, ys)))
    fig = go.Figure()
    edge_x, edge_y, pulse_x, pulse_y, pulse_size = [], [], [], [], []
    for e in edges:
        li = e["layer"]
        if li >= len(positions) - 1:
            continue
        src_pos = positions[li]
        dst_pos = positions[li + 1]
        si, di = e["src"], e["dst"]
        if si >= len(src_pos) or di >= len(dst_pos):
            continue
        x0, y0 = src_pos[si]
        x1, y1 = dst_pos[di]
        intensity = e["intensity"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        if intensity > 0.25:
            t = (pulse_phase + li * 0.17 + si * 0.03) % 1.0
            pulse_x.append(x0 + (x1 - x0) * t)
            pulse_y.append(y0 + (y1 - y0) * t)
            pulse_size.append(4 + 8 * intensity)
    if edge_x:
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=0.9, color="rgba(61,139,253,0.38)"), hoverinfo="skip", showlegend=False))
    if pulse_x:
        fig.add_trace(go.Scatter(x=pulse_x, y=pulse_y, mode="markers",
                                 marker=dict(size=pulse_size, color="rgba(61,214,140,0.85)", symbol="circle", line=dict(width=0)),
                                 hoverinfo="skip", showlegend=False))
    for li, layer in enumerate(layers):
        acts = np.array(layer["activations"])
        pos = positions[li]
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        a_norm = (acts - acts.min()) / (acts.max() - acts.min() + 1e-6) if len(acts) else acts
        sizes = 8 + 14 * a_norm
        colors = [f"rgba({int(61+a*100)},{int(139+a*50)},{int(253-a*80)},{0.5+0.5*a:.2f})" for a in a_norm]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                 marker=dict(size=sizes, color=colors, line=dict(width=1, color="#0a0e14"), opacity=0.9),
                                 name=f"L{li}", hovertemplate=f"Layer {li}<br>Act: %{{marker.size:.1f}}<extra></extra>", showlegend=False))
    labels = ["INPUT"] + [f"H{i}" for i in range(n_layers - 2)] + ["OUTPUT"]
    for li, lab in enumerate(labels):
        fig.add_annotation(x=li * 1.8, y=max_neurons * 0.32 + 0.4, text=lab, showarrow=False,
                           font=dict(size=9, color=COLORS["muted"], family="JetBrains Mono, monospace"))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"]),
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(visible=False, range=[-0.5, (n_layers - 1) * 1.8 + 0.5]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        height=420,
        title=dict(text=f"NEURAL CORE  ·  {decision}  ·  conf {confidence:.2f}", font=dict(size=12, color=COLORS["accent"]), x=0.02, y=0.98),
    )
    return fig
