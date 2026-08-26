"""
Live Neural Network Visualization
NEXUS-01 — Persistent STATIC topology + DYNAMIC activations/pulses

Architecture:
  build_static_neural(layer_sizes)  → Figure with fixed node positions, edges, labels
  update_neural_activity(fig, neural_data, pulse_phase) → mutates activations + pulses
"""

from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from config.settings import COLORS


def _compute_positions(layer_sizes: List[int]) -> List[List[Tuple[float, float]]]:
    """Deterministic neuron layout from layer sizes only (static)."""
    positions = []
    for li, size in enumerate(layer_sizes):
        xs = [li * 1.8] * size
        span = max(1.0, (size - 1) * 0.55)
        ys = list(np.linspace(-span / 2, span / 2, size)) if size > 1 else [0.0]
        positions.append(list(zip(xs, ys)))
    return positions


def _build_edge_geometry(positions: List[List[Tuple[float, float]]],
                         max_src: int = 12, max_dst: int = 10
                         ) -> Tuple[List[float], List[float], List[dict]]:
    """Build static edge line segments and edge descriptor list for pulses."""
    edge_x, edge_y = [], []
    edge_meta = []
    for li in range(len(positions) - 1):
        src_pos = positions[li]
        dst_pos = positions[li + 1]
        for si in range(min(len(src_pos), max_src)):
            for di in range(min(len(dst_pos), max_dst)):
                x0, y0 = src_pos[si]
                x1, y1 = dst_pos[di]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_meta.append({
                    "layer": li, "src": si, "dst": di,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                })
    return edge_x, edge_y, edge_meta


def build_static_neural(layer_sizes: List[int] = None) -> go.Figure:
    """Build STATIC neural topology once: node positions, edges, labels, layout."""
    if layer_sizes is None:
        layer_sizes = [16, 32, 24, 16, 9]

    n_layers = len(layer_sizes)
    positions = _compute_positions(layer_sizes)
    max_neurons = max(layer_sizes)
    edge_x, edge_y, edge_meta = _build_edge_geometry(positions)

    fig = go.Figure()

    # Trace 0 — static edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.9, color="rgba(61,139,253,0.38)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Trace 1 — dynamic pulses
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="markers",
        marker=dict(size=6, color="rgba(61,214,140,0.85)",
                    symbol="circle", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Traces 2.. — one Scatter per layer (positions fixed, size/color dynamic)
    for li, size in enumerate(layer_sizes):
        pos = positions[li]
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=[10] * size,
                color=["rgba(61,139,253,0.6)"] * size,
                line=dict(width=1, color="#0a0e14"),
                opacity=0.9,
            ),
            name=f"L{li}",
            hovertemplate=f"Layer {li}<extra></extra>",
            showlegend=False,
        ))

    labels = ["INPUT"] + [f"H{i}" for i in range(n_layers - 2)] + ["OUTPUT"]
    annotations = []
    for li, lab in enumerate(labels):
        annotations.append(dict(
            x=li * 1.8, y=max_neurons * 0.32 + 0.4,
            text=lab, showarrow=False,
            font=dict(size=9, color=COLORS["muted"],
                      family="JetBrains Mono, monospace"),
        ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=COLORS["text"]),
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(visible=False, range=[-0.5, (n_layers - 1) * 1.8 + 0.5], fixedrange=True),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        height=420,
        title=dict(text="NEURAL CORE", font=dict(size=12, color=COLORS["accent"]),
                   x=0.02, y=0.98),
        annotations=annotations,
        uirevision="nexus-neural-static",
        transition={"duration": 0},
    )

    fig._nexus_positions = positions
    fig._nexus_edge_meta = edge_meta
    fig._nexus_layer_sizes = layer_sizes
    return fig


def update_neural_activity(fig: go.Figure,
                           neural_data: Dict[str, Any],
                           pulse_phase: float = 0.0) -> go.Figure:
    """Mutate ONLY pulses, neuron size/color, and title."""
    layers = neural_data.get("layers", [])
    decision = neural_data.get("decision", "—")
    confidence = neural_data.get("confidence", 0.0)
    edges_dyn = neural_data.get("edges", [])

    if not layers or len(fig.data) < 3:
        return fig

    positions = getattr(fig, "_nexus_positions", None)
    edge_meta = getattr(fig, "_nexus_edge_meta", None)

    if positions is None:
        sizes = [l["size"] for l in layers]
        positions = _compute_positions(sizes)
        _, _, edge_meta = _build_edge_geometry(positions)

    intensity_map = {}
    for e in edges_dyn:
        intensity_map[(e["layer"], e["src"], e["dst"])] = e.get("intensity", 0.0)

    pulse_x, pulse_y, pulse_size = [], [], []
    if edge_meta:
        for meta in edge_meta:
            key = (meta["layer"], meta["src"], meta["dst"])
            intensity = intensity_map.get(key, 0.0)
            if intensity > 0.25:
                t = (pulse_phase + meta["layer"] * 0.17 + meta["src"] * 0.03) % 1.0
                pulse_x.append(meta["x0"] + (meta["x1"] - meta["x0"]) * t)
                pulse_y.append(meta["y0"] + (meta["y1"] - meta["y0"]) * t)
                pulse_size.append(4 + 8 * intensity)

    fig.data[1].x = pulse_x
    fig.data[1].y = pulse_y
    fig.data[1].marker.size = pulse_size if pulse_size else [6]

    for li, layer in enumerate(layers):
        trace_idx = 2 + li
        if trace_idx >= len(fig.data):
            break
        acts = np.array(layer["activations"], dtype=float)
        if len(acts) == 0:
            continue
        a_min, a_max = acts.min(), acts.max()
        a_norm = (acts - a_min) / (a_max - a_min + 1e-6)
        sizes = (8 + 14 * a_norm).tolist()
        colors = [
            f"rgba({int(61 + a * 100)},{int(139 + a * 50)},{int(253 - a * 80)},{0.5 + 0.5 * a:.2f})"
            for a in a_norm
        ]
        fig.data[trace_idx].marker.size = sizes
        fig.data[trace_idx].marker.color = colors

    fig.layout.title.text = f"NEURAL CORE  ·  {decision}  ·  conf {confidence:.2f}"
    return fig


def create_neural_figure(neural_data: Dict[str, Any],
                         pulse_phase: float = 0.0,
                         static_fig: Optional[go.Figure] = None) -> go.Figure:
    """Public API — reuses static topology when provided."""
    layers = neural_data.get("layers", [])
    if not layers:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            uirevision="nexus-neural-static",
        )
        return fig

    if static_fig is not None and len(static_fig.data) >= 3:
        fig = go.Figure(static_fig)
        if not hasattr(fig, "_nexus_positions") or fig._nexus_positions is None:
            sizes = [l["size"] for l in layers]
            fig._nexus_positions = _compute_positions(sizes)
            _, _, fig._nexus_edge_meta = _build_edge_geometry(fig._nexus_positions)
            fig._nexus_layer_sizes = sizes
        return update_neural_activity(fig, neural_data, pulse_phase)

    sizes = [l["size"] for l in layers]
    fig = build_static_neural(sizes)
    return update_neural_activity(fig, neural_data, pulse_phase)
