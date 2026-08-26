"""
Fleet Telemetry Table & Event Feed helpers
"""

from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd


def robots_to_dataframe(robots: List[Dict[str, Any]]) -> pd.DataFrame:
    if not robots:
        return pd.DataFrame()
    df = pd.DataFrame(robots)
    cols = ["id", "state", "battery", "temp", "load", "health", "decision", "confidence", "risk", "anomaly"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("id")


def format_events(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "_No recent events_"
    lines = []
    for e in events[:12]:
        sev = e.get("severity", "info").upper()
        icon = {"CRITICAL": "◆", "WARNING": "▲", "INFO": "●"}.get(sev, "○")
        rid = e.get("robot_id") or "SYS"
        lines.append(f"`{icon}` **{sev}** `{rid}` — {e.get('message', '')}")
    return "\n\n".join(lines)
