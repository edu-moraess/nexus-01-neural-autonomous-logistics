"""
Anomaly Detection Layer
NEXUS-01
"""

from __future__ import annotations
from typing import List, Dict, Any
from src.simulation.robot import RobotState


class AnomalyDetector:
    def detect(self, robots: List) -> List[Dict[str, Any]]:
        results = []
        for r in robots:
            score = 0.0
            reasons = []
            if r.temperature > 68:
                score += 0.35
                reasons.append("TEMP_HIGH")
            if r.battery < 18:
                score += 0.30
                reasons.append("BATTERY_LOW")
            if r.health < 40:
                score += 0.40
                reasons.append("HEALTH_LOW")
            if r.communication < 40:
                score += 0.25
                reasons.append("COMM_DEGRADED")
            if r.state in (RobotState.ALERT, RobotState.DEGRADED):
                score += 0.20
                reasons.append(r.state.value)
            if r.risk > 0.6:
                score += 0.15
                reasons.append("HIGH_RISK")
            score = min(1.0, score)
            is_anom = score > 0.45 or r.anomaly
            if is_anom:
                r.anomaly = True
                r.anomaly_score = max(r.anomaly_score, score)
                results.append({
                    "robot_id": r.id,
                    "score": round(score, 2),
                    "reasons": reasons,
                    "state": r.state.value,
                    "battery": round(r.battery, 1),
                    "temp": round(r.temperature, 1),
                    "health": round(r.health, 1),
                })
        return results
