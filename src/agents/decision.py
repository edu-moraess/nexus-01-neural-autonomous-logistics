"""
Decision Mapping & Task Reassignment Logic
NEXUS-01
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from src.simulation.robot import Robot, RobotState
from config.settings import DECISION_LABELS


def apply_operational_decision(robot: Robot, decision_idx: int, confidence: float, env: Dict[str, Any]) -> None:
    """Map neural output index to concrete robot behaviour."""
    robot.apply_decision(decision_idx, confidence, env)


def reassign_tasks(robots: List[Robot], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple multi-agent task reassignment.
    When a robot goes offline / charging / maintenance, try to hand off its load/task
    to the nearest capable peer.
    """
    log = []
    needy = [r for r in robots if r.state in (RobotState.ALERT, RobotState.CHARGING, RobotState.MAINTENANCE)
             and r.load > 5]
    capable = [r for r in robots if r.is_operational
               and r.state in (RobotState.IDLE, RobotState.WAITING, RobotState.MOVING)
               and r.load < 10 and r.battery > 40]

    for n in needy:
        if not capable:
            break
        peer = min(capable, key=lambda c: (c.x - n.x) ** 2 + (c.y - n.y) ** 2)
        transferred = min(n.load, peer.capacity - peer.load)
        if transferred > 5:
            peer.load += transferred
            n.load -= transferred
            peer.priority = max(peer.priority, n.priority)
            peer.state = RobotState.TRANSPORTING
            peer.task = f"reassigned_from_{n.id}"
            capable.remove(peer)
            log.append({
                "type": "TASK_REASSIGNED",
                "from": n.id,
                "to": peer.id,
                "load": round(transferred, 1),
                "message": f"Task/load from {n.id} reassigned to {peer.id} ({transferred:.0f} units)",
            })
    return log
