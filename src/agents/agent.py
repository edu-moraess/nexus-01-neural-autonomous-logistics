"""
Base Agent abstraction
NEXUS-01
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from src.simulation.robot import Robot, RobotState


class Agent:
    """
    Thin wrapper around Robot that exposes perception / decision / memory interface
    for multi-agent coordination.
    """

    def __init__(self, robot: Robot):
        self.robot = robot
        self.memory: list = []
        self.assigned_task: Optional[str] = None
        self.assisting: Optional[str] = None

    @property
    def id(self) -> str:
        return self.robot.id

    @property
    def state(self) -> RobotState:
        return self.robot.state

    def perceive(self, env_snapshot: Dict[str, Any]) -> None:
        self._last_snapshot = env_snapshot

    def remember(self, event: Dict[str, Any]) -> None:
        self.memory.append(event)
        if len(self.memory) > 30:
            self.memory = self.memory[-30:]

    def can_assist(self) -> bool:
        return (
            self.robot.is_operational
            and self.robot.state not in (RobotState.CHARGING, RobotState.MAINTENANCE, RobotState.ALERT)
            and self.robot.battery > 35
        )

    def to_dict(self) -> Dict[str, Any]:
        d = self.robot.to_dict()
        d["assigned_task"] = self.assigned_task
        d["assisting"] = self.assisting
        return d
