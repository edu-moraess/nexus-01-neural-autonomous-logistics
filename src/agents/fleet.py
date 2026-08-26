"""
Fleet Manager & Multi-Agent Coordination
NEXUS-01
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from src.simulation.robot import Robot, RobotState
from src.neural.network import NeuralCore
from src.neural.inference import InferenceEngine
from src.agents.decision import reassign_tasks
from config.settings import SimulationConfig


class Fleet:
    def __init__(self, config: SimulationConfig = None, seed: int = 42):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.robots: List[Robot] = []
        self.neural = NeuralCore(self.cfg, seed=seed)
        self.inference = InferenceEngine(self.neural, seed=seed)
        self.reassignment_log: List[Dict[str, Any]] = []
        self._spawn_robots()

    def _spawn_robots(self):
        for i in range(self.cfg.NUM_ROBOTS):
            x = self.rng.uniform(8, 72)
            y = self.rng.uniform(8, 42)
            r = Robot(
                id=f"R{i+1:02d}",
                x=x, y=y,
                battery=self.rng.uniform(55, 100),
                temperature=self.rng.uniform(32, 42),
                health=self.rng.uniform(75, 100),
                capacity=self.cfg.ROBOT_CAPACITY,
                max_speed=self.cfg.MAX_SPEED,
                speed=self.rng.uniform(1.2, 2.2),
                priority=self.rng.uniform(0.3, 0.7),
            )
            roll = self.rng.random()
            if roll < 0.15:
                r.state = RobotState.CHARGING
            elif roll < 0.35:
                r.state = RobotState.MOVING
            elif roll < 0.5:
                r.state = RobotState.TRANSPORTING
                r.load = self.rng.uniform(30, 80)
            self.robots.append(r)

    def step(self, env, dt: float):
        for robot in self.robots:
            if robot.state == RobotState.ALERT and robot.health < 10:
                continue
            snapshot = env.snapshot_for_robot(robot, self.robots)
            features = robot.sense(snapshot)
            decision_idx, confidence, _, _ = self.inference.infer(features)
            robot.apply_decision(decision_idx, confidence, snapshot)
            robot.update_physics(dt)
            if robot.anomaly and self.rng.random() < 0.03:
                robot.anomaly = False
                robot.anomaly_score *= 0.5
        re_log = reassign_tasks(self.robots, [])
        if re_log:
            self.reassignment_log.extend(re_log)
            if len(self.reassignment_log) > 40:
                self.reassignment_log = self.reassignment_log[-40:]
        env.update_congestion(self.robots)
        env.update_demand(dt)

    def get_telemetry(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.robots]

    def stats(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in RobotState}
        for r in self.robots:
            counts[r.state.value] += 1
        return counts

    def average_battery(self) -> float:
        return float(np.mean([r.battery for r in self.robots]))

    def average_health(self) -> float:
        return float(np.mean([r.health for r in self.robots]))

    def average_temp(self) -> float:
        return float(np.mean([r.temperature for r in self.robots]))

    def active_count(self) -> int:
        return sum(1 for r in self.robots if r.is_operational)
