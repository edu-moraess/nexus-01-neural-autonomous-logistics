# NEXUS-01 — Neural Autonomous Logistics System

**Experimental Robotics & AI Laboratory Platform**

A high-fidelity multi-agent digital twin of a futuristic autonomous logistics facility.  
The system continuously operates a heterogeneous robot fleet under dynamic disturbances while exposing a **live neural decision core** and full operational telemetry.

```
WORLD → SENSORS → NEURAL CORE → DECISION → ROBOTS → SYSTEM STATE → NEURAL CORE
```

## Concept

NEXUS-01 is **not** a single-mission path-planning demo.  
It models a **continuous logistics operation** with:

- 18 autonomous robots with individual state machines
- Real-time neural decision-making (16→32→24→16→9)
- Stochastic event engine that alters the world
- Multi-agent task reassignment
- Anomaly detection and System Efficiency Index (SEI)
- Live animated neural network visualization with signal pulses

## Architecture

```
nexus-01/
├── app.py
├── config/settings.py
├── src/
│   ├── simulation/   (environment, robot, events, engine)
│   ├── neural/       (network, inference, activations)
│   ├── agents/       (agent, fleet, decision)
│   ├── visualization/(warehouse, neural_network, telemetry, charts)
│   └── analytics/    (metrics, anomalies)
├── requirements.txt
└── README.md
```

## Key Capabilities

| Domain | Implementation |
|--------|----------------|
| Fleet | 18 robots — battery, thermal, health, load, risk, communication |
| States | IDLE · MOVING · PICKING · TRANSPORTING · CHARGING · MAINTENANCE · WAITING · ALERT · DEGRADED |
| Neural Core | 16 → 32 → 24 → 16 → 9 with live activations and moving signal pulses |
| Decisions | EXECUTE_TASK · WAIT · RECHARGE · REDUCE_SPEED · SWITCH_TASK · MAINTENANCE · AVOID_ZONE · ASSIST_PEER · PRIORITIZE_LOAD |
| Events | Battery degradation, sensor/motor failure, comms loss, congestion, priority task, station offline, overheating, obstacle, robot offline |
| Coordination | Automatic task/load reassignment |
| Digital Twin | Interactive tactical map + zones + congestion heatmap |
| Metrics | System Efficiency Index (SEI) + full operational KPIs |
| Controls | START / PAUSE / RESET / STEP · variable speed · reproducible seed |

## Installation

```bash
git clone https://github.com/edu-moraess/nexus-01-neural-autonomous-logistics.git
cd nexus-01-neural-autonomous-logistics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Neural Core

Architecture **16 → 32 → 24 → 16 → 9**. Forward pass with ReLU + Softmax. Live activation buffers, edge intensity and animated signal pulses. Designed as drop-in point for future PyTorch / RL / MARL policies.

## System Efficiency Index (SEI)

```
SEI = 0.25·Battery + 0.20·Health + 0.20·Utilization
    + 0.15·(1 − DegradedRatio) + 0.10·(1 − EventIntensity)
    + 0.10·DecisionConfidence
```

## Limitations

- Neural weights are static (not end-to-end trained)
- Soft congestion model
- Event recovery is probabilistic
- 2D Plotly visualization

## License

MIT

**NEXUS-01** — *Observe the brain. Command the fleet.*
