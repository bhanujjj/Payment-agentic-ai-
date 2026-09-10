# 🚀 Closed-Loop Autonomous Payment Routing AI Agent

A production-grade, risk-aware **Closed-Loop Autonomous Routing System** that monitors real-time payment network performance, diagnoses failures using generative LLMs (Gemini-2.5-flash), executes config-level routing mitigations, and leverages persistent reinforcement learning (via SQLite) to continuously optimize routing decisions.

Built to demonstrate what an AI-native reliability layer for a payments platform looks like end-to-end: **detect → diagnose → decide → act → learn**, with a human-approval gate on high-risk actions and a full audit trail of every decision.

![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688) ![React](https://img.shields.io/badge/frontend-React%2018-61DAFB)

---

## 🖥️ Live Demo — Control Dashboard

The system ships with an interactive **FastAPI + React control dashboard** (single process, zero build step) so the entire closed loop can be triggered and inspected from a browser. Below is the actual dashboard, running locally against the live agent — a real HDFC Bank outage scenario, diagnosed by Gemini, resolved automatically, and its outcome persisted to SQLite:

![Control Center — outage detected, diagnosed, and auto-resolved](docs/screenshots/control_center_outage_recovery.png)

*Top: live KPI strip aggregated from real scenario runs. Left: one-click failure injection. Right: the actual Gemini diagnosis, the decision engine's chosen action and risk level, and the pre/post recovery bars — all from a real request/response cycle, not mocked.*

Every intervention is written to a real SQLite database and surfaced in the **SQLite Memories** tab, so the reinforcement-learning signal is auditable, not a black box:

![SQLite Memories — full audit trail of past agent decisions and outcomes](docs/screenshots/sqlite_memories_log.png)

```bash
# Start the FastAPI + React Control Dashboard Server
python dashboard_server.py
```
Open **[http://localhost:8000/](http://localhost:8000/)** and you get:

*   **Control Center** — one-click failure injection (healthy traffic, single-bank degradation, full outage, UPI retry storm, or multiple simultaneous failures), a live KPI strip, the real-time routing map, and a step-by-step breakdown of the agent's diagnosis → decision → execution → learning cycle for the last run.
*   **SQLite Memories** — the full historical audit log of every intervention the agent has made, queried live from the database, with baseline vs. post-intervention deltas and outcome scores.
*   **Architecture** — an interactive map of the five-stage pipeline, each stage linked to its real source file, plus the safety guardrails and available routing actions.
*   Toast notifications and a confirm-before-destroy modal for state resets — this is built to be demoed live, not just curl'd.

---

## 📈 System Metrics & Validation Results

We executed a comprehensive 5-scenario multi-run validation suite (simulating **10 transaction windows** and **1,690 transaction events**) to measure diagnosis accuracy, recovery times, and metric transitions. These numbers come straight from [`run_multi_scenario_validation.py`](run_multi_scenario_validation.py) — re-run it yourself to reproduce them:

| Metric | Baseline (Outage) | Post-Intervention (Healed) | Delta / Outcome |
| :--- | :---: | :---: | :---: |
| **Transaction Success Rate** | **43.2%** | **97.5%** | **+54.3%** |
| **Transaction Failure Rate** | **56.8%** | **2.5%** | **-54.3%** (95.6% drop) |
| **Average Transaction Latency** | **10,318 ms** | **410 ms** | **-9,908 ms** reduction |
| **System Stabilization Speed** | — | — | **1 decision cycle (~5 min)** |
| **Problem Diagnosis Accuracy** | — | — | **60% (3/5 scenarios)** |
| **Average LLM Diagnosis Confidence** | — | — | **71.7%** |
| **Total Processed Transactions** | — | — | **1,690 simulated events** |

**Test suite:** 88/88 unit and integration tests passing (`pytest tests/ -v`) covering the metrics engine, reasoner, decision engine, executor, evaluator, memory/learning layer, and end-to-end agent loop.

---

## 🌀 Closed-Loop Architecture

The system operates on a continuous, five-stage **Observe → Reason → Decide → Act → Learn** loop:

```mermaid
graph TD
    subgraph Observable Environment
        A[Payment Ingestion Logs] -->|Compute Signals| B[MetricsEngine]
    end

    subgraph Cognitive Reasoning
        B -->|Anomaly Trigger| C[Gemini Reasoner]
        C -->|Scored Hypotheses| D[DecisionEngine]
        D -->|Evaluate Constraints & Risk| E[Decision Output]
    end

    subgraph Closed-Loop Action
        E -->|Execution Command| F[ActionExecutor]
        F -->|Real-Time Config Overrides| G[ROUTING_STATE]
        G -->|Dynamic Rerouting/Capping| A
    end

    subgraph Feedback Loop
        A -->|Post-Metrics Logs| H[OutcomeEvaluator]
        H -->|SUCCESS/FAILURE Classify| I[ActionLearner]
        I -->|Weight Adjustments| D
        I -->|Structured Insert| J[(SQLite database)]
    end

    style E fill:#4f46e5,stroke:#312e81,stroke-width:2px,color:#fff
    style G fill:#059669,stroke:#065f46,stroke-width:2px,color:#fff
    style J fill:#7c3aed,stroke:#5b21b6,stroke-width:2px,color:#fff
```

The **Architecture** tab in the live dashboard renders this same pipeline interactively, with each stage linked to its source file.

---

## 🛠️ Component Breakdown

### 1. Real-Time Ingest & Metrics Engine
*   **File:** [agent/metrics.py](agent/metrics.py)
*   Computes key performance indicators (KPIs) like success rate, latency (avg, p95), failure rate, and retry counts.
*   Calculates **Retry Effectiveness** to detect when excessive client-side retries are causing gateway load instead of resolving failures.

### 2. Multi-Hypothesis Diagnostic Reasoner
*   **File:** [agent/reasoner.py](agent/reasoner.py)
*   Uses generative AI (`gemini-2.5-flash`) to analyze signals, identify degraded gateways, and explain root causes.
*   Includes a **deterministic rule-based fallback** that takes over automatically if the LLM encounters rate limits or API key issues.

### 3. Constraint-Aware Decision Engine
*   **File:** [agent/decider.py](agent/decider.py)
*   Validates candidate actions against strict guardrails (`DecisionConstraints`) such as risk limits, minimum confidence levels, and human-in-the-loop approval triggers.
*   Applies a reinforcement multiplier to action weights dynamically based on historical outcomes of similar incidents.

### 4. Dynamic Action Executor
*   **File:** [agent/executor.py](agent/executor.py) & [simulation/routing_config.py](simulation/routing_config.py)
*   Actually closes the loop by modifying the active payment configuration.
*   Executes actions:
    *   `recommend_reroute`: Reroutes traffic away from degraded gateways.
    *   `recommend_path_suppression`: Suppresses pathways during complete outages.
    *   `recommend_retry_adjustment`: Restricts retry policies to prevent retry storms.

### 5. Persistent SQLite Learning Datastore
*   **File:** [agent/memory.py](agent/memory.py) & [agent/learner.py](agent/learner.py)
*   Replaced local JSON files with a structured SQLite database (`action_memory.db`).
*   Stores outcomes in a relational table, allowing the system to query past experiences, calculate learning rates, and show learning trends.

### 6. Control Dashboard
*   **File:** [dashboard_server.py](dashboard_server.py)
*   A single-file FastAPI backend serving a React 18 + Tailwind single-page app (no build step — Babel-in-browser).
*   Exposes `/api/metrics`, `/api/history`, `/api/reset`, and `/api/run_scenario` so the entire closed loop can be triggered and inspected live.

---

## 🧠 Production-Grade Safety Principles

*   **Causality-Safe Learning:** The learner skips reinforcement updates when non-intervention actions (`do_nothing` or `alert_ops`) are selected, preventing the agent from taking false credit/blame for natural performance variance.
*   **Graceful Fallback:** If the LLM service is degraded or quota-limited, the system falls back seamlessly to rule-based diagnostic routing.
*   **Risk-Gated Execution:** High-risk actions (e.g. full path suppression) are flagged `PENDING_HUMAN_APPROVAL` rather than executed blindly.
*   **State-Isolation Testing:** Built-in autouse fixtures ensure that simulation routing states are completely reset between unit tests, ensuring no cross-contamination.

---

## ⚡ Quickstart

### Setup & Requirements
```bash
# Install core dependencies
pip install -r requirements.txt

# Add your Gemini API key (optional, fallback engine active by default)
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Launch the Live Dashboard
```bash
python dashboard_server.py
# then open http://localhost:8000/
```

### Running the Automated Tests
Ensure the full agent loop, SQLite memory, and config execution work properly:
```bash
pytest tests/ -v
```

### Running the Validation Script
Execute the 5-scenario evaluation validation:
```bash
python run_multi_scenario_validation.py
```

### Running on Real Log Files
To run the agent sequential loop on historical transaction files (CSV or JSON):
```bash
python run_on_real_data.py --file your_transactions.csv --window 300
```
