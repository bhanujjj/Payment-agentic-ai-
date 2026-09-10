"""
FastAPI Backend Server for Payment Routing AI Agent Dashboard.
Exposes endpoints for fetching metrics, execution history, and triggering scenarios.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
import random
import sqlite3
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from simulation.generator import PaymentGenerator
from simulation.routing_config import ROUTING_STATE, reset_routing
from agent.metrics import MetricsEngine
from agent.reasoner import Reasoner
from agent.decider import DecisionEngine
from agent.executor import ActionExecutor
from agent.evaluator import OutcomeEvaluator
from agent.memory import ActionMemory
from agent.learning_models import ActionOutcome, OutcomeClassification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardServer")

app = FastAPI(title="Payment Routing AI Agent Dashboard")

# Enable CORS for easy local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared memory db path resolve
DB_PATH = "./data/memory/action_memory.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Helper function to get top hypothesis
def get_top_hyp(reasoning):
    top = reasoning.get_top_hypothesis()
    if top:
        return {"hypothesis": top[0], "confidence": top[1]}
    return {"hypothesis": "normal_operation", "confidence": 0.5}

@app.get("/api/metrics")
async def get_metrics():
    """Retrieve the current active, suppressed banks and retry limits."""
    return {
        "active_banks": list(ROUTING_STATE["active_banks"]),
        "suppressed_banks": list(ROUTING_STATE["suppressed_banks"]),
        "retry_limits": dict(ROUTING_STATE["retry_limits"]),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/history")
async def get_history():
    """Fetch the agent memory log (SQLite DB records) in descending order."""
    if not os.path.exists(DB_PATH):
        return []
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM action_memories ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "context_summary": r["context_summary"],
                "action": r["action"],
                "risk_level": r["risk_level"],
                "pre_success_rate": r["pre_success_rate"],
                "pre_latency_ms": r["pre_latency_ms"],
                "pre_retry_count": r["pre_retry_count"],
                "pre_error_rate": r["pre_error_rate"],
                "post_success_rate": r["post_success_rate"],
                "post_latency_ms": r["post_latency_ms"],
                "post_retry_count": r["post_retry_count"],
                "post_error_rate": r["post_error_rate"],
                "success_rate_delta": r["success_rate_delta"],
                "latency_delta": r["latency_delta"],
                "retry_delta": r["retry_delta"],
                "error_rate_delta": r["error_rate_delta"],
                "outcome": r["outcome"],
                "outcome_score": r["outcome_score"],
                "timestamp": r["timestamp"],
                "notes": r["notes"]
            })
        return history
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

@app.post("/api/reset")
async def reset_agent_state():
    """Clear memory database and reset the routing overrides."""
    reset_routing()
    
    # Clear memories table
    if os.path.exists(DB_PATH):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_memories")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            
    return {"status": "success", "message": "State reset successfully"}

@app.post("/api/run_scenario")
async def run_scenario(payload: Dict[str, str]):
    """
    Executes a single scenario end-to-end:
    1. Resets routing state.
    2. Injects failures.
    3. Triggers pre-intervention transaction metrics.
    4. Runs agent decision-making.
    5. ActionExecutor applies dynamic routing/retry adjustments.
    6. Triggers post-intervention metrics.
    7. Evaluates learning and returns full report.
    """
    scenario = payload.get("scenario", "healthy")
    logger.info(f"Running scenario: {scenario}")
    
    # Reset routing config to fresh defaults
    reset_routing()
    
    # Initialize components
    engine = MetricsEngine()
    reasoner = Reasoner()
    decider = DecisionEngine()
    executor = ActionExecutor()
    evaluator = OutcomeEvaluator()
    memory = ActionMemory(storage_path=DB_PATH)
    
    # 1. Setup Scenario & Generate Baseline (Failure Phase)
    gen = PaymentGenerator(config={'seed': 42})
    payments_pre = []
    
    if scenario == "healthy":
        payments_pre = gen.generate_batch(count=120, time_span_seconds=300)
        expected_hyp = "normal_operation"
    elif scenario == "degradation":
        gen.simulate_bank_degradation("ICICI Bank")
        # Skew 66% traffic to ICICI Bank to make drop visible
        for i in range(120):
            offset_seconds = (i / 120) * 300
            gen.current_time = datetime.utcnow() + timedelta(seconds=offset_seconds)
            bank = "ICICI Bank" if i % 3 != 0 else None
            p = gen.generate_payment(bank=bank)
            payments_pre.append(p)
        expected_hyp = "bank_degradation"
    elif scenario == "outage":
        gen.simulate_bank_outage("HDFC Bank")
        # Skew 66% traffic to HDFC
        for i in range(120):
            offset_seconds = (i / 120) * 300
            gen.current_time = datetime.utcnow() + timedelta(seconds=offset_seconds)
            bank = "HDFC Bank" if i % 3 != 0 else None
            p = gen.generate_payment(bank=bank)
            payments_pre.append(p)
        expected_hyp = "bank_outage"
    elif scenario == "retry_storm":
        gen.base_failure_rate = 0.4
        payments_pre = gen.generate_batch(count=100, time_span_seconds=300)
        # Add retry storm
        retry_payments = []
        for p in payments_pre:
            if p.is_failed():
                retries = gen.simulate_retry_storm(p, retry_count=5)
                retry_payments.extend(retries)
        payments_pre.extend(retry_payments)
        expected_hyp = "retry_storm"
    elif scenario == "multiple_issues":
        gen.simulate_bank_outage("HDFC Bank")
        gen.simulate_bank_degradation("ICICI Bank")
        # Skew traffic to both
        for i in range(100):
            offset_seconds = (i / 100) * 300
            gen.current_time = datetime.utcnow() + timedelta(seconds=offset_seconds)
            bank = "HDFC Bank" if i % 2 == 0 else "ICICI Bank"
            p = gen.generate_payment(bank=bank)
            payments_pre.append(p)
        # Add retry storm
        retry_payments = []
        for p in payments_pre[:50]:
            if p.is_failed():
                retries = gen.simulate_retry_storm(p, retry_count=5)
                retry_payments.extend(retries)
        payments_pre.extend(retry_payments)
        expected_hyp = "bank_degradation"
    else:
        raise HTTPException(status_code=400, detail="Invalid scenario name")
        
    pre_signals = engine.compute_signals(payments_pre)
    
    # 2. Agent Reasoning & Decision
    reasoning = await reasoner.reason(pre_signals)
    decision = decider.decide(reasoning, pre_signals)
    
    # 3. Action Execution (Alters ROUTING_STATE in real-time)
    execution_result = executor.execute(decision, pre_signals)
    
    # 4. Generate Post-Intervention (Recovery Phase)
    gen_post = PaymentGenerator(config={'seed': 43})
    
    # Rerouting simulation
    if decision.selected_action in ["recommend_reroute", "recommend_path_suppression", "recommend_circuit_breaker"]:
        active_banks = [b for b in gen_post.BANKS if b not in pre_signals.degraded_banks]
        if not active_banks:
            active_banks = [b for b in gen_post.BANKS if b not in ["HDFC Bank", "ICICI Bank"]]
            
        payments_post = []
        for i in range(120):
            offset_seconds = (i / 120) * 300
            gen_post.current_time = datetime.utcnow() + timedelta(seconds=offset_seconds)
            p = gen_post.generate_payment(bank=random.choice(active_banks))
            payments_post.append(p)
    elif decision.selected_action == "recommend_retry_adjustment":
        # Reduced failure rates and capped retries
        payments_post = gen_post.generate_batch(count=120, time_span_seconds=300)
        retry_payments = []
        for p in payments_post[:15]:
            if p.is_failed():
                retries = gen_post.simulate_retry_storm(p, retry_count=2)
                retry_payments.extend(retries)
        payments_post.extend(retry_payments)
    else:
        # Default scenario
        payments_post = gen_post.generate_batch(count=120, time_span_seconds=300)
        
    post_signals = engine.compute_signals(payments_post)
    
    # 5. Evaluate learning outcome & Persist to SQLite
    outcome_class, outcome_score = evaluator.evaluate_from_signals(pre_signals, post_signals, decision.selected_action)
    
    is_intervention = decision.selected_action not in ["do_nothing", "alert_ops"]
    saved_in_sqlite = False
    
    if is_intervention:
        outcome = ActionOutcome(
            context_summary=decider._summarize_context(pre_signals),
            action=decision.selected_action,
            risk_level=decision.risk_level.value,
            pre_success_rate=pre_signals.overall_success_rate,
            pre_latency_ms=pre_signals.avg_latency_ms,
            pre_retry_count=pre_signals.total_retries,
            pre_error_rate=pre_signals.overall_failure_rate,
            post_success_rate=post_signals.overall_success_rate,
            post_latency_ms=post_signals.avg_latency_ms,
            post_retry_count=post_signals.total_retries,
            post_error_rate=post_signals.overall_failure_rate,
            success_rate_delta=post_signals.overall_success_rate - pre_signals.overall_success_rate,
            latency_delta=post_signals.avg_latency_ms - pre_signals.avg_latency_ms,
            retry_delta=post_signals.total_retries - pre_signals.total_retries,
            error_rate_delta=post_signals.overall_failure_rate - pre_signals.overall_failure_rate,
            outcome=outcome_class,
            outcome_score=outcome_score,
            timestamp=datetime.utcnow(),
            notes=f"Interactive dashboard scenario: {scenario}"
        )
        memory.add(outcome)
        saved_in_sqlite = True
        
    top_hypothesis = get_top_hyp(reasoning)
    is_correct_diagnosis = False
    if top_hypothesis["hypothesis"]:
        is_correct_diagnosis = expected_hyp.lower() in top_hypothesis["hypothesis"].lower()
    elif expected_hyp == "normal_operation":
        is_correct_diagnosis = True
        
    return {
        "scenario": scenario,
        "pre_metrics": {
            "success_rate": pre_signals.overall_success_rate,
            "failure_rate": pre_signals.overall_failure_rate,
            "avg_latency": pre_signals.avg_latency_ms,
            "retries": pre_signals.total_retries,
            "degraded_banks": list(pre_signals.degraded_banks)
        },
        "post_metrics": {
            "success_rate": post_signals.overall_success_rate,
            "failure_rate": post_signals.overall_failure_rate,
            "avg_latency": post_signals.avg_latency_ms,
            "retries": post_signals.total_retries,
            "degraded_banks": list(post_signals.degraded_banks)
        },
        "diagnosis": {
            "top_hypothesis": top_hypothesis["hypothesis"],
            "confidence": top_hypothesis["confidence"],
            "explanation": reasoning.explanation,
            "is_correct": is_correct_diagnosis
        },
        "decision": {
            "action": decision.selected_action,
            "confidence": decision.confidence,
            "risk_level": decision.risk_level.value,
            "requires_human_approval": decision.requires_human_approval,
            "reasoning": decision.reasoning_summary
        },
        "execution": {
            "executed": execution_result.executed,
            "status": execution_result.status.value,
            "impact_scope": execution_result.impact_scope,
            "effect": execution_result.expected_effect
        },
        "learning": {
            "outcome": outcome_class.value,
            "score": outcome_score,
            "saved": saved_in_sqlite
        }
    }

# Serving the Single Page App (SPA)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Closed-Loop Payment Agent Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22><text y=%22.9em%22 font-size=%2222%22>⚡</text></svg>">
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Outfit', 'sans-serif'],
                    },
                }
            }
        }
    </script>
    <!-- React and Babel CDNs -->
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js" crossorigin></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js" crossorigin></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.5/babel.min.js" crossorigin></script>
    <style>
        body {
            background-color: #0b1120;
            color: #f1f5f9;
            background-image:
                radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.16), transparent 40%),
                radial-gradient(circle at 85% 8%, rgba(16, 185, 129, 0.12), transparent 38%),
                radial-gradient(circle at 50% 100%, rgba(147, 51, 234, 0.10), transparent 45%);
            background-attachment: fixed;
        }
        .glass-card {
            background: rgba(30, 41, 59, 0.55);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .glass-card:hover { border-color: rgba(255, 255, 255, 0.14); }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in-up { animation: fadeInUp 0.45s ease-out both; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.25); border-radius: 8px; }
        .kpi-value { font-variant-numeric: tabular-nums; }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;

        const SCENARIOS = [
            { key: "healthy", label: "Normal Healthy Traffic", desc: "Baseline UPI/card traffic across all 8 gateways — no anomalies.", dot: "bg-emerald-500" },
            { key: "degradation", label: "ICICI Bank Degradation", desc: "Partial slowdown & elevated failures on a single gateway.", dot: "bg-amber-500" },
            { key: "outage", label: "HDFC Bank Complete Outage", desc: "Full gateway blackout — agent must suppress the path fast.", dot: "bg-rose-500" },
            { key: "retry_storm", label: "Severe UPI Retry Storm", desc: "Client-side retries amplify load faster than they resolve it.", dot: "bg-purple-500" },
            { key: "multiple_issues", label: "Multiple Critical Failures", desc: "Simultaneous outage + degradation stress-tests the reasoner.", dot: "bg-red-400" },
        ];

        function StatPill({ label, value, accent, suffix }) {
            return (
                <div className="glass-card rounded-2xl px-5 py-4 flex-1 min-w-[150px] transition">
                    <div className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">{label}</div>
                    <div className={`text-2xl font-extrabold kpi-value ${accent}`}>{value}<span className="text-sm font-semibold text-slate-500 ml-0.5">{suffix}</span></div>
                </div>
            );
        }

        function ConfirmModal({ open, title, body, onConfirm, onCancel }) {
            if (!open) return null;
            return (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
                    <div className="glass-card bg-slate-900/95 rounded-2xl p-6 max-w-sm w-full shadow-2xl fade-in-up">
                        <h4 className="text-base font-bold text-white mb-2">{title}</h4>
                        <p className="text-sm text-slate-400 mb-5">{body}</p>
                        <div className="flex justify-end gap-3">
                            <button onClick={onCancel} className="px-4 py-2 text-sm font-medium rounded-lg text-slate-300 hover:bg-slate-800 transition">Cancel</button>
                            <button onClick={onConfirm} className="px-4 py-2 text-sm font-semibold rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition">Reset Everything</button>
                        </div>
                    </div>
                </div>
            );
        }

        const PIPELINE_STAGES = [
            { title: "1. Observe", file: "metrics.py", color: "indigo", desc: "MetricsEngine ingests raw transaction logs and computes success rate, p95 latency, failure rate, and retry-effectiveness signals per gateway." },
            { title: "2. Reason", file: "reasoner.py", color: "purple", desc: "Gemini-2.5-flash scores multiple root-cause hypotheses (bank outage, degradation, retry storm) with confidence — falls back to deterministic rules if the LLM is unavailable." },
            { title: "3. Decide", file: "decider.py", color: "sky", desc: "DecisionEngine validates candidate actions against risk guardrails, confidence thresholds, and human-approval constraints, applying a reinforcement multiplier from past outcomes." },
            { title: "4. Act", file: "executor.py", color: "emerald", desc: "ActionExecutor mutates the live ROUTING_STATE — suppressing gateways, rerouting traffic, or capping retries — closing the loop in real time." },
            { title: "5. Learn", file: "learner.py", color: "rose", desc: "OutcomeEvaluator classifies SUCCESS/FAILURE from pre/post metrics and persists the experience to SQLite for future reinforcement." },
        ];

        const COLOR_MAP = {
            indigo: { bg: "bg-indigo-500/15", border: "border-indigo-500/30", text: "text-indigo-400", dot: "bg-indigo-500" },
            purple: { bg: "bg-purple-500/15", border: "border-purple-500/30", text: "text-purple-400", dot: "bg-purple-500" },
            sky: { bg: "bg-sky-500/15", border: "border-sky-500/30", text: "text-sky-400", dot: "bg-sky-500" },
            emerald: { bg: "bg-emerald-500/15", border: "border-emerald-500/30", text: "text-emerald-400", dot: "bg-emerald-500" },
            rose: { bg: "bg-rose-500/15", border: "border-rose-500/30", text: "text-rose-400", dot: "bg-rose-500" },
        };

        function ArchitectureView() {
            return (
                <div className="space-y-6 fade-in-up">
                    <div className="glass-card rounded-2xl p-6 shadow-xl">
                        <h3 className="text-base font-semibold text-white mb-1">Closed-Loop Architecture</h3>
                        <p className="text-xs text-slate-400 mb-6">A continuous five-stage Observe → Reason → Decide → Act → Learn loop. Each stage below maps to a real module in this codebase.</p>
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                            {PIPELINE_STAGES.map((s, i) => {
                                const c = COLOR_MAP[s.color];
                                return (
                                    <div key={s.title} className="relative">
                                        <div className={`glass-card rounded-xl p-4 h-full border ${c.border}`}>
                                            <div className={`inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide mb-2 ${c.text}`}>
                                                <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`}></span>{s.title}
                                            </div>
                                            <div className="text-[11px] font-mono text-slate-500 mb-2">{s.file}</div>
                                            <p className="text-xs text-slate-300 leading-relaxed">{s.desc}</p>
                                        </div>
                                        {i < PIPELINE_STAGES.length - 1 && (
                                            <div className="hidden md:flex absolute top-1/2 -right-3 -translate-y-1/2 z-10 text-slate-600 text-lg">→</div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="glass-card rounded-2xl p-6 shadow-xl">
                            <h3 className="text-base font-semibold text-white mb-4">Production-Grade Safety Principles</h3>
                            <ul className="space-y-3 text-sm text-slate-300">
                                <li className="flex gap-3"><span className="text-emerald-400 font-bold">✓</span><span><b className="text-white">Causality-safe learning</b> — the learner skips reinforcement updates on non-intervention actions (do_nothing / alert_ops) to avoid false credit assignment.</span></li>
                                <li className="flex gap-3"><span className="text-emerald-400 font-bold">✓</span><span><b className="text-white">Graceful LLM fallback</b> — if Gemini is rate-limited or unavailable, a deterministic rule-based reasoner takes over seamlessly.</span></li>
                                <li className="flex gap-3"><span className="text-emerald-400 font-bold">✓</span><span><b className="text-white">Risk-gated execution</b> — high-risk actions (e.g. full path suppression) require human approval before the loop closes.</span></li>
                                <li className="flex gap-3"><span className="text-emerald-400 font-bold">✓</span><span><b className="text-white">State-isolated testing</b> — autouse pytest fixtures reset routing state between tests to prevent cross-contamination.</span></li>
                            </ul>
                        </div>
                        <div className="glass-card rounded-2xl p-6 shadow-xl">
                            <h3 className="text-base font-semibold text-white mb-4">Tech Stack</h3>
                            <div className="flex flex-wrap gap-2">
                                {["FastAPI", "React 18", "TailwindCSS", "SQLite", "Gemini 2.5 Flash", "Pydantic", "pytest", "uvicorn"].map(t => (
                                    <span key={t} className="text-xs font-medium px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">{t}</span>
                                ))}
                            </div>
                            <h3 className="text-base font-semibold text-white mt-6 mb-3">Available Actions</h3>
                            <div className="space-y-2 text-xs text-slate-300">
                                <div className="flex justify-between bg-slate-900/40 rounded-lg px-3 py-2 border border-slate-800"><span className="font-mono text-indigo-300">recommend_reroute</span><span className="text-slate-500">shift traffic away from a degraded gateway</span></div>
                                <div className="flex justify-between bg-slate-900/40 rounded-lg px-3 py-2 border border-slate-800"><span className="font-mono text-indigo-300">recommend_path_suppression</span><span className="text-slate-500">fully suppress a pathway during an outage</span></div>
                                <div className="flex justify-between bg-slate-900/40 rounded-lg px-3 py-2 border border-slate-800"><span className="font-mono text-indigo-300">recommend_retry_adjustment</span><span className="text-slate-500">cap client retries to stop a retry storm</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        function RecoveryBars({ run }) {
            const preSuccess = run.pre_metrics.success_rate * 100;
            const postSuccess = run.post_metrics.success_rate * 100;
            const preLatency = run.pre_metrics.avg_latency;
            const postLatency = run.post_metrics.avg_latency;
            const maxLatency = Math.max(preLatency, postLatency, 1);
            const [animate, setAnimate] = useState(false);
            useEffect(() => {
                setAnimate(false);
                const t = setTimeout(() => setAnimate(true), 60);
                return () => clearTimeout(t);
            }, [run]);

            const Bar = ({ label, value, display, max, colorClass }) => (
                <div className="flex-1">
                    <div className="h-40 flex items-end">
                        <div
                            className={`w-full rounded-t-lg ${colorClass} transition-all duration-700 ease-out flex items-start justify-center pt-1.5`}
                            style={{ height: animate ? `${Math.max((value / max) * 100, 4)}%` : "2%" }}
                        >
                            <span className="text-xs font-bold text-white drop-shadow">{display}</span>
                        </div>
                    </div>
                    <div className="text-[11px] text-slate-500 text-center mt-2">{label}</div>
                </div>
            );

            return (
                <div className="h-56 flex flex-col">
                    <div className="flex-1 flex gap-6 px-2">
                        <div className="flex-1 flex gap-3">
                            <Bar label="Pre" value={preSuccess} display={`${preSuccess.toFixed(0)}%`} max={100} colorClass="bg-gradient-to-t from-red-600 to-red-400" />
                            <Bar label="Post" value={postSuccess} display={`${postSuccess.toFixed(0)}%`} max={100} colorClass="bg-gradient-to-t from-emerald-600 to-emerald-400" />
                        </div>
                        <div className="w-px bg-slate-800"></div>
                        <div className="flex-1 flex gap-3">
                            <Bar label="Pre" value={preLatency} display={`${Math.round(preLatency)}ms`} max={maxLatency} colorClass="bg-gradient-to-t from-red-600 to-red-400" />
                            <Bar label="Post" value={postLatency} display={`${Math.round(postLatency)}ms`} max={maxLatency} colorClass="bg-gradient-to-t from-emerald-600 to-emerald-400" />
                        </div>
                    </div>
                    <div className="flex justify-around text-xs text-slate-400 font-semibold mt-3 pt-3 border-t border-slate-800/60">
                        <span>Success Rate</span>
                        <span>Avg Latency</span>
                    </div>
                </div>
            );
        }

        function App() {
            const [activeTab, setActiveTab] = useState("dashboard");
            const [scenarioRunning, setScenarioRunning] = useState(false);
            const [routingState, setRoutingState] = useState({ active_banks: [], suppressed_banks: [], retry_limits: {} });
            const [history, setHistory] = useState([]);
            const [latestRun, setLatestRun] = useState(null);
            const [activeScenario, setActiveScenario] = useState("None");
            const [showResetModal, setShowResetModal] = useState(false);
            const [toast, setToast] = useState(null);

            const showToast = (msg) => {
                setToast(msg);
                setTimeout(() => setToast(null), 3000);
            };

            const totalRuns = history.length;
            const avgUplift = totalRuns > 0
                ? (history.reduce((s, m) => s + (m.post_success_rate - m.pre_success_rate), 0) / totalRuns) * 100
                : 0;
            const avgScore = totalRuns > 0
                ? history.reduce((s, m) => s + m.outcome_score, 0) / totalRuns
                : 0;
            const successCount = history.filter(m => m.outcome === "SUCCESS").length;

            useEffect(() => {
                fetchMetrics();
                fetchHistory();
            }, []);

            const fetchMetrics = async () => {
                try {
                    const res = await fetch("/api/metrics");
                    const data = await res.json();
                    setRoutingState(data);
                } catch (e) {
                    console.error("Error fetching metrics", e);
                }
            };

            const fetchHistory = async () => {
                try {
                    const res = await fetch("/api/history");
                    const data = await res.json();
                    setHistory(data);
                } catch (e) {
                    console.error("Error fetching history", e);
                }
            };

            const resetState = async () => {
                setShowResetModal(false);
                try {
                    await fetch("/api/reset", { method: "POST" });
                    setLatestRun(null);
                    setActiveScenario("None");
                    fetchMetrics();
                    fetchHistory();
                    showToast("✓ System state reset — routing overrides cleared, memories wiped.");
                } catch (e) {
                    console.error("Error resetting state", e);
                    showToast("✗ Reset failed — see console.");
                }
            };

            const runScenario = async (name) => {
                setScenarioRunning(true);
                setActiveScenario(name.toUpperCase());
                setActiveTab("dashboard");
                try {
                    const res = await fetch("/api/run_scenario", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ scenario: name })
                    });
                    const data = await res.json();
                    setLatestRun(data);
                    fetchMetrics();
                    fetchHistory();
                    showToast(`✓ Scenario "${name}" resolved in ${data.learning.outcome === 'SUCCESS' ? 'one cycle' : 'partial recovery'} — action: ${data.decision.action.replace('recommend_', '')}`);
                } catch (e) {
                    console.error("Error running scenario", e);
                    showToast("✗ Scenario run failed — see console.");
                } finally {
                    setScenarioRunning(false);
                }
            };

            return (
                <div className="min-h-screen flex flex-col font-sans">
                    <ConfirmModal
                        open={showResetModal}
                        title="Reset all agent state?"
                        body="This clears every SQLite memory record and restores routing config to defaults. This cannot be undone."
                        onConfirm={resetState}
                        onCancel={() => setShowResetModal(false)}
                    />

                    {toast && (
                        <div className="fixed top-5 right-5 z-[110] glass-card bg-slate-900/95 rounded-xl px-4 py-3 shadow-2xl max-w-sm fade-in-up text-sm text-slate-200 border-l-4 border-l-indigo-500">
                            {toast}
                        </div>
                    )}

                    {/* Header */}
                    <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 py-4 flex flex-wrap items-center justify-between gap-3 sticky top-0 z-50">
                        <div className="flex items-center space-x-3">
                            <div className="h-10 w-10 bg-indigo-600 rounded-xl flex items-center justify-between p-2.5 shadow-lg shadow-indigo-500/20">
                                <svg xmlns="http://www.w3.org/2000/svg" className="text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                            </div>
                            <div>
                                <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                                    Closed-Loop Payment Routing Agent
                                    <span className="text-xs bg-indigo-500/20 text-indigo-400 font-medium px-2 py-0.5 rounded-full border border-indigo-500/25 uppercase">Autonomous</span>
                                </h1>
                                <p className="text-xs text-slate-400">Observe • Diagnose • Decide • Act • Learn — self-healing payment routing, powered by Gemini</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            <button onClick={() => setActiveTab("dashboard")} className={`px-4 py-2 text-sm font-medium rounded-lg transition ${activeTab === 'dashboard' ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`}>
                                Control Center
                            </button>
                            <button onClick={() => setActiveTab("history")} className={`px-4 py-2 text-sm font-medium rounded-lg transition ${activeTab === 'history' ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`}>
                                SQLite Memories
                            </button>
                            <button onClick={() => setActiveTab("architecture")} className={`px-4 py-2 text-sm font-medium rounded-lg transition ${activeTab === 'architecture' ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`}>
                                Architecture
                            </button>
                            <button onClick={() => setShowResetModal(true)} className="border border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500 hover:text-white px-4 py-2 text-sm font-medium rounded-lg transition">
                                Reset System State
                            </button>
                        </div>
                    </header>

                    {/* KPI strip */}
                    <div className="px-6 pt-6 max-w-7xl mx-auto w-full">
                        <div className="flex flex-wrap gap-4">
                            <StatPill label="Scenario Runs Logged" value={totalRuns} accent="text-white" suffix="" />
                            <StatPill label="Avg Success-Rate Uplift" value={totalRuns ? `+${avgUplift.toFixed(1)}` : "—"} accent="text-emerald-400" suffix={totalRuns ? "%" : ""} />
                            <StatPill label="Learning Outcome Score" value={totalRuns ? avgScore.toFixed(2) : "—"} accent="text-indigo-400" suffix="/ 1.00" />
                            <StatPill label="Interventions Marked Success" value={totalRuns ? `${successCount}/${totalRuns}` : "—"} accent="text-purple-400" suffix="" />
                            <StatPill label="Active Gateways" value={`${routingState.active_banks.length}`} accent="text-amber-400" suffix="/ 8" />
                        </div>
                    </div>

                    {/* Main Area */}
                    <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
                        {activeTab === "architecture" ? (
                            <ArchitectureView />
                        ) : activeTab === "dashboard" ? (
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Left column: Actions and Environment Health */}
                                <div className="space-y-6 lg:col-span-1">
                                    <div className="glass-card rounded-2xl p-6 shadow-xl">
                                        <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                                            <span className="h-2 w-2 bg-indigo-500 rounded-full"></span>
                                            Injected Failure Scenarios
                                        </h3>
                                        <p className="text-xs text-slate-400 mb-4">Click to trigger dynamic real-time traffic outages & test the agent loop.</p>
                                        <div className="space-y-2.5">
                                            {SCENARIOS.map(s => {
                                                const isActive = activeScenario === s.key.toUpperCase();
                                                return (
                                                    <button key={s.key} disabled={scenarioRunning} title={s.desc} onClick={() => runScenario(s.key)} className={`w-full py-2.5 px-4 rounded-xl text-sm font-medium flex items-center justify-between border transition disabled:opacity-50 disabled:cursor-not-allowed ${
                                                        isActive
                                                        ? "bg-indigo-600 border-indigo-400 text-white shadow-lg shadow-indigo-500/20"
                                                        : "bg-slate-800/80 hover:bg-slate-700/80 border-slate-700/50 text-slate-300"
                                                    }`}>
                                                        <span className="text-left">
                                                            <span className="block">{s.label}</span>
                                                            <span className={`block text-[11px] font-normal ${isActive ? "text-indigo-200" : "text-slate-500"}`}>{s.desc}</span>
                                                        </span>
                                                        <span className="flex items-center gap-2 shrink-0 pl-3">
                                                            {isActive && <span className="text-xs text-indigo-200">✓</span>}
                                                            <span className={`h-2 w-2 rounded-full ${s.dot} shadow-md ${isActive && scenarioRunning ? "animate-ping" : ""}`}></span>
                                                        </span>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    {/* Active Routing config state */}
                                    <div className="glass-card rounded-2xl p-6 shadow-xl">
                                        <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                                            <span className="h-2 w-2 bg-emerald-500 rounded-full"></span>
                                            Real-Time Routing Map (Simulator Reads)
                                        </h3>
                                        <div className="space-y-4">
                                            <div>
                                                <span className="text-xs text-slate-400 block mb-1">Active Gateway Count</span>
                                                <div className="text-lg font-bold text-emerald-400">
                                                    {routingState.active_banks.length} / 8 Providers
                                                </div>
                                            </div>
                                            <div>
                                                <span className="text-xs text-slate-400 block mb-2">Suppressed/De-routed Gateways</span>
                                                <div className="flex flex-wrap gap-2">
                                                    {routingState.suppressed_banks.length === 0 ? (
                                                        <span className="text-xs text-emerald-500 bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">All pathways active (healthy)</span>
                                                    ) : (
                                                        routingState.suppressed_banks.map(b => (
                                                            <span key={b} className="text-xs text-rose-400 bg-rose-500/10 px-2.5 py-1 rounded-lg border border-rose-500/20 flex items-center gap-1.5">
                                                                ⛔ {b} (suppressed)
                                                            </span>
                                                        ))
                                                    )}
                                                </div>
                                            </div>
                                            <div>
                                                <span className="text-xs text-slate-400 block mb-2">Retry Limits Capped</span>
                                                <div className="bg-slate-900/50 rounded-xl p-3 border border-slate-800">
                                                    {Object.keys(routingState.retry_limits).length === 0 ? (
                                                        <span className="text-xs text-slate-400">Standard default policy active (max 3 retries)</span>
                                                    ) : (
                                                        Object.entries(routingState.retry_limits).map(([m, lim]) => (
                                                            <div key={m} className="flex justify-between text-xs py-1 border-b border-slate-800/40 last:border-b-0">
                                                                <span className="text-slate-400">{m} retry limit</span>
                                                                <span className="text-indigo-400 font-bold">{lim} Max Retries</span>
                                                            </div>
                                                        ))
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Center/Right columns: Live analysis charts and logs */}
                                <div className="lg:col-span-2 space-y-6">
                                    {scenarioRunning && (
                                        <div className="glass-card rounded-2xl p-8 flex flex-col items-center justify-center text-center py-20 animate-pulse border border-indigo-500/30">
                                            <div className="h-12 w-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                                            <h4 className="text-lg font-bold text-white">Agent Running Step Cycle...</h4>
                                            <p className="text-xs text-slate-400 max-w-sm mt-1">Executing observing metrics calculation, querying Gemini reasoning model, and resolving optimal routing changes...</p>
                                        </div>
                                    )}

                                    {!scenarioRunning && !latestRun && (
                                        <div className="glass-card rounded-2xl p-12 text-center py-24 flex flex-col items-center justify-center">
                                            <div className="h-16 w-16 bg-slate-800 rounded-2xl flex items-center justify-center text-3xl mb-4 border border-slate-700">🖥️</div>
                                            <h4 className="text-lg font-bold text-slate-300">Ready to Monitor Routing Loop</h4>
                                            <p className="text-xs text-slate-400 max-w-md mt-2">Trigger a failure scenario on the left panel to test how the agent automatically detects, diagnoses, executes recovery routing configurations, and records learning feedback.</p>
                                        </div>
                                    )}

                                    {!scenarioRunning && latestRun && (
                                        <div className="space-y-6">
                                            {/* Baseline Comparison Card */}
                                            <div className="glass-card rounded-2xl p-6 shadow-xl">
                                                <div className="flex items-center justify-between mb-4">
                                                    <div>
                                                        <h3 className="text-base font-semibold text-white">Baseline vs. Agent-Healed Recovery</h3>
                                                        <p className="text-xs text-slate-400">Performance recovery comparison for scenario: {latestRun.scenario.toUpperCase()}</p>
                                                    </div>
                                                    <span className={`text-xs font-bold px-3 py-1 rounded-full border ${latestRun.learning.outcome === 'SUCCESS' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}`}>
                                                        {latestRun.learning.outcome} (Score: {latestRun.learning.score.toFixed(2)})
                                                    </span>
                                                </div>
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                                                    <RecoveryBars run={latestRun} />
                                                    <div className="grid grid-cols-2 gap-4">
                                                        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                                                            <span className="text-xs text-slate-400 block mb-1">Pre-Success Rate</span>
                                                            <div className="text-2xl font-bold text-red-400">
                                                                {(latestRun.pre_metrics.success_rate * 100).toFixed(1)}%
                                                            </div>
                                                        </div>
                                                        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                                                            <span className="text-xs text-slate-400 block mb-1">Post-Success Rate</span>
                                                            <div className="text-2xl font-bold text-emerald-400">
                                                                {(latestRun.post_metrics.success_rate * 100).toFixed(1)}%
                                                            </div>
                                                        </div>
                                                        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                                                            <span className="text-xs text-slate-400 block mb-1">Pre-Latency</span>
                                                            <div className="text-xl font-bold text-slate-300">
                                                                {Math.round(latestRun.pre_metrics.avg_latency)}ms
                                                            </div>
                                                        </div>
                                                        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                                                            <span className="text-xs text-slate-400 block mb-1">Post-Latency</span>
                                                            <div className="text-xl font-bold text-slate-300">
                                                                {Math.round(latestRun.post_metrics.avg_latency)}ms
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* AI Diagnosis and Action Card */}
                                            <div className="glass-card rounded-2xl p-6 shadow-xl space-y-5">
                                                <div className="border-b border-slate-800/60 pb-4">
                                                    <h4 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-2">1. AI Root-Cause Diagnosis (Gemini Reasoning)</h4>
                                                    <div className="flex items-center gap-3 mb-2">
                                                        <span className="text-xs text-slate-400">Top Hypothesis:</span>
                                                        <span className="text-xs font-bold text-white bg-slate-800 px-2.5 py-0.5 rounded-lg border border-slate-700">
                                                            {latestRun.diagnosis.top_hypothesis.toUpperCase()}
                                                        </span>
                                                        <span className="text-xs text-slate-400">Confidence:</span>
                                                        <span className="text-xs font-bold text-indigo-300">
                                                            {(latestRun.diagnosis.confidence * 100).toFixed(0)}%
                                                        </span>
                                                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${latestRun.diagnosis.is_correct ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20'}`}>
                                                            {latestRun.diagnosis.is_correct ? "✓ Correct Diagnosis" : "⚠️ Misdiagnosis"}
                                                        </span>
                                                    </div>
                                                    <p className="text-sm text-slate-200 bg-slate-900/40 p-3 rounded-xl border border-slate-800/80 leading-relaxed">
                                                        {latestRun.diagnosis.explanation}
                                                    </p>
                                                </div>

                                                <div className="border-b border-slate-800/60 pb-4">
                                                    <h4 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-2">2. Decision & Real-Time Config Change (Closed-Loop)</h4>
                                                    <div className="flex items-center gap-3 mb-2">
                                                        <span className="text-xs text-slate-400">Executed Action:</span>
                                                        <span className="text-xs font-bold text-white bg-slate-800 px-2.5 py-0.5 rounded-lg border border-slate-700">
                                                            {latestRun.decision.action.toUpperCase()}
                                                        </span>
                                                        <span className="text-xs text-slate-400">Risk Level:</span>
                                                        <span className="text-xs font-bold text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded border border-yellow-500/20">
                                                            {latestRun.decision.risk_level}
                                                        </span>
                                                    </div>
                                                    <div className="bg-slate-900/40 p-3 rounded-xl border border-slate-800/80 text-sm space-y-1">
                                                        <div className="flex justify-between"><span className="text-slate-400">Action Rationale:</span> <span className="text-slate-200 font-medium">{latestRun.decision.reasoning || "Standard routing override"}</span></div>
                                                        <div className="flex justify-between"><span className="text-slate-400">Simulation Update:</span> <span className="text-emerald-400 font-bold">{latestRun.execution.effect}</span></div>
                                                        <div className="flex justify-between"><span className="text-slate-400">Execution Status:</span> <span className="text-slate-300 font-medium">{latestRun.execution.status}</span></div>
                                                    </div>
                                                </div>

                                                <div>
                                                    <h4 className="text-sm font-semibold text-purple-400 uppercase tracking-wider mb-2">3. Causality-Safe Outcome Evaluation & Memory</h4>
                                                    <div className="text-sm text-slate-300 bg-slate-900/40 p-3 rounded-xl border border-slate-800/80 space-y-1">
                                                        <div className="flex justify-between"><span className="text-slate-400">Failure Rate Reduction:</span> <span className="text-emerald-400">-{((latestRun.pre_metrics.failure_rate - latestRun.post_metrics.failure_rate)*100).toFixed(1)}% drop</span></div>
                                                        <div className="flex justify-between"><span className="text-slate-400">Persistence Update:</span> <span className="text-slate-200">{latestRun.learning.saved ? "✓ Stored in SQLite memories database" : "Skipped (Causality safety active on non-intervention)"}</span></div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <div className="glass-card rounded-2xl p-6 shadow-xl">
                                <div className="flex items-center justify-between mb-6">
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">SQLite Memory Database Log (`action_memories`)</h3>
                                        <p className="text-xs text-slate-400">Historical records retrieved dynamically from the local SQLite datastore.</p>
                                    </div>
                                    <button onClick={fetchHistory} className="border border-slate-700 bg-slate-800 text-white hover:bg-slate-700 px-4 py-2 text-sm font-medium rounded-lg transition">
                                        Refresh Log
                                    </button>
                                </div>
                                
                                {history.length === 0 ? (
                                    <div className="text-center py-20 text-slate-500 text-sm">
                                        No experiences stored in memory database yet. Run failure scenarios to populate memories.
                                    </div>
                                ) : (
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-left text-sm text-slate-300 border-collapse">
                                            <thead>
                                                <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase font-semibold">
                                                    <th className="py-3 px-4">ID</th>
                                                    <th className="py-3 px-4">Timestamp</th>
                                                    <th className="py-3 px-4">Scenario / Action</th>
                                                    <th className="py-3 px-4 text-center">Baseline SR</th>
                                                    <th className="py-3 px-4 text-center">Post SR</th>
                                                    <th className="py-3 px-4 text-center">SR Delta</th>
                                                    <th className="py-3 px-4 text-center">Outcome Score</th>
                                                    <th className="py-3 px-4">Evaluation</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {history.map(m => (
                                                    <tr key={m.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition">
                                                        <td className="py-3.5 px-4 font-mono text-slate-500">#{m.id}</td>
                                                        <td className="py-3.5 px-4 text-xs text-slate-400">{new Date(m.timestamp).toLocaleString()}</td>
                                                        <td className="py-3.5 px-4">
                                                            <div className="font-semibold text-white">{m.action.replace("recommend_", "").toUpperCase()}</div>
                                                            <div className="text-xs text-slate-500 font-mono truncate max-w-xs">{m.context_summary}</div>
                                                        </td>
                                                        <td className="py-3.5 px-4 text-center text-red-400 font-semibold">{(m.pre_success_rate * 100).toFixed(0)}%</td>
                                                        <td className="py-3.5 px-4 text-center text-emerald-400 font-semibold">{(m.post_success_rate * 100).toFixed(0)}%</td>
                                                        <td className="py-3.5 px-4 text-center font-bold text-emerald-400">+{((m.post_success_rate - m.pre_success_rate)*100).toFixed(0)}%</td>
                                                        <td className="py-3.5 px-4 text-center font-mono font-bold text-slate-200">{m.outcome_score.toFixed(2)}</td>
                                                        <td className="py-3.5 px-4">
                                                            <span className={`text-xs px-2.5 py-0.5 rounded-full border ${m.outcome === 'SUCCESS' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}`}>
                                                                {m.outcome}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}
                    </main>

                    {/* Footer */}
                    <footer className="border-t border-slate-800 bg-slate-950 py-4 px-6 text-center text-xs text-slate-500">
                        Closed-Loop Autonomous Payment Routing Agent — FastAPI + React control dashboard, SQLite-backed reinforcement learning.
                    </footer>
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById("root"));
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse(content=INDEX_HTML, status_code=200)

def main():
    logger.info("Starting Dashboard server on http://localhost:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
