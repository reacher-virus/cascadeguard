# 🛡️ CHARM Multi-Agent Reliability Guardrail & Explainable AI (XAI)
## Plain-Language Explainability and Remediation Guardrail Layer for Cascading Hallucination Detection in No-Code Agentic RAG Platforms

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40%2B-yellow.svg)](https://huggingface.co/)
[![NIST AI 600-1](https://img.shields.io/badge/Compliance-NIST%20AI%20600--1-green.svg)](https://www.nist.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Summary

Autonomous multi-agent pipelines (such as AutoAgent, CrewAI, LangGraph) are acutely vulnerable to **cascading hallucinations**: when an early retrieval or reasoning agent makes a subtle factual error, downstream execution agents accept this premise as ground truth. The error compounds across stages, culminating in critical operational failures (e.g. invalid medical dosages, unauthorized financial transactions, corrupted infrastructure changes).

This repository contains the complete implementation of a **100% offline, local edge-deployable guardrail system** combining:
1. **CHARM Neural Detection Backend**: Multi-signal detection ($a_{sfv}, a_{csct}, a_{cpm}$) powered by local DeBERTa-v3 and MPNet.
2. **Deterministic State Interceptor Middleware**: Non-blocking asynchronous worker queue, Two-Phase Commit (2PC) action gating, and Positional Causal Matrix DAG surgical rollback.
3. **Local Qwen 2.5 Explainable AI (XAI)**: Fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` via 4-bit NF4 QLoRA translating technical telemetry into 2-sentence plain-English NIST AI 600-1 alerts with 1-click human remedies.
4. **Interactive Reliability Studio**: Streamlit dashboard featuring live error injection presets, real-time stage handoffs, and closed-loop working self-healing.

---

## 🚀 Key Performance & Validation Metrics

| Metric | Measured Score | Target Specification | Validation Description |
| :--- | :---: | :---: | :--- |
| **Cascade Detection Rate (CDR)** | **89.4%** | $\ge 85.0\%$ | High sensitivity against multi-stage failure propagation |
| **False Positive Rate (FPR)** | **5.3%** | $\le 6.0\%$ | Proven via balanced 50/50 clean control benchmarks |
| **Early Prevention Rate (EPR)** | **82.1%** | $\ge 80.0\%$ | Halts corruption at Step 1 or 2 before tool execution |
| **Perceived Latency Overhead ($LO/s$)** | **< 12.0 ms** | $< 50.0\text{ ms}$ | Asynchronous speculative execution hides neural inference |
| **XAI Inference Latency** | **112 ms** | $< 200\text{ ms}$ | 4-bit NF4 quantized local inference on Tesla T4 |
| **Peak GPU VRAM Footprint** | **2.25 GB** | $< 4.0\text{ GB}$ | Runs seamlessly on commodity 16 GB GPUs (Tesla T4) |
| **Statistical Significance** | **$p < 0.001$** | $p < 0.01$ | $N=10,000$ paired bootstrap resamples |

---

## 📂 Repository Layout

```
c:/harsh/CAPSTONE/
├── README.md                              # Master project overview & quick start (this file)
├── requirements.txt                       # Pinned Python package dependencies
├── docs/                                  # Comprehensive architectural & reproduction guides
│   ├── README.md                          # Documentation index and navigation map
│   ├── architecture.md                    # Core mathematical formulation, 2PC lock & causal DAG
│   ├── colab_deployment_guide.md          # Step-by-step Google Colab T4 GPU execution runbook
│   ├── qwen_xai_model.md                  # Qwen 2.5 0.5B NF4 QLoRA training dynamics & loss logs
│   ├── simulation_studio_guide.md         # Operator manual for the Streamlit Simulation Studio
│   ├── comprehensive_evaluation_report.md # Master benchmark methodology, equations, comparisons & XAI
│   ├── frontend_backend_spec.md           # Decoupled FastAPI + React.js REST/WebSocket spec
│   └── viva_defense_faq.md                # Scripted defense answers & NIST AI 600-1 profile
├── src/                                   # Production-ready Python modules (Phase 1 to 9)
│   ├── 01_environment_setup.py            # Environment initialization & GPU verification
│   ├── 02_state_interceptor.py            # Wiretap interceptor, async queue & causal rollback
│   ├── 03_charm_detector.py               # Dual-anchor CHARM detector & entailment gate
│   ├── 04_qwen_xai_translator.py          # Local Qwen plain-English translation layer
│   ├── 05_simulation_dashboard.py         # Streamlit Interactive Reliability Studio & UI
│   ├── 06_generate_charm_eval_dataset.py  # Dual-engine neural red-teaming & balanced benchmark
│   ├── 07_evaluate_charm_performance.py   # Metric computation (CDR, FPR, EPR) & bootstrap test
│   ├── 08_generate_qwen_training_dataset.py # 1,200 multi-domain ChatML dataset synthesizer
│   └── 09_train_qwen.py                   # PyTorch 4-bit NF4 QLoRA fine-tuning loop
├── results/                               # Empirical logs, training telemetry & paper tables
│   ├── optimization_research_log.md       # Master ablation log with publication LaTeX tables
│   ├── qwen_training_report.md            # Formal Qwen training loss convergence report
│   └── qwen_training_metrics.json         # Raw optimizer step metrics
└── research/                              # Foundational academic reference papers
    ├── charm.pdf                          # Original CHARM research paper
    ├── Autoagent.pdf                      # Multi-agent framework specification
    └── capstone pitch.pdf                 # Project pitch deck
```

---

## ⚡ Quick Start: Google Colab Deployment

To reproduce all experiments and launch the Interactive Simulation Studio on a free Google Colab Tesla T4 GPU:

1. **Upload `src/` and `requirements.txt`** to `/content/drive/MyDrive/Capstone_Project/`.
2. Follow the cell-by-cell runbook in [`docs/colab_deployment_guide.md`](file:///c:/harsh/CAPSTONE/docs/colab_deployment_guide.md):
   - **Cell 1**: Mount Drive & install requirements (`!pip install -r requirements.txt`).
   - **Cell 2–4**: Initialize environment and run smoke tests for `01`, `02`, and `03`.
   - **Cell 5**: Generate dataset (`08_generate_qwen_training_dataset.py`). *(Skip `09_train_qwen.py` if adapter weights are already in Drive!)*
   - **Cell 6**: Test local Qwen translator (`04_qwen_xai_translator.py`).
   - **Cell 7**: Launch Interactive Studio via Localtunnel (`05_simulation_dashboard.py`).
   - **Cell 8**: Synthesize benchmark and evaluate performance (`06_generate_charm_eval_dataset.py` & `07_evaluate_charm_performance.py`).

---

## 📖 Recommended Documentation Reading Flow

1. **Architecture & Formulation**: [`docs/architecture.md`](file:///c:/harsh/CAPSTONE/docs/architecture.md)
2. **Colab Runbook**: [`docs/colab_deployment_guide.md`](file:///c:/harsh/CAPSTONE/docs/colab_deployment_guide.md)
3. **Interactive UI Guide**: [`docs/simulation_studio_guide.md`](file:///c:/harsh/CAPSTONE/docs/simulation_studio_guide.md)
4. **Qwen Model Training**: [`docs/qwen_xai_model.md`](file:///c:/harsh/CAPSTONE/docs/qwen_xai_model.md)
5. **Master Evaluation & Benchmarks**: [`docs/comprehensive_evaluation_report.md`](file:///c:/harsh/CAPSTONE/docs/comprehensive_evaluation_report.md)
6. **Oral Defense & Faculty Review FAQ**: [`docs/viva_defense_faq.md`](file:///c:/harsh/CAPSTONE/docs/viva_defense_faq.md)
7. **Publication Optimization Log**: [`results/optimization_research_log.md`](file:///c:/harsh/CAPSTONE/results/optimization_research_log.md)
