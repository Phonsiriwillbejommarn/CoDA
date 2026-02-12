# 🧠 RED-CoDA: Recall-Extend Dynamics for Small Retrieval-Augmented Agents

**Enhancing CoDA (Context-Decoupled Hierarchical Agent) with RED (Recall-Extend Dynamics) to train small language models as effective retrieval-augmented reasoning agents.**

Built on [CoDA](https://arxiv.org/abs/2505.xxxxx) + [RED](https://arxiv.org/abs/2505.xxxxx) frameworks, using Gemma-2-2B with GRPO reinforcement learning.

[![Model on HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Phonsiri/CoDA-Gemma2-RED-v3)
[![W&B Dashboard](https://img.shields.io/badge/W%26B-Dashboard-blue)](https://wandb.ai)

---

## 📌 Overview

Small Language Models (SLMs) struggle with complex multi-hop QA tasks that require retrieval. Standard approaches either:
- **SFT only** → overfits to teacher patterns, poor generalization
- **RL only** → insufficient exploration space for small models
- **SFT → RL** → catastrophic forgetting of learned patterns

**RED-CoDA** solves this by **jointly training SFT + RL** with dynamic weighting controlled by two mechanisms:

| RED Component | What it does |
|---|---|
| **Part 1: Dynamic Entropy Regulation** | Monitors entropy changes to balance exploration (RL) vs exploitation (SFT) |
| **Part 2: Accuracy-Aware Policy Shift** | When model answers poorly → more SFT; when it answers well → more RL |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│              Gemma-2-2B (Single LLM)         │
│                                              │
│   🧠 Planner          ⚡ Executor            │
│   (Strategic)         (Ephemeral)            │
│   Plans long-term     Executes subtasks      │
│   Keeps context       Forgets after done     │
└──────┬───────────────────────┬───────────────┘
       │                       │
       ▼                       ▼
  search(query)          answer(result)
       │
       ▼
  ┌─────────────┐
  │ FAISS Index │ ← Wikipedia (21M docs)
  │ (CPU)       │
  └─────────────┘
```

### RED Training Loop

```
┌─────────────────────────────────────────────────────┐
│  1 Training Step                                     │
│                                                      │
│  🎲 RL Rollout (vLLM) → Generate responses           │
│  📊 Compute Reward (F1 + format + refine)            │
│  📈 GRPO Advantage (group normalization)             │
│                                                      │
│  ┌─── RED Weight Computation ───────────────────┐   │
│  │  Part 1: entropy_weight = f(δH_sft / δH_rl)  │   │
│  │  Part 2: accuracy_factor = G^(1 - 2·acc)     │   │
│  │  final_weight = entropy × accuracy            │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  🧠 Actor Update:                                    │
│     RL loss (policy gradient)                        │
│     + final_weight × SFT loss (cross-entropy)        │
│     → single optimizer step                          │
└─────────────────────────────────────────────────────┘
```

### Composite Reward

| Component | Weight | Description |
|-----------|--------|-------------|
| **Answer Quality** | `6 × F1 - 3` | F1 score vs ground truth (dominant) |
| **Format Compliance** | `0.1 × score` | Graduated XML tag scoring (0.25/tag) |
| **Refinement Quality** | `0.1 × score` | Search summarization quality |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.x compatible GPU (H100 recommended)
- ~140GB disk space (for retriever index + Wikipedia corpus)

### 1. Clone & Install

```bash
git clone https://github.com/Phonsiriwillbejommarn/CoDA.git
cd CoDA
pip install -e .
```

### 2. Login Services

```bash
wandb login            # For training dashboard
huggingface-cli login  # For checkpoint push
```

### 3. Download Data

```bash
# Download retriever index + Wikipedia corpus (~130GB)
bash preprocess/download_and_process.sh

# Process training data (NQ, HotpotQA, etc.)
bash preprocess/scripts/data_process.sh

# Generate SFT training data
python cmd/generate_sft_data.py
```

### 4. Start Training

```bash
# Terminal 1: Start Retrieval Server
bash retrieval_launch.sh

# Terminal 2: Start Training
bash cmd/train.sh
```

---

## ⚙️ Configuration

All configs in [`cmd/train.sh`](cmd/train.sh):

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_batch_size` | 32 | Prompts per training step |
| `n_agent` | 2 | Responses per prompt (GRPO group size) |
| `max_turns` | 2 | Search rounds per sample |
| `total_training_steps` | 480 | Total training steps |
| `learning_rate` | 1e-6 | Actor learning rate |

### RED Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sft.enabled` | true | Enable SFT co-training |
| `sft.loss_coef` | 0.1 | Base SFT loss coefficient |
| `red.G` | 5.0 | Upper bound for RED weight |
| `red.sft_entropy_ema_decay` | 0.99 | SFT entropy EMA smoothing |
| `red.rl_entropy_ema_decay` | 0.99 | RL entropy EMA smoothing |
| `algorithm.accuracy_aware_policy_shift` | true | Enable Part 2 |

### Ablation Configurations

```bash
# Run 1: GRPO only (baseline)
sft.enabled=false

# Run 2: GRPO + fixed SFT (no dynamic weighting)
sft.enabled=true red.G=1.0

# Run 3: GRPO + RED (full)
sft.enabled=true red.G=5.0 algorithm.accuracy_aware_policy_shift=true
```

---

## 📁 Project Structure

```
CoDA/
├── cmd/
│   ├── train.sh                 # Main training script & config
│   ├── auto_resume.py           # Auto-resume from HF Hub checkpoints
│   └── generate_sft_data.py     # Generate SFT training data
├── search_r1/
│   ├── llm_agent/
│   │   └── generation.py        # Hierarchical agent (Planner/Executor)
│   └── search/
│       └── retrieval_server.py  # FastAPI retrieval server (FAISS)
├── verl/
│   ├── trainer/
│   │   ├── main_ppo.py          # Entry point + RewardManager
│   │   └── ppo/
│   │       ├── ray_trainer.py   # Training loop + RED integration
│   │       └── core_algos.py    # GRPO + RED algorithms
│   ├── workers/
│   │   └── actor/
│   │       └── dp_actor.py      # Actor update (RL + SFT loss)
│   └── utils/
│       ├── reward_score/
│       │   └── qa_em.py         # Reward functions (F1, EM, format)
│       └── dataset/
│           ├── rl_dataset.py    # RL training dataset
│           └── sft_dataset.py   # SFT co-training dataset
├── data/                        # Training data (generated)
└── requirements.txt
```

---

## 📊 W&B Metrics

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| `critic/rewards/mean` | Overall reward per step |
| `answer_f1/mean` | Answer quality (F1 score) |
| `format_scores/mean` | XML format compliance |
| `red/entropy_weight` | RED Part 1 — entropy-based weight |
| `red/accuracy_factor` | RED Part 2 — accuracy-based multiplier |
| `red/final_weight` | Combined RED weight |
| `red/batch_accuracy` | Fraction of correct answers |

---

## 🔧 Restart After Server Reboot

```bash
cd CoDA
git pull origin main
bash preprocess/scripts/data_process.sh    # Recreate parquet files
python cmd/generate_sft_data.py            # Recreate SFT data
bash retrieval_launch.sh &                 # Start retriever
bash cmd/train.sh                          # Auto-resumes from HF Hub
```

> **Note:** If `wiki-18.jsonl` and `e5_Flat.index` are also missing, run `bash preprocess/download_and_process.sh` first.

---

## 📝 License

Apache License 2.0

## 🙏 Acknowledgments

- Based on [CoDA](https://github.com/xxx/CoDA) — Context-Decoupled Hierarchical Agent
- RED framework adapted from [RED](https://arxiv.org/abs/2505.xxxxx) — Recall-Extend Dynamics
- Built on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [verl](https://github.com/volcengine/verl)
- Model: [Google Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)