# 🧠 CoDA: Context-Decoupled Hierarchical Agent

**CoDA-Gemma2-RED** — A single Gemma-2-2B model trained as a hierarchical RAG agent using GRPO reinforcement learning.

[![Model on HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Phonsiri/CoDA-Gemma2-RED-v1)
[![W&B Dashboard](https://img.shields.io/badge/W%26B-Dashboard-blue)](https://wandb.ai)

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
  search(query)          finish(answer)
       │
       ▼
  ┌─────────────┐
  │ FAISS Index │ ← Wikipedia (21M docs)
  │ (CPU)       │
  └─────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Context-Decoupled** | Separates Planner (strategic) from Executor (ephemeral) contexts to prevent context explosion |
| **PECO Training** | Planner-Executor Co-Optimization — trains both roles simultaneously with RL |
| **GRPO** | Group Relative Policy Optimization for reward-based learning |
| **RED** | Recall-Extend Dynamics for balancing SFT/RL training |

### Composite Reward (3 components)

1. **Correctness** — F1 score vs ground truth answer (primary)
2. **Format Compliance** — Correct XML tag usage (+0.1)
3. **Refinement Quality** — Effective search summarization (+0.1)

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
wandb login          # For training dashboard
huggingface-cli login  # For checkpoint push
```

### 3. Download Data

```bash
# Download retriever index + Wikipedia corpus (~130GB)
bash preprocess/download_and_process.sh

# Process training data (NQ, HotpotQA, TriviaQA, PopQA, Musique, etc.)
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

## ⚙️ Training Configuration

All configs are in [`cmd/train.sh`](cmd/train.sh):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_batch_size` | 32 | Prompts per training step |
| `n_agent` | 2 | Responses per prompt (GRPO group size) |
| `max_turns` | 2 | Search rounds per sample |
| `save_freq` | 5 | Checkpoint push frequency (steps) |
| `total_training_steps` | 480 | Total training steps |
| `max_prompt_length` | 3072 | Max prompt token length |
| `max_response_length` | 1024 | Max response token length |
| `learning_rate` | 1e-6 | Actor learning rate |

### Speed Tuning
- **Faster:** Reduce `max_turns`, `n_agent`, `train_batch_size`
- **Better learning:** Increase `max_turns` (slower per step)

### Checkpoint Management
- Saves every `save_freq` steps to local + [HF Hub](https://huggingface.co/Phonsiri/CoDA-Gemma2-RED-v1)
- **Keeps only 2 latest checkpoints** (auto-deletes old ones)
- Auto-resumes from the latest checkpoint on restart

---

## 📁 Project Structure

```
CoDA/
├── cmd/
│   ├── train.sh                 # Main training script & config
│   ├── auto_resume.py           # Auto-resume from HF Hub checkpoints
│   └── generate_sft_data.py     # Generate SFT training data
├── preprocess/
│   ├── download_and_process.sh  # Download retriever data
│   └── scripts/
│       └── data_process.sh      # Process QA datasets
├── search_r1/
│   ├── llm_agent/
│   │   └── generation.py        # Agent generation logic (Planner/Executor)
│   └── search/
│       └── retrieval_server.py  # FastAPI retrieval server (FAISS)
├── verl/
│   ├── trainer/
│   │   ├── main_ppo.py          # Training entry point
│   │   ├── config/
│   │   │   └── grpo_trainer.yaml # Default config
│   │   └── ppo/
│   │       ├── ray_trainer.py   # Main training loop + checkpointing
│   │       └── core_algos.py    # GRPO algorithm implementation
│   ├── workers/
│   │   ├── actor/
│   │   │   └── dp_actor.py      # Actor policy update
│   │   ├── fsdp_workers.py      # FSDP distributed workers
│   │   └── rollout/
│   │       └── vllm_rollout/    # vLLM inference engine
│   └── utils/
│       ├── reward_score/
│       │   └── qa_em.py         # Reward functions (F1, EM)
│       ├── dataset/
│       │   ├── rl_dataset.py    # RL training dataset
│       │   └── sft_dataset.py   # SFT co-training dataset
│       └── padding_utils.py     # SDPA padding utilities
├── data/                        # Training data (generated, not in git)
├── retrieval_launch.sh          # Launch retrieval server
└── requirements.txt             # Python dependencies
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Step Time | ~4-5 min (H100 1x) |
| Total Steps | 480 |
| Estimated Duration | ~35 hours |
| Samples per Step | 64 (32 prompts × 2 responses) |
| Model Size | 2B parameters |

---

## 🔧 Restart After Server Reboot

Data files are ephemeral on cloud servers. After restart:

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

- Based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework
- Uses [verl](https://github.com/volcengine/verl) for RL training
- Model: [Google Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)