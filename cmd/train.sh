#!/bin/bash
# =============================================================
# CoDA Training Script (Gemma-2-2b + RED Logic) สำหรับ H100
# =============================================================

# 1. การตั้งค่า Environment สำหรับ H100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

# 2. ข้อมูลโมเดลและไดเรกทอรี
num_gpus=1
export DATA_DIR=data
export BASE_MODEL='google/gemma-2-2b'
export EXPERIMENT_NAME="CoDA-Gemma2-RED-v1"

# 3. API Keys
MY_WANDB_KEY="API"
MY_HF_TOKEN="API"

# 4. Login W&B
export WANDB_API_KEY=$MY_WANDB_KEY
export WANDB_PROJECT="CoDA_RED_Project"
export WANDB_MODE="disabled" # เปลี่ยนเป็น "online" หากต้องการดูผ่านเว็บ

python3 -c "
try:
    key = '${MY_WANDB_KEY}'
    if key:
        import wandb
        wandb.login(key=key)
        print('W&B logged in successfully!')
    else:
        print('W&B Key is empty. Skipping W&B login.')
except Exception as e:
    print(f'Failed to login to W&B: {e}')
"

# 5. Login Hugging Face
export HF_TOKEN=$MY_HF_TOKEN

python3 -c "
try:
    token = '${MY_HF_TOKEN}'
    if token:
        from huggingface_hub import login
        login(token=token)
        print('Hugging Face logged in successfully!')
    else:
        print('HF Token is empty. Skipping HF login.')
except Exception as e:
    print(f'Failed to login to Hugging Face: {e}')
"

# สร้างโฟลเดอร์สำหรับเก็บ Log
mkdir -p log/

# 7. Auto-resume from HF Hub checkpoint (if available)
HF_REPO="Phonsiri/$EXPERIMENT_NAME"
RESUME_STEP=0
MODEL_PATH=$BASE_MODEL

echo "Checking for existing checkpoints on HF Hub..."
RESUME_OUTPUT=$(python3 cmd/auto_resume.py --repo $HF_REPO 2>&1)
RESUME_EXIT=$?

if [ $RESUME_EXIT -eq 0 ]; then
    RESUME_PATH=$(echo "$RESUME_OUTPUT" | grep "RESUME_PATH=" | cut -d'=' -f2)
    RESUME_STEP=$(echo "$RESUME_OUTPUT" | grep "RESUME_STEP=" | cut -d'=' -f2)
    if [ -n "$RESUME_PATH" ] && [ -d "$RESUME_PATH" ]; then
        MODEL_PATH=$RESUME_PATH
        echo "✅ Resuming from checkpoint: step $RESUME_STEP"
        echo "   Model path: $MODEL_PATH"
    else
        echo "⚠️ No valid checkpoint found, starting from base model."
    fi
else
    echo "⚠️ No checkpoint found on HF Hub, training from scratch."
fi

# 8. คำสั่งรันการฝึก (Consolidated Command) — CoDA + RED Logic
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    reward_model.reward_style="F1" \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/valid_500.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=3072 \
    max_turns=6 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    algorithm.red_enabled=true \
    algorithm.entropy_weight_regulation=true \
    algorithm.accuracy_aware_policy_shift=true \
    trainer.total_training_steps=480 \
    actor_rollout_ref.actor.refine_score=0.1 \
    actor_rollout_ref.actor.format_score=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['tensorboard'] \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.resume_step=$RESUME_STEP \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    sft.enabled=true \
    sft.train_files=$DATA_DIR/sft_train.parquet \
    sft.loss_coef=0.1 \
    sft.micro_batch_size=4 \
    sft.max_length=4096 \
    red.G=5.0 \
    red.sft_entropy_ema_decay=0.99 \
    red.rl_entropy_ema_decay=0.99 \
    2>&1 | tee log/$EXPERIMENT_NAME.log