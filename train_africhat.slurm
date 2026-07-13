#!/bin/bash
###############################################################################
# SLURM — Fine-tuning AfriChat (Mistral-Nemo-12B-Instruct + QLoRA)
# Serveur : joyeux.univ-avignon.fr
# Projet  : /home/data/projets-aps/projet6/africhat
###############################################################################
#SBATCH --job-name=africhat-ft
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=heracles
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

PROJECT="/home/data/projets-aps/projet6/africhat"
PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PIP="/home/data/projets-aps/projet6/env_aps/bin/pip"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PIP_CACHE_DIR="/home/data/projets-aps/projet6/tmp/pip_cache"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
mkdir -p "$PROJECT/logs" "$PROJECT/data" "$PROJECT/checkpoints"
cd "$PROJECT"

log_msg "========================================="
log_msg "Job ID       : ${SLURM_JOB_ID:-local}"
log_msg "Node         : ${SLURM_NODELIST:-$(hostname)}"
log_msg "GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log_msg "Python       : $($PYTHON --version 2>&1)"
log_msg "Project      : $PROJECT"
log_msg "Time limit   : 7 days"
log_msg "========================================="

if [[ -f "/home/data/projets-aps/projet6/start_conda_env.sh" ]]; then
    export LD_LIBRARY_PATH="/home/data/projets-aps/projet6/env_aps/lib:/home/data/projets-aps/projet6/env_aps/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
fi

log_msg "Vérification des dépendances Python..."
if ! "$PYTHON" -c "import peft, bitsandbytes" 2>/dev/null; then
    log_msg "Installation des dépendances AfriChat (sans rebuild pyarrow)..."
    "$PIP" install --no-cache-dir --no-deps peft bitsandbytes
else
    log_msg "Dépendances déjà installées."
fi

"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

log_msg "Préparation du dataset (conversations + salutations)..."
"$PYTHON" prepare_dataset.py \
    --input datasets_greetings.json \
    --input datasets_greetings.json \
    --input datasets_greetings.json \
    --input datasets.json \
    --output-dir data \
    --val-ratio 0.1 \
    --seed 42

log_msg "Lancement du fine-tuning Mistral-Nemo-12B (QLoRA)..."
"$PYTHON" train.py \
    --model-name mistralai/Mistral-Nemo-Instruct-2407 \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --output-dir checkpoints/africhat-lora \
    --epochs 4 \
    --batch-size 1 \
    --grad-accum 8 \
    --learning-rate 2e-4 \
    --max-seq-length 1536 \
    --save-steps 100 \
    --eval-steps 100 \
    2>&1 | tee "logs/train_${SLURM_JOB_ID:-local}.log"

log_msg "Fine-tuning terminé."
if [[ -f "checkpoints/africhat-lora/adapter_config.json" ]]; then
    log_msg "Adaptateur LoRA sauvegardé : checkpoints/africhat-lora"
    ls -lah checkpoints/africhat-lora/
else
    log_msg "ATTENTION : adapter_config.json introuvable — vérifier les logs."
    exit 1
fi

log_msg "Test rapide d'inférence..."
"$PYTHON" chat_infer.py \
    --adapter-path checkpoints/africhat-lora \
    --question "Tonton eh, rester motivé ça me tue, aide-moi un peu." \
    2>&1 | tee -a "logs/train_${SLURM_JOB_ID:-local}.log"

log_msg "Job AfriChat terminé avec succès."
