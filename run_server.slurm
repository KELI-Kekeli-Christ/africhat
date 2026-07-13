#!/bin/bash
###############################################################################
# SLURM — Serveur web AfriChat (interface chat GPT)
# Accès local via tunnel SSH (voir connect.sh)
###############################################################################
#SBATCH --job-name=africhat-web
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=heracles
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err

set -euo pipefail

PROJECT="/home/data/projets-aps/projet6/africhat"
PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PORT="${AFRICHAT_PORT:-7860}"
NODE="$(hostname -s)"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export AFRICHAT_PORT="$PORT"
export AFRICHAT_HOST="0.0.0.0"
export LD_LIBRARY_PATH="/home/data/projets-aps/projet6/env_aps/lib:/home/data/projets-aps/projet6/env_aps/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/home/data/projets-aps/projet6/env_aps/lib/libstdc++.so.6.0.34"

mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : $NODE"
echo "Port     : $PORT"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    /home/data/projets-aps/projet6/env_aps/bin/pip install --no-cache-dir fastapi "uvicorn[standard]"
fi

"$PYTHON" -m pip install --no-cache-dir -q "sniffio>=1.3.0" "anyio>=4.0.0" 2>/dev/null || true

INFO_FILE="$PROJECT/logs/server.info"
cat > "$INFO_FILE" <<EOF
job_id=${SLURM_JOB_ID:-local}
node=$NODE
port=$PORT
url_local=http://localhost:$PORT
tunnel=ssh -L ${PORT}:${NODE}:${PORT} uapv2600500@joyeux.univ-avignon.fr
EOF

echo "Infos connexion écrites dans $INFO_FILE"
cat "$INFO_FILE"

exec "$PYTHON" app.py
