#!/bin/bash
# Ouvre un tunnel SSH pour accéder à AfriChat depuis ton PC local.
# Usage : ./connect.sh [port]

set -euo pipefail

REMOTE_USER="uapv2600500"
REMOTE_HOST="joyeux.univ-avignon.fr"
PORT="${1:-7860}"
PROJECT="/home/data/projets-aps/projet6/africhat"

echo "==> Recherche du serveur AfriChat sur le cluster..."
INFO=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "cat ${PROJECT}/logs/server.info 2>/dev/null" || true)

if [[ -z "$INFO" ]]; then
    echo "Aucun serveur actif. Lance d'abord sur le cluster :"
    echo "  cd ${PROJECT} && sbatch run_server.slurm"
    exit 1
fi

NODE=$(echo "$INFO" | grep '^node=' | cut -d= -f2-)
JOB=$(echo "$INFO" | grep '^job_id=' | cut -d= -f2-)
FILE_PORT=$(echo "$INFO" | grep '^port=' | cut -d= -f2-)
NODE="${NODE:-heracles}"
JOB="${JOB:-?}"
PORT="${FILE_PORT:-$PORT}"

echo "Job      : $JOB"
echo "Node GPU : $NODE"
echo "Port     : $PORT"
echo ""
echo "==> Tunnel SSH (laisse ce terminal ouvert)..."
echo "    Puis ouvre dans ton navigateur : http://localhost:${PORT}"
echo ""

exec ssh -N -L "${PORT}:${NODE}:${PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
