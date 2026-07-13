#!/bin/bash
# Déploie le projet AfriChat sur le serveur joyeux.univ-avignon.fr
set -euo pipefail

REMOTE_USER="uapv2600500"
REMOTE_HOST="joyeux.univ-avignon.fr"
REMOTE_DIR="/home/data/projets-aps/projet6/africhat"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Création du dossier distant : ${REMOTE_DIR}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/logs ${REMOTE_DIR}/data ${REMOTE_DIR}/checkpoints"

echo "==> Synchronisation des fichiers..."
rsync -avz --progress \
    "${LOCAL_DIR}/datasets.json" \
    "${LOCAL_DIR}/datasets_greetings.json" \
    "${LOCAL_DIR}/prepare_dataset.py" \
    "${LOCAL_DIR}/train.py" \
    "${LOCAL_DIR}/chat_infer.py" \
    "${LOCAL_DIR}/model_engine.py" \
    "${LOCAL_DIR}/build_ivoirien_dataset.py" \
    "${LOCAL_DIR}/prompts.py" \
    "${LOCAL_DIR}/app.py" \
    "${LOCAL_DIR}/requirements.txt" \
    "${LOCAL_DIR}/train_africhat.slurm" \
    "${LOCAL_DIR}/run_server.slurm" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

rsync -avz --progress \
    "${LOCAL_DIR}/static/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/static/"

echo "==> Permissions d'exécution..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "chmod +x ${REMOTE_DIR}/train_africhat.slurm ${REMOTE_DIR}/run_server.slurm"

echo ""
echo "Déploiement terminé."
echo ""
echo "Pour lancer l'entraînement sur le serveur :"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  sbatch train_africhat.slurm"
echo ""
echo "Suivre le job :"
echo "  squeue -u ${REMOTE_USER}"
echo "  tail -f logs/slurm_<JOB_ID>.out"
