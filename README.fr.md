<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md"><b>Français</b></a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

Chatbot affiné en QLoRA sur Mistral-Nemo-12B-Instruct pour répondre en français d’Afrique de l’Ouest (nouchi, camfranglais, Lomé). L’interface web est en FastAPI.

```bash
pip install -r requirements.txt
python app.py
```

`app.py` démarre uvicorn. `train.py` est l’entraînement, `chat_infer.py` l’inférence, `prepare_dataset.py` et `build_ivoirien_dataset.py` construisent les JSONL. Les fichiers cluster `train_africhat.slurm` et `run_server.slurm` sont aussi dans le dépôt.


La documentation anglaise déjà présente dans ce dépôt reste dans le fichier anglais.
