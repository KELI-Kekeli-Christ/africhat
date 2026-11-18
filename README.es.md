<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md"><b>Español</b></a>
</p>

Chatbot ajustado con QLoRA sobre Mistral-Nemo-12B-Instruct para responder en francés de África occidental (nouchi, camfranglais, Lomé). La interfaz web es FastAPI.

```bash
pip install -r requirements.txt
python app.py
```

`app.py` arranca uvicorn. `train.py` es el entrenamiento, `chat_infer.py` la inferencia, y `prepare_dataset.py` / `build_ivoirien_dataset.py` construyen los JSONL. Los archivos de clúster `train_africhat.slurm` y `run_server.slurm` también están en el repositorio.


La documentación en inglés que ya estaba en este repositorio sigue en el archivo en inglés.
