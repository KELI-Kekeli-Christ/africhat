#!/usr/bin/env python3
"""Interface web AfriChat — API + chat style GPT."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from model_engine import AfriChatEngine

STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("AFRICHAT_PORT", "7860"))
HOST = os.environ.get("AFRICHAT_HOST", "0.0.0.0")

engine = AfriChatEngine(
    base_model=os.environ.get("AFRICHAT_BASE_MODEL", "mistralai/Mistral-Nemo-Instruct-2407"),
    adapter_path=os.environ.get("AFRICHAT_ADAPTER", "checkpoints/africhat-lora"),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Chargement du modèle AfriChat...")
    engine.load()
    print("Modèle prêt.")
    yield


app = FastAPI(title="AfriChat", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=96, ge=32, le=200)
    temperature: float = Field(default=0.82, ge=0.1, le=1.5)
    top_p: float = Field(default=0.92, ge=0.1, le=1.0)
    stream: bool = True


@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Interface introuvable.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if engine._loaded else "loading",
        "model": engine.base_model,
        "adapter": engine.adapter_path,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not engine._loaded:
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement.")

    messages = [{"role": m.role, "content": m.content.strip()} for m in request.messages]
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Le dernier message doit venir de l'utilisateur.")

    if request.stream:
        def event_stream():
            try:
                for chunk in engine.generate_stream(
                    messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ):
                    payload = json.dumps({"content": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    reply = engine.generate(
        messages,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return {"role": "assistant", "content": reply}


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
