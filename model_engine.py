"""Chargement et génération avec le modèle AfriChat (base + LoRA)."""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Iterator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from prompts import (
    ADVICE_PROMPT,
    CASUAL_PROMPT,
    CASUAL_SCRIPT_PHRASES,
    FAKE_USER_MARKERS,
    PROBLEM_KEYWORDS,
    SYSTEM_PROMPT,
    TAUNT_OPENERS,
    USER_SENTENCE_PATTERNS,
)

DEFAULT_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
DEFAULT_ADAPTER = "checkpoints/africhat-lora"


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def is_casual_message(text: str) -> bool:
    lower = text.strip().lower()
    if not lower:
        return True
    if any(kw in lower for kw in PROBLEM_KEYWORDS):
        return False
    if len(lower) <= 35:
        return True
    return False


def build_system_prompt(messages: list[dict[str, str]]) -> str:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    extra = CASUAL_PROMPT if is_casual_message(last_user) else ADVICE_PROMPT
    return f"{SYSTEM_PROMPT}\n{extra}"


def looks_like_advice_script(text: str) -> bool:
    lower = text.lower()
    if any(text.strip().startswith(t) for t in TAUNT_OPENERS):
        return True
    return any(phrase in lower for phrase in CASUAL_SCRIPT_PHRASES)


def truncate_scripted_output(text: str, casual: bool) -> str:
    text = re.sub(r"([.!?])([A-Za-zÀ-ÿ])", r"\1 \2", text).strip()
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    limit = 2 if casual else 3

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if any(re.match(pat, s) for pat in USER_SENTENCE_PATTERNS):
            continue
        if casual and (any(s.startswith(t) for t in TAUNT_OPENERS) or looks_like_advice_script(s)):
            continue
        kept.append(s)
        if len(kept) >= limit:
            break

    if kept:
        return " ".join(kept)

    if casual:
        return ""

    cut = len(text)
    for marker in FAKE_USER_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def generation_params(casual: bool, attempt: int = 0) -> dict[str, float | int]:
    if casual:
        return {
            "max_new_tokens": 64,
            "temperature": min(0.95, 0.88 + attempt * 0.04),
            "top_p": 0.93,
            "top_k": 50,
            "repetition_penalty": 1.2,
        }
    return {
        "max_new_tokens": 96,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.15,
    }


class AfriChatEngine:
    def __init__(
        self,
        base_model: str = DEFAULT_MODEL,
        adapter_path: str = DEFAULT_ADAPTER,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.tokenizer: AutoTokenizer | None = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        compute_dtype = get_compute_dtype()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        self.model = PeftModel.from_pretrained(model, self.adapter_path)
        self.model.eval()
        self._loaded = True

    @contextmanager
    def _use_adapter(self, enabled: bool):
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé.")
        if enabled:
            self.model.enable_adapter_layers()
        else:
            self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()

    def _generate_once(
        self,
        prompt: str,
        params: dict[str, float | int],
        max_new_tokens: int | None,
        temperature: float,
        top_p: float,
    ) -> str:
        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or int(params["max_new_tokens"]),
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(params["top_k"]),
            "repetition_penalty": params["repetition_penalty"],
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            if chunk:
                accumulated += chunk
        thread.join()
        return accumulated

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Iterator[str]:
        if not self._loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé.")

        last_user = messages[-1]["content"] if messages else ""
        casual = is_casual_message(last_user)
        chat_messages = [{"role": "system", "content": build_system_prompt(messages)}, *messages]
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        attempts = 3 if casual else 1
        result = ""

        with self._use_adapter(not casual):
            for attempt in range(attempts):
                params = generation_params(casual, attempt)
                effective_temp = params["temperature"] if casual else (
                    temperature if temperature is not None else params["temperature"]
                )
                effective_top_p = params["top_p"] if casual else (
                    top_p if top_p is not None else params["top_p"]
                )
                raw = self._generate_once(
                    prompt,
                    params,
                    max_new_tokens,
                    effective_temp,
                    effective_top_p,
                )
                result = truncate_scripted_output(raw, casual)
                if result and (not casual or not looks_like_advice_script(result)):
                    break

        if result:
            yield result

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        return "".join(
            self.generate_stream(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
