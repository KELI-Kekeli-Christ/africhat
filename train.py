#!/usr/bin/env python3
"""Fine-tuning QLoRA de Mistral-Nemo-12B-Instruct sur le dataset AfriChat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/africhat-lora"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_chat_dataset(train_file: Path, val_file: Path):
    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)
    return load_dataset("json", data_files=data_files)


def tokenize_example(example: dict, tokenizer, max_seq_length: int) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype = get_compute_dtype()
    use_bf16 = compute_dtype == torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_chat_dataset(args.train_file, args.val_file)
    tokenized = dataset.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenisation",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=args.eval_steps if "validation" in tokenized else None,
        save_total_limit=3,
        load_best_model_at_end="validation" in tokenized,
        metric_for_best_model="eval_loss" if "validation" in tokenized else None,
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "model_name": args.model_name,
        "train_samples": len(tokenized["train"]),
        "val_samples": len(tokenized["validation"]) if "validation" in tokenized else 0,
        "epochs": args.epochs,
        "global_step": train_result.global_step,
        "train_loss": train_result.training_loss,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
