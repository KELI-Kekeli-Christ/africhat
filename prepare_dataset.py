#!/usr/bin/env python3
"""Prépare datasets.json (conversations multi-tours) pour le fine-tuning SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompts import REGISTER_HINTS, SYSTEM_PROMPT


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} ligne {line_no}: JSON invalide") from exc
            rows.append(row)
    return rows


def load_many(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if path.exists():
            rows.extend(load_jsonl(path))
    return rows


def system_for_register(register: str | None) -> str:
    hint = REGISTER_HINTS.get(register or "generic", REGISTER_HINTS["generic"])
    return f"{SYSTEM_PROMPT}\n{hint}"


def merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            merged.append(msg)
            continue
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = f"{merged[-1]['content']}\n{msg['content']}"
        else:
            merged.append(dict(msg))
    return merged


def row_to_messages(row: dict) -> dict | None:
    if "discussion" in row:
        discussion = row["discussion"]
        if not discussion:
            return None
        messages = [{"role": "system", "content": system_for_register(row.get("register"))}]
        for turn in discussion:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})
        if len(messages) < 2:
            return None
        messages = merge_consecutive_roles(messages)
        # Mistral exige user/assistant alternés après le system
        conv = messages[1:]
        if not conv or conv[0]["role"] != "user":
            return None
        for i in range(1, len(conv)):
            expected = "assistant" if conv[i - 1]["role"] == "user" else "user"
            if conv[i]["role"] != expected:
                return None
        return {"messages": messages, "theme": row.get("theme"), "register": row.get("register")}

    if "instruction" in row and "response" in row:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instruction"].strip()},
                {"role": "assistant", "content": row["response"].strip()},
            ]
        }

    return None


def split_dataset(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    import random

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    val_size = max(1, int(len(rows) * val_ratio))
    val_idx = set(indices[:val_size])

    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = args.input or [Path("datasets.json")]
    raw_rows = load_many(inputs)

    chat_rows: list[dict] = []
    skipped = 0
    for row in raw_rows:
        converted = row_to_messages(row)
        if converted is None:
            skipped += 1
            continue
        chat_rows.append(converted)

    if not chat_rows:
        raise SystemExit("Aucun exemple valide trouvé.")

    train_rows, val_rows = split_dataset(chat_rows, args.val_ratio, args.seed)

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)

    registers: dict[str, int] = {}
    for row in chat_rows:
        reg = row.get("register") or "legacy"
        registers[reg] = registers.get(reg, 0) + 1

    meta = {
        "sources": [str(p) for p in inputs if p.exists()],
        "total_raw": len(raw_rows),
        "total_conversations": len(chat_rows),
        "skipped": skipped,
        "train": len(train_rows),
        "val": len(val_rows),
        "registers": registers,
        "system_prompt": SYSTEM_PROMPT,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Dataset préparé : {len(train_rows)} train, {len(val_rows)} val "
        f"({skipped} ignorés) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
