"""
将 MathQA 原始数据转为 Qwen2.5 chat 模板所需的 jsonl 格式。

输入:
    data/dataset/simple/train.json  (JSON 数组, 每条含 Problem/Rationale/options/correct 等字段)
    data/dataset/simple/test.json

输出:
    data/train.jsonl
    data/test.jsonl

每行结构:
    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<problem>...</problem><operations>...</options>"},
        {"role": "assistant", "content": "<rationale>...</rationale><correct>x</correct>"}
    ]}
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = "You are a helpful assistant."

# 本地算力有限, 按文档要求限制样本量
TRAIN_LIMIT = 500
TEST_LIMIT = 100
RANDOM_SEED = 42


def load_raw(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    # 原始 MathQA simple 数据是一整个 JSON 数组, 不是 jsonl
    if text.startswith("["):
        return json.loads(text)
    # 容错处理 jsonl
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def to_chat_record(item: dict) -> dict | None:
    problem = (item.get("Problem") or "").strip()
    rationale = (item.get("Rationale") or "").strip()
    options = (item.get("options") or "").strip()
    correct = (item.get("correct") or "").strip()

    if not problem or not options or not correct:
        return None

    user_content = f"<problem>{problem}</problem><operations>{options}</options>"
    assistant_content = f"<rationale>{rationale}</rationale><correct>{correct}</correct>"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert(input_file: Path, output_file: Path, limit: int):
    raw = load_raw(input_file)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(raw)

    written = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fout:
        for item in raw:
            if written >= limit:
                break
            record = to_chat_record(item)
            if record is None:
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"{input_file} -> {output_file}: {written} records")


def main():
    base = Path(__file__).resolve().parent.parent
    convert(base / "data" / "dataset" / "simple" / "train.json",
            base / "data" / "train.jsonl",
            TRAIN_LIMIT)
    convert(base / "data" / "dataset" / "simple" / "test.json",
            base / "data" / "test.jsonl",
            TEST_LIMIT)


if __name__ == "__main__":
    main()
