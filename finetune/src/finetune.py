"""
Qwen2.5-0.5B-Instruct + LoRA 监督微调主脚本.

运行:
    python ./src/finetune.py

输出:
    ./saved_models/qwen2.5-0.5b-sft/   LoRA 适配器权重与分词器
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "Qwen2.5-0.5B-Instruct")
TRAIN_FILE = str(ROOT / "data" / "train.jsonl")
CHECKPOINT_DIR = str(ROOT / "saved_models" / "qwen2.5-0.5b-math-sft-checkpoint")
ADAPTER_DIR = str(ROOT / "saved_models" / "qwen2.5-0.5b-sft")
LOG_DIR = str(ROOT / "logs")

MAX_LENGTH = 1024
IGNORE_INDEX = -100


def build_label_masked_example(messages, tokenizer):
    """
    只让 assistant 段参与 loss 计算, system + user 段标签置为 -100.
    """
    input_ids = []
    labels = []
    # 系统提示与用户提示部分不算 loss
    prefix_messages = [m for m in messages if m["role"] != "assistant"]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    input_ids.extend(prefix_ids)
    labels.extend([IGNORE_INDEX] * len(prefix_ids))

    assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
    if assistant_msg is None:
        return None

    assistant_text = assistant_msg["content"] + tokenizer.eos_token
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
    input_ids.extend(assistant_ids)
    labels.extend(assistant_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
    )

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")

    def preprocess(batch):
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for messages in batch["messages"]:
            example = build_label_masked_example(messages, tokenizer)
            if example is None:
                continue
            out["input_ids"].append(example["input_ids"])
            out["labels"].append(example["labels"])
            out["attention_mask"].append(example["attention_mask"])
        return out

    processed = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        batched=True,
        desc="tokenize",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=LOG_DIR,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed,
        data_collator=collator,
    )

    print("start training")
    trainer.train()
    print("training finished")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
