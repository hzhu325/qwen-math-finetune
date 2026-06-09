"""
评估 LoRA 微调后的模型在测试集上的表现.

用法:
    python ./src/evaluator.py \
        --model_path ./Qwen2.5-0.5B-Instruct \
        --adapter_path ./saved_models/qwen2.5-0.5b-sft/ \
        --eval_dataset_path ./data/test.jsonl

输出三类指标:
    ROUGE 与 BLEU       生成文本与参考答案的整体相似度
    Choice Accuracy     从 <correct>x</correct> 抽取的选项是否命中标准答案
"""

import argparse
import re
import json
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CORRECT_PATTERN = re.compile(r"<correct>\s*([a-eA-E])\s*</correct>")


def extract_choice(text: str) -> str | None:
    m = CORRECT_PATTERN.search(text)
    return m.group(1).lower() if m else None


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if args.adapter_path:
        print(f"loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()

    eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    predictions, references = [], []
    hit, total = 0, 0
    samples_log = []

    for item in tqdm(eval_dataset, desc="generate"):
        messages = item["messages"]
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        ref_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if not user_msg or not ref_msg:
            continue

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        predictions.append(response)
        references.append(ref_msg["content"])

        pred_choice = extract_choice(response)
        gold_choice = extract_choice(ref_msg["content"])
        if gold_choice is not None:
            total += 1
            if pred_choice == gold_choice:
                hit += 1

        if len(samples_log) < 5:
            samples_log.append({
                "prompt": user_msg["content"],
                "reference": ref_msg["content"],
                "prediction": response,
                "pred_choice": pred_choice,
                "gold_choice": gold_choice,
            })

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    accuracy = hit / total if total else 0.0

    print("\n=== eval results ===")
    print("ROUGE:")
    for k, v in rouge_results.items():
        print(f"  {k}: {v:.4f}")
    print(f"BLEU : {bleu_results['bleu']:.4f}")
    print(f"Choice Accuracy: {accuracy:.4f}  ({hit}/{total})")

    report = {
        "rouge": {k: float(v) for k, v in rouge_results.items()},
        "bleu": float(bleu_results["bleu"]),
        "choice_accuracy": accuracy,
        "choice_hit": hit,
        "choice_total": total,
        "samples": samples_log,
    }
    report_path = Path(args.adapter_path or args.model_path) / "eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
