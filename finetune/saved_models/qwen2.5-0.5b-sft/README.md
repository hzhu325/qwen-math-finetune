---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - sft
  - math-qa
  - qwen2.5
---

# Qwen2.5-0.5B Math LoRA Adapter

LoRA adapter fine-tuned on a 500-sample subset of MathQA, using
`Qwen/Qwen2.5-0.5B-Instruct` as the base model. This is the artefact
produced by `finetune/src/finetune.py`.

## Adapter Details

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** PEFT / LoRA
- **Rank:** 16 · **alpha:** 32 · **dropout:** 0.1
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`
- **Trainable parameters:** 8.80 M (1.75 % of base)
- **Precision:** bf16
- **Training samples:** 500 (MathQA `simple/train.json`, seed 42)
- **Epochs:** 3

## Evaluation

On a 100-sample held-out test set (same seed):

| Metric | Base | + this adapter |
| --- | --- | --- |
| ROUGE-L | 0.148 | 0.365 |
| BLEU | 0.020 | 0.204 |
| Choice Accuracy | 0 / 100 | 26 / 100 |

The base model never emitted a valid `<correct>x</correct>` tag in 100
samples, so the choice-accuracy gain is **largely a format-compliance
effect**, not a math-reasoning gain. 26 / 100 is only marginally above
the 20 % random baseline for a 5-option multiple-choice task.

Full training and evaluation report:
[`../../TRAINING_REPORT.md`](../../TRAINING_REPORT.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "./").merge_and_unload()
```

Or via the project CLI:

```bash
python finetune/src/inference.py \
    --model_path ./finetune/Qwen2.5-0.5B-Instruct \
    --adapter_path ./finetune/saved_models/qwen2.5-0.5b-sft
```

## Limitations

This is an internship-scale fine-tuning exercise, not a math-reasoning
model. The SFT taught the model the **output format**, not the math.
For genuine reasoning gains on this task, the natural next step is
verifiable-reward RL (GRPO / DPO) on the `<correct>x</correct>` tag.

## Framework Versions

- PEFT 0.19.1
