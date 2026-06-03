# Qwen2.5-0.5B Math LoRA Fine-Tuning

A LoRA-based supervised fine-tuning (SFT) pipeline on
`Qwen/Qwen2.5-0.5B-Instruct` and the MathQA dataset, covering data
preparation, PEFT/LoRA training, evaluation (ROUGE / BLEU / choice accuracy)
and an interactive CLI for inference.

The goal of the project is to **walk through the full LLM fine-tuning loop**
end-to-end on a consumer GPU — not to chase final accuracy. At 0.5B
parameters and 500 training samples, the model is not large enough to
acquire genuine multi-step math reasoning, and the report says so plainly.

This project was developed during my data science internship at BONC
(东方国信), April – June 2026. The full methodology, hyperparameters,
loss curve, evaluation numbers and sample-level comparisons are documented
in `finetune/TRAINING_REPORT.md`.

---

## Highlights

- **PEFT / LoRA, not full fine-tuning**: rank=16, alpha=32, dropout=0.1.
  Target modules cover both attention (q/k/v/o) and FFN (gate/up/down).
  8.80 M trainable parameters, 1.75 % of total.
- **Label masking is enforced**: `system` and `user` segments are set to
  `-100` so loss is only computed over the `assistant` segment. This is the
  detail that most "quick LoRA scripts" online get wrong.
- **Base vs. SFT comparison, on the same eval harness**:
  ROUGE-L 0.148 → 0.365, BLEU 0.020 → 0.204, Choice Accuracy 0/100 → 26/100.
  The base model never produced a valid `<correct>x</correct>` tag once,
  which is what the SFT actually fixes.
- **Honest framing of what was learned**: the report states explicitly that
  the SFT taught the model the **output format**, not math reasoning. The
  26 % choice accuracy is only marginally above the 20 % random baseline
  and the report does not oversell it.

## Quick Start

```powershell
# 1. Install a CUDA-matched PyTorch (adjust the wheel channel for your GPU)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Install the remaining dependencies
python -m pip install -r finetune/requirements.txt

# 3. Pull the base model into ./finetune/Qwen2.5-0.5B-Instruct/
#    (HuggingFace direct, or hf-mirror for users inside mainland China)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./finetune/Qwen2.5-0.5B-Instruct')"

# 4. Prepare data (raw MathQA JSON → Qwen chat-format jsonl)
python ./finetune/scripts/prepare_data.py

# 5. Train (LoRA SFT, ~95 s on a single RTX 5090 Laptop)
python ./finetune/src/finetune.py

# 6. Evaluate (LoRA vs. base on the same 100-sample test set)
python ./finetune/src/evaluator.py `
    --model_path ./finetune/Qwen2.5-0.5B-Instruct `
    --adapter_path ./finetune/saved_models/qwen2.5-0.5b-sft `
    --eval_dataset_path ./finetune/data/test.jsonl

# 7. Chat with the fine-tuned model
python ./finetune/src/inference.py `
    --model_path ./finetune/Qwen2.5-0.5B-Instruct `
    --adapter_path ./finetune/saved_models/qwen2.5-0.5b-sft
```

The trained LoRA adapter is already checked in under
`finetune/saved_models/qwen2.5-0.5b-sft/`, so steps 4 and 5 can be skipped
if you only want to reproduce evaluation or run inference.

## Project Layout

| Path | Purpose |
| --- | --- |
| `finetune/scripts/prepare_data.py` | MathQA raw JSON → Qwen chat-format jsonl (train 500 / test 100, seed 42) |
| `finetune/data/dataset/simple/` | Raw MathQA train / dev / test splits |
| `finetune/data/train.jsonl` · `test.jsonl` | Processed splits actually consumed by training and evaluation |
| `finetune/src/finetune.py` | LoRA SFT entrypoint (PEFT + HuggingFace `Trainer`) |
| `finetune/src/evaluator.py` | ROUGE / BLEU / choice-accuracy evaluation, works on base or LoRA adapter |
| `finetune/src/inference.py` | Multi-turn CLI chat; `/exit`, `/reset` |
| `finetune/saved_models/qwen2.5-0.5b-sft/` | Final LoRA adapter, tokenizer and `eval_report.json` |
| `finetune/TRAINING_REPORT.md` | Full training report — setup, loss curve, metrics, sample comparison, limitations |
| `finetune/README.md` | Detailed Chinese developer notes (mirrors this README at a finer level) |

The base model weights (`finetune/Qwen2.5-0.5B-Instruct/`, ~940 MB) and
intermediate training checkpoints with optimizer state
(`finetune/saved_models/qwen2.5-0.5b-math-sft-checkpoint/`, ~220 MB) are
deliberately git-ignored. The base model is re-downloaded from HuggingFace
in step 3 of Quick Start; the intermediate checkpoints are not needed to
reproduce the result.

## Limitations

This is an internship-scale fine-tuning exercise, not a math-reasoning
model.

- **0.5B parameters, 500 training samples, 3 epochs.** Far below the scale
  at which next-token SFT teaches genuine multi-step reasoning.
- **What the model actually learned is the output format**, not the math.
  The SFT model reliably emits `<rationale>...</rationale><correct>x</correct>`
  inside the option set `a-e`, but the algebraic steps inside `<rationale>`
  are still frequently wrong.
- **Choice accuracy 26 %** is only marginally above the 20 % five-option
  random baseline. The improvement over the base model is real, but it is
  largely an artefact of format compliance rather than reasoning gain.
- **One known prompt-format quirk**: the `user` segment uses
  `<operations>...</options>` — an asymmetric tag pair that comes from the
  internship task spec. It is preserved here for spec alignment, not because
  it is good practice. `TRAINING_REPORT.md` flags this as the first item on
  the improvement list.

`TRAINING_REPORT.md` discusses follow-ups in order of effort-to-value:
larger training subset, larger base model (1.5B / 3B), step-wise eval
during training, and the natural next step — verifiable-reward RL on the
`<correct>x</correct>` tag (GRPO / DPO style), which is what would actually
move the needle on reasoning rather than format.

## Acknowledgements and Development Notes

I designed and led this project during my BONC internship, including the
choice of LoRA over full fine-tuning, the attention + FFN target-module
set, the strict label masking on system / user segments, the three-metric
eval harness (ROUGE / BLEU / choice accuracy via regex on `<correct>`),
and the honest framing that the SFT only teaches output format rather
than reasoning. AI coding assistants were used during implementation;
the design, evaluation and analysis work was mine.

`finetune/TRAINING_REPORT.md` is the most honest description of what worked
and what did not in this experiment, including the gap between the
ROUGE / BLEU gains and the actual reasoning quality. If you are reading
this as part of an application review, that report is the document I would
point you to first.

**A note on the Git history**: during the internship the project was
developed locally and originally pushed in a few large commits authored
under a generic placeholder identity. After the internship I grouped the
files into a small number of cleanup commits under my own name to make the
repository easier to read. The commits therefore share a recent date, but
the code they record reflects the work done between April and June 2026.

---

Author: Hongtao Zhu · hzhu325@wisc.edu · github.com/hzhu325
