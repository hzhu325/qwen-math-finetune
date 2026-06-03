# Qwen2.5-0.5B + LoRA 微调实验报告

朱宏涛 · 2026-05-28

## 1. 实验设置

### 硬件与软件

- Windows 11 + Python 3.13
- NVIDIA RTX 5090 Laptop GPU，24 GB VRAM
- CUDA 13.2 驱动 / PyTorch 2.11+cu128
- transformers 5.9 / peft 0.19 / datasets 4.8

### 数据

数据来自 [MathQA](https://math-qa.github.io/) 完整训练集，
按文档要求随机抽取 **训练 500 条 / 测试 100 条**（随机种子 42）。
原始字段保留 `Problem / Rationale / options / correct`，按 Qwen2.5 chat 模板封装为三段 `system / user / assistant`。

每条样本的 user 段格式：

```
<problem>...</problem><operations>...</options>
```

assistant 段格式：

```
<rationale>...</rationale><correct>x</correct>
```

### 模型与训练超参

- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`（约 5 亿参数）
- LoRA rank=16, alpha=32, dropout=0.1
- target_modules：注意力 4 个（q/k/v/o）+ FFN 3 个（gate/up/down）
- 可训练参数 8.80 M，占总参数 1.75 %
- batch size 2，梯度累积 2，等效 batch 4
- 学习率 2e-5，cosine 调度，warmup 比例 0.03
- 3 epoch，约 375 个 optimizer step
- 精度 bf16
- label mask：system + user 段标签置 -100，loss 只在 assistant 段计算

## 2. 训练过程

整体训练耗时 95.4 秒，平均 3.93 step/s。loss 曲线：

| step (≈) | epoch | training loss |
| - | - | - |
| 20  | 0.16 | 1.857 |
| 40  | 0.32 | 1.301 |
| 60  | 0.48 | 0.992 |
| 80  | 0.64 | 0.932 |
| 100 | 0.80 | 0.883 |
| 140 | 1.12 | 0.943 |
| 180 | 1.44 | 0.836 |
| 220 | 1.76 | 0.851 |
| 260 | 2.08 | 0.808 |
| 300 | 2.40 | 0.850 |
| 340 | 2.72 | 0.814 |
| 375 | 3.00 | 0.798 |

最终 `train_loss = 0.946`。
loss 在第一个 epoch 内快速下降，第二、三 epoch 趋于稳定，没有明显过拟合迹象。

## 3. 评估结果

在测试集（100 条）上分别评估 base 模型与 LoRA 微调后的模型。

| 指标 | Base | SFT | 变化 |
| - | - | - | - |
| ROUGE-1   | 0.1977 | 0.4133 | +109 % |
| ROUGE-2   | 0.0616 | 0.1823 | +196 % |
| ROUGE-L   | 0.1480 | 0.3650 | +147 % |
| BLEU      | 0.0200 | 0.2036 | +918 % |
| Choice Accuracy | 0 / 100 | 26 / 100 | 0 → 26 % |

Choice Accuracy 的口径是从模型输出中正则匹配 `<correct>x</correct>` 并与标准选项比对。
base 模型在 100 条样本中一次都没产生符合格式的 `<correct>` 标签，所以命中数为 0。
微调后命中 26%，略高于 5 选 1 随机基线（20%）。

## 4. 样本对照

### 样本 A：鸭与牛的腿数问题

prompt：

> in a group of ducks and cows, the total number of legs are 26 more than twice the no. of heads. find the total no. of buffaloes.
> options: a) 12, b) 14, c) 13, d) 11, e) none

参考答案：`<correct>c</correct>`（推理：`4x + 2y = 2(x+y) + 26 → 2x = 26 → x = 13`）

**base 模型输出**（节选）：

> Let's denote D as the number of ducks, C as the number of cows.
> Each duck has 2 legs, each cow has 4 legs, so the total number of legs is given by: 2D + 4C ...

格式上完全不按 `<rationale>/<correct>` 输出，没有最终选项。

**SFT 模型输出**：

```
<rationale>"let the no of ducks be x and cows be y. then, we have:
4x + 2y = 2(x+y) + 26
4x + 2y = 2x + 2y + 26
2x = 26 - 26 = 0
x = 0
so, no of cows is 0. answer: a"</rationale><correct>a</correct>
```

格式完全对齐，但代数化简出错（把 26 减成了 0），所以选错了选项。

### 样本 B：good 的排列数

prompt：

> how many words, with or without meaning, can be formed using all letters of the word "good" using each letter exactly once?
> options: a) 18, b) 16, c) 24, d) 12, e) 4

参考答案：`<correct>e</correct>`（推理：4 个字母两个 o 重复，4!/2! = 12，但参考文本给的答案是 e=4，疑似数据集本身存在标注问题）

**base 模型输出**：

> The word "good" consists of 5 distinct letters: G, O, D, A, and I.

5 个字母都搞错了，不知道存在选项列表，更没有 `<correct>` 输出。

**SFT 模型输出**：

```
<rationale>"explanation: total no. of ways to arrange 5 letters = 5! = 120
answer is d"</rationale><correct>d</correct>
```

依然误把 good 当作 5 个字母，但已经知道按选项作答并输出 `<correct>d</correct>`。

## 5. 观察与结论

**实际学到了什么。**
对照 base 与 SFT 的输出可以明显看出，3 epoch / 500 条样本的微调主要让模型学到了 **输出格式**：

- 严格遵守 `<rationale>...</rationale><correct>x</correct>` 包装
- 知道输出必须落在选项 a–e 之内，而不是给一段开放式解释
- Rationale 段的写作风格（短句、`=` 推导、`answer : x`）也接近训练集

但 **数学推理本身没有真正提升**。SFT 命中率 26% 仅略高于随机基线 20%，
大量错误来自代数化简、字母计数、单位换算等基础步骤。

**为什么会这样。**
模型容量小（0.5B）、样本量小（500 条）、训练步数有限，
仅靠 next-token 监督很难让模型学到长链式的数学推理。
文档中也明确指出这一点：真正的推理能力通常需要更大模型 + RL/RLHF/RLVR 等后训练策略。

**这次实验的价值。**
跑通了 LLM 调优的完整工程链路——
环境配置、数据格式化、PEFT/LoRA 配置、训练 / 评估 / 推理三段脚本、指标统计、报告产出。
作为零基础到能独立完成一次 LoRA 微调闭环的基线，目标达成。

## 6. 后续可改进项

按改进性价比从高到低排：

1. **指令格式细化**：把 user 段中的 `<operations>` 与 `</options>` 这一对不匹配的标签纠正，
   原文档示例里就有这个 typo，沿用是为了与 P4 文档对齐，但实际工程中应统一。
2. **拉大训练规模**：直接用 MathQA full train（约 29k 条）跑 1 epoch，预期 Choice Accuracy 能进一步上升 5–10 个百分点。
3. **换更大底座**：升到 Qwen2.5-1.5B 或 3B，推理能力会有量变。
4. **加 step 评估**：训练时引入 eval_dataset 与 eval_strategy="steps"，画 train/eval loss 双曲线，便于判断过拟合时机。
5. **引入 RLVR**：用 `<correct>` 字段做 verifiable reward，对推理链路做 GRPO/DPO 之类的强化训练，
   这才是把 base 模型推到接近 DeepSeek-R1 推理风格的方向。

## 7. 复现清单

| 文件 | 说明 |
| - | - |
| `scripts/prepare_data.py` | MathQA 原始 JSON → Qwen chat jsonl |
| `src/finetune.py` | LoRA SFT 主脚本 |
| `src/evaluator.py` | ROUGE / BLEU / Choice Accuracy 评估 |
| `src/inference.py` | 多轮命令行对话 |
| `saved_models/qwen2.5-0.5b-sft/eval_report.json` | SFT 评估完整结果 |
| `Qwen2.5-0.5B-Instruct/eval_report.json` | base 评估完整结果 |

复现命令：

```powershell
python ./scripts/prepare_data.py
python ./src/finetune.py
python ./src/evaluator.py --model_path ./Qwen2.5-0.5B-Instruct --adapter_path ./saved_models/qwen2.5-0.5b-sft --eval_dataset_path ./data/test.jsonl
python ./src/evaluator.py --model_path ./Qwen2.5-0.5B-Instruct --eval_dataset_path ./data/test.jsonl
```
