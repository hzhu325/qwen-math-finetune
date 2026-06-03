# Qwen2.5-0.5B 数学问答 LoRA 微调

基于 Qwen2.5-0.5B-Instruct 与 MathQA 数据集的监督微调（SFT）实战，
使用 PEFT/LoRA 在消费级显卡上完成训练、评估、命令行推理三段流程。

本项目目的是走通 LLM 从数据准备到部署推理的全流程，
而不是追求最终问答准确率，模型参数量与数据规模都不足以学到真正的数学推理能力。

## 作者

朱宏涛 (hzhu325@wisc.edu, GitHub: hzhu325)

## 目录结构

```
finetune/
├── data/
│   ├── dataset/simple/     # MathQA 原始 JSON
│   ├── train.jsonl         # 转换后的训练集 (500 条)
│   └── test.jsonl          # 转换后的测试集 (100 条)
├── Qwen2.5-0.5B-Instruct/  # 基础模型权重
├── scripts/
│   └── prepare_data.py     # 数据转换脚本
├── src/
│   ├── finetune.py         # LoRA 微调主脚本
│   ├── evaluator.py        # 评估脚本 (ROUGE / BLEU / 选项正确率)
│   └── inference.py        # 命令行推理脚本
├── saved_models/
│   └── qwen2.5-0.5b-sft/   # LoRA 适配器与评估报告
├── logs/                   # 训练日志
├── requirements.txt
└── README.md
```

## 环境

测试环境：

- Windows 11 + Python 3.13
- NVIDIA RTX 5090 Laptop (CUDA 13.2 驱动)
- PyTorch 2.11 + CUDA 12.8

依赖安装：

```powershell
# 1. 装匹配的 GPU 版 PyTorch (按显卡换 wheels 通道)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 装其他依赖
python -m pip install -r requirements.txt
```

## 数据与模型准备

1. 拉模型权重（任选其一）：

   ```powershell
   # 走 HuggingFace 直连
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./Qwen2.5-0.5B-Instruct')"

   # 国内可走 hf-mirror
   $env:HF_ENDPOINT="https://hf-mirror.com"
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./Qwen2.5-0.5B-Instruct')"
   ```

2. 拉 MathQA 原始数据：

   ```powershell
   curl -L -o MathQA.zip https://math-qa.github.io/math-QA/data/MathQA.zip
   # 解压后把 train.json / test.json / dev.json 放到 ./data/dataset/simple/
   ```

3. 转换数据：

   ```powershell
   python ./scripts/prepare_data.py
   ```

   默认按文档要求只取训练 500 / 测试 100 条，可在脚本顶部调整 `TRAIN_LIMIT` 与 `TEST_LIMIT`。

## 训练

```powershell
python ./src/finetune.py
```

关键超参（在 `src/finetune.py` 中可调）：

| 项 | 值 |
| - | - |
| LoRA rank / alpha | 16 / 32 |
| target_modules | q/k/v/o + gate/up/down (Attention + FFN) |
| LoRA dropout | 0.1 |
| batch size | 2 × gradient_accumulation_steps 2 = 4 |
| learning rate | 2e-5 |
| epochs | 3 |
| 调度器 | cosine, warmup_ratio 0.03 |
| 精度 | bf16 优先, 否则 fp16 |
| label mask | system + user 段不算 loss |

训练结束后，LoRA 适配器保存在 `./saved_models/qwen2.5-0.5b-sft/`。

## 评估

```powershell
python ./src/evaluator.py `
    --model_path ./Qwen2.5-0.5B-Instruct `
    --adapter_path ./saved_models/qwen2.5-0.5b-sft `
    --eval_dataset_path ./data/test.jsonl
```

输出三类指标：

- **ROUGE-1 / ROUGE-2 / ROUGE-L**：生成文本与参考答案 n-gram 召回相似度
- **BLEU**：n-gram 精确度
- **Choice Accuracy**：从 `<correct>x</correct>` 中提取出的选项是否命中标准答案

评估结果同时落盘到 `./saved_models/qwen2.5-0.5b-sft/eval_report.json`。

省略 `--adapter_path` 即评估未微调的 base 模型，用于对照。

## 命令行推理

```powershell
# 加载微调后的模型
python ./src/inference.py `
    --model_path ./Qwen2.5-0.5B-Instruct `
    --adapter_path ./saved_models/qwen2.5-0.5b-sft

# 仅加载 base 模型
python ./src/inference.py --model_path ./Qwen2.5-0.5B-Instruct
```

命令：`/exit` 或 `/quit` 退出；`/reset` 清空对话历史。

## 实测结果

参见 [`TRAINING_REPORT.md`](./TRAINING_REPORT.md)。
