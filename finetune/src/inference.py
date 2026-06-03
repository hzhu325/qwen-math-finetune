"""
命令行多轮对话推理.

用法:
    python ./src/inference.py --model_path ./Qwen2.5-0.5B-Instruct \
        --adapter_path ./saved_models/qwen2.5-0.5b-sft/
    python ./src/inference.py --model_path ./Qwen2.5-0.5B-Instruct          # 不加载适配器

退出: /exit 或 /quit
重置上下文: /reset
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a helpful assistant."


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
    else:
        print("running base model without adapter")

    model.eval()
    print("ready, type /exit to quit, /reset to clear context")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_input = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            break
        if user_input.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("context cleared")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
