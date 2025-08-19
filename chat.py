#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def get_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def bnb_and_cuda_available() -> bool:
    """Return True if bitsandbytes is importable and CUDA is available."""
    try:
        import bitsandbytes  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def load_model(model_id_or_path: str,
               load_in_4bit: bool,
               dtype: str,
               token: Optional[str]) -> (AutoModelForCausalLM, AutoTokenizer):
    torch_dtype = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, token=token)
    # Ensure pad token exists to prevent generation errors
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "token": token,
    }

    if load_in_4bit:
        if not bnb_and_cuda_available():
            print("[Info] 4-bit отключен: bitsandbytes/CUDA недоступны на этой системе.")
            load_in_4bit = False

        # 4-bit quantization via bitsandbytes (recommended for 12B on Colab)
        if load_in_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "torch_dtype": torch.float16,
            })
    else:
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **model_kwargs)
    return model, tokenizer


def format_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    # Use chat template if available (Gemma supports it)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt")
    # Fallback: simple concatenation
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
    return tokenizer(prompt, return_tensors="pt")


def chat_loop(model: AutoModelForCausalLM,
              tokenizer: AutoTokenizer,
              system_prompt: str,
              temperature: float,
              top_p: float,
              max_new_tokens: int):
    print("\n[Chat] Введите ваш вопрос. Команды: /exit или /quit для выхода, /clear для очистки контекста.\n")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Завершение.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("[Chat] Пока!")
            break
        if user_text.lower() == "/clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[Chat] Контекст очищен.")
            continue

        messages.append({"role": "user", "content": user_text})

        inputs = format_chat(tokenizer, messages).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Try to extract only the last assistant turn when using chat template
        if hasattr(tokenizer, "apply_chat_template"):
            # Heuristic: decode only the newly generated tokens
            new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            # Fallback: best-effort split
            answer = text.split("ASSISTANT:")[-1].strip()

        print(f"Модель: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with an HF model (optimized for Gemma 3 12B)")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it",
                        help="Model repo_id or local path (default: google/gemma-3-12b-it)")
    parser.add_argument("--system_prompt", type=str,
                        default="Ты — полезный краткий русскоязычный ассистент.",
                        help="System prompt to steer behavior")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens per reply")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype when not using 4-bit (default: float16)")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading (use more VRAM)")
    parser.add_argument("--token", type=str, default=None, help="HF token for gated/private models")
    return parser.parse_args()


def main():
    args = parse_args()

    # Default to 4-bit in Colab to save VRAM
    load_in_4bit = not args.no_4bit

    token = get_token(args.token)
    if "gemma" in str(args.model).lower() and not token:
        print("[WARN] Gemma модели требуют принятия лицензии и токен Hugging Face (HF_TOKEN).")

    print(f"[Info] Loading model: {args.model} | 4-bit: {load_in_4bit}")
    model, tokenizer = load_model(
        model_id_or_path=args.model,
        load_in_4bit=load_in_4bit,
        dtype=args.dtype,
        token=token,
    )

    chat_loop(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
