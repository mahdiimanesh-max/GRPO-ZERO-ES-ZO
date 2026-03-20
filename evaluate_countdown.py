"""
Universal Evaluation Script for Countdown Task
================================================
Evaluates any model checkpoint (ES, ZO, GRPO, or base model) on the countdown task.
All methods now save in HuggingFace format, so this script works uniformly.

Usage:
  # Evaluate base model
  python evaluate_countdown.py --model_path ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct

  # Evaluate any fine-tuned checkpoint (ES, ZO, or GRPO - all HuggingFace format)
  python evaluate_countdown.py --model_path ./finetuned_es_pop10_iter50_final

  # Compare base vs fine-tuned
  python evaluate_countdown.py --model_path ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct \
      --compare ./finetuned_grpo_countdown_final
"""
import json
import os
import sys
import time
import re
import argparse
from pathlib import Path

import numpy as np
import torch

# Add repo root and countdown to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'countdown'))
from countdown_task import reward_function


# ─── Prompt Template (same as training) ───
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def format_prompt_hf(tokenizer, numbers, target):
    """Format prompt using HuggingFace tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(numbers=numbers, target=target)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "Let me solve this step by step.\n<think>"
    return prompt


def format_prompt_grpo(tokenizer, numbers, target):
    """Format prompt using GRPO custom tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(numbers=numbers, target=target)},
    ]
    from grpo.grpo_tokenizer import Tokenizer as GRPOTokenizer
    prompt = tokenizer.encode_chat_with_response_prompt(
        messages,
        "Let me solve this step by step.\n<think>"
    )
    return prompt


def load_test_data(data_path, test_size=50):
    """Load test data from countdown.json."""
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    test_data = all_data[-test_size:]
    return test_data


def evaluate_hf_model(model, tokenizer, test_data, device, max_new_tokens=512, verbose=False):
    """Evaluate a HuggingFace model on the countdown task."""
    model.eval()
    results = []

    for idx, item in enumerate(test_data):
        numbers = item["numbers"]
        target = int(item["target"])

        prompt = format_prompt_hf(tokenizer, numbers, target)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Compute reward
        reward_result = reward_function(
            response=response,
            numbers=numbers,
            target=target,
            end_token=tokenizer.eos_token,
        )

        result = {
            "idx": idx,
            "numbers": numbers,
            "target": target,
            "response": response,
            "reward": reward_result["reward"],
            "answer_reward": reward_result["reward_info"]["answer_reward"],
            "format_reward": reward_result["reward_info"]["format_reward"],
        }
        results.append(result)

        if verbose or (idx + 1) % 10 == 0:
            print(f"\r  Evaluating: {idx+1}/{len(test_data)} "
                  f"(running success: {np.mean([r['answer_reward'] for r in results]):.2%})",
                  end="", flush=True)

    print()  # newline after progress
    return results


def evaluate_grpo_model(model, tokenizer, test_data, device, dtype, max_gen_len=512, verbose=False):
    """Evaluate a GRPO custom Transformer on the countdown task."""
    from grpo.grpo_core import rollout
    from grpo.data_types import MiniBatch

    model.eval()
    all_success = []
    all_format = []
    all_rewards = []
    all_responses = []

    # Process in batches
    batch_size = 2
    for start in range(0, len(test_data), batch_size):
        end = min(start + batch_size, len(test_data))
        batch_items = test_data[start:end]

        # Build MiniBatch
        numbers_list = [item["numbers"] for item in batch_items]
        targets = [int(item["target"]) for item in batch_items]
        prefixes = []
        prefix_tokens_list = []
        prefix_token_ids_list = []

        for item in batch_items:
            prefix = format_prompt_grpo(tokenizer, item["numbers"], int(item["target"]))
            tokens = tokenizer.tokenize(prefix)
            prefixes.append(prefix)
            prefix_tokens_list.append(tokens.tokens)
            prefix_token_ids_list.append(tokens.ids)

        batch = MiniBatch(
            numbers=numbers_list,
            target=targets,
            prefix=prefixes,
            prefix_tokens=prefix_tokens_list,
            prefix_token_ids=prefix_token_ids_list,
        )

        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=max_gen_len,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )

        for ep in episodes:
            all_success.append(ep.reward_info["answer_reward"])
            all_format.append(ep.reward_info["format_reward"])
            all_rewards.append(ep.reward)

        if (start + batch_size) % 10 < batch_size or verbose:
            print(f"\r  Evaluating: {end}/{len(test_data)} "
                  f"(running success: {np.mean(all_success):.2%})",
                  end="", flush=True)

    print()
    results = []
    for i, item in enumerate(test_data[:len(all_success)]):
        results.append({
            "idx": i,
            "numbers": item["numbers"],
            "target": int(item["target"]),
            "reward": all_rewards[i],
            "answer_reward": all_success[i],
            "format_reward": all_format[i],
        })
    return results


def print_results(results, label="Model"):
    """Print evaluation results summary."""
    rewards = [r["reward"] for r in results]
    success = [r["answer_reward"] for r in results]
    format_r = [r["format_reward"] for r in results]

    print(f"\n{'='*60}")
    print(f"Results for: {label}")
    print(f"{'='*60}")
    print(f"  Samples evaluated:  {len(results)}")
    print(f"  Mean reward:        {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"  Success rate:       {np.mean(success):.2%} ({sum(1 for s in success if s > 0)}/{len(success)})")
    print(f"  Format reward:      {np.mean(format_r):.4f}")
    print(f"  Best reward:        {max(rewards):.4f}")
    print(f"  Worst reward:       {min(rewards):.4f}")
    print(f"{'='*60}")

    # Show a few examples
    successes = [r for r in results if r.get("answer_reward", 0) > 0 and "response" in r]
    failures = [r for r in results if r.get("answer_reward", 0) == 0 and "response" in r]

    if successes:
        print(f"\n  ✓ Example success (#{successes[0]['idx']}):")
        print(f"    Numbers: {successes[0]['numbers']}, Target: {successes[0]['target']}")
        resp = successes[0]['response'][:200]
        print(f"    Response: {resp}...")

    if failures:
        print(f"\n  ✗ Example failure (#{failures[0]['idx']}):")
        print(f"    Numbers: {failures[0]['numbers']}, Target: {failures[0]['target']}")
        resp = failures[0]['response'][:200]
        print(f"    Response: {resp}...")

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(success)),
        "format_reward": float(np.mean(format_r)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Countdown Task Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (HuggingFace format or GRPO base model)')
    parser.add_argument('--grpo_ckpt', type=str, default=None,
                        help='Path to GRPO .pt checkpoint (requires --model_path as base model)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to second model for comparison (HuggingFace format)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to countdown.json (default: countdown/data/countdown.json)')
    parser.add_argument('--test_size', type=int, default=50,
                        help='Number of test samples')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max new tokens for generation')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-sample results')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    model_path = Path(os.path.expanduser(args.model_path))
    data_path = args.data_path or os.path.join(os.path.dirname(__file__), 'countdown/data/countdown.json')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Determine device
    if args.use_cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load test data
    test_data = load_test_data(data_path, args.test_size)
    print(f"Loaded {len(test_data)} test samples")

    all_summaries = {}

    # ─── Evaluate main model ───
    if args.grpo_ckpt:
        # GRPO model: load custom Transformer + state dict
        print(f"\nLoading GRPO model from {model_path} + {args.grpo_ckpt}...")
        from grpo.qwen2_model import Transformer
        from grpo.grpo_tokenizer import Tokenizer as GRPOTokenizer

        model = Transformer.from_pretrained(model_path, device=device)
        ckpt = torch.load(args.grpo_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        model = model.to(torch.float32).eval()

        tokenizer = GRPOTokenizer(str(model_path / "tokenizer.json"))
        dtype = torch.float32

        results = evaluate_grpo_model(
            model, tokenizer, test_data, device, dtype,
            max_gen_len=args.max_new_tokens, verbose=args.verbose,
        )
        label = f"GRPO ({Path(args.grpo_ckpt).stem})"
        summary = print_results(results, label)
        all_summaries[label] = summary

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        import gc; gc.collect()
    else:
        # HuggingFace model
        print(f"\nLoading HuggingFace model from {model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

        print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")

        results = evaluate_hf_model(
            model, tokenizer, test_data, device,
            max_new_tokens=args.max_new_tokens, verbose=args.verbose,
        )
        label = f"Model ({model_path.name})"
        summary = print_results(results, label)
        all_summaries[label] = summary

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # ─── Evaluate comparison model ───
    if args.compare:
        compare_path = Path(os.path.expanduser(args.compare))
        print(f"\nLoading comparison model from {compare_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()

        tokenizer2 = AutoTokenizer.from_pretrained(str(compare_path), trust_remote_code=True)
        model2 = AutoModelForCausalLM.from_pretrained(
            str(compare_path),
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

        results2 = evaluate_hf_model(
            model2, tokenizer2, test_data, device,
            max_new_tokens=args.max_new_tokens, verbose=args.verbose,
        )
        label2 = f"Fine-tuned ({compare_path.name})"
        summary2 = print_results(results2, label2)
        all_summaries[label2] = summary2

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for name, s in all_summaries.items():
            print(f"  {name:40s} → success: {s['success_rate']:.2%}, reward: {s['mean_reward']:.4f}")
        print(f"{'='*60}")

        del model2
        import gc; gc.collect()

    # ─── Save results ───
    if args.save_results:
        output = {
            "config": {
                "model_path": str(model_path),
                "grpo_ckpt": args.grpo_ckpt,
                "compare": args.compare,
                "test_size": args.test_size,
                "device": str(device),
            },
            "summaries": all_summaries,
        }
        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
