"""
GRPO (Group Relative Policy Optimization) Fine-tuning for Countdown Task
Mac Compatible Version - Uses HuggingFace Transformers (AutoModelForCausalLM)

This is a true RL approach: it computes log-probabilities and backpropagates
through the policy network, unlike ES/ZO which are gradient-free.

Based on https://github.com/policy-gradient/GRPO-Zero, adapted to use
HuggingFace models for consistency with the ES and ZO scripts.
"""
import json
import os
import sys
import time
import gc
import argparse
import math
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Add countdown directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'countdown'))
from countdown_task import reward_function

hf_logging.set_verbosity_error()


# ─── Data Types ───
@dataclass
class Episode:
    """Store all relevant information of an episode."""
    prefix: str
    text: str
    prefix_token_ids: List[int]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


# ─── Prompt templates (same as GRPO-Zero) ───
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
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


# ─── Utility Functions ───
def force_memory_cleanup(device):
    """Force aggressive memory cleanup."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def format_prompt(tokenizer, numbers, target):
    """Format a countdown prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(numbers=numbers, target=target)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += RESPONSE_PROMPT
    return prompt


# ─── Rollout (Generation) ───
@torch.no_grad()
def rollout(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    max_gen_len: int,
    num_answers_per_question: int,
    device: torch.device,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[Episode]:
    """
    Generate rollout episodes using HuggingFace model.generate().
    For each question, generates num_answers_per_question different responses.
    """
    model.eval()
    episodes = []

    for q_idx, question in enumerate(questions):
        numbers = question["numbers"]
        # Handle both int and float targets (some targets are floats in the data)
        target = int(float(question["target"]))
        prefix = format_prompt(tokenizer, numbers, target)

        # Tokenize the prefix
        inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
        prefix_token_ids = inputs["input_ids"][0].tolist()
        prefix_len = inputs["input_ids"].shape[1]

        # Repeat for multiple answers
        input_ids = inputs["input_ids"].repeat(num_answers_per_question, 1)
        attention_mask = inputs["attention_mask"].repeat(num_answers_per_question, 1)

        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen_len,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        print(
            f"\r* Generated {q_idx+1}/{len(questions)} questions "
            f"({num_answers_per_question} answers each)",
            flush=True, end="",
        )

        for ans_idx in range(num_answers_per_question):
            generated_ids = outputs[ans_idx][prefix_len:].tolist()

            # Remove padding
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            if pad_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(pad_id)]

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            is_finished = (
                len(generated_ids) < max_gen_len
                or (tokenizer.eos_token_id in generated_ids)
            )

            # Compute reward
            reward_result = reward_function(
                response=generated_text,
                numbers=numbers,
                target=target,
                end_token=tokenizer.eos_token,
            )

            episode = Episode(
                prefix=prefix,
                text=prefix + generated_text,
                prefix_token_ids=prefix_token_ids,
                generated_token_ids=generated_ids,
                is_finished=is_finished,
                reward=reward_result["reward"],
                reward_info=reward_result["reward_info"],
            )
            episodes.append(episode)

        del input_ids, attention_mask, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\r" + " " * 80, end="\r", flush=True)
    return episodes


# ─── GRPO Policy Update ───
def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for ep in episodes:
        groups[ep.prefix].append(ep)
    output = []
    for group in groups.values():
        group_rewards = [ep.reward for ep in group]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        for ep in group:
            normalized = (ep.reward - mean_r) / (std_r + 1e-4)
            output.append(replace(ep, reward=normalized))
    return output


def update_policy(
    model,
    optimizer,
    tokenizer,
    episodes: List[Episode],
    micro_batch_size: int,
    max_grad_norm: float,
    device: torch.device,
):
    """
    Update the policy using the GRPO algorithm with HuggingFace model.

    1. Normalize rewards within each question group
    2. Forward pass to compute log-probabilities of generated tokens
    3. Policy gradient: loss = -mean(log_prob * advantage)
    4. Backpropagate and update
    """
    model.train()
    episodes = normalize_rewards_per_group(episodes)
    # Sort by length for efficient batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    num_target_tokens = sum(len(ep.generated_token_ids) for ep in episodes)
    if num_target_tokens == 0:
        return {"loss": 0.0, "grad_norm": 0.0}

    for i in range(0, len(episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(episodes))
        batch_eps = episodes[i:j]

        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True, end="",
        )

        batch_lengths = [
            len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in batch_eps
        ]
        batch_max_length = max(batch_lengths)

        # Build padded token sequences and masks
        batch_token_ids = []
        batch_masks = []
        for bi, ep in enumerate(batch_eps):
            seq = ep.prefix_token_ids + ep.generated_token_ids
            pad_len = batch_max_length - len(seq)
            batch_token_ids.append(seq + [pad_token_id] * pad_len)
            # Mask: 0 for prefix, 1 for generated, 0 for padding
            mask = ([0] * len(ep.prefix_token_ids)
                    + [1] * len(ep.generated_token_ids)
                    + [0] * pad_len)
            batch_masks.append(mask)

        batch_advantages = [ep.reward for ep in batch_eps]

        input_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

        # Forward pass (shift for next-token prediction)
        input_token_ids = input_ids[:, :-1]
        target_token_ids = input_ids[:, 1:]
        target_masks = masks[:, 1:]

        outputs = model(input_token_ids)
        logits = outputs.logits.float()

        # Compute per-token log probabilities
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        # Policy gradient objective: log_prob * advantage
        obj = log_probs * advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

        del input_ids, masks, advantages, logits, log_probs, outputs

    # Clip gradients and update
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print("\r" + " " * 60, end="\r", flush=True)

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
    }


# ─── Evaluation ───
def evaluate_model(model, tokenizer, test_data, device, max_gen_len, temperature=0.7):
    """Evaluate model on a held-out test set."""
    model.eval()
    success_list = []
    format_list = []
    reward_list = []

    for idx, item in enumerate(test_data):
        numbers = item["numbers"]
        # Handle both int and float targets (some targets are floats in the data)
        target = int(float(item["target"]))
        prefix = format_prompt(tokenizer, numbers, target)

        inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        result = reward_function(
            response=response,
            numbers=numbers,
            target=target,
            end_token=tokenizer.eos_token,
        )
        success_list.append(result["reward_info"]["answer_reward"])
        format_list.append(result["reward_info"]["format_reward"])
        reward_list.append(result["reward"])

        if (idx + 1) % 10 == 0:
            print(f"\r  Eval: {idx+1}/{len(test_data)} "
                  f"(success: {np.mean(success_list):.2%})", end="", flush=True)

    print("\r" + " " * 60, end="\r", flush=True)
    return {
        "success_rate": float(np.mean(success_list)) if success_list else 0.0,
        "format_reward": float(np.mean(format_list)) if format_list else 0.0,
        "mean_reward": float(np.mean(reward_list)) if reward_list else 0.0,
    }


# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description='GRPO Fine-tuning for Countdown Task (Mac Compatible)')
    parser.add_argument('--model_path', type=str, default='~/Desktop/Qwen/Qwen2.5-1.5B-Instruct',
                        help='Path to pretrained model')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU (most stable on Mac)')
    parser.add_argument('--precision', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Model precision (float32 recommended for CPU/MPS)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Total batch size (num_questions × num_answers)')
    parser.add_argument('--num_questions', type=int, default=2,
                        help='Number of questions per batch')
    parser.add_argument('--micro_batch_size', type=int, default=1,
                        help='Micro-batch size for gradient accumulation')
    parser.add_argument('--max_gen_len', type=int, default=512,
                        help='Max generation length')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--data_sample', type=int, default=50,
                        help='Number of training samples to use')
    parser.add_argument('--test_size', type=int, default=50,
                        help='Number of samples for test set')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate every N steps (0 to disable)')
    parser.add_argument('--ckpt_interval', type=int, default=50,
                        help='Save checkpoint every N steps')
    parser.add_argument('--skip_unfinished', action='store_true',
                        help='Skip episodes that did not finish generating')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature for generation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose logs')
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    data_path = os.path.join(os.path.dirname(__file__), 'countdown/data/countdown.json')

    # Determine device
    if args.use_cpu:
        device = torch.device("cpu")
        print("Using CPU (forced)")
    elif torch.backends.mps.is_available():
        try:
            test_t = torch.randn(1, device="mps")
            del test_t
            device = torch.device("mps")
            print("Using MPS")
        except Exception:
            device = torch.device("cpu")
            print("MPS failed, using CPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.precision, torch.float32)
    if device.type == "mps" and dtype != torch.float32:
        print(f"Warning: MPS may not fully support {args.precision}, using float32")
        dtype = torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    NUM_ANSWERS_PER_QUESTION = args.batch_size // args.num_questions

    print(f"\n{'='*60}")
    print(f"GRPO Fine-tuning for Countdown Task (HuggingFace)")
    print(f"{'='*60}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Model: {model_path}")
    print(f"Batch: {args.num_questions} questions × {NUM_ANSWERS_PER_QUESTION} answers = {args.batch_size}")
    print(f"Max gen len: {args.max_gen_len}, LR: {args.lr}")
    print(f"Eval interval: {args.eval_interval}, Ckpt interval: {args.ckpt_interval}")
    print(f"{'='*60}\n")

    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, 'r') as f:
        all_data = json.load(f)

    # Split into train and test
    train_data = all_data[:args.data_sample]
    test_data = all_data[-args.test_size:]
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Make sure the model is downloaded to ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct")
        return

    model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({n_params:.1f}M parameters, dtype={dtype})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    # Setup checkpoint directory
    ckpt_dir = Path("ckpt_grpo")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create batches from training data (each batch = num_questions items)
    np.random.shuffle(train_data)
    batches = []
    for i in range(0, len(train_data), args.num_questions):
        batch = train_data[i:i + args.num_questions]
        if len(batch) == args.num_questions:
            batches.append(batch)

    total_steps = len(batches)
    print(f"\nStarting GRPO training ({total_steps} steps)...\n")

    # Training loop
    training_start_time = time.time()

    for step_idx, batch_questions in enumerate(batches):
        step = step_idx + 1
        iter_start = time.time()
        force_memory_cleanup(device)

        # 1. Generate rollouts (model in eval mode)
        model.eval()
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            questions=batch_questions,
            max_gen_len=args.max_gen_len,
            num_answers_per_question=NUM_ANSWERS_PER_QUESTION,
            device=device,
            temperature=args.temperature,
        )

        if args.skip_unfinished:
            episodes = [ep for ep in episodes if ep.is_finished]

        if len(episodes) == 0:
            print(f"Step {step}: No finished episodes, skipping...")
            continue

        # 2. Update policy (model in train mode)
        results = update_policy(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            episodes=episodes,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        force_memory_cleanup(device)

        iter_time = time.time() - iter_start

        # 3. Compute and log metrics
        reward_list = [ep.reward for ep in episodes]
        answer_rewards = [ep.reward_info["answer_reward"] for ep in episodes]
        format_rewards = [ep.reward_info["format_reward"] for ep in episodes]
        num_finished = sum(ep.is_finished for ep in episodes)
        mean_reward = np.mean(reward_list)
        success_rate = np.mean(answer_rewards)
        format_reward = np.mean(format_rewards)
        mean_resp_len = np.mean([len(ep.generated_token_ids) for ep in episodes])

        # Get GPU memory usage
        if device.type == "cuda":
            gpu_allocated = torch.cuda.memory_allocated(device) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_max = torch.cuda.max_memory_allocated(device) / 1024**3
            gpu_info = f", GPU: {gpu_allocated:.2f}GB alloc / {gpu_reserved:.2f}GB reserved (max: {gpu_max:.2f}GB)"
        else:
            gpu_info = ""

        print(
            f"Step {step}/{total_steps}, reward: {mean_reward:.3f}, "
            f"success: {success_rate:.2f}, format: {format_reward:.2f}, "
            f"loss: {results['loss']:.4f}, grad_norm: {results['grad_norm']:.2f}, "
            f"time: {iter_time:.1f}s, finished: {num_finished}/{len(episodes)}, "
            f"resp_len: {mean_resp_len:.0f}{gpu_info}"
        )

        if args.verbose:
            for ep in episodes[:2]:
                print(f"    Response: {ep.text[-200:]}")

        # 4. Periodic evaluation
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating model at step {step}...")
            eval_results = evaluate_model(
                model, tokenizer, test_data, device,
                args.max_gen_len, args.temperature,
            )
            print(f"Evaluation Results:")
            print(f"  Success rate:  {eval_results['success_rate']:.2%}")
            print(f"  Format reward: {eval_results['format_reward']:.4f}")
            print(f"  Mean reward:   {eval_results['mean_reward']:.4f}")
            print(f"{'='*60}\n")

        # 5. Save checkpoint
        if step % args.ckpt_interval == 0:
            ckpt_path = str(ckpt_dir / f"grpo_ckpt_step_{step:06d}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        gc.collect()

    total_time = time.time() - training_start_time

    # Final evaluation
    model.eval()
    print(f"\n{'='*60}")
    print("Final Evaluation:")
    eval_results = evaluate_model(
        model, tokenizer, test_data, device,
        args.max_gen_len, args.temperature,
    )
    print(f"  Success rate:  {eval_results['success_rate']:.2%}")
    print(f"  Format reward: {eval_results['format_reward']:.4f}")
    print(f"  Mean reward:   {eval_results['mean_reward']:.4f}")
    print(f"{'='*60}\n")

    # Save final model
    save_dir = f"finetuned_grpo_countdown_final"
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Saving final model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Final model saved successfully to {save_dir}")


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
