"""
Zero-Order (ZO) Optimization Fine-tuning for Countdown Task - Mac Compatible Version
Uses finite differences to estimate gradients, then standard gradient descent
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
import time
import gc
import json
import sys

# Add countdown directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'countdown'))
from countdown_task import reward_function

logging.set_verbosity_error()

parser = argparse.ArgumentParser(description='ZO Fine-tuning for Countdown Task (Mac Compatible)')
parser.add_argument('--model_path', type=str, default='~/Desktop/Qwen/Qwen2.5-1.5B-Instruct', 
                    help='Path to model (default: ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct)')
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                    help='Model precision (float32 for CPU/MPS, float16/bfloat16 if supported)')
parser.add_argument('--num_threads', type=int, default=4, 
                    help='Number of parallel threads for perturbation processing')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=50, 
                    help='Number of data samples to use for training (reduced for demo)')
parser.add_argument('--use_cpu', action='store_true', 
                    help='Force CPU usage (more stable than MPS for this task)')
parser.add_argument('--grad_method', type=str, default='central', choices=['forward', 'central'],
                    help='Gradient estimation method: forward or central difference (default: central)')
parser.add_argument('--eval_iterations', type=int, default=10,
                    help='Evaluate model every N iterations (0 to disable, default: 10)')
parser.add_argument('--eval_data_sample', type=int, default=None,
                    help='Number of samples for evaluation (default: same as training data)')
parser.add_argument('--iterations', type=int, default=500,
                    help='Number of ZO iterations (default: 500)')
parser.add_argument('--num_perturbations', type=int, default=30,
                    help='Number of perturbations per iteration (default: 30)')
parser.add_argument('--mu', type=float, default=0.001,
                    help='Perturbation step size μ for finite differences (default: 0.001)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate for gradient descent (default: 0.0005)')
parser.add_argument('--max_new_tokens', type=int, default=1024,
                    help='Maximum tokens to generate (default: 1024)')
parser.add_argument('--seed', type=int, default=33,
                    help='Random seed (default: 33)')
args = parser.parse_args()

# Hyperparameters for ZO
NUM_ITERATIONS = args.iterations
NUM_PERTURBATIONS = args.num_perturbations
MU = args.mu
LR = args.lr
max_new_tokens = args.max_new_tokens
do_sample = False                # Greedy decoding
initial_seed = args.seed

# Determine device (CPU is more stable for this task on Mac)
if args.use_cpu:
    device = torch.device("cpu")
    print("Using CPU (forced)")
elif torch.backends.mps.is_available() and not args.use_cpu:
    try:
        test_tensor = torch.randn(1, device="mps")
        del test_tensor
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    except Exception as e:
        print(f"MPS test failed ({e}), falling back to CPU")
        device = torch.device("cpu")
        print("Using CPU (MPS fallback)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device: {device}")
print(f"Gradient method: {args.grad_method}")

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()
        try:
            torch.mps.synchronize()
        except:
            pass

def evaluate_model_performance(model, tokenizer, dataset, verbose=False):
    """
    Evaluate model performance on a dataset.
    Returns metrics: average_loss, average_reward, accuracy (reward > 0.9)
    """
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    
    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # Compute rewards and metrics
    rewards = []
    correct_count = 0
    
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        numbers = None
        target = None
        
        # Extract numbers from input
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if tgt_text.isdigit():
            target = int(tgt_text)

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        # Use reward_function from countdown_task.py
        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        rewards.append(reward)
        
        # Count correct (reward > 0.9 means mostly correct answer)
        if reward > 0.9:
            correct_count += 1

    average_loss = -sum(rewards) / len(rewards)  # Loss is negative reward
    average_reward = sum(rewards) / len(rewards)
    accuracy = correct_count / len(rewards)
    
    return average_loss, average_reward, accuracy

def compute_loss(model, tokenizer, input_texts, target_texts, verbose=False):
    """
    Compute loss (negative reward) for the model on given inputs.
    Returns average loss across all samples.
    """
    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # Compute losses (negative rewards)
    losses = []
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        numbers = None
        target = None
        
        # Extract numbers from input
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if tgt_text.isdigit():
            target = int(tgt_text)

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        # Use reward_function from countdown_task.py
        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        # Convert reward to loss (negative reward, since we want to minimize)
        loss = -reward  # Higher reward = lower loss
        losses.append(loss)

    average_loss = sum(losses) / len(losses)
    return average_loss

def process_perturbation(pert_args):
    """
    Process a single perturbation to estimate directional gradient.
    Returns (pert_idx, dir_grad) where dir_grad is the directional gradient estimate.
    """
    pert_idx, seed, model, tokenizer, verbose, train_dataset = pert_args

    if verbose:
        print(f"Thread processing perturbation {pert_idx} (seed: {seed})")

    input_texts = [input_text for input_text, _ in train_dataset]
    target_texts = [target_text for _, target_text in train_dataset]

    # Generate perturbation vector using seed
    perturbations = {}
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        perturbations[name] = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )

    # Forward difference method
    if args.grad_method == 'forward':
        # Evaluate original model (baseline)
        loss_0 = compute_loss(model, tokenizer, input_texts, target_texts, verbose)
        
        # Perturb forward: θ + μ*ε
        for name, param in model.named_parameters():
            param.data.add_(MU * perturbations[name])
        
        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass
        
        # Evaluate perturbed model
        loss_plus = compute_loss(model, tokenizer, input_texts, target_texts, verbose)
        
        # Restore original weights
        for name, param in model.named_parameters():
            param.data.add_(-MU * perturbations[name])
        
        # Compute directional gradient: (loss_plus - loss_0) / mu
        dir_grad = (loss_plus - loss_0) / MU

    # Central difference method (more accurate)
    elif args.grad_method == 'central':
        # Perturb forward: θ + μ*ε
        for name, param in model.named_parameters():
            param.data.add_(MU * perturbations[name])
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass
        
        # Evaluate forward perturbation
        loss_plus = compute_loss(model, tokenizer, input_texts, target_texts, verbose)
        
        # Perturb backward: θ - μ*ε (from current position, so -2*μ total)
        for name, param in model.named_parameters():
            param.data.add_(-2 * MU * perturbations[name])
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass
        
        # Evaluate backward perturbation
        loss_minus = compute_loss(model, tokenizer, input_texts, target_texts, verbose)
        
        # Restore original weights: back to θ
        for name, param in model.named_parameters():
            param.data.add_(MU * perturbations[name])
        
        # Compute directional gradient: (loss_plus - loss_minus) / (2*mu)
        dir_grad = (loss_plus - loss_minus) / (2 * MU)

    # Synchronize final restore
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except:
            pass

    force_memory_cleanup()

    if verbose:
        print(f"Completed perturbation {pert_idx} with dir_grad: {dir_grad:.6f}")

    return pert_idx, dir_grad

def main():
    # Expand model path
    model_path = os.path.expanduser(args.model_path)
    
    # Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), 'countdown/data/countdown.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    dataset = []
    for item in data_json:
        context = item['context']
        target = item['target']
        dataset.append((context, target))

    # Split into training and evaluation sets
    total_samples = len(dataset)
    train_samples = args.data_sample
    eval_samples = args.eval_data_sample if args.eval_data_sample is not None else train_samples
    
    train_dataset = dataset[:train_samples]
    # Use different samples for evaluation if available
    if total_samples > train_samples:
        eval_dataset = dataset[train_samples:train_samples + eval_samples]
    else:
        eval_dataset = train_dataset  # Use same data if not enough samples
    
    print(f"Loaded {total_samples} total countdown samples from {data_path}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    print(f"Number of perturbations: {NUM_PERTURBATIONS}, Iterations: {NUM_ITERATIONS}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Mu (perturbation scale): {MU}, Learning rate: {LR}")
    print(f"Parallel threads: {args.num_threads}")

    # Load model
    print(f"Loading model from {model_path}...")
    
    # Determine dtype
    if args.precision == 'float16':
        dtype = torch.float16
    elif args.precision == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # For MPS, use float32
    if device.type == "mps" and dtype != torch.float32:
        print(f"Warning: MPS may not fully support {args.precision}, using float32")
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Make sure the model is downloaded to ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct")
        return

    # Move model to device
    model = model.to(device)
    model.eval()  # Turn off dropout, etc.

    print("Model loaded successfully", flush=True)
    print(
        "Training: each iteration runs many forward passes; the first lines may take 10–20+ minutes.",
        flush=True,
    )
    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()
    np.random.seed(initial_seed)

    # Main ZO loop
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        print(f"Starting iteration {iteration + 1}/{NUM_ITERATIONS} (perturbations)...", flush=True)

        # Generate random seeds for perturbations
        seeds = np.random.randint(0, 2**30, size=NUM_PERTURBATIONS, dtype=np.int64).tolist()

        # Process perturbations in parallel batches
        from concurrent.futures import ThreadPoolExecutor
        
        local_dir_grads = []
        batch_size = max(1, min(args.num_threads, NUM_PERTURBATIONS))

        for batch_start in range(0, NUM_PERTURBATIONS, batch_size):
            batch_end = min(batch_start + batch_size, NUM_PERTURBATIONS)
            batch_pert = [(pert_idx, seeds[pert_idx], model, tokenizer, args.verbose, dataset) 
                         for pert_idx in range(batch_start, batch_end)]

            with ThreadPoolExecutor(max_workers=len(batch_pert)) as executor:
                thread_args = [
                    (pert_idx, seed, model, tokenizer, args.verbose, train_dataset)
                    for pert_idx, seed in [(i, seeds[i]) for i in range(batch_start, batch_end)]
                ]
                results = list(executor.map(process_perturbation, thread_args))
                local_dir_grads.extend(results)

            print(
                f"  Perturbation batch done: {batch_start}–{batch_end - 1} / {NUM_PERTURBATIONS - 1}",
                flush=True,
            )
            force_memory_cleanup()

        # Sort by pert_idx
        local_dir_grads.sort(key=lambda x: x[0])
        dir_grads = [dir_grad for _, dir_grad in local_dir_grads]

        # Construct full gradient estimate: gradient = (1/P) * Σ_p (dir_grad_p * ε_p)
        if args.verbose:
            print("Constructing gradient estimate...")
        
        for name, param in model.named_parameters():
            gradient_estimate = torch.zeros_like(param)
            
            for pert_idx in range(NUM_PERTURBATIONS):
                seed = seeds[pert_idx]
                dir_grad = dir_grads[pert_idx]
                
                # Regenerate the same perturbation
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                perturbation = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                
                # Add to gradient estimate: dir_grad * perturbation
                gradient_estimate.add_(perturbation, alpha=float(dir_grad))
                del perturbation
            
            # Average over perturbations
            gradient_estimate.div_(NUM_PERTURBATIONS)
            
            # Gradient descent update: θ_new = θ - lr * gradient
            param.data.add_(-LR * gradient_estimate)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        
        # Compute current loss for monitoring
        # Compute training metrics
        input_texts = [input_text for input_text, _ in train_dataset]
        target_texts = [target_text for _, target_text in train_dataset]
        train_loss = compute_loss(model, tokenizer, input_texts, target_texts)
        train_reward = -train_loss  # Convert back to reward for display

        mean_dir_grad = np.mean([abs(dg) for dg in dir_grads])
        std_dir_grad = np.std(dir_grads)

        print(
            f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Train Reward: {train_reward:.4f}, "
            f"Mean |dir_grad|: {mean_dir_grad:.6f}, Std: {std_dir_grad:.6f}",
            flush=True,
        )

        # Evaluation on separate dataset
        if args.eval_iterations > 0 and (iteration + 1) % args.eval_iterations == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating model at iteration {iteration + 1}...")
            eval_loss, eval_reward, eval_accuracy = evaluate_model_performance(
                model, tokenizer, eval_dataset, verbose=args.verbose
            )
            print(f"Evaluation Results:")
            print(f"  Loss: {eval_loss:.4f}")
            print(f"  Reward: {eval_reward:.4f}")
            print(f"  Accuracy (reward > 0.9): {eval_accuracy * 100:.2f}%")
            print(f"{'='*60}\n")

        # Save checkpoint when periodic eval runs (same cadence as ES; zo prefix avoids clobbering ES dirs)
        if args.eval_iterations > 0 and (iteration + 1) % args.eval_iterations == 0:
            checkpoint_dir = f"checkpoint_zo_iter_{iteration + 1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")

    total_time = time.time() - training_start_time

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation:")
    print(f"{'='*60}")
    final_loss, final_reward, final_accuracy = evaluate_model_performance(
        model, tokenizer, eval_dataset, verbose=args.verbose
    )
    print(f"Final Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Reward: {final_reward:.4f}")
    print(f"  Accuracy (reward > 0.9): {final_accuracy * 100:.2f}%")
    print(f"{'='*60}\n")

    # Save final model
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    save_dir = f"finetuned_zo_pert{NUM_PERTURBATIONS}_iter{NUM_ITERATIONS}_{args.grad_method}_final"
    print(f"Saving final model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Final model saved successfully to {save_dir}")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
