"""
Evolution Strategies Fine-tuning for Countdown Task - Mac Compatible Version
Optimized for Apple Silicon (MPS) or CPU
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

parser = argparse.ArgumentParser(description='ES Fine-tuning for Countdown Task (Mac Compatible)')
parser.add_argument('--model_path', type=str, default='~/Desktop/Qwen/Qwen2.5-1.5B-Instruct', 
                    help='Path to model (default: ~/Desktop/Qwen/Qwen2.5-1.5B-Instruct)')
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                    help='Model precision (float32 for CPU/MPS, float16/bfloat16 if supported)')
parser.add_argument('--num_threads', type=int, default=4, 
                    help='Number of parallel threads for seed processing')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=50, 
                    help='Number of data samples to use for training (reduced for demo)')
parser.add_argument('--use_cpu', action='store_true', 
                    help='Force CPU usage (more stable than MPS for this task)')
parser.add_argument('--iterations', type=int, default=50,
                    help='Number of ES iterations (default: 50)')
parser.add_argument('--eval_iterations', type=int, default=10,
                    help='Evaluate model every N iterations (0 to disable, default: 10)')
parser.add_argument('--eval_data_sample', type=int, default=None,
                    help='Number of samples for evaluation (default: same as training data)')
parser.add_argument('--population_size', type=int, default=30,
                    help='Population size (number of perturbations per iteration, default: 30)')
parser.add_argument('--sigma', type=float, default=0.001,
                    help='Standard deviation for weight perturbations (default: 0.001)')
parser.add_argument('--alpha', type=float, default=0.0005,
                    help='Learning rate (default: 0.0005)')
parser.add_argument('--max_new_tokens', type=int, default=1024,
                    help='Maximum tokens to generate (default: 1024)')
parser.add_argument('--seed', type=int, default=33,
                    help='Random seed (default: 33)')
args = parser.parse_args()

# Hyperparameters for ES
NUM_ITERATIONS = args.iterations
POPULATION_SIZE = args.population_size
SIGMA = args.sigma
ALPHA = args.alpha
max_new_tokens = args.max_new_tokens
do_sample = False                 # Greedy decoding
initial_seed = args.seed

# Determine device (CPU is more stable for this task on Mac)
if args.use_cpu:
    device = torch.device("cpu")
    print("Using CPU (forced)")
elif torch.backends.mps.is_available() and not args.use_cpu:
    # Try MPS but be cautious
    try:
        # Test MPS with a simple operation
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

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()

def evaluate_model(model, tokenizer, input_texts, target_texts, seed_idx=None, verbose=False, return_text=False):
    """
    Generate responses from the model and compute rewards.
    """
    if verbose:
        print(f"Evaluating seed {seed_idx}")

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # Only synchronize if needed (CPU doesn't need it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            # MPS synchronization can be problematic, skip it or use try/except
            try:
                torch.mps.synchronize()
            except:
                pass  # MPS sync can fail, but operations are usually complete

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

    # Compute rewards
    rewards = []
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

    if return_text:
        return rewards, generated_texts
    else:
        return rewards


def evaluate_model_performance(model, tokenizer, dataset, verbose=False):
    """
    Evaluate model performance on a held-out dataset.
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
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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
    format_rewards = []
    answer_rewards = []
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
        
        # Extract format_reward and answer_reward for detailed metrics
        if "reward_info" in reward_result:
            format_rewards.append(reward_result["reward_info"].get("format_reward", 0.0))
            answer_rewards.append(reward_result["reward_info"].get("answer_reward", 0.0))
        
        # Count correct (reward > 0.9 means mostly correct answer)
        if reward > 0.9:
            correct_count += 1

    average_loss = -sum(rewards) / len(rewards)
    average_reward = sum(rewards) / len(rewards)
    accuracy = correct_count / len(rewards)
    
    # Compute format and answer reward averages
    avg_format_reward = sum(format_rewards) / len(format_rewards) if format_rewards else 0.0
    avg_answer_reward = sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0.0
    success_rate = avg_answer_reward  # answer_reward is 1.0 for correct, 0.0 otherwise
    
    return average_loss, average_reward, accuracy, avg_format_reward, success_rate


def process_seed(seed_args):
    """Function to process a single seed"""
    seed_idx, seed, model, tokenizer, verbose, dataset = seed_args

    if verbose:
        print(f"Thread processing seed {seed_idx} (value: {seed})")

    # Weight Perturbation
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Synchronize if using GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except:
            pass  # MPS sync can fail, but operations are usually complete

    # Evaluate all prompts with perturbed weights
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, 
                           seed_idx=seed_idx, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    # Synchronize if using GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except:
            pass  # MPS sync can fail, but operations are usually complete

    average_reward = total_reward / len(dataset)
    force_memory_cleanup()

    if verbose:
        print(f"Completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward

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

    print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
    print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Parallel threads: {args.num_threads}")
    if args.eval_iterations > 0:
        print(f"Evaluation every: {args.eval_iterations} iterations")
    else:
        print(f"Periodic evaluation: disabled")

    # Load model
    print(f"Loading model from {model_path}...")
    
    # Determine dtype
    if args.precision == 'float16':
        dtype = torch.float16
    elif args.precision == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # For MPS, use float32 (bfloat16/float16 may not be fully supported)
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

    # Create multiple model copies for parallel processing
    # This allows true parallelism without thread contention
    num_model_copies = min(args.num_threads, POPULATION_SIZE)
    if num_model_copies > 1:
        print(f"Creating {num_model_copies} model copies for parallel processing...")
        model_list = [model]  # First copy is the original
        for i in range(1, num_model_copies):
            # Deep copy the model for parallel processing
            model_copy = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model_copy = model_copy.to(device)
            model_copy.eval()
            model_list.append(model_copy)
        print(f"Created {len(model_list)} model copies (using ~{len(model_list) * 3.5:.1f} GB GPU memory)")
    else:
        model_list = [model]
        print("Using single model (sequential processing)")

    print("Model loaded successfully")
    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()
    np.random.seed(initial_seed)

    # Main ES loop
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate random seeds
        seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()

        # Process seeds in parallel batches using multiple model copies
        # Each model copy can process perturbations independently
        local_rewards = []
        batch_size = len(model_list)  # Process as many as we have model copies
        
        if args.verbose:
            print(f"Processing {POPULATION_SIZE} perturbations in batches of {batch_size}...")
        
        from concurrent.futures import ThreadPoolExecutor
        
        for batch_start in range(0, POPULATION_SIZE, batch_size):
            batch_end = min(batch_start + batch_size, POPULATION_SIZE)
            batch_seeds = [(seed_idx, seeds[seed_idx]) for seed_idx in range(batch_start, batch_end)]
            
            if args.verbose:
                print(f"  Processing batch: perturbations {batch_start+1}-{batch_end}/{POPULATION_SIZE}")
            
            # Use ThreadPoolExecutor with separate model copies
            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = [
                    (seed_idx, seed, model_list[i % len(model_list)], tokenizer, args.verbose, train_dataset)
                    for i, (seed_idx, seed) in enumerate(batch_seeds)
                ]
                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)
            
            # Cleanup between batches
            force_memory_cleanup()

        # Sort rewards by seed_idx
        local_rewards.sort(key=lambda x: x[0])
        rewards = [reward for _, reward in local_rewards]

        # Normalize rewards
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Update model weights (update all copies to keep them in sync)
        if args.verbose:
            print("Updating model weights...")
        
        # Update the first model (primary)
        for name, param in model_list[0].named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # Copy updated weights to all other model copies to keep them in sync
        if len(model_list) > 1:
            for model_copy in model_list[1:]:
                for (name, param_copy), (_, param_orig) in zip(model_copy.named_parameters(), model_list[0].named_parameters()):
                    param_copy.data.copy_(param_orig.data)
        
        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass  # MPS sync can fail, but operations are usually complete

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, "
              f"Mean: {mean_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}")

        # Periodic evaluation on held-out data
        if args.eval_iterations > 0 and (iteration + 1) % args.eval_iterations == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating model at iteration {iteration + 1}...")
            eval_loss, eval_reward, eval_accuracy, eval_format_reward, eval_success_rate = evaluate_model_performance(
                model_list[0], tokenizer, eval_dataset, verbose=args.verbose
            )
            print(f"Evaluation Results:")
            print(f"  Loss: {eval_loss:.4f}")
            print(f"  Reward: {eval_reward:.4f}")
            print(f"  Accuracy (reward > 0.9): {eval_accuracy * 100:.2f}%")
            print(f"  Success Rate (answer_reward): {eval_success_rate * 100:.2f}%")
            print(f"  Format Reward: {eval_format_reward:.4f}")
            print(f"{'='*60}\n")

        # Save checkpoint every 10 iterations
        if (iteration + 1) % 10 == 0:
            checkpoint_dir = f"checkpoint_iter_{iteration + 1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_list[0].save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")

    total_time = time.time() - training_start_time

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation:")
    print(f"{'='*60}")
    final_loss, final_reward, final_accuracy, final_format_reward, final_success_rate = evaluate_model_performance(
        model_list[0], tokenizer, eval_dataset, verbose=args.verbose
    )
    print(f"Final Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Reward: {final_reward:.4f}")
    print(f"  Accuracy (reward > 0.9): {final_accuracy * 100:.2f}%")
    print(f"  Success Rate (answer_reward): {final_success_rate * 100:.2f}%")
    print(f"  Format Reward: {final_format_reward:.4f}")
    print(f"{'='*60}\n")

    # Save final model
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    save_dir = f"finetuned_es_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_final"
    print(f"Saving final model to {save_dir}...")
    model_list[0].save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Final model saved successfully to {save_dir}")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
