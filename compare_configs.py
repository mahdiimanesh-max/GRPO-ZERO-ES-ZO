#!/usr/bin/env python3
"""
Compare ES configurations: Original vs Current
"""

print("=" * 100)
print("ES Configuration Comparison: Original Repository vs Current Setup")
print("=" * 100)

# Original configuration (from countdown/es_fine-tuning_countdown.py)
original_config = {
    "NUM_ITERATIONS": 500,
    "POPULATION_SIZE": 30,
    "SIGMA": 0.001,
    "ALPHA": 0.0005,
    "MAX_NEW_TOKENS": 1024,
    "DATA_SAMPLE": 1000,  # Default from argparse
    "MODEL": "Qwen/Qwen2.5-3B-Instruct",
}

# Current configuration (from run_es_gpu.sh)
current_config = {
    "NUM_ITERATIONS": 15,
    "POPULATION_SIZE": 30,
    "SIGMA": 0.001,
    "ALPHA": 0.0005,
    "MAX_NEW_TOKENS": 256,
    "DATA_SAMPLE": 32,
    "MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
}

print("\n" + "=" * 100)
print("Configuration Comparison")
print("=" * 100)
print(f"{'Parameter':<25} {'Original':<30} {'Current':<30} {'Change':<15}")
print("-" * 100)

for key in original_config.keys():
    orig_val = original_config[key]
    curr_val = current_config[key]
    
    if isinstance(orig_val, (int, float)):
        if orig_val != curr_val:
            if orig_val > 0:
                change_pct = ((curr_val - orig_val) / orig_val) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
        else:
            change_str = "Same"
    else:
        change_str = "Different" if orig_val != curr_val else "Same"
    
    print(f"{key:<25} {str(orig_val):<30} {str(curr_val):<30} {change_str:<15}")

print("\n" + "=" * 100)
print("Critical Differences")
print("=" * 100)

iter_ratio = current_config["NUM_ITERATIONS"] / original_config["NUM_ITERATIONS"]
data_ratio = current_config["DATA_SAMPLE"] / original_config["DATA_SAMPLE"]
tokens_ratio = current_config["MAX_NEW_TOKENS"] / original_config["MAX_NEW_TOKENS"]

print(f"\n1. ITERATIONS: {current_config['NUM_ITERATIONS']} vs {original_config['NUM_ITERATIONS']}")
print(f"   - Reduction: {iter_ratio:.1%} ({1/iter_ratio:.1f}x fewer iterations)")
print(f"   - Impact: ES needs many iterations to converge. 15 iterations is likely insufficient.")

print(f"\n2. DATA_SAMPLE: {current_config['DATA_SAMPLE']} vs {original_config['DATA_SAMPLE']}")
print(f"   - Reduction: {data_ratio:.1%} ({1/data_ratio:.1f}x fewer samples)")
print(f"   - Impact: With only 32 samples, ES has very limited signal per iteration.")

print(f"\n3. MAX_NEW_TOKENS: {current_config['MAX_NEW_TOKENS']} vs {original_config['MAX_NEW_TOKENS']}")
print(f"   - Reduction: {tokens_ratio:.1%} ({1/tokens_ratio:.1f}x fewer tokens)")
print(f"   - Impact: May truncate some responses, but less critical.")

print(f"\n4. MODEL: {current_config['MODEL']} vs {original_config['MODEL']}")
print(f"   - Smaller model (1.5B vs 3B)")
print(f"   - Impact: Less capacity, but shouldn't explain poor performance alone.")

print("\n" + "=" * 100)
print("Why ES is Performing Poorly")
print("=" * 100)
print("""
The main issues are:

1. **Insufficient Iterations (15 vs 500)**
   - ES is a black-box optimizer that needs many iterations to converge
   - With only 15 iterations, ES hasn't had time to learn effectively
   - Original: 500 iterations allows ES to explore and converge
   - Current: 15 iterations is only 3% of the original!

2. **Too Few Training Samples (32 vs 1000)**
   - ES evaluates rewards on training samples to estimate gradients
   - With only 32 samples, the reward signal is very noisy
   - Original: 1000 samples provide stable reward estimates
   - Current: 32 samples = 3.2% of original data

3. **Combined Effect**
   - Total gradient estimates: Original = 500 × 30 × 1000 = 15M samples
   - Total gradient estimates: Current = 15 × 30 × 32 = 14.4K samples
   - That's ~1000x fewer gradient estimates!

4. **Why GRPO Works Better with Same Setup**
   - GRPO uses explicit gradients (backpropagation)
   - More efficient: learns from each sample directly
   - Doesn't need as many iterations or samples
   - 15 steps × 32 samples is sufficient for gradient-based learning

5. **ES Needs More Resources**
   - ES is fundamentally less sample-efficient than gradient methods
   - It needs more iterations and/or more samples to compete
   - The "fair comparison" setup (15 iterations, 32 samples) is unfair to ES
""")

print("=" * 100)
print("Recommendations")
print("=" * 100)
print("""
To get fair ES performance:

Option 1: Increase Iterations (keep 32 samples)
  - Use 100-200 iterations (still less than 500, but more reasonable)
  - ES needs time to converge

Option 2: Increase Training Samples (keep 15 iterations)
  - Use 200-500 samples per iteration
  - More stable reward estimates

Option 3: Both (closer to original)
  - 100-200 iterations
  - 200-500 samples
  - This would be a more fair comparison

The current setup (15 iterations, 32 samples) is optimized for GRPO,
not ES. ES needs more resources to learn effectively.
""")
print("=" * 100)
