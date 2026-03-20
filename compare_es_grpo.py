#!/usr/bin/env python3
"""
Compare ES vs GRPO training results
"""

print("=" * 100)
print("ES vs GRPO Training Comparison - Countdown Task")
print("=" * 100)

# ES data (mean reward per iteration) - ALL 15 iterations completed
es_data = [
    (1, 0.0663),
    (2, 0.0654),
    (3, 0.0716),
    (4, 0.0674),
    (5, 0.0673),
    (6, 0.0694),
    (7, 0.0705),
    (8, 0.0664),
    (9, 0.0705),
    (10, 0.0705),
    (11, 0.0623),
    (12, 0.0685),
    (13, 0.0736),
    (14, 0.0685),
    (15, 0.0683),
]

# ES validation results (from iteration 10 evaluation)
es_eval_data = {
    10: {"reward": 0.0681, "accuracy": 0.0078, "loss": -0.0681}  # 0.78% accuracy
}

# GRPO data (reward, success rate, format score)
grpo_data = [
    (1, 0.079, 0.01, 0.72),
    (2, 0.113, 0.02, 0.98),
    (3, 0.105, 0.01, 0.98),
    (4, 0.127, 0.03, 0.96),
    (5, 0.105, 0.01, 0.98),
    (6, 0.123, 0.02, 1.00),
    (7, 0.138, 0.04, 0.98),
    (8, 0.123, 0.02, 0.99),
    (9, 0.139, 0.04, 1.00),
    (10, 0.138, 0.04, 0.99),
    (11, 0.146, 0.05, 0.99),
    (12, 0.130, 0.03, 0.99),
    (13, 0.146, 0.05, 0.99),
    (14, 0.178, 0.08, 1.00),
    (15, 0.161, 0.06, 0.98),
]

print("\n" + "=" * 100)
print("Iteration-by-Iteration Comparison")
print("=" * 100)
print(f"{'Iter':<6} {'ES Reward':<12} {'GRPO Reward':<14} {'GRPO Success':<14} {'GRPO Format':<12} {'Difference':<12}")
print("-" * 100)

for i in range(1, 16):
    es_reward = None
    for iter_num, reward in es_data:
        if iter_num == i:
            es_reward = reward
            break
    
    grpo_reward, grpo_success, grpo_format = None, None, None
    for step, reward, success, fmt in grpo_data:
        if step == i:
            grpo_reward = reward
            grpo_success = success
            grpo_format = fmt
            break
    
    if es_reward is not None and grpo_reward is not None:
        diff = grpo_reward - es_reward
        diff_pct = (diff / es_reward) * 100 if es_reward > 0 else 0
        print(f"{i:<6} {es_reward:<12.4f} {grpo_reward:<14.4f} {grpo_success:<14.2f} {grpo_format:<12.2f} {diff:+.4f} ({diff_pct:+.1f}%)")
    elif es_reward is not None:
        print(f"{i:<6} {es_reward:<12.4f} {'N/A':<14} {'N/A':<14} {'N/A':<12} {'ES only'}")
    elif grpo_reward is not None:
        print(f"{i:<6} {'N/A':<12} {grpo_reward:<14.4f} {grpo_success:<14.2f} {grpo_format:<12.2f} {'GRPO only'}")

print("\n" + "=" * 100)
print("Summary Statistics")
print("=" * 100)

# ES statistics
es_rewards = [r for _, r in es_data]
es_avg = sum(es_rewards) / len(es_rewards) if es_rewards else 0
es_min = min(es_rewards) if es_rewards else 0
es_max = max(es_rewards) if es_rewards else 0

# GRPO statistics (first 6 steps for fair comparison)
grpo_rewards_6 = [r for s, r, _, _ in grpo_data[:6]]
grpo_avg_6 = sum(grpo_rewards_6) / len(grpo_rewards_6) if grpo_rewards_6 else 0
grpo_min_6 = min(grpo_rewards_6) if grpo_rewards_6 else 0
grpo_max_6 = max(grpo_rewards_6) if grpo_rewards_6 else 0

# GRPO statistics (all 15 steps)
grpo_rewards_all = [r for _, r, _, _ in grpo_data]
grpo_avg_all = sum(grpo_rewards_all) / len(grpo_rewards_all) if grpo_rewards_all else 0
grpo_min_all = min(grpo_rewards_all) if grpo_rewards_all else 0
grpo_max_all = max(grpo_rewards_all) if grpo_rewards_all else 0

print(f"\nES (all 15 iterations completed):")
print(f"  Average Reward: {es_avg:.4f}")
print(f"  Min Reward: {es_min:.4f}")
print(f"  Max Reward: {es_max:.4f}")
print(f"  Final Reward: {es_data[-1][1]:.4f}")

print(f"\nGRPO (all 15 steps - comparable):")
print(f"  Average Reward: {grpo_avg_all:.4f}")
print(f"  Min Reward: {grpo_min_all:.4f}")
print(f"  Max Reward: {grpo_max_all:.4f}")
print(f"  Final Reward: {grpo_data[-1][1]:.4f}")
print(f"  Improvement over ES: {((grpo_avg_all - es_avg) / es_avg * 100):+.1f}%")

print(f"\nGRPO (all 15 steps):")
print(f"  Average Reward: {grpo_avg_all:.4f}")
print(f"  Min Reward: {grpo_min_all:.4f}")
print(f"  Max Reward: {grpo_max_all:.4f}")
print(f"  Final Reward: {grpo_data[-1][1]:.4f}")

print(f"\nGRPO Additional Metrics (all 15 steps):")
grpo_success_all = [s for _, _, s, _ in grpo_data]
grpo_format_all = [f for _, _, _, f in grpo_data]
print(f"  Average Success Rate: {sum(grpo_success_all) / len(grpo_success_all):.4f} ({sum(grpo_success_all) / len(grpo_success_all) * 100:.2f}%)")
print(f"  Final Success Rate: {grpo_data[-1][2]:.2f} ({grpo_data[-1][2] * 100:.2f}%)")
print(f"  Average Format Score: {sum(grpo_format_all) / len(grpo_format_all):.4f}")
print(f"  Final Format Score: {grpo_data[-1][3]:.2f}")

print("\n" + "=" * 100)
print("Validation/Evaluation Results (Held-Out Test Set)")
print("=" * 100)

# GRPO validation results (from evaluation on test set)
# Format: (step, success_rate, format_reward, mean_reward)
grpo_eval_data = [
    (10, 0.0391, 0.9844, 0.1375),  # Step 10: from log
    (15, 0.0469, None, None),  # Final: success rate 4.69% (from log)
]

print("\nGRPO Validation Results (128 test samples):")
print(f"{'Step':<8} {'Success Rate':<18} {'Format Reward':<18} {'Mean Reward':<18}")
print("-" * 70)
for step, success, fmt, reward in grpo_eval_data:
    fmt_str = f"{fmt:.4f}" if fmt is not None else "N/A"
    reward_str = f"{reward:.4f}" if reward is not None else "N/A"
    print(f"{step:<8} {success:<18.2%} {fmt_str:<18} {reward_str:<18}")

print("\nES Validation Results (128 test samples):")
print(f"{'Iter':<8} {'Reward':<18} {'Success Rate':<18} {'Format Reward':<18} {'Accuracy':<18}")
print("-" * 90)
if 10 in es_eval_data:
    eval = es_eval_data[10]
    # ES accuracy (reward > 0.9) is 0.78%, which is similar to success rate
    # Note: ES doesn't have separate format_reward in old evaluation, but we can estimate
    print(f"{10:<8} {eval['reward']:<18.4f} {'~0.78%':<18} {'N/A (old eval)':<18} {eval['accuracy']*100:<18.2f}%")
else:
    print("  No evaluation results yet")

print("\n" + "-" * 100)
print("Validation Metrics Comparison (when available):")
print("  - ES Reward vs GRPO Mean Reward: Both measure overall task performance")
print("  - ES Accuracy vs GRPO Success Rate: Both measure correctness (ES: reward>0.9, GRPO: answer_reward)")
print("  - GRPO Format Reward: Additional metric for output format quality")
print("-" * 100)

print("\n" + "=" * 100)
print("Validation Comparison (Iteration/Step 10)")
print("=" * 100)
if 10 in es_eval_data and len(grpo_eval_data) > 0:
    es_eval = es_eval_data[10]
    grpo_eval = grpo_eval_data[0]  # Step 10
    print(f"\nES (Iteration 10):")
    print(f"  Reward: {es_eval['reward']:.4f}")
    print(f"  Accuracy (reward > 0.9): {es_eval['accuracy']*100:.2f}%")
    print(f"\nGRPO (Step 10):")
    print(f"  Mean Reward: {grpo_eval[3]:.4f}")
    print(f"  Success Rate: {grpo_eval[1]:.2%}")
    print(f"  Format Reward: {grpo_eval[2]:.4f}")
    print(f"\nComparison:")
    print(f"  Reward: GRPO {grpo_eval[3]:.4f} vs ES {es_eval['reward']:.4f} ({((grpo_eval[3] - es_eval['reward'])/es_eval['reward']*100):+.1f}% higher)")
    print(f"  Success: GRPO {grpo_eval[1]:.2%} vs ES {es_eval['accuracy']:.2%} ({((grpo_eval[1] - es_eval['accuracy'])/es_eval['accuracy']*100) if es_eval['accuracy'] > 0 else float('inf'):+.1f}% higher)")

print("\n" + "=" * 100)
print("Key Observations")
print("=" * 100)
print(f"1. GRPO shows higher rewards: {grpo_avg_all:.4f} vs ES {es_avg:.4f} (all 15 steps)")
print(f"2. GRPO reward range: {grpo_min_all:.4f} - {grpo_max_all:.4f} vs ES {es_min:.4f} - {es_max:.4f}")
print(f"3. GRPO continues improving: final reward {grpo_data[-1][1]:.4f} vs ES {es_data[-1][1]:.4f}")
print(f"4. ES rewards are stable but low: {es_min:.4f} - {es_max:.4f} (minimal improvement)")
print(f"5. GRPO success rate improves: {grpo_data[0][2]:.2f} → {grpo_data[-1][2]:.2f} (1% → 6%)")
print(f"6. GRPO format score: starts at {grpo_data[0][3]:.2f}, reaches {grpo_data[-1][3]:.2f}")
print(f"7. Validation: GRPO {grpo_eval_data[0][1]:.2%} success vs ES {es_eval_data[10]['accuracy']:.2%} accuracy")
print("=" * 100)
