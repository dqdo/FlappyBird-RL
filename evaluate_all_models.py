"""
evaluate_all_models.py

Evaluates all trained Flappy Bird RL models across:
  1 dynamic environment
  6 fixed environments

For each model/environment pair, this reports:
  - average score
  - score standard deviation
  - average survival time in seconds
  - survival time standard deviation
  - generalization gap

Generalization gap is calculated as:

  actual_train_dist average score - fixed environment average score

Positive gap  = model did worse on that fixed environment than on actual_train_dist
Negative gap  = model did better on that fixed environment than on actual_train_dist

Run:
  py evaluate_all_models.py

Make sure you have trained models saved in the model_weights folder first.
"""

import csv
import os
from typing import Dict, List, Optional

import numpy as np

from environment import DynamicFlappyEnv, FixedFlappyEnv, FPS
from models import DQNAgent, build_agent, MODEL_WEIGHTS_DIR


# ------------------------------------------------------------
# TEST ENVIRONMENT DEFINITIONS
# ------------------------------------------------------------

DYNAMIC_TEST_ENV = {
    "name": "dynamic_train_dist",
    "description": "Oscillating speed and gap, same as training distribution",
    "env_class": DynamicFlappyEnv,
    "kwargs": {
        "speed_min": 0.75,
        "speed_max": 1.25,
        "gap_min": 150,
        "gap_max": 280,
        "speed_period": 300,
        "gap_period": 450,
    },
}


FIXED_TEST_ENVS = [
    {
        "name": "fixed_ood_slow_wide",
        "description": "Fixed: speed=0.5x, gap=300px — below min speed, above max gap",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 0.5, "gap_size": 300},
    },
    {
        "name": "fixed_ood_slow_narrow",
        "description": "Fixed: speed=0.5x, gap=130px — below min speed, below min gap",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 0.5, "gap_size": 130},
    },
    {
        "name": "fixed_ood_fast_wide",
        "description": "Fixed: speed=1.5x, gap=300px — above max speed, above max gap",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.5, "gap_size": 300},
    },
    {
        "name": "fixed_ood_fast_narrow",
        "description": "Fixed: speed=1.5x, gap=130px — above max speed, below min gap, hardest",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.5, "gap_size": 130},
    },
    {
        "name": "fixed_ood_very_fast",
        "description": "Fixed: speed=1.75x, gap=180px — well above max speed, mid gap",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.75, "gap_size": 180},
    },
    {
        "name": "fixed_classic",
        "description": "Fixed: speed=1.0x, gap=140px — classic Flappy Bird feel",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.0, "gap_size": 140},
    },
]


ALL_TEST_ENVS = [DYNAMIC_TEST_ENV] + FIXED_TEST_ENVS


# ------------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "label": "Vanilla DQN",
        "agent_type": "dqn",
        "use_per": False,
        "filename": "flappy_dynamic_dqn.pth",
    },
    {
        "label": "Vanilla DQN + PER",
        "agent_type": "dqn",
        "use_per": True,
        "filename": "flappy_dynamic_dqn_per.pth",
    },
    {
        "label": "Double DQN",
        "agent_type": "ddqn",
        "use_per": False,
        "filename": "flappy_dynamic_ddqn.pth",
    },
    {
        "label": "Double DQN + PER",
        "agent_type": "ddqn",
        "use_per": True,
        "filename": "flappy_dynamic_ddqn_per.pth",
    },
    {
        "label": "Dueling DQN",
        "agent_type": "dueling",
        "use_per": False,
        "filename": "flappy_dynamic_dueling.pth",
    },
    {
        "label": "Dueling DQN + PER",
        "agent_type": "dueling",
        "use_per": True,
        "filename": "flappy_dynamic_dueling_per.pth",
    },
    {
        "label": "Fixed Vanilla DQN",
        "agent_type": "dqn",
        "use_per": False,
        "filename": "flappy_fixed_dqn.pth",
    },
    {
        "label": "Fixed Vanilla DQN + PER",
        "agent_type": "dqn",
        "use_per": True,
        "filename": "flappy_fixed_dqn_per.pth",
    },
    {
        "label": "Fixed Double DQN",
        "agent_type": "ddqn",
        "use_per": False,
        "filename": "flappy_fixed_ddqn.pth",
    },
    {
        "label": "Fixed Double DQN + PER",
        "agent_type": "ddqn",
        "use_per": True,
        "filename": "flappy_fixed_ddqn_per.pth",
    },
    {
        "label": "Fixed Dueling DQN",
        "agent_type": "dueling",
        "use_per": False,
        "filename": "flappy_fixed_dueling.pth",
    },
    {
        "label": "Fixed Dueling DQN + PER",
        "agent_type": "dueling",
        "use_per": True,
        "filename": "flappy_fixed_dueling_per.pth",
    },
]


# ------------------------------------------------------------
# AGENT LOADER
# ------------------------------------------------------------

def load_agent(
    model_path: str,
    agent_type: str,
    use_per: bool,
) -> DQNAgent:
    """
    Loads one trained model checkpoint and returns an evaluation-ready agent.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    dummy_env = DynamicFlappyEnv()

    class _Args:
        agent = agent_type
        per = use_per

    agent = build_agent(_Args(), dummy_env)
    dummy_env.close()

    agent.load(model_path)
    agent.policy.eval()

    return agent


# ------------------------------------------------------------
# EPISODE-LEVEL EVALUATION
# ------------------------------------------------------------

def run_one_episode(
    agent: DQNAgent,
    env_cfg: Dict,
    seed: Optional[int] = None,
    max_steps_per_episode: int = 1000000,
) -> Dict[str, float]:
    """
    Runs one episode and returns score and survival time.
    Caps the episode length so evaluation cannot run forever.
    """

    kwargs = dict(env_cfg["kwargs"])

    if seed is not None:
        kwargs["seed"] = seed

    env = env_cfg["env_class"](**kwargs)

    state = env.reset()
    done = False
    steps_survived = 0
    final_score = 0

    try:
        while not done and steps_survived < max_steps_per_episode:
            action = agent.act(state, greedy=True)
            state, _, done, info = env.step(action)

            steps_survived += 1
            final_score = info["score"]

            if steps_survived % 10000 == 0:
                print(
                    f"      still alive after {steps_survived} steps "
                    f"({steps_survived / FPS:.1f}s), score={final_score}",
                    flush=True,
                )

    finally:
        env.close()

    survival_seconds = steps_survived / FPS
    timed_out = not done

    if timed_out:
        print(
            f"      episode capped at {steps_survived} steps "
            f"({survival_seconds:.1f}s), score={final_score}",
            flush=True,
        )

    return {
        "score": float(final_score),
        "survival_steps": float(steps_survived),
        "survival_seconds": float(survival_seconds),
    }


def evaluate_model_on_env(
    agent: DQNAgent,
    env_cfg: Dict,
    episodes: int = 200,
    base_seed: int = 12345,
    env_index: int = 0,
) -> Dict[str, float]:
    """
    Evaluates one model on one environment over many episodes.
    """

    scores: List[float] = []
    survival_steps: List[float] = []
    survival_seconds: List[float] = []

    for ep in range(episodes):
        print(f"    Episode {ep + 1}/{episodes}...", flush=True)
        # Use repeatable seeds so each model sees comparable starting conditions.
        seed = base_seed + env_index * 100_000 + ep

        result = run_one_episode(agent, env_cfg, seed=seed)

        scores.append(result["score"])
        survival_steps.append(result["survival_steps"])
        survival_seconds.append(result["survival_seconds"])

    score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    survival_steps_std = float(np.std(survival_steps, ddof=1)) if len(survival_steps) > 1 else 0.0
    survival_seconds_std = float(np.std(survival_seconds, ddof=1)) if len(survival_seconds) > 1 else 0.0

    return {
        "avg_score": float(np.mean(scores)),
        "std_score": score_std,
        "avg_survival_steps": float(np.mean(survival_steps)),
        "std_survival_steps": survival_steps_std,
        "avg_survival_seconds": float(np.mean(survival_seconds)),
        "std_survival_seconds": survival_seconds_std,
    }



# ------------------------------------------------------------
# FULL EVALUATION
# ------------------------------------------------------------

def evaluate_all_models(
    episodes_per_env: int = 200,
    base_seed: int = 12345,
    output_csv: str = "evaluation_results.csv",
) -> List[Dict[str, float]]:
    """
    Evaluates every trained model across every test environment.
    """

    all_rows: List[Dict[str, float]] = []

    print("\n" + "=" * 90)
    print("FULL MODEL EVALUATION")
    print("=" * 90)
    print(f"Episodes per environment: {episodes_per_env}")
    print(f"Output CSV: {output_csv}")
    print("=" * 90)

    for model_cfg in MODEL_CONFIGS:
        model_label = model_cfg["label"]
        model_path = os.path.join(MODEL_WEIGHTS_DIR, model_cfg["filename"])

        print(f"\n\n{'=' * 90}")
        print(f"Evaluating model: {model_label}")
        print(f"Checkpoint: {model_path}")
        print(f"{'=' * 90}")

        if not os.path.exists(model_path):
            print(f"[SKIP] Missing checkpoint: {model_path}")
            continue

        agent = load_agent(
            model_path=model_path,
            agent_type=model_cfg["agent_type"],
            use_per=model_cfg["use_per"],
        )

        model_rows: List[Dict[str, float]] = []

        for env_index, env_cfg in enumerate(ALL_TEST_ENVS):
            metrics = evaluate_model_on_env(
                agent=agent,
                env_cfg=env_cfg,
                episodes=episodes_per_env,
                base_seed=base_seed,
                env_index=env_index,
            )

            row = {
                "model": model_label,
                "agent_type": model_cfg["agent_type"],
                "use_per": model_cfg["use_per"],
                "checkpoint": model_cfg["filename"],
                "environment": env_cfg["name"],
                "description": env_cfg["description"],
                **metrics,
            }

            model_rows.append(row)

        # Calculate generalization gap relative to actual_train_dist.
        dynamic_rows = [
            r for r in model_rows
            if r["environment"] == "dynamic_train_dist"
        ]

        if dynamic_rows:
            dynamic_avg_score = dynamic_rows[0]["avg_score"]
        else:
            dynamic_avg_score = 0.0

        for row in model_rows:
            row["generalization_gap"] = dynamic_avg_score - row["avg_score"]

            if row["environment"] == "dynamic_train_dist":
                row["gap_interpretation"] = "baseline"
            elif row["generalization_gap"] > 0:
                row["gap_interpretation"] = "worse than dynamic baseline"
            elif row["generalization_gap"] < 0:
                row["gap_interpretation"] = "better than dynamic baseline"
            else:
                row["gap_interpretation"] = "same as dynamic baseline"

        print_model_table(model_label, model_rows)

        all_rows.extend(model_rows)

    print_full_comparison_table(all_rows)
    print_hardest_fixed_vs_dynamic_table(all_rows)
    save_results_csv(all_rows, output_csv)

    print("\n" + "=" * 90)
    print("Evaluation complete.")
    print(f"Saved results to: {output_csv}")
    print("=" * 90 + "\n")

    return all_rows


def print_model_table(model_label: str, rows: List[Dict[str, float]]) -> None:
    """
    Prints a readable results table for one model.
    """

    print(f"\nResults for {model_label}")
    print("-" * 120)
    print(
        f"{'Environment':<24} "
        f"{'Avg Score':>10} "
        f"{'Std Score':>10} "
        f"{'Avg Survival(s)':>16} "
        f"{'Std Survival(s)':>16} "
        f"{'Gen Gap':>10} "
        f"Interpretation"
    )
    print("-" * 120)

    for r in rows:
        print(
            f"{r['environment']:<24} "
            f"{r['avg_score']:>10.2f} "
            f"{r['std_score']:>10.2f} "
            f"{r['avg_survival_seconds']:>16.2f} "
            f"{r['std_survival_seconds']:>16.2f} "
            f"{r['generalization_gap']:>10.2f} "
            f"{r['gap_interpretation']}"
        )

    print("-" * 120)

def print_full_comparison_table(rows: List[Dict[str, float]]) -> None:
    """
    Prints one combined comparison table containing all models,
    all test environments, and all evaluation metrics.
    """

    if not rows:
        print("\n[comparison] No evaluation results to print.")
        return

    print("\n" + "=" * 160)
    print("FULL COMPARISON TABLE - ALL MODELS ACROSS ALL TEST ENVIRONMENTS")
    print("=" * 160)

    print(
        f"{'Model':<22} "
        f"{'Environment':<24} "
        f"{'Avg Score':>10} "
        f"{'Std Score':>10} "
        f"{'Avg Steps':>12} "
        f"{'Std Steps':>12} "
        f"{'Avg Time(s)':>12} "
        f"{'Std Time(s)':>12} "
        f"{'Gen Gap':>10} "
        f"{'Interpretation':<28}"
    )

    print("-" * 160)

    for r in rows:
        print(
            f"{r['model']:<22} "
            f"{r['environment']:<24} "
            f"{r['avg_score']:>10.2f} "
            f"{r['std_score']:>10.2f} "
            f"{r['avg_survival_steps']:>12.2f} "
            f"{r['std_survival_steps']:>12.2f} "
            f"{r['avg_survival_seconds']:>12.2f} "
            f"{r['std_survival_seconds']:>12.2f} "
            f"{r['generalization_gap']:>10.2f} "
            f"{r['gap_interpretation']:<28}"
        )

    print("=" * 160 + "\n")

def print_hardest_fixed_vs_dynamic_table(
    rows: List[Dict[str, float]],
    dynamic_env_name: str = "dynamic_train_dist",
    hardest_env_name: str = "fixed_ood_fast_narrow",
) -> None:
    """
    Compares each model's metrics on the hardest fixed environment
    against its metrics on the dynamic training-distribution environment.
    """

    if not rows:
        print("\n[hardest comparison] No evaluation results to compare.")
        return

    models = sorted(set(r["model"] for r in rows))

    print("\n" + "=" * 150)
    print("HARDEST FIXED ENVIRONMENT VS DYNAMIC BASELINE")
    print("=" * 150)
    print(
        f"{'Model':<22} "
        f"{'Dyn Avg Score':>14} "
        f"{'Hard Avg Score':>15} "
        f"{'Score Diff':>12} "
        f"{'Dyn Avg Time(s)':>16} "
        f"{'Hard Avg Time(s)':>17} "
        f"{'Time Diff(s)':>13} "
        f"{'Interpretation':<30}"
    )
    print("-" * 150)

    for model in models:
        dynamic_rows = [
            r for r in rows
            if r["model"] == model and r["environment"] == dynamic_env_name
        ]

        hardest_rows = [
            r for r in rows
            if r["model"] == model and r["environment"] == hardest_env_name
        ]

        if not dynamic_rows or not hardest_rows:
            print(f"{model:<22} Missing dynamic or hardest fixed environment result.")
            continue

        dyn = dynamic_rows[0]
        hard = hardest_rows[0]

        score_diff = hard["avg_score"] - dyn["avg_score"]
        time_diff = hard["avg_survival_seconds"] - dyn["avg_survival_seconds"]

        if score_diff < 0:
            interpretation = "Worse on hardest fixed env"
        elif score_diff > 0:
            interpretation = "Better on hardest fixed env"
        else:
            interpretation = "Same average score"

        print(
            f"{model:<22} "
            f"{dyn['avg_score']:>14.2f} "
            f"{hard['avg_score']:>15.2f} "
            f"{score_diff:>12.2f} "
            f"{dyn['avg_survival_seconds']:>16.2f} "
            f"{hard['avg_survival_seconds']:>17.2f} "
            f"{time_diff:>13.2f} "
            f"{interpretation:<30}"
        )

    print("=" * 150 + "\n")

def save_results_csv(rows: List[Dict[str, float]], path: str) -> None:
    """
    Saves all evaluation rows to a CSV file.
    """

    if not rows:
        print("[warning] No rows to save.")
        return

    fieldnames = [
        "model",
        "agent_type",
        "use_per",
        "checkpoint",
        "environment",
        "description",
        "avg_score",
        "std_score",
        "avg_survival_steps",
        "std_survival_steps",
        "avg_survival_seconds",
        "std_survival_seconds",
        "generalization_gap",
        "gap_interpretation",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# RUN CONFIGURATION
# ------------------------------------------------------------

if __name__ == "__main__":
    evaluate_all_models(
        episodes_per_env=50,
        base_seed=12345,
        output_csv="evaluation_results.csv",
    )