"""
test_model.py
Evaluates a trained agent across:
  1 dynamic environment  (oscillating speed and gap, matching training distribution)
  5 fixed environments   (each with a different constant speed and gap combination)

To run, edit the configuration block at the bottom of this file and run:
  python test_model.py

Dependencies:
  pip install torch numpy pygame
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np

from environment import DynamicFlappyEnv, FixedFlappyEnv
from models import DQNAgent, evaluate, build_agent, MODEL_WEIGHTS_DIR


# TEST SUITE DEFINITION

# dynamic environment that mirrors the training distribution
DYNAMIC_TEST_ENV = {
    "name": "dynamic_train_dist",
    "description": "Oscillating speed (0.75-1.25x) and gap (150-280 px) - same as training",
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

# five fixed environments covering easy, medium, and hard difficulty combinations
# slow + wide   -> easiest: agent should perform well here
# slow + narrow -> medium: slow but tight gap
# mid  + mid    -> baseline: centre of the difficulty space
# fast + wide   -> medium: fast but the gap is forgiving
# fast + narrow -> hardest: out-of-distribution stress test
FIXED_TEST_ENVS = [
    {
        "name": "fixed_slow_wide",
        "description": "Fixed: speed=0.75x, gap=260 px (slow + wide, easiest)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 0.75, "gap_size": 260},
    },
    {
        "name": "fixed_slow_narrow",
        "description": "Fixed: speed=0.75x, gap=140 px (slow + narrow)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 0.75, "gap_size": 140},
    },
    {
        "name": "fixed_mid_mid",
        "description": "Fixed: speed=1.00x, gap=200 px (centre of fixed space)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.0, "gap_size": 200},
    },
    {
        "name": "fixed_fast_wide",
        "description": "Fixed: speed=1.25x, gap=260 px (fast + wide)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.25, "gap_size": 260},
    },
    {
        "name": "fixed_fast_narrow",
        "description": "Fixed: speed=1.25x, gap=130 px (fast + narrow, hardest)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.25, "gap_size": 130},
    },
    {
        "name": "fixed_classic",
        "description": "Fixed: speed=1.0x, gap=140 px (closer to original Flappy Bird)",
        "env_class": FixedFlappyEnv,
        "kwargs": {"speed": 1.0, "gap_size": 140},
    },
]

# combine dynamic and all fixed environments into a single list for iteration
ALL_TEST_ENVS = [DYNAMIC_TEST_ENV] + FIXED_TEST_ENVS


# AGENT LOADER

# load a trained agent from a checkpoint file and return it ready for evaluation
def load_agent(
    model_path: str,
    agent_type: str = "ddqn", # must match how the model was trained: dqn, ddqn, or dueling
    use_per: bool = False, # must match whether the model was trained with PER
) -> DQNAgent:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            "Train a model first with:  python models.py --agent ddqn"
        )

    # create a temporary environment just to get the state and action sizes
    dummy_env = DynamicFlappyEnv()

    # lightweight args object so build_agent can read the type and per flag
    class _Args:
        agent = agent_type
        per = use_per

    agent = build_agent(_Args(), dummy_env)
    dummy_env.close()

    # load the saved weights and training state into the agent
    agent.load(model_path)
    # switch the policy network to evaluation mode so dropout and batchnorm behave correctly
    agent.policy.eval()
    return agent


# EVALUATION

# run the agent across all test environments and print a results table with generalisation gaps
def run_evaluation(
    agent: DQNAgent,
    episodes_per_env: int = 200, # number of episodes to average per environment
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    print(f"\n{'='*65}")
    print("  EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"  {'Environment':<26}  {'Avg Score':>10}  Description")
    print(f"  {'-'*26}  {'-'*10}  {'-'*28}")

    # evaluate on each environment and record the average score
    for cfg in ALL_TEST_ENVS:
        env = cfg["env_class"](**cfg["kwargs"])
        sc = evaluate(agent, env, episodes=episodes_per_env)
        env.close()

        results[cfg["name"]] = sc
        print(f"  {cfg['name']:<26}  {sc:>10.2f}  {cfg['description']}")

    # show how each fixed environment score compares to the dynamic training environment
    train_score = results.get("dynamic_train_dist", 0.0)
    print(f"\n{'-'*65}")
    print("  GENERALISATION GAP  (dynamic_train_dist minus fixed env)")
    print(f"{'-'*65}")
    for cfg in FIXED_TEST_ENVS:
        name = cfg["name"]
        gap = train_score - results[name]
        direction = "worse" if gap > 0 else "better"
        print(f"  {name:<26}  {gap:>+8.2f}  {direction}")

    print(f"{'='*65}\n")
    return results


# VISUALISATION

# open a pygame window for a single environment and let the agent play until time runs out
def _visualize_env(
    agent: DQNAgent,
    cfg: dict,
    duration_seconds: int = 10,
) -> None:
    name = cfg["name"]
    print(f"\n  [viz] '{name}'  ({duration_seconds}s)")

    env = cfg["env_class"](**cfg["kwargs"], render_mode="human")
    state = env.reset()
    t_end = time.time() + duration_seconds
    done = False

    episode_scores: List[float] = []
    ep_score = 0

    try:
        while time.time() < t_end:
            # when an episode ends, record the score and start a new one
            if done:
                episode_scores.append(ep_score)
                state = env.reset()
                done = False
                ep_score = 0

            action = agent.act(state, greedy=True)
            state, _, done, info = env.step(action)
            ep_score = info["score"]

        # record the score from the last episode even if it did not finish
        episode_scores.append(ep_score)

    finally:
        env.close()

    if episode_scores:
        print(
            f"  [viz] done - {len(episode_scores)} episode(s) | "
            f"scores: {episode_scores} | best: {max(episode_scores)}"
        )


# open a pygame window for each selected test environment one after another
def run_visualization(
    agent: DQNAgent,
    duration_seconds: int = 10, # seconds to keep each window open
    env_filter: Optional[List[str]] = None, # list of env names to show, or None for all
) -> None:
    # filter to only the requested environments, or use all if no filter is given
    targets = [
        cfg for cfg in ALL_TEST_ENVS
        if env_filter is None or cfg["name"] in env_filter
    ]

    if not targets:
        print("[viz] No matching environments found. Check env_filter names.")
        return

    print(f"\n{'='*65}")
    print(f"  VISUALIZATION  ({len(targets)} environment(s), {duration_seconds}s each)")
    print(f"  Close a window early to skip to the next environment.")
    print(f"{'='*65}")

    for cfg in targets:
        try:
            _visualize_env(agent, cfg, duration_seconds=duration_seconds)
        except SystemExit:
            # user closed the window early, move on to the next environment
            print(f"  [viz] '{cfg['name']}' skipped.")

    print(f"\n{'='*65}")
    print("  Visualization complete.")
    print(f"{'='*65}\n")


# RUN CONFIGURATION - edit this section to control what runs

if __name__ == "__main__":
    # load the trained model from the model_weights directory
    agent = load_agent(
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "flappy_dynamic_dqn_per.pth"),
        agent_type = "dqn",
        use_per = True,
    )

    # set to True to just watch the agent play, False to run a full evaluation first
    WATCH_ONLY = False

    if WATCH_ONLY:
        run_visualization(
            agent = agent,
            duration_seconds = 30, # how long to watch each environment
            env_filter = ["fixed_slow_wide"], # or None to watch all environments
        )
    else:
        run_evaluation(
            agent = agent,
            episodes_per_env = 200,
        )
        run_visualization(
            agent = agent,
            duration_seconds = 10,
            env_filter = None,
        )