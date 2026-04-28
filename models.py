"""
models.py
Neural networks, replay buffers, DQNAgent, training loop, and CLI entry point.

Three agent architectures are supported, each optionally combined with PER:

  Variant                  | use_double | use_dueling | use_per | Saves to
  Vanilla DQN              |   False    |    False    |  False  | flappy_dynamic_dqn.pth
  Vanilla DQN + PER        |   False    |    False    |  True   | flappy_dynamic_dqn_per.pth
  Double DQN               |   True     |    False    |  False  | flappy_dynamic_ddqn.pth
  Double DQN + PER         |   True     |    False    |  True   | flappy_dynamic_ddqn_per.pth
  Dueling DQN              |   False    |    True     |  False  | flappy_dynamic_dueling.pth
  Dueling DQN + PER        |   False    |    True     |  True   | flappy_dynamic_dueling_per.pth

Quick-start (CLI)
  python models.py --agent dqn
  python models.py --agent dqn     --per
  python models.py --agent ddqn
  python models.py --agent ddqn    --per
  python models.py --agent dueling
  python models.py --agent dueling --per
  python models.py --train-all
  python models.py --agent ddqn    --steps 500_000 --model model_weights/flappy_dynamic_ddqn.pth

Dependencies
  pip install torch numpy pygame
"""

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import DynamicFlappyEnv, FixedFlappyEnv

# directory where all model checkpoint files will be saved
MODEL_WEIGHTS_DIR = "model_weights"

# create the model weights folder if it does not already exist
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)


# NEURAL NETWORKS
# DQNNetwork is used by Vanilla DQN and Double DQN
# DuelingDQNNetwork is used by Dueling DQN
# The dueling version splits into two heads: one for state value, one for action advantage
# These are then combined to get the final Q-values

# standard Q-network with two hidden layers that maps a state to a Q-value for each action
class DQNNetwork(nn.Module):

    # set up the layers of the network
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        # build a simple 3-layer network: input -> hidden -> hidden -> output
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    # pass input through the network and return Q-values
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# dueling Q-network that splits into a value stream and an advantage stream
class DuelingDQNNetwork(nn.Module):

    # set up the shared trunk and the two separate output heads
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        # shared layers that process the raw state before splitting
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        h2 = hidden // 2
        # value head: estimates how good the current state is overall
        self.value = nn.Sequential(nn.Linear(hidden, h2), nn.ReLU(), nn.Linear(h2, 1))
        # advantage head: estimates how much better each action is relative to average
        self.advantage = nn.Sequential(nn.Linear(hidden, h2), nn.ReLU(), nn.Linear(h2, out_dim))

    # combine value and advantage into a single Q-value per action
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        value = self.value(shared)
        advantage = self.advantage(shared)
        # subtract the mean advantage so value and advantage stay independently meaningful
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# REPLAY BUFFERS
# ReplayBuffer samples experiences uniformly at random
# PrioritizedReplayBuffer samples experiences that had large prediction errors more often

# a single stored experience with state, action, reward, next state, and done flag
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# converts a list of Transition objects into batched numpy arrays for training
def _collate(transitions: List[Transition]):
    return (
        np.array([t.state for t in transitions], dtype=np.float32),
        np.array([t.action for t in transitions], dtype=np.int64),
        np.array([t.reward for t in transitions], dtype=np.float32),
        np.array([t.next_state for t in transitions], dtype=np.float32),
        np.array([t.done for t in transitions], dtype=np.float32),
    )


# simple replay buffer that samples experiences with equal probability
class ReplayBuffer:

    # create a buffer that holds up to capacity experiences
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    # add a new experience to the buffer
    def push(self, *args):
        self.buf.append(Transition(*args))

    # randomly sample n experiences from the buffer
    def sample(self, n: int):
        return _collate(random.sample(self.buf, n)), None, None

    def __len__(self):
        return len(self.buf)


# prioritized replay buffer that replays high-error experiences more frequently
class PrioritizedReplayBuffer:

    # set up the sum-tree data structure and priority parameters
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6, # controls how strongly to prioritize high-error samples
        beta_start: float = 0.4, # initial strength of bias correction weights
        beta_end: float = 1.0, # final strength of bias correction weights
        beta_anneal_steps: int = 1_000_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal = beta_anneal_steps
        self._step = 0
        # sum-tree stores priorities in a binary tree for fast weighted sampling
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.size = 0
        self.ptr = 0
        self.max_p = 1.0

    # beta is annealed from beta_start to beta_end over training to correct sampling bias
    @property
    def beta(self) -> float:
        t = min(1.0, self._step / self.beta_anneal)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    # store a new experience and assign it the current max priority
    def push(self, *args):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = Transition(*args)
        self._update(idx, self.max_p ** self.alpha)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # sample n experiences proportional to their priority and compute IS weights
    def sample(self, n: int):
        self._step += 1
        tree_indices, priorities = [], []
        # divide the total priority range into n equal segments and sample one from each
        seg = self.tree[0] / n
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            ti = self._retrieve(0, s)
            tree_indices.append(ti)
            priorities.append(self.tree[ti])
        # compute importance-sampling weights to correct for the non-uniform sampling
        probs = np.array(priorities) / (self.tree[0] + 1e-9)
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()
        transitions = [self.data[ti - self.capacity + 1] for ti in tree_indices]
        return _collate(transitions), weights.astype(np.float32), tree_indices

    # update priorities for a batch of experiences after learning from them
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for ti, err in zip(indices, td_errors):
            p = (abs(float(err)) + 1e-6) ** self.alpha
            self.max_p = max(self.max_p, p)
            self._update(ti, p)

    # update a single node in the sum-tree and propagate the change upward
    def _update(self, ti: int, p: float):
        delta = p - self.tree[ti]
        self.tree[ti] = p
        # walk up the tree updating parent sums
        while ti > 0:
            ti = (ti - 1) // 2
            self.tree[ti] += delta

    # recursively walk down the tree to find the leaf node for a given priority value
    def _retrieve(self, ti: int, s: float) -> int:
        left = 2 * ti + 1
        if left >= len(self.tree):
            return ti
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(left + 1, s - self.tree[left])

    def __len__(self):
        return self.size


# AGENT
# DQNAgent supports all six variants via three flags:
# use_double=False, use_dueling=False -> Vanilla DQN
# use_double=True,  use_dueling=False -> Double DQN
# use_double=False, use_dueling=True  -> Dueling DQN
# adding use_per=True to any of the above enables prioritized replay
#
# The key difference between Vanilla DQN and Double DQN is how the next Q-value is computed:
# Vanilla DQN: the target network both picks and scores the best next action
# Double DQN:  the policy network picks the action, the target network scores it
# Double DQN reduces overestimation of Q-values

# agent that handles all DQN variants through configuration flags
class DQNAgent:

    # set up networks, optimizer, replay buffer, and training hyperparameters
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 200_000,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        use_double: bool = False, # True enables Double DQN
        use_dueling: bool = False, # True uses the Dueling network architecture
        use_per: bool = False, # True uses Prioritized Experience Replay
        device: str = "auto",
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        # how much to reduce epsilon each step
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double = use_double
        self.use_per = use_per
        self._learn_steps = 0

        # automatically pick the best available hardware
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # use the dueling architecture if requested, otherwise use the standard network
        Net = DuelingDQNNetwork if use_dueling else DQNNetwork
        # policy network is the one being actively trained
        self.policy = Net(state_size, action_size).to(self.device)
        # target network is a lagging copy used to compute stable training targets
        self.target = Net(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # Adam optimizer for updating the policy network weights
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        # use prioritized replay if requested, otherwise use uniform replay
        self.memory = (
            PrioritizedReplayBuffer(buffer_size) if use_per else ReplayBuffer(buffer_size)
        )

    # choose an action using epsilon-greedy exploration or greedily if specified
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        # explore randomly with probability epsilon unless greedy mode is on
        if not greedy and random.random() < self.eps:
            return random.randrange(self.action_size)
        # otherwise pick the action with the highest predicted Q-value
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy(s).argmax(1).item())

    # reduce epsilon by one step to gradually shift from exploring to exploiting
    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    # store a new experience in the replay buffer
    def push(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    # sample a batch from memory and update the policy network weights
    def learn(self) -> Optional[float]:
        # wait until the buffer has enough experiences to fill a batch
        if len(self.memory) < self.batch_size:
            return None

        batch, is_weights, tree_ids = self.memory.sample(self.batch_size)
        s, a, r, ns, d = batch

        # convert all batch arrays to tensors on the correct device
        S = torch.FloatTensor(s).to(self.device)
        A = torch.LongTensor(a).to(self.device)
        R = torch.FloatTensor(r).to(self.device)
        NS = torch.FloatTensor(ns).to(self.device)
        D = torch.FloatTensor(d).to(self.device)

        # get the Q-value the policy network predicted for each action that was taken
        q_curr = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                # Double DQN: policy net picks the best next action, target net scores it
                best_actions = self.policy(NS).argmax(1)
                q_next = self.target(NS).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN: target net both picks and scores the best next action
                q_next = self.target(NS).max(1)[0]

            # compute the target Q-value using the Bellman equation
            q_target = R + self.gamma * q_next * (1 - D)

        # compute TD errors to measure how wrong our predictions were
        td_err = (q_curr - q_target).detach().cpu().numpy()

        if is_weights is not None:
            # scale each sample's loss by its importance-sampling weight to correct PER bias
            W = torch.FloatTensor(is_weights).to(self.device)
            loss = (W * F.mse_loss(q_curr, q_target, reduction="none")).mean()
        else:
            loss = F.mse_loss(q_curr, q_target)

        # backpropagate the loss and update the policy network
        self.opt.zero_grad()
        loss.backward()
        # clip gradients to prevent exploding gradient issues
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        # update the priorities in the PER buffer based on new TD errors
        if self.use_per and tree_ids is not None:
            self.memory.update_priorities(tree_ids, td_err)

        self._learn_steps += 1
        # periodically copy the policy network weights into the target network
        if self._learn_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    # save the agent's networks, optimizer state, and training metadata to a file
    def save(self, path: str, best_eval: float = None, best_ep_score: float = None):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "opt": self.opt.state_dict(),
                "eps": self.eps,
                "learn_steps": self._learn_steps,
                "best_eval": best_eval,
                "best_ep_score": best_ep_score,
            },
            path,
        )
        print(f"[save] {path}")

    # load a previously saved checkpoint and restore all training state
    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.eps = ck["eps"]
        self._learn_steps = ck["learn_steps"]
        self._loaded_best_eval = ck.get("best_eval", None)
        self._loaded_best_ep_score = ck.get("best_ep_score", None)
        print(f"[load] {path}")


# EVALUATION

# run the agent greedily for a fixed number of episodes and return the average score
def evaluate(
    agent: DQNAgent,
    env: DynamicFlappyEnv,
    episodes: int = 200,
) -> float:
    scores = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        # play a full episode without exploration
        while not done:
            a = agent.act(s, greedy=True)
            s, _, done, info = env.step(a)
        scores.append(info["score"])
    return float(np.mean(scores))


# VISUALISATION

# open a game window and watch the agent play for a set number of seconds
def visualize_agent(
    agent: DQNAgent,
    env_kwargs: Dict,
    duration_seconds: int = 10,
    step: int = 0,
) -> None:
    print(f"\n  [vis] rendering agent at step {step:,} for {duration_seconds}s")
    vis_env = DynamicFlappyEnv(**env_kwargs, render_mode="human")
    state = vis_env.reset()
    t_end = time.time() + duration_seconds
    done = False
    try:
        while time.time() < t_end:
            # reset the environment when an episode ends and keep playing
            if done:
                state = vis_env.reset()
                done = False
            action = agent.act(state, greedy=True)
            state, _, done, _ = vis_env.step(action)
    finally:
        vis_env.close()
    print("  [vis] done.\n")


# TRAINING LOOP

# run the main training loop, saving the best model and logging progress along the way
def train(
    agent: DQNAgent,
    env: DynamicFlappyEnv,
    total_steps: int = 1_000_000,
    eval_interval: int = 20_000,
    eval_episodes: int = 20,
    log_interval: int = 2_000,
    save_path: str = "model.pth",
    viz_interval: int = 100_000,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
    label: str = ""
) -> Tuple[List[float], List[float]]:
    if env_kwargs is None:
        env_kwargs = {}

    # print a summary of the training configuration before starting
    print(f"\n{'-'*60}")
    print(f" Training - device: {agent.device}")
    print(f" double={agent.use_double}  dueling={hasattr(agent.policy, 'value')}"
          f"  per={agent.use_per}  steps={total_steps:,}")
    print(f" viz every {viz_interval:,} steps for {viz_duration}s")
    print(f"{'-'*60}\n")

    state = env.reset()
    ep_scores: List[float] = []
    losses: List[float] = []

    # restore best scores from checkpoint if resuming, otherwise start fresh
    best_eval = getattr(agent, "_loaded_best_eval", None) or -float("inf")
    best_ep_score = getattr(agent, "_loaded_best_ep_score", None) or -float("inf")
    ep = 0

    for step in range(1, total_steps + 1):
        # pick an action, step the environment, and store the transition
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.push(state, action, reward, next_state, done)
        # learn from a batch of past experiences
        loss = agent.learn()
        # reduce exploration probability over time
        agent.decay_epsilon()

        state = next_state
        if done:
            ep_score = info["score"]
            ep_scores.append(ep_score)
            ep += 1

            # save the model whenever the agent achieves a new personal best episode score
            if ep_score > best_ep_score:
                best_ep_score = ep_score
                agent.save(save_path, best_eval=best_eval, best_ep_score=best_ep_score)
                print(f"  new best episode score: {ep_score}  (saved)")

            state = env.reset()

        if loss is not None:
            losses.append(loss)

        # print a progress summary every log_interval steps
        if step % log_interval == 0:
            sc = np.mean(ep_scores[-50:]) if ep_scores else 0.0
            ls = np.mean(losses[-200:]) if losses else 0.0
            print(
                f"step {step:>8,} | ep {ep:>5,} | "
                f"score(50) {sc:>6.2f} | best {best_ep_score:.0f} | loss {ls:.4f} | eps {agent.eps:.3f}"
            )
            with open(label+".csv", "a") as f:
                f.write(f"{step},{ep},{sc},{best_ep_score},{ls},{agent.eps}\n")

        # run a formal evaluation on a separate environment every eval_interval steps
        if step % eval_interval == 0:
            eval_env = DynamicFlappyEnv(**env_kwargs)
            ev = evaluate(agent, eval_env, episodes=eval_episodes)
            eval_env.close()
            print(f"  eval  avg_score={ev:.2f}")
            # save the model if this is the best average evaluation score so far
            if ev > best_eval:
                best_eval = ev
                agent.save(save_path, best_eval=best_eval, best_ep_score=best_ep_score)
                print(f"  new best avg eval: {ev:.2f}  (saved)")

        # open a visualisation window every viz_interval steps if enabled
        if viz_interval > 0 and step % viz_interval == 0:
            visualize_agent(agent, env_kwargs=env_kwargs, duration_seconds=viz_duration, step=step)

    return ep_scores, losses


# AGENT BUILDER

# create a DQNAgent configured for the requested architecture type and settings
def build_agent(args, env: DynamicFlappyEnv) -> DQNAgent:
    # map agent type string to the correct combination of flags
    # dqn     -> Vanilla DQN (use_double=False, use_dueling=False)
    # ddqn    -> Double DQN  (use_double=True,  use_dueling=False)
    # dueling -> Dueling DQN (use_double=False, use_dueling=True)
    return DQNAgent(
        state_size = env.observation_space_size,
        action_size = env.action_space_size,
        lr = 1e-3,
        gamma = 0.99,
        eps_start = 1.0,
        eps_end = 0.01,
        eps_decay_steps = 100_000,
        buffer_size = 100_000,
        batch_size = 64,
        target_update_freq = 1_000,
        use_double = (args.agent == "ddqn"),
        use_dueling = (args.agent == "dueling"),
        use_per = args.per,
    )


# DEFAULT ENVIRONMENT KWARGS
# default settings for the Flappy Bird environment's difficulty parameters
DEFAULT_ENV_KWARGS = dict(
    speed_min = 0.75,
    speed_max = 1.25,
    gap_min = 150,
    gap_max = 280,
    speed_period = 300,
    gap_period = 450,
)


# TRAINING FUNCTIONS
# _run_training() is the shared internal helper used by all public train functions
# Each public function just calls _run_training with the right agent type and PER flag

# internal helper that builds the agent, optionally loads a checkpoint, and starts training
def _run_training(
    agent_type: str,
    use_per: bool,
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
    label: str = ""
):
    if env_kwargs is None:
        env_kwargs = DEFAULT_ENV_KWARGS.copy()

    # build the filename based on agent type and whether PER is used
    suffix = "_per" if use_per else ""
    filename = f"flappy_dynamic_{agent_type}{suffix}.pth"
    # store the file inside the model_weights directory
    save_path = model_path or os.path.join(MODEL_WEIGHTS_DIR, filename)
    #train_env = DynamicFlappyEnv(**env_kwargs)
    train_env = FixedFlappyEnv(speed=1.25, gap_size=100)

    # lightweight args-like object so build_agent can read agent type and per flag
    class _Args:
        agent = agent_type
        per = use_per

    agent = build_agent(_Args(), train_env)

    # resume from an existing checkpoint if one is found at the save path
    if os.path.exists(save_path):
        print(f"\n[resume] Found '{save_path}' - resuming training.")
        agent.load(save_path)
    else:
        print(f"\n[fresh] No checkpoint at '{save_path}' - starting from scratch.")

    start = time.time()
    train(
        agent,
        train_env,
        total_steps = total_steps,
        save_path = save_path,
        viz_interval = 0 if no_viz else 100_000,
        viz_duration = viz_duration,
        env_kwargs = env_kwargs,
        label=label
    )
    end = time.time()
    delta = end - start
    print(f"Trained for {delta} seconds.")
    with open("times.txt", "a") as f:
        f.write(f"{label},{delta}\n")

    print(f"\n[done] {save_path} training complete.")
    train_env.close()


# train a Vanilla DQN agent
def train_dqn(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("dqn", use_per=False,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train a Vanilla DQN agent with Prioritized Experience Replay
def train_dqn_per(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("dqn", use_per=True,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train a Double DQN agent
def train_ddqn(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("ddqn", use_per=False,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train a Double DQN agent with Prioritized Experience Replay
def train_ddqn_per(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("ddqn", use_per=True,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train a Dueling DQN agent
def train_dueling(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("dueling", use_per=False,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train a Dueling DQN agent with Prioritized Experience Replay
def train_dueling_per(
    total_steps: int = 1_000_000,
    model_path: str = None,
    no_viz: bool = False,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    _run_training("dueling", use_per=True,
                  total_steps=total_steps, model_path=model_path,
                  no_viz=no_viz, viz_duration=viz_duration, env_kwargs=env_kwargs)


# train all six variants one after another
def train_all(
    total_steps: int = 1_000_000,
    no_viz: bool = True,
    viz_duration: int = 10,
    env_kwargs: Dict = None,
):
    # list of all six agent configurations with their labels
    configs = [
        ("dqn", False, "Vanilla DQN"),
        ("dqn", True, "Vanilla DQN + PER"),
        ("ddqn", False, "Double DQN"),
        ("ddqn", True, "Double DQN + PER"),
        ("dueling", False, "Dueling DQN"),
        ("dueling", True, "Dueling DQN + PER"),
    ]

    print(f"\n{'='*60}")
    print(f"  TRAIN ALL - {len(configs)} variants x {total_steps:,} steps each")
    print(f"{'='*60}\n")

    # train each variant in order, printing which one is starting
    for i, (agent_type, use_per, label) in enumerate(configs, 1):
        with open(label + ".csv", "w") as f:
            f.write("step,ep,score(50),best,loss,eps\n")
        print(f"\n{'-'*60}")
        print(f"  [{i}/{len(configs)}]  {label}")
        print(f"{'-'*60}")
        _run_training(
            agent_type = agent_type,
            use_per = use_per,
            total_steps = total_steps,
            no_viz = no_viz,
            viz_duration = viz_duration,
            env_kwargs = env_kwargs,
            label=label
        )

    print(f"\n{'='*60}")
    print("  All variants trained.")
    print(f"{'='*60}\n")


# CLI ENTRY POINT

# parse command line arguments and launch the appropriate training run
def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Flappy Bird - Training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "ddqn", "dueling"],
        default="ddqn",
        help=(
            "dqn     -> Vanilla DQN\n"
            "ddqn    -> Double DQN\n"
            "dueling -> Dueling DQN"
        ),
    )
    parser.add_argument("--per", action="store_true", help="Add Prioritized Experience Replay to selected agent")
    parser.add_argument("--train-all", action="store_true", help="Train all 6 variants sequentially")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path (ignored with --train-all)")
    parser.add_argument("--no-viz", action="store_true", help="Disable periodic visualisation windows")
    parser.add_argument("--viz-duration", type=int, default=10)
    parser.add_argument("--speed-min", type=float, default=0.75)
    parser.add_argument("--speed-max", type=float, default=1.25)
    parser.add_argument("--gap-min", type=int, default=150)
    parser.add_argument("--gap-max", type=int, default=280)
    parser.add_argument("--speed-period", type=int, default=300)
    parser.add_argument("--gap-period", type=int, default=450)
    args = parser.parse_args()

    # collect environment settings into a dict to pass along
    env_kwargs = dict(
        speed_min = args.speed_min,
        speed_max = args.speed_max,
        gap_min = args.gap_min,
        gap_max = args.gap_max,
        speed_period = args.speed_period,
        gap_period = args.gap_period,
    )

    # run all variants if --train-all was passed, otherwise run just the selected one
    if args.train_all:
        train_all(
            total_steps = args.steps,
            no_viz = args.no_viz,
            viz_duration = args.viz_duration,
            env_kwargs = env_kwargs,
        )
    else:
        _run_training(
            agent_type = args.agent,
            use_per = args.per,
            total_steps = args.steps,
            model_path = args.model,
            no_viz = args.no_viz,
            viz_duration = args.viz_duration,
            env_kwargs = env_kwargs,
        )


if __name__ == "__main__":
    main()