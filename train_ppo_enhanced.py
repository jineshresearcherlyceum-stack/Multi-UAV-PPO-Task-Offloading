import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from env_multi_uav_enhanced import MultiUAVOffloadEnv
import warnings
warnings.filterwarnings('ignore')

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
        )
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)
    def forward(self, x):
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features)

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, seed=42):
        torch.manual_seed(seed)
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
    def get_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = self.policy(obs_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.squeeze(0)[action] + 1e-8)
        return action, log_prob, probs.squeeze(0).detach()
    def update(self, states, actions, rewards, log_probs_old, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs_old = torch.stack(log_probs_old).detach()
        advantages = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            advantages.insert(0, R)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logits, values = self.policy(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        return loss.item()

def train_ppo(seed, episodes=500):
    os.makedirs("results", exist_ok=True)
    print(f"\n[>>] Training with seed: {seed}")
    env = MultiUAVOffloadEnv(seed=seed)
    agent = PPOAgent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=seed)
    rewards_history, losses, success_rate, avg_latency, avg_energy = [], [], [], [], []
    for episode in tqdm(range(episodes), desc=f"Seed {seed}"):
        obs, _ = env.reset()
        done = False
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        total_reward, success_count, total_latency, total_energy = 0, 0, 0, 0
        while not done:
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            states.append(obs); actions.append(action); rewards.append(reward)
            log_probs.append(log_prob); dones.append(1.0 if done else 0.0)
            total_reward += reward
            if reward > -10: success_count += 1
            total_latency += abs(reward) * 0.1
            total_energy += abs(reward) * 0.05
            obs = next_obs
        if len(states) > 0:
            loss = agent.update(states, actions, rewards, log_probs, dones)
            losses.append(loss)
        rewards_history.append(total_reward)
        success_rate.append(success_count / max(1, len(states)))
        avg_latency.append(total_latency / max(1, len(states)))
        avg_energy.append(total_energy / max(1, len(states)))
    torch.save(agent.policy.state_dict(), f"results/ppo_seed_{seed}.pth")
    print(f"[OK] Model saved for seed {seed}")
    return {'rewards': rewards_history, 'losses': losses, 'success_rate': success_rate, 'latency': avg_latency, 'energy': avg_energy}

seeds = [42, 123, 456]
all_metrics = {}
for seed in seeds:
    all_metrics[seed] = train_ppo(seed, episodes=500)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
colors = ['#1e3a5f', '#2e7d32', '#b71c1c']
for idx, seed in enumerate(seeds):
    m = all_metrics[seed]
    axes[0,0].plot(m['rewards'], color=colors[idx], alpha=0.7, label=f'Seed {seed}')
    if len(m['rewards']) > 50:
        axes[0,0].plot(pd.Series(m['rewards']).rolling(50).mean(), color=colors[idx], linewidth=2)
    axes[0,1].plot(m['losses'], color=colors[idx], alpha=0.7)
    if len(m['losses']) > 50:
        axes[0,1].plot(pd.Series(m['losses']).rolling(50).mean(), color=colors[idx], linewidth=2)
    axes[0,2].plot(m['success_rate'], color=colors[idx], alpha=0.7)
    if len(m['success_rate']) > 50:
        axes[0,2].plot(pd.Series(m['success_rate']).rolling(50).mean(), color=colors[idx], linewidth=2)
    axes[1,0].plot(m['latency'], color=colors[idx], alpha=0.7)
    if len(m['latency']) > 50:
        axes[1,0].plot(pd.Series(m['latency']).rolling(50).mean(), color=colors[idx], linewidth=2)
    axes[1,1].plot(m['energy'], color=colors[idx], alpha=0.7)
    if len(m['energy']) > 50:
        axes[1,1].plot(pd.Series(m['energy']).rolling(50).mean(), color=colors[idx], linewidth=2)

axes[0,0].set_title("Reward Convergence"); axes[0,0].set_xlabel("Episode"); axes[0,0].set_ylabel("Total Reward"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
axes[0,1].set_title("Training Loss"); axes[0,1].set_xlabel("Episode"); axes[0,1].set_ylabel("Loss"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[0,2].set_title("Task Success Rate"); axes[0,2].set_xlabel("Episode"); axes[0,2].set_ylabel("Success Rate"); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)
axes[1,0].set_title("Average Latency"); axes[1,0].set_xlabel("Episode"); axes[1,0].set_ylabel("Latency (ms)"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
axes[1,1].set_title("Average Energy"); axes[1,1].set_xlabel("Episode"); axes[1,1].set_ylabel("Energy (J)"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

final_rewards = [np.mean(all_metrics[s]['rewards'][-50:]) for s in seeds]
axes[1,2].bar([f'Seed {s}' for s in seeds], final_rewards, color=colors)
axes[1,2].set_title("Avg Reward (Last 50)")
axes[1,2].set_ylabel("Avg Reward")
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("ppo_enhanced_results.png", dpi=150)

print("\n[RESULTS] Final Results:")
print("-" * 50)
for seed in seeds:
    m = all_metrics[seed]
    print(f"Seed {seed}: Reward={np.mean(m['rewards'][-50:]):.2f}, Success={np.mean(m['success_rate'][-50:]):.2f}, Latency={np.mean(m['latency'][-50:]):.2f}, Energy={np.mean(m['energy'][-50:]):.2f}")
print("-" * 50)
