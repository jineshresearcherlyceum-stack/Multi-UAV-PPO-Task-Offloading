
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from scipy.stats import rayleigh
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 56 * 56, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

class MultiUAVOffloadEnv(gym.Env):
    def __init__(self, num_uavs=3, num_mec=2, seed=42):
        super().__init__()
        self.num_uavs = num_uavs
        self.num_mec = num_mec
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.max_steps = 100
        self.energy_init = 100.0
        self.F_max = 10.0
        self.B = 20e6
        self.p_n = 0.1
        self.sigma2 = 1e-10
        self.uav_positions = None
        self.mec_positions = None
        self.energy = None
        self.tasks = None
        self.images = None
        self.step_count = 0
        obs_dim = self.num_uavs * 3 + self.num_uavs + self.num_uavs * 4 + self.num_uavs * self.num_mec
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_mec + 2)
        self.cnn = CNNFeatureExtractor()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.reset()
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.uav_positions = np.random.rand(self.num_uavs, 3) * np.array([100, 100, 50])
        self.mec_positions = np.random.rand(self.num_mec, 3) * np.array([100, 100, 10])
        self.energy = np.ones(self.num_uavs) * self.energy_init
        self.tasks = []
        self.images = []
        for _ in range(self.num_uavs):
            self.tasks.append({'img_size': random.randint(50, 200), 'cpu_cycles': random.randint(100, 500), 'deadline': random.randint(80, 120)})
            self.images.append(Image.new('RGB', (224, 224), color='blue'))
        return self._get_obs(), {}
    def _get_channel_gain(self, distance):
        path_loss = 20 * np.log10(4 * np.pi * 2.4e9 * max(distance, 1) / 3e8)
        fading = rayleigh.rvs(scale=1.0, size=1)[0]
        return 10 ** (-path_loss / 10) * fading ** 2
    def _compute_snr(self, gain):
        return 10 * np.log10(self.p_n * gain / self.sigma2)
    def _compute_image_complexity(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image.resize((64, 64)).convert('L'))
        else:
            img_array = np.array(image)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 255))
        hist = hist / (hist.sum() + 1e-8)
        return -np.sum(hist * np.log2(hist + 1e-8)) / 8.0
    def _get_obs(self):
        obs = []
        for i in range(self.num_uavs):
            obs.extend(self.uav_positions[i] / 100)
        obs.extend(self.energy / self.energy_init)
        for i, t in enumerate(self.tasks):
            obs.append(t['img_size'] / 200)
            obs.append(t['cpu_cycles'] / 500)
            obs.append(t['deadline'] / 120)
            obs.append(self._compute_image_complexity(self.images[i]))
        for i in range(self.num_uavs):
            for j in range(self.num_mec):
                dist = np.linalg.norm(self.uav_positions[i] - self.mec_positions[j])
                obs.append(self._get_channel_gain(dist) / 1e-6)
        return np.array(obs, dtype=np.float32)
    def step(self, action):
        self.step_count += 1
        uav_idx = self.step_count % self.num_uavs
        task = self.tasks[uav_idx]
        done = False
        reward = 0
        dists = [np.linalg.norm(self.uav_positions[uav_idx] - self.mec_positions[j]) for j in range(self.num_mec)]
        gains = [self._get_channel_gain(d) for d in dists]
        snrs = [self._compute_snr(g) for g in gains]
        if action == 0:
            energy_cost = task['cpu_cycles'] * 0.002
            latency = task['cpu_cycles'] / 500.0
            detection_accuracy = 0.6
        elif action <= self.num_mec:
            mec_idx = action - 1
            energy_cost = task['img_size'] * 0.02 + dists[mec_idx] * 0.2
            latency = dists[mec_idx] / 20.0 + task['cpu_cycles'] / 2000.0
            detection_accuracy = 0.85 if snrs[mec_idx] > 10 else 0.7
        else:
            energy_cost = task['img_size'] * 0.05 + min(dists) * 0.3
            latency = min(dists) / 15.0 + task['cpu_cycles'] / 3000.0
            complexity = self._compute_image_complexity(self.images[uav_idx])
            detection_accuracy = 0.9 - 0.3 * complexity
        f_det = 2.0
        f_off = min(8.0, self.F_max - f_det)
        if action > 0 and f_det + f_off > self.F_max:
            reward -= 10
        self.energy[uav_idx] -= energy_cost
        reward -= 0.1 * energy_cost + 0.05 * latency + 2.0 * detection_accuracy
        if latency > task['deadline']:
            reward -= 10
        if self.energy[uav_idx] <= 0:
            done = True
            reward -= 50
        if self.step_count >= self.max_steps:
            done = True
        self.tasks[uav_idx] = {'img_size': random.randint(50, 200), 'cpu_cycles': random.randint(100, 500), 'deadline': random.randint(80, 120)}
        self.images[uav_idx] = Image.new('RGB', (224, 224), color='green')
        return self._get_obs(), reward, done, False, {}
    def render(self):
        pass
