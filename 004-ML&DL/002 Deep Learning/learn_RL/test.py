

# %%
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import cv2


# %% [markdown]
# ## 定义DQN网络
# 
# 我们使用一个简单的卷积神经网络（CNN）来估计状态-动作值函数（Q值）。这个Q值函数将帮助智能体选择最好的动作。

# %%
from torch import nn 


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %% [markdown]
# 解释：
# - 卷积层（Conv2d）：提取游戏画面的特征。
# - 全连接层（Fully connected layer）：最后一层输出Q值，用于每个动作的价值评估。

# %% [markdown]
# ##  经验回放（Replay Buffer）
# 
# 强化学习中的经验回放是为了避免在训练时对连续数据的过度拟合。我们将通过经验回放来存储智能体的经验，并随机抽取这些经验来训练网络。

# %%
import random


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# %% [markdown]
# ## 定义智能体（Agent）
# 
# 智能体将与环境进行交互，并通过DQN来进行决策。它将包括epsilon-greedy策略（平衡探索与利用）。

# %%
from torch import optim 
from torch import nn 
import torch 


class Agent:
    def __init__(self, env, model, target_model, buffer, epsilon=1.0, gamma=0.99, lr=0.0001, batch_size=64):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.buffer = buffer
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Exploration: random action
        else:
            state = torch.tensor(state).unsqueeze(0).float()  # Convert to tensor
            q_values = self.model(state)
            return torch.argmax(q_values).item()  # Exploitation: best action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).float()
        next_states = torch.tensor(next_states).float()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # Compute target Q-values using the Bellman equation
        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# %% [markdown]
# ## 训练循环
# 我们将通过与游戏环境的交互来训练智能体。

# %%
import gym
import flappy_bird_gym


def train():
    env = gym.make('FlappyBird-v0')  # 假设环境已经被包装为gym环境
    input_shape = (3, 84, 84)  # 输入形状（3个通道，84x84的图像）
    num_actions = env.action_space.n

    model = DQN(input_shape, num_actions)
    target_model = DQN(input_shape, num_actions)
    agent = Agent(env, model, target_model, ReplayBuffer(10000))

    episodes = 1000
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(episodes):
        state = env.reset()
        state = cv2.resize(state, (84, 84))  # 调整图像大小
        state = np.transpose(state, (2, 0, 1))  # 调整图像通道顺序
        state = torch.tensor(state).unsqueeze(0).float()
        total_reward = 0

        for t in range(10000):  # 每个回合的最大时间步数
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            next_state = cv2.resize(next_state, (84, 84))
            next_state = np.transpose(next_state, (2, 0, 1))

            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        if agent.epsilon > epsilon_min:
            agent.epsilon *= epsilon_decay

        # Update target network every few episodes
        if episode % 10 == 0:
            agent.update_target()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()


# %% [markdown]
# 解释：
# - 训练循环：智能体与环境交互，收集经验，并通过update方法优化网络。
# - epsilon衰减：每回合结束后逐渐减少探索概率（epsilon）。
# - 目标网络更新：每隔一段时间将目标网络更新为当前网络，以稳定训练。

# %%
train()


