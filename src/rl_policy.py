import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class RLPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(RLPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class RLAgent:
    def __init__(self, policy_network, learning_rate=1e-3):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

    def update_policy(self, log_probs, rewards):
        discounted_rewards = []
        G = 0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            discounted_rewards.insert(0, G)

        discounted_rewards = torch.tensor(discounted_rewards).to('cuda')
        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)

        loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()