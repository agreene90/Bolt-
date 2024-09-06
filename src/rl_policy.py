import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class RLPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(RLPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Regularization
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class RLAgent:
    def __init__(self, policy_network, learning_rate=1e-3, gamma=0.99):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        
        # Experience Replay Buffer
        self.replay_buffer = []

    def select_action(self, state):
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action

    def store_transition(self, reward):
        # Store reward in buffer
        self.rewards.append(reward)

    def update_policy(self):
        # Discounted rewards calculation
        discounted_rewards = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)

        discounted_rewards = torch.tensor(discounted_rewards).to('cuda')
        log_probs = torch.cat(self.log_probs)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Compute loss
        loss = -log_probs * discounted_rewards
        loss = loss.sum()

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear stored rewards and log probabilities after update
        self.rewards.clear()
        self.log_probs.clear()

    def optimize_model(self):
        if len(self.replay_buffer) > 100:  # Only optimize after buffer is sufficiently filled
            batch = random.sample(self.replay_buffer, 32)  # Sample a batch from the buffer
            # Process the batch
            # (Add logic here to train on batch if experience replay is used)

    def save_model(self, path="policy_model.pth"):
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path="policy_model.pth"):
        self.policy_network.load_state_dict(torch.load(path))