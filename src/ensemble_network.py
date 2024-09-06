import torch
from eeg_preprocessing import EEGPreprocessingLayer
from snn_layer import SNNLayer
from attention_layer import MultiHeadAttentionLayer
from rl_policy import RLPolicyNetwork, RLAgent

class EnsembleNetwork(torch.nn.Module):
    def __init__(self, input_size, snn_hidden_size, attention_embed_size, attention_heads, rl_state_size, rl_action_size):
        super(EnsembleNetwork, self).__init__()
        self.preprocess = EEGPreprocessingLayer()
        self.snn = SNNLayer(input_size, snn_hidden_size, attention_embed_size).to('cuda')
        self.attention = MultiHeadAttentionLayer(attention_embed_size, attention_heads).to('cuda')
        self.policy_net = RLPolicyNetwork(rl_state_size, rl_action_size).to('cuda')
        self.rl_agent = RLAgent(self.policy_net)

    def forward(self, eeg_data, state, task_goal):
        processed_eeg = self.preprocess(eeg_data).to('cuda')
        snn_output = self.snn(processed_eeg)
        attention_output = self.attention(snn_output, snn_output, snn_output)
        action, log_prob = self.rl_agent.select_action(state)
        reward = abs(torch.mean(eeg_data) - task_goal).item()  # Reward function
        self.rl_agent.update_policy([log_prob], [reward])
        return attention_output, action
