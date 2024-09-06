import torch
from src.ensemble_network import EnsembleNetwork

def main():
    eeg_data = torch.randn(32, 100, 64).to('cuda')  # Simulated EEG data
    initial_state = torch.randn(32, 128).to('cuda')  # Simulated initial state
    task_goal = 0.85  # Task goal for EEG processing

    network = EnsembleNetwork(input_size=64, snn_hidden_size=128, attention_embed_size=256, attention_heads=8, rl_state_size=128, rl_action_size=10)
    output, action = network(eeg_data, initial_state, task_goal)

    print(f"Processed Output: {output}")
    print(f"Selected Action: {action}")

if __name__ == "__main__":
    main()