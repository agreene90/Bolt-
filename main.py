import torch
import time
from eeg_processing import EEGPreprocessingLayer
from snn_layer import SNNLayer
from rl_policy import RLAgent, RLPolicyNetwork
from hardware_interface import GlassesHardwareInit, EEGSensorInterface

# Initialize hardware
hardware = GlassesHardwareInit()
eeg_interface = EEGSensorInterface()

# Initialize neural networks
eeg_preprocessor = EEGPreprocessingLayer().to('cuda')
snn = SNNLayer(input_size=64, hidden_size=128, output_size=10, beta=0.9).to('cuda')
policy_network = RLPolicyNetwork(state_size=10, action_size=4).to('cuda')
rl_agent = RLAgent(policy_network)

# Power on the hardware
hardware.power_on()

try:
    # Main loop for real-time EEG data processing and control
    while True:
        # Step 1: Read EEG data
        eeg_data = eeg_interface.read_eeg_data()
        if eeg_data is not None:
            # Step 2: Preprocess the EEG data
            eeg_data_tensor = torch.tensor([eeg_data], dtype=torch.float32).unsqueeze(0).to('cuda')
            preprocessed_data = eeg_preprocessor(eeg_data_tensor)
            
            # Step 3: Process data through SNN (Spiking Neural Network)
            snn_output = snn(preprocessed_data)
            
            # Step 4: Select an action using the RL agent
            action = rl_agent.select_action(snn_output)
            
            # Step 5: Execute the action (e.g., send command to external device)
            hardware.bluetooth_send(action)
            
            # Optional: Visualize the EEG data or system state
            # Visualization code can be added here using OpenCV or matplotlib
            
        # Sleep briefly to maintain real-time performance
        time.sleep(0.1)

except KeyboardInterrupt:
    # Power off the hardware when stopping the system
    hardware.power_off()
    print("System powered down.")