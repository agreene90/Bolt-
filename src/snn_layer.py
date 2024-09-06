import torch
import torch.nn as nn
import snntorch as snn

class SNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.9):
        super(SNNLayer, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Leaky integrate-and-fire neuron model with STDP for adaptive learning
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True).to('cuda')
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Second Leaky integrate-and-fire neuron
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True).to('cuda')
        
        # STDP mechanism for learning neuron-to-neuron spikes
        self.stdp = snn.STDP(self.fc1, snn.optim.STDPConfig(learning_rate=1e-3), snn.optim.STDPConfig(learning_rate=1e-3))
        
    def forward(self, x):
        # Generate spikes from the input signal (rate encoding)
        spikes = snn.spikegen.rate(x).to('cuda')
        
        # Pass through the first layer and the first LIF neuron
        mem1, spk1 = self.lif1(self.fc1(spikes))
        
        # Apply STDP for plasticity
        self.stdp(spk1, x)
        
        # Pass through the second layer and the second LIF neuron
        mem2, spk2 = self.lif2(self.fc2(spk1))
        
        return spk2

    def reset(self):
        # Reset membrane potentials after each forward pass (for real-time applications)
        self.lif1.reset()
        self.lif2.reset()