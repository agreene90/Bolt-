import torch
import torch.nn as nn
import snntorch as snn

class SNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.9):
        super(SNNLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta).to('cuda')
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta).to('cuda')

    def forward(self, x):
        spikes = snn.spikegen.rate(x).to('cuda')
        mem1, spk1 = self.lif1(self.fc1(spikes))
        mem2, spk2 = self.lif2(self.fc2(spk1))
        return spk2
