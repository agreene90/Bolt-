import torch
import torch.nn as nn

class EEGPreprocessingLayer(nn.Module):
    def __init__(self):
        super(EEGPreprocessingLayer, self).__init__()
        self.filter = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(64)

    def forward(self, eeg_data):
        processed_eeg = self.filter(eeg_data)
        processed_eeg = self.batch_norm(processed_eeg)
        return processed_eeg
