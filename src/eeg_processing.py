import torch
import torch.nn as nn

class EEGPreprocessingLayer(nn.Module):
    def __init__(self):
        super(EEGPreprocessingLayer, self).__init__()
        
        # Layer to remove noise from the EEG signal
        self.noise_reduction = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Band-pass filter to focus on EEG frequency bands of interest (e.g., alpha, beta)
        self.bandpass_filter = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        
        # Batch normalization for stabilizing the activations
        self.batch_norm = nn.BatchNorm1d(64)
        
        # Final filtering layer to fine-tune the signal
        self.final_filter = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, eeg_data):
        # Apply noise reduction
        eeg_data = self.noise_reduction(eeg_data)
        
        # Apply band-pass filter to focus on specific brainwave frequencies
        eeg_data = self.bandpass_filter(eeg_data)
        
        # Normalize the signal using batch normalization
        eeg_data = self.batch_norm(eeg_data)
        
        # Apply the final filter to clean up the data further
        processed_eeg = self.final_filter(eeg_data)
        
        return processed_eeg