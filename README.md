# Bolt: Real-Time EEG Processing and Reinforcement Learning with Docker and Hardware Integration
___
![img](https://github.com/agreene90/Bolt-/blob/main/Boltframe.jpg)
___

## Overview
Bolt is a fully-implemented system designed to process real-time EEG signals using Spiking Neural Networks (SNN), multi-head attention, and reinforcement learning. The system is optimized for GPU execution and can be deployed on EEG wearable devices through hardware interfaces.

## Features
- **Advanced EEG Preprocessing**: Real-time filtering, noise reduction, and band-pass filtering of EEG signals.
- **Spiking Neural Network (SNN)**: Processes time-based EEG signals with biologically inspired neuron models.
- **Multi-Head Attention**: Enhances temporal patterns in EEG signals with multi-scale attention.
- **Reinforcement Learning Agent**: Learns and performs actions based on EEG states with Proximal Policy Optimization (PPO) and experience replay.
- **Ensemble Learning**: Combines multiple models to improve accuracy and robustness.
- **FPGA Integration**: Processes EEG data with hardware-level filtering and signal normalization.
- **Docker Support**: Full containerization for running the project with PyTorch, CUDA, and MicroPython.
- **MicroPython Integration**: Interfacing with EEG hardware (e.g., wearable EEG glasses) for real-time communication and control.

## Requirements
1. Docker
2. GPU-enabled system with CUDA support
3. Python 3.7+
4. PyTorch with CUDA support
5. MicroPython for hardware interaction

## Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/agreene90/Bolt.git
cd Bolt
```

### 2. Build the Docker Container
```bash
docker build -t bolt-eeg .
```

### 3. Run the Docker Container
```bash
docker run --gpus all bolt-eeg
```

## Project Structure

```
Bolt/
│
├── README.md
│
├── docker/
│   └── Dockerfile
│
├── embedded/
│   ├── init.py
│   └── eeg_sensor_interface.py
│
├── src/
│   ├── eeg_preprocessing.py
│   ├── snn_layer.py
│   ├── attention_layer.py
│   ├── rl_policy.py
│   └── ensemble_network.py
│
└── main.py
```

## Authors and Contributors
@LoQiseaking69
Developed by HermiTech.
