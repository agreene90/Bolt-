# Bolt: Real-Time EEG Processing and Reinforcement Learning with Docker and Hardware Integration

## Overview
Bolt is a fully-implemented system designed to process real-time EEG signals using Spiking Neural Networks (SNN), multi-head attention, and reinforcement learning. The system is optimized for GPU execution and can be deployed on EEG wearable devices through hardware interfaces.

## Features
- **EEG Preprocessing**: Real-time filtering and normalization of EEG signals.
- **Spiking Neural Network**: Processes time-based EEG signals.
- **Multi-Head Attention**: Enhances temporal patterns in EEG signals.
- **Reinforcement Learning Agent**: Learns actions based on EEG states.
- **Docker Support**: Full containerization for running the project with PyTorch and CUDA support.
- **MicroPython Integration**: Interfacing with EEG hardware (e.g., wearable EEG glasses).

## Requirements
1. Docker
2. GPU-enabled system with CUDA support
3. Python 3.7+
4. PyTorch with CUDA support
5. MicroPython for hardware interaction

## Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/HermiTech-LLC/Bolt.git
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
Developed by HermiTech.
