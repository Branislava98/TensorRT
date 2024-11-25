# Disaster Management Project

This project focuses on real-time disaster management using UAV-assisted edge computing. It leverages an optimized Swin Transformer for aerial image classification, integrating advanced preprocessing and quantization techniques to ensure efficient deployment on resource-constrained devices.

## Features
- **Real-Time Processing:** Optimized Swin Transformer using TensorRT INT8 and FP16 precision for low-latency inference.
- **Novel Dataset:** DisasterEye dataset featuring UAV-captured disaster scenes and ground-level images for realistic scenarios.
- **Scalability:** Designed for deployment on hardware-limited platforms like drones.

## Requirements
- Python 3.10.12
- TensorRT 10.2.0
- PyTorch 2.5.1+cu121
- Torchvision 0.20.1+cu121
- NumPy 1.26.4
- PyCUDA 12.2.0
- Pillow 11.0.0

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/disaster-management.git
cd disaster-management
