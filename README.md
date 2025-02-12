# Disaster Management Project

This project focuses on real-time disaster management using UAV-assisted edge computing. It leverages an optimized Swin Transformer for aerial image classification, integrating advanced preprocessing and quantization techniques to ensure efficient deployment on resource-constrained devices. Moreover, to tackle the relevance of the optimized model in real-world scenarios, we introduce a novel database of disaster cases taken by UAVs and individuals on sight. The dataset can be found here: 

The following guidelines will help you install TensorRT on your device. If you would like a quick test, you can just run our [Google Colab Notebook](https://github.com/Branislava98/TensorRT/blob/main/TensortRT.ipynb) (do not forget to update paths for images and models).

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
git clone https://github.com/Branislava98/TensorRT.git
cd TensorRT
```
### Step 2: Create a Virtual Environment
```bash
python3 -m venv disaster_env
source disaster_env/bin/activate
```
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Install TensorRT

[Download](https://developer.nvidia.com/tensorrt) the TensorRT local repo file that matches the Ubuntu version and CPU architecture that you are using.

Install TensorRT from the Debian local repo package. Replace ubuntuxx04, 10.x.x, and cuda-x.x with your specific OS, TensorRT, and CUDA versions. For ARM SBSA and JetPack users, replace amd64 with arm64. JetPack users also need to replace `nv-tensorrt-local-repo`
 with `nv-tensorrt-local-tegra-repo`.

```bash
os="ubuntuxx04"
tag="10.x.x-cuda-x.x"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-key add /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg
sudo apt-get update
```

Ubuntu will install TensorRT for the latest CUDA version by default when using the CUDA network repository. The following commands will install tensorrt and related TensorRT packages for an older CUDA version and hold these packages at this version. Replace 10.x.x.x with your version of TensorRT and cudax.x with your CUDA version for your installation.

```bash
version="10.x.x.x-1+cudax.x"
sudo apt-get install libnvinfer-bin=${version} libnvinfer-dev=${version} libnvinfer-dispatch-dev=${version} libnvinfer-dispatch10=${version} libnvinfer-headers-dev=${version} libnvinfer-headers-plugin-dev=${version} libnvinfer-lean-dev=${version} libnvinfer-lean10=${version} libnvinfer-plugin-dev=${version} libnvinfer-plugin10=${version} libnvinfer-samples=${version} libnvinfer-vc-plugin-dev=${version} libnvinfer-vc-plugin10=${version} libnvinfer10=${version} libnvonnxparsers-dev=${version} libnvonnxparsers10=${version} python3-libnvinfer-dev=${version} python3-libnvinfer-dispatch=${version} python3-libnvinfer-lean=${version} python3-libnvinfer=${version} tensorrt-dev=${version} tensorrt-libs=${version} tensorrt=${version}

sudo apt-mark hold libnvinfer-bin libnvinfer-dev libnvinfer-dispatch-dev libnvinfer-dispatch10 libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvinfer-lean-dev libnvinfer-lean10 libnvinfer-plugin-dev libnvinfer-plugin10 libnvinfer-samples libnvinfer-vc-plugin-dev libnvinfer-vc-plugin10 libnvinfer10 libnvonnxparsers-dev libnvonnxparsers10 python3-libnvinfer-dev python3-libnvinfer-dispatch python3-libnvinfer-lean python3-libnvinfer tensorrt-dev tensorrt-libs tensorrt
```
```bash
dpkg -l | grep TensorRT
```
Locate your TensorRT files and copy `tensorrt.so`, `tensorrt_lean.so` and `tensorrt_dispatch.so` from `/usr/lib/python$PYTHON_VERSION/dist-packages/` to `/usr/local/lib/python$PYTHON_VERSION/dist-packages/`.

```bash
sudo find / -name "*tensorrt*"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
sudo cp -r /usr/lib/python$PYTHON_VERSION/dist-packages/tensorrt /usr/local/lib/python$PYTHON_VERSION/dist-packages/
sudo cp -r /usr/lib/python$PYTHON_VERSION/dist-packages/tensorrt_lean /usr/local/lib/python$PYTHON_VERSION/dist-packages/
sudo cp -r /usr/lib/python$PYTHON_VERSION/dist-packages/tensorrt_dispatch /usr/local/lib/python$PYTHON_VERSION/dist-packages/
```
### Step 5: Download pre-trained models

Download pre-trained models [disastereye](https://drive.google.com/file/d/1c75OmjyS5bLFso2nZ4aeoLCdq_5pRN4p/view?usp=sharing), [dfan](https://drive.google.com/file/d/1yzFPfQRS85Vl2fLvXVXx_g0Lsl1TtbuS/view?usp=sharing), [aider](https://drive.google.com/file/d/1CR_Hbk4kaPymMoWAUlst2DlfJOwPN2bD/view?usp=sharing) and put them into models directory.

Run the following commands to create TensorRT engines:
```bash
/usr/src/tensorrt/bin/trtexec --int8 --onnx=./models/disastereye.onnx  --saveEngine=./models/disastereye.trt
/usr/src/tensorrt/bin/trtexec --int8 --onnx=./models/dfan.onnx  --saveEngine=./models/dfan.trt
/usr/src/tensorrt/bin/trtexec --int8 --onnx=./models/aider.onnx  --saveEngine=./models/aider.trt
```
### Step 6: Run code to measure FPS on your device
```bash
python tensorrt_demo.py
```



