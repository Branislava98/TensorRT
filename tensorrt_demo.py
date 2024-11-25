import tensorrt as trt
# print(trt.__version__)
import torch, torchvision
import numpy as np
import time
import os
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
import math
from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore")

def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host

def evaluate_trt(engine_path, image, dataset):
    trt_inference = TensorRTInference(engine_path)

    if dataset == "disastereye":
      labels = ['Conflict', 'Fire', 'Flood', 'Landslide', 'Mudslide', 'Normal', 'Post Earthquake', 'Traffic Accident']
    elif dataset == "dfan":
      labels = ['Boad Fire', 'Building Fire', 'Bus Fire', 'Car Fire', 'Cargo Fire', 'Electric Fire', 'Forest Fire', 'Non Fire', 'PickUp Fire', 'SUV Fire', 'Train Fire', 'Van Fire']
    else:
      labels = ['Collapsed Building', 'Fire', 'Flood', 'Normal', 'Traffic Accident']
    labels = labels

    start_time = time.time()
    image = image.numpy()

    # Run inference
    output = trt_inference.infer(image)
    predicted = np.argmax(output)
    duration = time.time() - start_time

    print("Prediction: {}, Time: {:.2f}ms, FPS: {:.2f}".format(labels[predicted], (duration * 1000), (1 / duration)))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_pct = 0.9
transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

image_paths = ['./images/DisasterEye_flood.png', './images/DFAN_forest_fire.jpg', './images/AIDER_traffic_accident.jpg']
engine_paths = ["./models/disastereye.trt", "./models/dfan.trt", "./models/aider.trt"]
dataset = ["disastereye", "dfan", "aider"]

for i in range(3):
  print("Dataset:", dataset[i])
  img = Image.open(image_paths[i]).convert('RGB')
  img_transformed = transform(img)
  img_transformed = img_transformed.unsqueeze(0)

  evaluate_trt(engine_paths[i], img_transformed, dataset[i])
  model_size = os.path.getsize(engine_paths[i])
  print("Model size: {:.2f}MB".format(model_size / 1e6))
  print()
