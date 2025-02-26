# -*- coding: utf-8 -*-
"""ASL-Test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gqvfi9ots3-nAE07Yo1AzjZWbIjyB_S8
"""

import os
import zipfile

dataset_path = "/content/American Sign Language Letters.v6-raw.yolov8.zip"  # Update with your actual file name
extract_path = "/content/dataset"

# Extract the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully!")

pip install ultralytics streamlit opencv-python-headless torch torchvision

with open("/content/dataset/data.yaml", "r") as f:
    data_yaml_content = f.read()

print("✅ Current data.yaml contents:\n")
print(data_yaml_content)

from ultralytics import YOLO

# Load YOLOv8 model (use yolov8n.pt for speed, yolov8s.pt for better accuracy)
model = YOLO("yolov8n.pt")

# Start training
model.train(data="/content/dataset/data.yaml", epochs=50, imgsz=224, batch=16)

print("✅ Training Complete! Model is saved.")

from google.colab import files
files.download("/content/runs/detect/train/weights/best.pt")

model.export(format="onnx")       # For fast inference with OpenCV, TensorRT, etc.
model.export(format="torchscript") # For mobile or edge deployment