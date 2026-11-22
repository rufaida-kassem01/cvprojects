from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import resnet18
import torch.nn as nn


CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "resnet18_cifar10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource

def load_model_for_inference(checkpoint_path, device='cpu'):
    # Load the deployment checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct the modified ResNet-18 architecture
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Recreate the exact validation/testing transformation using the saved parameters
    mean = checkpoint['normalization']['mean']
    std = checkpoint['normalization']['std']
    inference_transform = transforms.Compose([
        transforms.Resize(checkpoint['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    class_names = checkpoint['class_names']
    
    print(f"Successfully loaded model with {len(class_names)} classes. Saved Accuracy: {checkpoint.get('best_accuracy', 0.0):.2f}%")
    # print model information
    print(f"Model architecture: {model}")
    # print class names
    print(f"Class names: {class_names}")
    return model, class_names, inference_transform

# Verification block to test loading the model
loaded_model, labels, transform = load_model_for_inference(CHECKPOINT, DEVICE)