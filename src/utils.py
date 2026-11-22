from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(checkpoint_path), map_location=device)


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_inference_transform(checkpoint: dict) -> transforms.Compose:
    normalization = checkpoint["normalization"]
    return transforms.Compose(
        [
            transforms.Resize(checkpoint["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                normalization["mean"],
                normalization["std"],
            ),
        ]
    )


def preprocess_image(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        probabilities = model(image_tensor.to(device)).softmax(dim=1)[0]

    scores, indices = probabilities.topk(min(top_k, len(class_names)))
    return [
        (class_names[index.item()], score.item())
        for score, index in zip(scores, indices)
    ]

