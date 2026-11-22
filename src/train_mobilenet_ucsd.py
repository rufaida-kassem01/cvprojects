"""Fine-tune a torchvision MobileNet on sampled frames from the UCSD traffic clips."""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


CLASS_NAMES = ("heavy", "light", "medium")


class VideoFrameDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, frame_index, label_index = self.samples[index]
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        return self.transform(image), label_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/ucsd/manifest.csv")
    parser.add_argument("--video-dir", default="data/ucsd")
    parser.add_argument("--output", default="outputs/mobilenet_ucsd.pth")
    parser.add_argument("--metrics", default="eval/mobilenet_ucsd.json")
    parser.add_argument("--architecture", choices=("mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"),
                        default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def read_clips(manifest_path, video_dir):
    clips = {name: [] for name in CLASS_NAMES}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["traffic_class"]
            video_path = video_dir / row["file"]
            if label in clips and video_path.is_file():
                clips[label].append(video_path)
    return clips


def sample_clip_frames(video_path, count, rng):
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if total <= 1:
        raise RuntimeError(f"Video has too few frames: {video_path}")
    start = 1
    usable = list(range(start, total))
    if len(usable) <= count:
        return usable
    positions = [round(i * (len(usable) - 1) / (count - 1)) for i in range(count)]
    indices = [usable[position] for position in positions]
    rng.shuffle(indices)
    return indices


def make_model(architecture, num_classes):
    if architecture == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def evaluate_clips(model, clips, transform, class_to_index, device):
    model.eval()
    clip_results = []
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    frame_correct = 0
    frame_total = 0
    for true_name in CLASS_NAMES:
        true_index = class_to_index[true_name]
        for video_path in clips:
            if clips[video_path] != true_name:
                continue
            probabilities = []
            capture = cv2.VideoCapture(str(video_path))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            positions = [1 + round(i * (total - 2) / max(15 - 1, 1)) for i in range(min(15, max(total - 1, 1)))]
            for frame_index in positions:
                capture = cv2.VideoCapture(str(video_path))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                capture.release()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
                probability = model(image).softmax(dim=1)[0].cpu()
                probabilities.append(probability)
                frame_correct += int(probability.argmax().item() == true_index)
                frame_total += 1
            if not probabilities:
                continue
            clip_prediction = torch.stack(probabilities).mean(dim=0).argmax().item()
            confusion[true_index][clip_prediction] += 1
            clip_results.append({
                "clip": video_path.name,
                "true_label": true_name,
                "predicted_label": CLASS_NAMES[clip_prediction],
                "correct": clip_prediction == true_index,
            })
    clip_correct = sum(result["correct"] for result in clip_results)
    return {
        "clip_accuracy": clip_correct / len(clip_results) if clip_results else 0.0,
        "frame_accuracy": frame_correct / frame_total if frame_total else 0.0,
        "clips": clip_results,
        "confusion_matrix": confusion,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    clips_by_class = read_clips(manifest_path, video_dir)

    for label in CLASS_NAMES:
        if len(clips_by_class[label]) < 2:
            raise SystemExit(f"Need at least 2 clips in class '{label}', found {len(clips_by_class[label])}")

    train_clips = []
    test_clips = {}
    for label in CLASS_NAMES:
        ordered = sorted(clips_by_class[label])
        test_clips[video_dir / ordered[-1].name] = label
        train_clips.extend(ordered[:-1])

    rng = random.Random(args.seed)
    train_samples = []
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    for video_path in train_clips:
        label = next(name for name in CLASS_NAMES if video_path in clips_by_class[name])
        for frame_index in sample_clip_frames(video_path, args.frames_per_clip, rng):
            train_samples.append((video_path, frame_index, class_to_index[label]))

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoFrameDataset(train_samples, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = make_model(args.architecture, len(CLASS_NAMES)).to(device)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier"):
                parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Train clips: {[path.name for path in train_clips]}")
    print(f"Test clips: {[path.name for path in test_clips]}")
    print(f"Train frames: {len(train_samples)}")

    history = []
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, frame_accuracy={accuracy:.4f}")

    test_metrics = evaluate_clips(model, test_clips, eval_transform, class_to_index, device)
    print(f"Test clip accuracy: {test_metrics['clip_accuracy']:.4f}")
    print(f"Test frame accuracy: {test_metrics['frame_accuracy']:.4f}")

    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": args.architecture,
        "class_names": list(CLASS_NAMES),
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, output_path)

    metrics = {
        "device": str(device),
        "architecture": args.architecture,
        "train_clips": [path.name for path in train_clips],
        "test_clips": [path.name for path in test_clips],
        "history": history,
        **test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
