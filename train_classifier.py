#!/usr/bin/env python3
"""
Image Classification Training Script using Pre-trained CNN
Menggunakan PyTorch dengan pre-trained models dari torchvision
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# Daftar pre-trained model yang tersedia
AVAILABLE_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "alexnet": models.alexnet,
    "squeezenet1_0": models.squeezenet1_0,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "inception_v3": models.inception_v3,
    "googlenet": models.googlenet,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
}


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Load pre-trained model dan modifikasi layer terakhir untuk num_classes."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' tidak tersedia. Pilih dari: {list(AVAILABLE_MODELS.keys())}"
        )

    model_fn = AVAILABLE_MODELS[model_name]
    weights = "DEFAULT" if pretrained else None
    model = model_fn(weights=weights)

    # Modifikasi layer terakhir berdasarkan arsitektur model
    if "resnet" in model_name or "resnext" in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif "vgg" in model_name or "alexnet" in model_name:
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif "densenet" in model_name:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif "squeezenet" in model_name:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif "inception" in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif "googlenet" in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif "mobilenet" in model_name:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif "efficientnet" in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def get_transforms(model_name: str, image_size: int = 224):
    """Mendapatkan transform untuk training dan validation."""
    # Inception_v3 menggunakan input 299x299
    if "inception" in model_name:
        image_size = 299

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


class TrainingCallback:
    """Callback untuk memantau proses training."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = None

    def on_train_begin(self, num_batches: int):
        """Dipanggil saat training dimulai."""
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("TRAINING DIMULAI")
        print("=" * 60)
        print(f"Total epochs: {self.total_epochs}")
        print(f"Total batches per epoch: {num_batches}")
        print("=" * 60 + "\n")

    def on_epoch_begin(self, epoch: int):
        """Dipanggil di awal setiap epoch."""
        print(f"\n--- Epoch {epoch + 1}/{self.total_epochs} ---")

    def on_batch_end(self, batch: int, total_batches: int, loss: float, lr: float):
        """Dipanggil setiap selesai batch (dipanggil setiap N batch untuk efisiensi)."""
        if (batch + 1) % max(1, total_batches // 10) == 0 or batch == total_batches - 1:
            progress = (batch + 1) / total_batches * 100
            bar_length = 30
            filled = int(bar_length * (batch + 1) / total_batches)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r  Batch [{bar}] {progress:.1f}% | Loss: {loss:.4f} | LR: {lr:.6f}", end="")

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time: float,
    ):
        """Dipanggil di akhir setiap epoch."""
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (self.total_epochs - epoch - 1) if epoch < self.total_epochs - 1 else 0
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Waktu epoch: {epoch_time:.2f}s | ETA: {eta:.1f}s")

    def on_train_end(self, best_acc: float, total_time: float):
        """Dipanggil saat training selesai."""
        print("\n" + "=" * 60)
        print("TRAINING SELESAI")
        print("=" * 60)
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Total waktu training: {total_time / 60:.2f} menit")
        print("=" * 60 + "\n")


def train_one_epoch(model, dataloader, criterion, optimizer, device, callback, epoch, total_epochs):
    """Training satu epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        lr = optimizer.param_groups[0]["lr"]
        callback.on_batch_end(batch_idx, total_batches, loss.item(), lr)

    avg_loss = running_loss / total_batches
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validasi model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(
    dataset_path: str,
    model_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    num_workers: int = 4,
    image_size: int = 224,
    output_dir: str = "output",
    save_best_only: bool = True,
):
    """
    Training model klasifikasi gambar.

    Args:
        dataset_path: Path ke folder dataset (format: dataset/label/image.jpg)
        model_name: Nama pre-trained model (lihat AVAILABLE_MODELS)
        epochs: Jumlah epoch training
        batch_size: Ukuran batch
        learning_rate: Learning rate
        val_split: Rasio data validasi (0-1)
        num_workers: Jumlah worker untuk DataLoader
        image_size: Ukuran input gambar
        output_dir: Folder untuk menyimpan model
        save_best_only: Simpan hanya model terbaik
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path tidak ditemukan: {dataset_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # Load dataset
    train_transform, val_transform = get_transforms(model_name, image_size)

    train_dataset = datasets.ImageFolder(
        root=str(dataset_path),
        transform=train_transform,
    )

    num_classes = len(train_dataset.classes)
    print(f"Jumlah kelas: {num_classes}")
    print(f"Label: {train_dataset.classes}")

    val_dataset = datasets.ImageFolder(
        root=str(dataset_path),
        transform=val_transform,
    )

    # Split indices
    indices = list(range(len(train_dataset)))
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset)).tolist()
    val_size = int(len(train_dataset) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Load model
    model = get_model(model_name, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Callback
    callback = TrainingCallback(epochs)
    callback.on_train_begin(len(train_loader))

    # Training loop
    best_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        callback.on_epoch_begin(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, callback, epoch, epochs
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        callback.on_epoch_end(
            epoch, train_loss, train_acc, val_loss, val_acc, epoch_time
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = output_path / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "num_classes": num_classes,
                "class_to_idx": train_dataset.class_to_idx,
                "model_name": model_name,
            }, save_path)
            print(f"  ✓ Model terbaik disimpan ke {save_path}")

        if not save_best_only:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "num_classes": num_classes,
                "class_to_idx": train_dataset.class_to_idx,
                "model_name": model_name,
            }, output_path / f"model_epoch_{epoch + 1}.pt")

    total_time = time.time() - callback.start_time
    callback.on_train_end(best_acc, total_time)

    return model, best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Training Image Classification dengan Pre-trained CNN"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path ke folder dataset (format: dataset/label/image.jpg)",
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Nama pre-trained model yang akan digunakan",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Rasio validasi 0-1 (default: 0.2)")
    parser.add_argument("--num-workers", type=int, default=4, help="Jumlah DataLoader workers (default: 4)")
    parser.add_argument("--image-size", type=int, default=224, help="Ukuran input gambar (default: 224)")
    parser.add_argument("--output-dir", type=str, default="output", help="Folder output (default: output)")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Simpan model setiap epoch (default: hanya terbaik)",
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        image_size=args.image_size,
        output_dir=args.output_dir,
        save_best_only=not args.save_all,
    )


if __name__ == "__main__":
    main()
