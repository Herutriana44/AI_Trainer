#!/usr/bin/env python3
"""
Inferensi model klasifikasi gambar yang sudah dilatih.
Input: path model + path label mapping (JSON).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from train_classifier import get_model, get_transforms


def load_label_mapping(label_mapping_path: str) -> dict:
    """
    Load label mapping dari file JSON.

    Args:
        label_mapping_path: Path ke file label_mapping.json

    Returns:
        idx_to_class: dict {int: str} mapping index ke nama kelas
    """
    path = Path(label_mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping tidak ditemukan: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Support format: {"idx_to_class": {"0": "apel", "1": "jeruk"}} atau {"class_to_idx": {"apel": 0, "jeruk": 1}}
    if "idx_to_class" in data:
        raw = data["idx_to_class"]
        idx_to_class = {int(k): v for k, v in raw.items()}
    elif "class_to_idx" in data:
        class_to_idx = data["class_to_idx"]
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    else:
        # Fallback: assume root is idx_to_class
        idx_to_class = {int(k): v for k, v in data.items()}

    return idx_to_class


def load_checkpoint(model_path: str):
    """
    Load checkpoint model (weights + metadata).
    Tidak termasuk label mapping - gunakan load_label_mapping terpisah.

    Returns:
        dict dengan keys: model, model_name, num_classes, image_size, device
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", 299 if "inception" in model_name else 224)

    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": image_size,
        "device": device,
    }


def get_inference_transform(model_name: str, image_size: int = None):
    """Transform untuk inferensi (sama dengan validasi)."""
    _, val_transform = get_transforms(model_name, image_size or 224)
    return val_transform


def predict(model_path: str, label_mapping_path: str, image_input, top_k: int = 5):
    """
    Prediksi kelas dari gambar.

    Args:
        model_path: Path ke file model (.pt)
        label_mapping_path: Path ke file label_mapping.json
        image_input: Path ke gambar (str/Path) atau PIL.Image
        top_k: Jumlah prediksi teratas yang dikembalikan

    Returns:
        list of (class_name, probability) sorted by probability descending
    """
    loaded = load_checkpoint(model_path)
    idx_to_class = load_label_mapping(label_mapping_path)

    model = loaded["model"]
    model_name = loaded["model_name"]
    image_size = loaded["image_size"]
    device = loaded["device"]

    # Load image
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    transform = get_inference_transform(model_name, image_size)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(idx_to_class)))

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        class_name = idx_to_class.get(idx, f"class_{idx}")
        results.append((class_name, prob))

    return results


def predict_single(model_path: str, label_mapping_path: str, image_input):
    """
    Prediksi kelas paling probable.

    Returns:
        (class_name, probability)
    """
    results = predict(model_path, label_mapping_path, image_input, top_k=1)
    return results[0] if results else (None, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Inferensi model klasifikasi gambar")
    parser.add_argument("model_path", type=str, help="Path ke file model (.pt)")
    parser.add_argument("label_mapping_path", type=str, help="Path ke file label_mapping.json")
    parser.add_argument("image_path", type=str, help="Path ke gambar")
    parser.add_argument("--top-k", type=int, default=5, help="Jumlah prediksi teratas (default: 5)")

    args = parser.parse_args()

    results = predict(
        args.model_path,
        args.label_mapping_path,
        args.image_path,
        top_k=args.top_k,
    )

    print(f"\nPrediksi untuk: {args.image_path}")
    print("-" * 40)
    for i, (class_name, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"{i}. {class_name}: {prob:.2%} [{bar}]")
    print("-" * 40)
    print(f"Prediksi: {results[0][0]} ({results[0][1]:.2%})")


if __name__ == "__main__":
    main()
