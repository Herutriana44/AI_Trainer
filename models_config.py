"""
Konfigurasi Pre-trained CNN Models untuk Image Classification
Dapat di-import di kode Python manapun.
"""

from torchvision import models

# Daftar pre-trained model yang tersedia
# Format: "nama_model": fungsi_model_torchvision
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


def get_model_names():
    """Mengembalikan list nama model yang tersedia."""
    return list(AVAILABLE_MODELS.keys())
