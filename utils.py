"""
Utility functions untuk AI Trainer
"""

import re
import unicodedata
from pathlib import Path


def preprocess_dir_name(text: str) -> str:
    """
    Preprocessing nama untuk direktori yang aman.
    - Lowercase
    - Replace spasi dengan underscore
    - Hapus karakter khusus
    - Normalize unicode
    """
    if not text or not text.strip():
        return "dataset"
    # Normalize unicode (e.g. é -> e)
    text = unicodedata.normalize("NFKD", text.strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    text = text.lower()
    # Replace spasi dan karakter tidak valid dengan underscore
    text = re.sub(r"[^\w\-]", "_", text)
    # Hapus underscore berulang
    text = re.sub(r"_+", "_", text)
    # Hapus underscore di awal/akhir
    text = text.strip("_")
    return text if text else "dataset"
