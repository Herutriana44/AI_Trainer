#!/usr/bin/env python3
"""
Install requirements dengan konfigurasi untuk koneksi internet lambat.
Mencegah timeout saat download package besar (torch, dll).
"""

import subprocess
import sys
from pathlib import Path

# Timeout dalam detik (1 jam untuk koneksi sangat lambat)
TIMEOUT = 3600
RETRIES = 5

def main():
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print(f"Error: {req_file} tidak ditemukan.")
        sys.exit(1)

    print("Installing requirements...")
    print(f"Timeout: {TIMEOUT}s | Retries: {RETRIES}")
    print()

    cmd = [
        sys.executable, "-m", "pip", "install", "-r", str(req_file),
        "--default-timeout", str(TIMEOUT),
        "--retries", str(RETRIES),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("\nInstallation selesai.")


if __name__ == "__main__":
    main()
