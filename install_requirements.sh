#!/bin/bash
# Install requirements dengan konfigurasi untuk koneksi internet lambat
# Mencegah timeout saat download package besar (torch, etc)

# Timeout dalam detik (default pip: 15 detik, kita set 3600 = 1 jam untuk koneksi lambat)
TIMEOUT=3600
# Retry saat gagal
RETRIES=5
# Tidak cache untuk menghemat disk (opsional, bisa dihapus jika ingin cache)
# CACHE_OPT="--no-cache-dir"
CACHE_OPT=""

echo "Installing requirements..."
echo "Timeout: ${TIMEOUT}s | Retries: ${RETRIES}"
echo ""

pip install -r requirements.txt \
    --default-timeout=${TIMEOUT} \
    --retries ${RETRIES} \
    ${CACHE_OPT}

echo ""
echo "Installation selesai."
