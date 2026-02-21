#!/bin/bash
# Menjalankan Streamlit dengan ngrok (port forwarding)
# Pastikan ngrok CLI terpasang: https://ngrok.com/download

PORT=8501
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Memulai Streamlit di port $PORT..."
streamlit run "$SCRIPT_DIR/app_streamlit.py" \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true &

STREAMLIT_PID=$!
sleep 3

echo "Memulai ngrok tunnel..."
ngrok http $PORT

kill $STREAMLIT_PID 2>/dev/null
