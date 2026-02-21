#!/usr/bin/env python3
"""
Menjalankan Streamlit dengan ngrok untuk port forwarding.
Akses aplikasi dari internet via URL ngrok.
"""

import subprocess
import sys
import time
from pathlib import Path

# Default port Streamlit
STREAMLIT_PORT = 8501


def main():
    script_dir = Path(__file__).parent
    streamlit_script = script_dir / "app_streamlit.py"

    if not streamlit_script.exists():
        print(f"Error: {streamlit_script} tidak ditemukan.")
        sys.exit(1)

    # Jalankan Streamlit di background
    print("Memulai Streamlit...")
    streamlit_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_script),
            "--server.port", str(STREAMLIT_PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Tunggu Streamlit siap
    time.sleep(3)

    # Jalankan ngrok
    print("Memulai ngrok tunnel...")
    try:
        import pyngrok
        from pyngrok import ngrok

        # Buka tunnel ke port Streamlit
        public_url = ngrok.connect(STREAMLIT_PORT, "http")
        print("\n" + "=" * 60)
        print("✅ Aplikasi berjalan!")
        print(f"   Local:  http://localhost:{STREAMLIT_PORT}")
        print(f"   Public: {public_url}")
        print("=" * 60)
        print("\nTekan Ctrl+C untuk menghentikan.\n")

        streamlit_proc.wait()
    except ImportError:
        print("pyngrok tidak terpasang. Install: pip install pyngrok")
        print("Atau jalankan ngrok manual:")
        print(f"  ngrok http {STREAMLIT_PORT}")
        streamlit_proc.wait()
    except KeyboardInterrupt:
        streamlit_proc.terminate()
        streamlit_proc.wait()
        print("\nDihentikan.")


if __name__ == "__main__":
    main()
