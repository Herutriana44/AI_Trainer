#!/usr/bin/env python3
"""
Streamlit App - Dataset Builder & Model Training untuk Image Classification
Custom label, upload gambar per label, training model dengan pre-trained CNN
"""

import time
import uuid
from pathlib import Path

import streamlit as st

from models_config import get_model_names
from train_classifier import train
from utils import preprocess_dir_name

# Konfigurasi
BASE_DATASET_DIR = Path("datasets")
BASE_OUTPUT_DIR = Path("outputs")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class StreamlitTrainingCallback:
    """Callback training untuk update UI Streamlit."""

    def __init__(self, total_epochs: int, progress_container, status_container, metrics_container):
        self.total_epochs = total_epochs
        self.total_batches = 1
        self.current_epoch = 0
        self.progress_container = progress_container
        self.status_container = status_container
        self.metrics_container = metrics_container
        self.start_time = None

    def on_train_begin(self, num_batches: int):
        self.start_time = time.time()
        self.total_batches = num_batches
        self.status_container.info("🔄 Training dimulai...")
        self.progress_container.progress(0, text="Memulai...")
        self.metrics_container.write(f"Total epochs: {self.total_epochs} | Batches/epoch: {num_batches}")

    def on_epoch_begin(self, epoch: int):
        self.current_epoch = epoch
        self.status_container.info(f"⏳ Epoch {epoch + 1}/{self.total_epochs}...")

    def on_batch_end(self, batch: int, total_batches: int, loss: float, lr: float):
        batch_progress = (batch + 1) / total_batches
        overall = (self.current_epoch + batch_progress) / self.total_epochs
        self.progress_container.progress(
            min(overall, 1.0),
            text=f"Epoch {self.current_epoch + 1} | Batch {batch + 1}/{total_batches} | Loss: {loss:.4f}",
        )

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time: float,
    ):
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (self.total_epochs - epoch - 1) if epoch < self.total_epochs - 1 else 0
        self.metrics_container.markdown(f"""
        **Epoch {epoch + 1}/{self.total_epochs}**
        - Train Loss: `{train_loss:.4f}` | Train Acc: `{train_acc:.2f}%`
        - Val Loss: `{val_loss:.4f}` | Val Acc: `{val_acc:.2f}%`
        - Waktu: `{epoch_time:.2f}s` | ETA: `{eta:.1f}s`
        """)
        self.progress_container.progress((epoch + 1) / self.total_epochs, text=f"Epoch {epoch + 1} selesai")

    def on_train_end(self, best_acc: float, total_time: float):
        self.status_container.success(f"✅ Training selesai! Best accuracy: **{best_acc:.2f}%**")
        self.progress_container.progress(1.0, text="Selesai")
        self.metrics_container.markdown(f"**Total waktu:** {total_time / 60:.2f} menit")


def init_session_state():
    """Inisialisasi session state."""
    if "labels" not in st.session_state:
        st.session_state.labels = []
    if "project_title" not in st.session_state:
        st.session_state.project_title = ""
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None
    if "training_done" not in st.session_state:
        st.session_state.training_done = False
    if "last_best_acc" not in st.session_state:
        st.session_state.last_best_acc = None


def get_dataset_dir(project_title: str) -> Path:
    """Mendapatkan path direktori dataset dari project title."""
    dir_name = preprocess_dir_name(project_title)
    return BASE_DATASET_DIR / dir_name


def save_uploaded_files(uploaded_files, label_dir: Path):
    """Menyimpan file yang di-upload ke direktori label."""
    saved_count = 0
    for f in uploaded_files:
        if Path(f.name).suffix.lower() in ALLOWED_EXTENSIONS:
            # Generate unique filename untuk menghindari overwrite
            ext = Path(f.name).suffix
            unique_name = f"{uuid.uuid4().hex[:8]}_{Path(f.name).stem}{ext}"
            save_path = label_dir / unique_name
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            saved_count += 1
    return saved_count


def main():
    st.set_page_config(
        page_title="AI Trainer - Dataset Builder",
        page_icon="📁",
        layout="wide",
    )

    init_session_state()

    # --- Sidebar: Project Title ---
    with st.sidebar:
        st.header("⚙️ Pengaturan Project")
        project_title = st.text_input(
            "Judul Project",
            value=st.session_state.project_title,
            placeholder="Contoh: Klasifikasi Buah",
            help="Judul akan digunakan untuk membuat nama direktori dataset",
        )

        if project_title:
            dir_name = preprocess_dir_name(project_title)
            dataset_path = BASE_DATASET_DIR / dir_name
            st.info(f"📂 Direktori: `{dataset_path}`")
            st.session_state.project_title = project_title
            st.session_state.dataset_path = dataset_path

    # --- Main: Title Project ---
    st.title("📁 AI Trainer - Dataset Builder")
    st.markdown("Buat dataset untuk training model klasifikasi gambar dengan custom label.")

    if not project_title or not project_title.strip():
        st.warning("⚠️ Masukkan **Judul Project** di sidebar untuk memulai.")
        st.markdown("""
        ### Cara penggunaan:
        1. Masukkan **Judul Project** di sidebar (contoh: "Klasifikasi Buah")
        2. Klik **Buat Project** untuk membuat direktori dataset
        3. Tambahkan **Custom Label** (nama kelas)
        4. Pilih label, lalu upload gambar sesuai label
        """)
        return

    dataset_path = get_dataset_dir(project_title)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # --- Custom Label Section ---
    st.subheader("🏷️ Custom Label")

    col1, col2 = st.columns([2, 1])
    with col1:
        new_label = st.text_input(
            "Tambah label baru",
            placeholder="Contoh: apel, jeruk, pisang",
            key="new_label_input",
        )
    with col2:
        st.write("")  # spacing
        st.write("")
        if st.button("➕ Tambah Label", type="primary"):
            if new_label and new_label.strip():
                label_clean = preprocess_dir_name(new_label.strip())
                if label_clean and label_clean not in st.session_state.labels:
                    st.session_state.labels.append(label_clean)
                    (dataset_path / label_clean).mkdir(exist_ok=True)
                    st.success(f"Label '{label_clean}' ditambahkan!")
                    st.rerun()
            elif new_label:
                st.error("Label tidak valid setelah preprocessing.")

    # Tampilkan labels yang ada
    if st.session_state.labels:
        st.markdown("**Label yang tersedia:**")
        label_cols = st.columns(min(5, len(st.session_state.labels)))
        for i, label in enumerate(st.session_state.labels):
            with label_cols[i % 5]:
                num_images = len(list((dataset_path / label).glob("*"))) if (dataset_path / label).exists() else 0
                if st.button(f"🗑️ {label} ({num_images})", key=f"del_{label}"):
                    st.session_state.labels.remove(label)
                    st.rerun()

    st.divider()

    # --- Upload Section ---
    st.subheader("📤 Upload Gambar")

    if not st.session_state.labels:
        st.info("Tambahkan minimal 1 label terlebih dahulu.")
        return

    selected_label = st.selectbox(
        "Pilih label untuk upload",
        options=st.session_state.labels,
        key="selected_label",
    )

    uploaded_files = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and selected_label:
        if st.button("💾 Simpan Gambar"):
            label_dir = dataset_path / selected_label
            label_dir.mkdir(parents=True, exist_ok=True)
            saved = save_uploaded_files(uploaded_files, label_dir)
            st.success(f"✅ {saved} gambar disimpan ke label '{selected_label}'")
            st.rerun()

    # --- Summary ---
    st.divider()
    st.subheader("📊 Ringkasan Dataset")

    if st.session_state.labels:
        total_images = 0
        for label in st.session_state.labels:
            label_path = dataset_path / label
            count = len(list(label_path.glob("*"))) if label_path.exists() else 0
            total_images += count
            st.write(f"- **{label}**: {count} gambar")

        st.write(f"**Total**: {total_images} gambar")
        st.code(f"Path dataset: {dataset_path.absolute()}", language=None)

    # --- Training Section ---
    st.divider()
    st.subheader("🤖 Training Model")

    if len(st.session_state.labels) < 2:
        st.warning("⚠️ Minimal 2 label diperlukan untuk training klasifikasi.")
        return

    if total_images < 4:
        st.warning("⚠️ Minimal 4 gambar diperlukan untuk training (train + validasi). Upload lebih banyak gambar.")
        return

    # Cek setiap label punya minimal 1 gambar
    labels_with_images = sum(1 for lbl in st.session_state.labels if (dataset_path / lbl).exists() and list((dataset_path / lbl).glob("*")))
    if labels_with_images < 2:
        st.warning("⚠️ Minimal 2 label harus memiliki gambar. Pastikan setiap label punya minimal 1 gambar.")
        return

    with st.expander("⚙️ Pengaturan Training", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_name = st.selectbox(
                "Pre-trained Model",
                options=get_model_names(),
                index=get_model_names().index("resnet18") if "resnet18" in get_model_names() else 0,
                help="Model CNN pre-trained dari torchvision",
            )
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=0.1,
                value=0.001,
                format="%.5f",
                step=1e-4,
            )
            val_split = st.slider("Rasio Validasi", 0.1, 0.5, 0.2, 0.05)
            image_size = st.number_input("Image Size", min_value=64, max_value=512, value=224, step=32)
        with col3:
            output_dir_name = preprocess_dir_name(project_title)
            output_dir = BASE_OUTPUT_DIR / output_dir_name
            st.text_input("Output Dir", value=str(output_dir), disabled=True)
            num_workers = st.number_input("DataLoader Workers", min_value=0, max_value=8, value=0, help="0 = main thread (lebih aman di Streamlit)")

    if st.button("🚀 Mulai Training", type="primary"):
        output_dir = BASE_OUTPUT_DIR / preprocess_dir_name(project_title)
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_container = st.container()
        status_container = st.container()
        metrics_container = st.container()

        callback = StreamlitTrainingCallback(
            total_epochs=epochs,
            progress_container=progress_container,
            status_container=status_container,
            metrics_container=metrics_container,
        )

        with st.spinner("Training berjalan... Jangan tutup halaman."):
            try:
                _, best_acc = train(
                    dataset_path=str(dataset_path),
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    val_split=val_split,
                    num_workers=num_workers,
                    image_size=image_size,
                    output_dir=str(output_dir),
                    save_best_only=True,
                    callback=callback,
                )
                st.session_state.training_done = True
                st.session_state.last_best_acc = best_acc
                st.success(f"✅ Model disimpan ke `{output_dir / 'best_model.pt'}`")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
