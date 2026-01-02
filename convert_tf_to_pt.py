import tensorflow as tf
from transformers import T5ForConditionalGeneration
import torch
import os

print("="*50)
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
print("="*50)

# Konfigurasi path
tf_model_path = "kompas_summarization_model"
pt_model_path = "kompas_summarization_model_pt"

try:
    print(f"[1/3] Memuat model TensorFlow dari {tf_model_path}...")
    model = T5ForConditionalGeneration.from_pretrained(
        tf_model_path,
        from_tf=True
    )
    
    print(f"[2/3] Menyimpan model PyTorch ke {pt_model_path}...")
    model.save_pretrained(pt_model_path)
    
    print("[3/3] Verifikasi file hasil konversi:")
    print(os.listdir(pt_model_path))
    print("="*50)
    print("✅ Konversi berhasil!")
    
except Exception as e:
    print("❌ Gagal mengonversi:")
    print(str(e))
    print("\nSolusi:")
    print("1. Pastikan folder model berisi tf_model.h5")
    print("2. Cek permission folder")
    print("3. Hubungi developer jika error persist")