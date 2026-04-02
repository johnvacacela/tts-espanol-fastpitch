import os
import torch
import numpy as np
from tqdm import tqdm

OUT_DIR = "/root/fastpitch_data/victor_villarraza/mels_sinteticos"
archivos = [f for f in os.listdir(OUT_DIR) if f.endswith('.pt')]

print(f"🔄 Convirtiendo {len(archivos)} archivos .pt a .npy para HiFi-GAN...")

for f in tqdm(archivos):
    ruta_pt = os.path.join(OUT_DIR, f)
    ruta_npy = ruta_pt.replace('.pt', '.npy')
    
    try:
        # 1. Cargar tensor de PyTorch
        mel_tensor = torch.load(ruta_pt, map_location='cpu')
        
        # 2. Convertir a arreglo de NumPy
        mel_numpy = mel_tensor.numpy()
        
        # 3. Guardar como .npy
        np.save(ruta_npy, mel_numpy)
        
    except Exception as e:
        print(f"Error convirtiendo {f}: {e}")

print("✅ ¡Conversión completada! Los Mels ya hablan el idioma de HiFi-GAN.")
