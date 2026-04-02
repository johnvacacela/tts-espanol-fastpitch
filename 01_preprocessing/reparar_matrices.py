import os
import numpy as np
from tqdm import tqdm

OUT_DIR = "/root/fastpitch_data/victor_villarraza/mels_sinteticos"
archivos = [f for f in os.listdir(OUT_DIR) if f.endswith('.npy')]

print(f"🔧 Reparando forma de {len(archivos)} matrices para HiFi-GAN...")

for f in tqdm(archivos):
    ruta = os.path.join(OUT_DIR, f)
    mel = np.load(ruta)
    
    # Si la matriz está al revés [Tiempo, 80], la rotamos a [80, Tiempo]
    if len(mel.shape) == 2 and mel.shape[1] == 80:
        mel_corregido = mel.T  # Transposición matemática
        np.save(ruta, mel_corregido)

print("✅ Matrices rotadas. Listos para entrenar.")
