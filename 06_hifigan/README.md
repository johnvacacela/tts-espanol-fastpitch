# HiFi-GAN Vocoder — Entrenamiento desde cero

## Archivos
- `train.py` — Script de entrenamiento modificado para GPU single y mels sintéticos
- `meldataset.py` — Dataset modificado para cargar pares mel_sintetico|wav_real
- `config_v1.json` — Configuración: batch_size=8, num_workers=4, fine_tuning=True
- `inference_e2e.py` — Inferencia end-to-end mel→audio

## Setup
Input: mels generados por FastPitch (80×T)
Target: wavs reales del corpus (22050 Hz)
