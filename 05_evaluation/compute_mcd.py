"""
Calcula Distancia Mel Espectral entre mels sintéticos y reales.
Uso: python compute_mcd.py --synth /path/mel.npy --real /path/mel.pt
"""
import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt

def compute_mcd(mel_synth_path, mel_real_path):
    mel_s = np.load(mel_synth_path).T      # [80, T]
    mel_r = torch.load(mel_real_path, weights_only=True).numpy()
    min_len = min(mel_s.shape[1], mel_r.shape[1])
    mel_s = mel_s[:, :min_len]
    mel_r = mel_r[:, :min_len]
    diff = mel_s - mel_r
    return np.mean(np.sqrt(np.sum(diff**2, axis=0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth', required=True, help='Mel sintético .npy')
    parser.add_argument('--real', required=True, help='Mel real .pt')
    args = parser.parse_args()
    mcd = compute_mcd(args.synth, args.real)
    print(f'MCD: {mcd:.4f}')
