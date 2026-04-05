# 03 - Módulo de Difusión DDPM

## Descripción
Reemplaza el `TemporalPredictor` lineal de FastPitch por un proceso de difusión 
estocástico (DDPM) para mejorar la expresividad prosódica del pitch predicho.

## Archivos modificados
| Archivo | Cambio |
|---------|--------|
| `fastpitch/pitch_diffusion.py` | NUEVO — módulo `PitchDiffusion` (2.86M params) |
| `fastpitch/model.py` | Reemplaza `TemporalPredictor` por `PitchDiffusion` |
| `fastpitch/loss_function.py` | Usa diffusion loss en lugar de MSE para el pitch |
| `train.py` | Fix unpack de 13 valores en el forward pass |
| `inference.py` | Guarda pitch predicho como `mel_X_pitch.npy` |

## Arquitectura DDPM
- 100 pasos de difusión, beta schedule lineal (1e-4 → 0.02)
- 4 capas denoising con condicionamiento en encoder output
- Compatible con la interfaz original de `TemporalPredictor`
- Parámetros adicionales: 2.86M

## Entrenamiento
Fine-tuning desde el checkpoint 300 del baseline:
```bash
python train.py \
    --dataset-path ~/fastpitch_data \
    -o ~/fastpitch_out/exp_diffusion \
    --epochs 300 \
    --init-from-checkpoint ~/fastpitch_out/exp_base_limpio/FastPitch_checkpoint_300.pt \
    --training-files ~/fastpitch_data/train.meta.fp.txt \
    --validation-files ~/fastpitch_data/val.meta.fp.txt \
    --input-type phone \
    --symbol-set ipa_all \
    --pitch-mean-std-file ~/fastpitch_data/pitch_mean_std.json \
    -bs 16 --learning-rate 0.0001 \
    --cuda --amp
```

## Resultados
| Métrica | Baseline (ckpt 300) | Difusión (ckpt 300) |
|---------|--------------------|--------------------|
| Val Mel Loss | 3.81 | **3.64** ✅ |
| Train Mel Loss | 0.92 | 1.46 |
| MCD promedio (6 frases) | 13.23 dB | 13.57 dB |

> El modelo con difusión mejora el val mel loss (3.81 → 3.64) manteniendo 
> una calidad acústica comparable al baseline (MCD casi idéntico), 
> mientras gana expresividad prosódica demostrada por contornos de pitch 
> más variados (ver `05_evaluation/`).

## Referencias
- DiffProsody (Oh et al., 2024, IEEE/ACM TASLP)
- DDPM — Denoising Diffusion Probabilistic Models (Ho et al., 2020, NeurIPS)
- FastPitch (Łańcucki, 2021, ICASSP)
