# Módulo BERT + Difusión DDPM

## Componentes
- `bert_conditioner.py` — BERTSemanticConditioner: BETO congelado + Linear(768→384) + Tanh
- `pitch_diffusion.py` — Módulo DDPM 100 pasos para predicción de pitch
- `model.py` — FastPitch modificado con DDPM + BETO integrados
- `data_function.py` — DataLoader con soporte para embeddings BETO precomputados

## Arquitectura
BETO lee el párrafo completo → vector CLS (768d) → proyección (256d) → inyección aditiva al predictor de pitch

## Resultados
- Val Mel Loss: 3.62 (mejor entre los 3 modelos)
- MCD: 13.64 dB
