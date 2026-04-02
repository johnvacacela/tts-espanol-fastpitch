# Sistema TTS en Español con FastPitch

**Tesis:** Mejora de la Expresividad Prosódica en Síntesis de Voz en Español mediante Modelos de Difusión y Condicionamiento Semántico con BERT

**Autor:** John Vacacela  
**Institución:** Universidad de Cuenca  
**Dataset:** Victor Villarraza (M-AILABS) — ~12h audio narrativo en español

## Pipeline
```
Audio + Transcripciones
    ↓
01_preprocessing/  → Alineación MFA + extracción features
    ↓
02_fastpitch_baseline/  → Entrenamiento FastPitch con IPA español
    ↓
03_evaluation/  → MCD, curvas de convergencia, audios
```

## Resultados baseline (500 épocas, CEDIA A100)

| Métrica | Valor |
|---------|-------|
| Train Mel Loss | 0.92 |
| Val Mel Loss | 3.81 |
| Mejor checkpoint (MCD) | Época 300 (13.23 dB) |

## Entorno

- Python 3.10
- PyTorch 2.1.2+cu121
- NVIDIA A100-SXM4-40GB (CEDIA HPC)

## Dataset

No incluido por tamaño. Estadísticas:
- 7,175 frases entrenamiento
- 147 frases validación
- Pitch mean: 207.25 Hz, std: 161.67 Hz
