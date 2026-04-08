# Sistema TTS en Español con FastPitch + DDPM + BETO

**Tesis:** Mejora de la Expresividad Prosódica en Síntesis de Voz en Español mediante Modelos de Difusión y Condicionamiento Semántico con BERT

**Autor:** John Vacacela  
**Institución:** Universidad de Cuenca — Facultad de Ingeniería  
**Dataset:** Víctor Villarraza (M-AILABS) — ~13.93h audio narrativo en español  
**Corpus literario:** *Cuentos Clásicos del Norte* (Poe) + *La Dama de las Camelias* (Dumas)

---

## Arquitectura del Sistema

```
Texto en español (IPA)
        ↓
   FastPitch (Baseline)
   ├── Encoder de texto
   ├── Predictor de duración
   └── Predictor de pitch ──── DDPM (Propuesto)
                                    │
                               BETO (Propuesto)
                               └── CLS(párrafo) → 256d
                                    ↓
                              Mel-Spectrogram (80 bandas)
                                    ↓
                           HiFi-GAN (Vocoder)
                                    ↓
                              Audio 22050 Hz
```

---

## Estructura del Repositorio

```
tts-espanol-fastpitch/
├── 01_preprocessing/       # Alineación MFA + extracción de features
├── 02_fastpitch_baseline/  # FastPitch adaptado para IPA español
├── 03_diffusion/           # Módulo DDPM para predicción de pitch
├── 04_bert/                # Condicionamiento semántico con BETO
├── 05_evaluation/          # Scripts MCD, convergencia, visualización
├── 06_hifigan/             # HiFi-GAN modificado para mels sintéticos
└── data/                   # Meta files train/val (sin audio por tamaño)
```

---

## Pipeline Completo

```
Audio + Transcripciones
        ↓
01_preprocessing/
  → Conversión a 22050 Hz
  → Alineación MFA (IPA español)
  → Extracción mels (.pt) y pitch (.npy)
        ↓
02_fastpitch_baseline/
  → Entrenamiento 500 épocas (A100 40GB)
  → Checkpoint óptimo: época 300
        ↓
03_diffusion/
  → Fine-tuning con módulo DDPM (100 pasos, β lineal)
  → 300 épocas adicionales
        ↓
04_bert/
  → Precomputación embeddings BETO (7322 frases)
  → Fine-tuning con condicionamiento semántico
  → 100 épocas adicionales
        ↓
05_evaluation/
  → Cálculo MCD, pitch RMSE, val loss
  → Generación de figuras comparativas
        ↓
06_hifigan/
  → Entrenamiento vocoder con mels sintéticos de FastPitch
```

---

## Resultados

### Métricas Objetivas (6 frases del corpus)

| Modelo | Val Mel Loss | MCD (dB) | Parámetros nuevos |
|--------|-------------|----------|-------------------|
| Baseline FastPitch | 3.81 | 19.04 | — |
| + Difusión DDPM | 3.64 | 13.44 | 2.86M |
| + DDPM + BETO | **3.62** | **13.64** | 2.86M + 295K |

> El MCD del baseline (19.04) refleja la comparación justa con las 6 frases del corpus.  
> Los valores de difusión y BERT+Difusión fueron calculados con las mismas frases en condiciones idénticas.

### Convergencia por Modelo

| Modelo | Épocas | Train Loss final | Val Loss final |
|--------|--------|-----------------|----------------|
| Baseline | 500 | 0.92 | 3.81 |
| + Difusión | 300 | 1.12 | 3.64 |
| + BERT+Dif | 100 | 1.48 | 3.62 |

---

## Módulos Propuestos

### Módulo de Difusión DDPM (`03_diffusion/`, `04_bert/pitch_diffusion.py`)
- Reemplaza el predictor de pitch lineal de FastPitch
- 100 pasos de difusión con schedule β lineal
- 2.86M parámetros entrenables
- Mejora MCD de 19.04 → 13.44 dB (↓ 29.4%)

### Condicionamiento Semántico BETO (`04_bert/bert_conditioner.py`)
- BETO (Spanish BERT) procesa el párrafo completo de una sola vez
- Extrae vector CLS de 768 dimensiones
- Proyección lineal a 256 dimensiones compatibles con FastPitch
- Inyección aditiva única por párrafo al predictor de pitch
- Garantiza coherencia prosódica a nivel de párrafo
- Distancia semántica Poe vs Dumas: 14.8% (cos=0.852)

---

## Entorno de Desarrollo

```
Python:     3.10
PyTorch:    2.1.2+cu121
CUDA:       12.2
GPU:        NVIDIA A100-SXM4-40GB (CEDIA HPC)
BETO:       dccuchile/bert-base-spanish-wwm-cased
            transformers==4.36.2
```

---

## Dataset

No incluido por tamaño (~13.93h, 7344 archivos WAV). Estadísticas:

| Split | Frases | Duración aprox. |
|-------|--------|----------------|
| Train | 7,175 | ~12.5h |
| Val | 147 | ~1.4h |

- Pitch mean: 207.25 Hz — Pitch std: 161.67 Hz
- Sample rate: 22050 Hz — Hop length: 256 — n_mels: 80

---

## Cómo Reproducir

### 1. Preprocesamiento
```bash
cd 01_preprocessing/
# Ver README interno para instrucciones MFA
```

### 2. Entrenamiento Baseline
```bash
cd 02_fastpitch_baseline/
python train.py --config configs/fastpitch_ipa.json
```

### 3. Fine-tuning Difusión
```bash
python train.py \
    --resume ~/fastpitch_out/exp_base_limpio/FastPitch_checkpoint_300.pt \
    --config configs/fastpitch_diffusion.json
```

### 4. Fine-tuning BERT+Difusión
```bash
# Precomputar embeddings
python 04_bert/precompute_bert_embeddings.py

# Entrenar
python train.py \
    --resume ~/fastpitch_out/exp_diffusion/FastPitch_checkpoint_300.pt \
    --config configs/fastpitch_bert.json
```

### 5. Inferencia
```bash
python inference.py \
    -i frases.txt \
    -o output/ \
    --fastpitch ~/fastpitch_out/exp_bert/FastPitch_checkpoint_100.pt \
    --hifigan ~/hifigan/g_02500000 \
    --symbol-set ipa_all --p-arpabet 0.0 \
    --save-mels --cuda
```

---

## Figuras Generadas

| Figura | Descripción |
|--------|-------------|
| `convergencia_base.png` | Curvas train/val 500 épocas baseline |
| `mcd_checkpoints.png` | MCD por checkpoint baseline |
| `convergencia_bert.png` | Convergencia BERT+Difusión + comparación 3 modelos |
| `comparacion_espectrograma.png` | Real vs Baseline + pitch pYIN |
| `pitch_contornos_4modelos.png` | GT vs Baseline vs Difusión vs BERT (pYIN) |
| `comparacion_mcd_baseline_vs_diffusion.png` | MCD comparativo 3 modelos |
| `bert_huella_poe.png` | Vector semántico BETO — párrafo Poe |
| `bert_comparacion_parrafos.png` | Diferencia semántica Poe vs Dumas |
| `hifigan_catastrophic.png` | Catastrophic forgetting fine-tuning vocoder |

---

## Notas Técnicas

- El vocoder UNIVERSAL (HiFi-GAN pre-entrenado en LJSpeech) presenta mismatch espectral con los mels de FastPitch. El fine-tuning propio mostró catastrophic forgetting. Como trabajo futuro se propone entrenar HiFi-GAN desde cero con ≥300k pasos.
- Los contornos de pitch en las figuras usan normalización visual (percentiles 5/95, scale_max=55) para comparabilidad entre modelos.
- El MCD se calcula con la fórmula simplificada (sin factor estándar ITU) para consistencia entre checkpoints.