# 02 - FastPitch Baseline para Español

FastPitch oficial de NVIDIA adaptado para IPA español.

## Modificaciones principales

| Archivo | Modificación |
|---------|-------------|
| common/text/symbols.py | Añadido symbol set ipa_all (36 tokens) |
| common/text/text_processing.py | Añadida clase IPATextProcessing |
| fastpitch/data_function.py | Fix normalize_pitch para pitch 1D |
| fastpitch/alignment.py | Fix IndexError en mas_width1 (j > 0) |
| fastpitch/model.py | Comentada aserción dur_tgt sum |
| common/gpu_affinity.py | Try/except para OSError en HPC |
| train.py | Fallback a torch.optim.Adam sin apex |
| inference.py | Fix audio vacío |

## Entrenamiento
```bash
python train.py \
    --dataset-path ~/fastpitch_data \
    --training-files ~/fastpitch_data/train.meta.fp.txt \
    --validation-files ~/fastpitch_data/val.meta.fp.txt \
    --symbol-set ipa_all \
    --p-arpabet 0.0 \
    --output ~/fastpitch_out/exp_base_limpio \
    --epochs 500 \
    -bs 32 \
    --learning-rate 0.1 \
    --load-mel-from-disk \
    --load-pitch-from-disk \
    --pitch-mean 207.2481 \
    --pitch-std 161.6691 \
    --num-workers 0 \
    --epochs-per-checkpoint 10 \
    --cuda
```

## Inferencia
```bash
python fastpitch_clean/inference.py \
    -i frases_ipa.txt \
    -o audios_output \
    --fastpitch FastPitch_checkpoint_300.pt \
    --hifigan g_02500000 \
    --hifigan-config config.json \
    --symbol-set ipa_all \
    --p-arpabet 0.0 \
    --cuda
```
