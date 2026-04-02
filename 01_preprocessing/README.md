# 01 - Preprocesamiento y Alineación

## Pipeline

1. **MFA** (Montreal Forced Aligner) alinea audio con transcripciones fonéticas IPA
2. **textgrids_to_meta.py** — extrae duraciones de los TextGrids generados por MFA
3. **convertir_mels.py** — convierte mels de formato numpy a PyTorch (.pt)
4. **reparar_matrices.py** — corrige matrices de pitch con >50% de ceros usando pYIN

## Uso
```bash
# Extraer transcripciones de TextGrids MFA
python textgrids_to_meta.py /path/to/textgrids meta.txt

# Convertir mels
python convertir_mels.py

# Re-extraer pitch problemático
python reparar_matrices.py
```

## Archivos meta generados

Formato: `mel_path|pitch_path|texto_ipa`

Ejemplo:
```
mels/audio_001.pt|pitches_expanded/audio_001.pt|s i l k a s a sil
```
