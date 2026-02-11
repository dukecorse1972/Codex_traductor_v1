# Reconocimiento aislado de signos LSE (SWL-LSE) con PyTorch + TCN

Proyecto completo para entrenar un clasificador de 300 clases usando landmarks de MediaPipe preextraídos (`MEDIAPIPE/*.pkl`) y hacer demo en tiempo real con webcam.

## Estructura esperada del dataset

```text
DATA_ROOT/
  train.csv
  val.csv
  test.csv
  videos_ref_annotations.csv
  MEDIAPIPE/
    <FILENAME>.pkl
    ...
```

- `train.csv/val.csv/test.csv`: sin cabecera, columnas `[FILENAME, CLASS_ID]`.
- `videos_ref_annotations.csv`: con cabecera `FILENAME,CLASS_ID,LABEL`.
- Cada `.pkl` es una lista de frames, cada frame dict con `pose`, `hands`, `holistic_legacy`.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Paso 0: inspección automática (obligatorio)

Genera `artifacts/feature_spec.json` con `T_fixed`, dimensiones y políticas de procesamiento:

```bash
python tools/inspect_dataset.py --splits_dir /ruta/DATA_ROOT --mediapipe_dir /ruta/DATA_ROOT/MEDIAPIPE --n_samples 20
```

## Entrenamiento

```bash
python train.py \
  --data_root /ruta/DATA_ROOT \
  --splits_dir /ruta/DATA_ROOT \
  --mediapipe_dir /ruta/DATA_ROOT/MEDIAPIPE \
  --annotations_csv /ruta/DATA_ROOT/videos_ref_annotations.csv \
  --epochs 40 --batch_size 32 --lr 1e-3 --model tcn
```

Salidas principales:
- `checkpoints/best.pt`
- `artifacts/label_map.json`
- `artifacts/feature_spec.json`
- `artifacts/train_log.csv`
- `artifacts/confusion_matrix.png`

## Demo en tiempo real con webcam

```bash
python realtime_webcam.py \
  --checkpoint checkpoints/best.pt \
  --feature_spec artifacts/feature_spec.json \
  --label_map artifacts/label_map.json \
  --model tcn --infer_every 2 --smooth_k 5 --threshold 0.4
```

- Muestra etiqueta (`LABEL`) + confianza + FPS.
- Si confianza < `threshold`: muestra `Desconocido`.
- Mientras el buffer no está lleno: muestra `Calibrando…`.

## Features por frame (orden fijo)

`D_frame = 291`:
1. Pose: 33 landmarks × `[x,y,z,visibility,presence]` = 165
2. Hands (slots fijos):
   - Left: 21 × `[x,y,z]` = 63
   - Right: 21 × `[x,y,z]` = 63

Total: 165 + 63 + 63 = 291.

Normalización:
- Centro corporal: midpoint caderas (`LEFT_HIP`, `RIGHT_HIP`) si visibles/presentes, si no midpoint hombros.
- Escala: distancia entre hombros.
- Pose y manos se transforman con el mismo centro/escala.

Secuencias:
- Salida por muestra: `(T_fixed, 291)`.
- Si `T < T_fixed`: padding de ceros al final.
- Si `T > T_fixed`: subsampling uniforme.

Augmentations (train):
- dropout temporal,
- jitter gaussiano ligero,
- landmark dropout de baja probabilidad.

## Tests

```bash
pytest -q
```

Incluye test de:
- dimensión 291,
- determinismo de orden de features.
