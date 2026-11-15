cat > docs/models/cardd-yolo11n.md <<'EOF'
# CarDD YOLO11n Damage Detection Model

## Model details

- Architecture: YOLO11n
- Dataset: CarDD
- Training framework: Ultralytics 8.4.191
- Classes: 6
- Epochs: 10
- Image size: 512
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.001

## Damage classes

| ID | Class |
|---|---|
| 0 | dent |
| 1 | scratch |
| 2 | crack |
| 3 | glass_damage |
| 4 | lamp_damage |
| 5 | tire_damage |

## Validation results

Preliminary validation on the CarDD validation split:

| Metric | Result |
|---|---:|
| Precision | 0.626 |
| Recall | 0.570 |
| mAP@50 | 0.597 |
| mAP@50-95 | 0.476 |

These are preliminary results from a 10-epoch training run, not test-set results.

## Model weights

The trained `best.pt` file is not committed to this repository because model weights are excluded by `.gitignore`.

Place the downloaded weights at:

`backend/models/weights/best.pt`

The model owner must share the checkpoint separately with collaborators.

## Backend configuration

From the `backend/` directory, configure:

```env
DETECTOR_BACKEND=ultralytics
DETECTOR_WEIGHTS_PATH=models/weights/best.pt
DETECTOR_MODEL_NAME=cardd-yolo11n
DETECTOR_MODEL_VERSION=cardd-10epochs-img512
DETECTOR_WEIGHTS_SHA256=d6ede1135427c68ad2b8d3d58da9e0ffc08f6879f623c26aec52b60ab6742768
DETECTOR_DEVICE=cpu
