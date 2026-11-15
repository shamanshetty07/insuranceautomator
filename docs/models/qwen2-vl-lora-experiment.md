# Qwen2-VL LoRA fine-tuning experiment

## Purpose

This note records a small local experiment to adapt a Qwen vision-language model to the vehicle-damage categories used by this project. It is an experiment log, not a claim that a working adapter was produced or that model quality improved.

## Data preparation

- Source: [tahaman/DamageCarDataset](https://huggingface.co/datasets/tahaman/DamageCarDataset), a small public image dataset.
- Working copy used 150 labeled images across six categories: crack, dent, glass shatter, lamp broken, scratch, and tire flat (25 images per category).
- Each category label was converted into a short JSON claim following the project's structured-claim style. These generated targets are weak labels: they do not provide reviewed damage locations, severity, repair cost, or undamaged examples.
- A class-balanced image split was prepared locally: 102 training images, 24 validation images, and 24 test images (17/4/4 per class). The working image copies and JSONL files were not added by this documentation commit.

## Model and method

- Base checkpoint: mlx-community/Qwen2-VL-2B-Instruct-4bit (about 1.25 GB of weight data).
- Training framework: MLX-VLM on Apple Silicon, using LoRA rather than updating all model weights.
- The diagnostic trained the assistant completion only and used one image per example. LoRA changes a small set of trainable adapter weights while keeping the base checkpoint fixed.

## Result

The run did **not** produce a usable adapter. In the diagnostic runs, the first training microbatch had a finite loss of 1.7603; after the first optimizer update the next loss became NaN (not a number). The problem reproduced with both Adam and SGD and with lower learning rates. The saved diagnostic adapter tensors included non-finite values, so they must not be used for inference or described as a successful fine-tune.

No held-out accuracy, JSON-validity score, or before/after model comparison is reported. A finite first-batch loss alone is not evidence of successful training.

## Integration and next steps

The backend's configured inference runner uses Hugging Face Transformers with a local Qwen2-VL or Qwen2.5-VL model snapshot. The experiment used MLX-VLM and its adapter format; compatibility with the backend runner has not been established. Before considering integration, reproduce and diagnose the NaN gradients/weights, save a finite adapter, then run held-out image evaluation for both JSON validity and per-class damage recognition. Do not commit model weights or claim improvement until those checks succeed.

## Reproduction caveat

The image dataset and prepared JSONL files are not part of this documentation change. To reproduce, obtain the dataset from the linked source, document the exact data revision and split seed, and follow the MLX-VLM training instructions for the selected checkpoint. Keep model weights and image binaries out of Git; commit scripts and non-sensitive experiment metadata instead.
