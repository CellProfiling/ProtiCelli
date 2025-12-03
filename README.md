# ProtVL Inference Pipeline

A multi-GPU inference pipeline for generating protein expression images using ProtVL.

## Overview

This script performs conditional image generation for proteins. It takes reference microscopy channels (DAPI, tubulin, ER) as input and generates predicted protein expression patterns. Supports distributed inference across multiple GPUs via HuggingFace Accelerate.

## Requirements

- Python 3.x
- PyTorch
- HuggingFace Diffusers & Accelerate
- timm
- NumPy, Pandas, SciPy
- tifffile
- tqdm

## Usage

CPU:
```bash
python ordinary_sampler_standalone.py \
    --csv_file_path", "p4ha2_example.csv \
    --model_path, ./checkpoint-1020000/ \
    --vae_path ./vae \
    --antibody_map_path ./antibody_map.pkl \
    --cell_line_map_path ./cell_line_dict.pkl \
    --antibody_map_path ./antibody_dict.pkl \
    --mixed_precision ./example_output\
    --batch_size 16 \
    --num_workers, 4 \
    --num_inference_steps 100
```

Single GPU:
```bash
python ordinary_sampler_standalone.py \
    --csv_file_path", "p4ha2_example.csv \
    --model_path, ./checkpoint-1020000/ \
    --vae_path ./vae \
    --antibody_map_path ./antibody_map.pkl \
    --cell_line_map_path ./cell_line_dict.pkl \
    --antibody_map_path ./antibody_dict.pkl \
    --mixed_precision ./example_output\
    --batch_size 16 \
    --num_workers, 4 \
    --num_inference_steps 100
```

Multi-GPU with Accelerate:
```bash
accelerate launch --num_processes 4 ordinary_sampler_standalone.py \
    --csv_file_path", "p4ha2_example.csv \
    --model_path, ./checkpoint-1020000/ \
    --vae_path ./vae \
    --antibody_map_path ./antibody_map.pkl \
    --cell_line_map_path ./cell_line_dict.pkl \
    --antibody_map_path ./antibody_dict.pkl \
    --mixed_precision ./example_output\
    --batch_size 16 \
    --num_workers 4 \
    --num_inference_steps 100
```


### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to pretrained DiT model |
| `--vae_path` | Required | Path to VAE checkpoint |
| `--csv_file_path` | Required | CSV with image paths and metadata |
| `--cell_line_map_path` | Required | Cell line name-to-index mapping |
| `--antibody_map_path` | Required | Antibody name-to-index mapping |
| `--output_dir` | `output` | Output directory for generated images |
| `--batch_size` | 4 | Samples per GPU |
| `--num_inference_steps` | 50 | Diffusion sampling steps |
| `--mixed_precision` | `no` | Mixed precision mode (`no`, `fp16`, `bf16`) |

## Input Format

**CSV file** must contain columns:
- `image_path`: Path to input TIFF
- `cell_line_name`: Cell line identifier
- `gene_name`: Target protein/antibody name

**Image format**: Normalized (-1 to 1) 3 or 4-channel TIFF (DAPI, Antibody (Optional), Tubulin, ER) with shape (H, W, C)

## Output

For each input image, generates:
- `{basename}_{cell_line}_{protein}_pred.tif`: Predicted protein + reference channels
- `{basename}_{cell_line}_{protein}_real.tif`: Ground truth + reference channels (if available)

Output TIFFs have 4 channels in order: DAPI, Protein, Tubulin, ER

## Logging

Synchronized logs across all GPUs are written to `--log_dir`:
- `inference_log_{timestamp}.txt`: Human-readable log
- `metrics_{timestamp}.json`: Machine-parseable metrics
