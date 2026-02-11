# Cell Generation v2

Conditional cell image generation using DiT (Diffusion Transformer) with EDM training.

## Features

- **DiT-based architecture** with protein and cell line conditioning
- **EDM (Elucidated Diffusion Model)** training and sampling
- **Classifier-free guidance** support via label dropout
- Clean, modular codebase with proper package structure

## Installation

```bash
# Clone or copy the repository
cd cell_generation_v2

# Install dependencies
pip install -r requirements.txt

# Install as editable package (optional)
pip install -e .
```

## Quick Start

### Training

```bash
python train.py \
    --data_root /path/to/data \
    --train_csv /path/to/train.csv \
    --test_csv /path/to/test.csv \
    --cellline_map /path/to/cell_line_map.pkl \
    --antibody_map /path/to/antibody_map.pkl \
    --vae_path /path/to/vae \
    --output_dir ./output \
    --train_batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --use_ema \
    --mixed_precision bf16 \
    --logger wandb
```

### Resume Training from Checkpoint

To resume training from a specific checkpoint:

```bash
python train.py \
    --data_root /path/to/images/ \
    --train_csv /path/to/train.csv \
    --test_csv /path/to/test.csv \
    --cellline_map /path/to/cell_line_map.pkl \
    --antibody_map /path/to/antibody_map.pkl \
    --vae_path /path/to/vae_model \
    --output_dir ./output \
    --resume_from_checkpoint ./output/checkpoint-1500 \
    --train_batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --use_ema \
    --mixed_precision bf16 \
    --logger wandb
```

Or resume from the latest checkpoint automatically:

```bash
python train.py \
    [... other arguments ...] \
    --resume_from_checkpoint latest
```

**Check available checkpoints:**
```bash
ls -la ./output/checkpoint-*
```

The script automatically restores model weights, optimizer state, EMA model, and resumes from the correct epoch.

### Sampling

```bash
python sample.py \
    --checkpoint ./output \
    --vae_path /path/to/vae \
    --protein_id 123 \
    --cell_line_id 5 \
    --cond_image /path/to/reference.tif \
    --num_samples 8 \
    --num_inference_steps 250 \
    --output_dir ./generated_images
```

## Project Structure

```
cell_generation_v2/
├── config/
│   ├── __init__.py
│   └── config.py              # Dataclass-based configuration
├── models/
│   ├── __init__.py
│   ├── dit.py                 # DiT model with label conditioning
│   └── transformer_block.py   # Custom transformer block with AdaLN
├── samplers/
│   ├── __init__.py
│   └── conditional_sampler.py # Conditional sampling with labels
├── data/
│   ├── __init__.py
│   └── dataset.py             # Dataset classes
├── utils/
│   ├── __init__.py
│   ├── edm_utils.py           # EDM preconditioning and loss
│   ├── checkpoint_utils.py    # Checkpoint management
│   └── logging_utils.py       # Logging utilities
├── schedulers/
│   ├── __init__.py
│   └── edm_scheduler.py       # EDM scheduler wrapper
├── train.py                   # Training entry point
├── sample.py                  # Sampling entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Model Architecture

The DiT model uses adaptive layer normalization to condition on:

- **Protein labels**: 12,810 protein classes embedded to 512 dimensions
- **Cell line labels**: 42 cell line classes embedded to 512 dimensions

The embeddings are combined with timestep information and used to modulate the transformer layers via `AdaLayerNormZeroContinuous`.

### Forward Pass

```python
model_output = model(
    hidden_states,           # (B, C, H, W) latent input
    timestep,                # (B,) diffusion timestep
    protein_labels=labels,   # (B,) protein label indices
    cell_line_labels=cells,  # (B,) cell line label indices
    encoder_hidden_states=encoder_states,  # Optional CLIP features
)
```

## Configuration

### Data Format

The dataset expects:
- TIFF images with 4 channels: [reference1, protein, reference2, reference3]
- CSV files with image filenames for train/test splits
- Pickle files mapping cell line and protein names to indices

### EDM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_min` | 0.002 | Minimum noise level |
| `sigma_max` | 80.0 | Maximum noise level |
| `sigma_data` | 0.5 | Data standard deviation |
| `rho` | 7.0 | EDM schedule parameter |

## Training Tips

1. **Label dropout**: Use `--label_dropout_prob 0.2` for classifier-free guidance
2. **Mixed precision**: Use `--mixed_precision bf16` for faster training
3. **EMA**: Enable `--use_ema` for better sample quality
4. **Checkpointing**: Adjust `--checkpointing_steps` based on dataset size

## Key Differences from v1

1. **Simplified architecture**: Only DiT model (removed UNet variants)
2. **Removed classifier guidance**: No twisted/SMC sampling
3. **Consolidated samplers**: 4 files merged into 1
4. **Removed hardcoded paths**: All paths are configurable
5. **Proper package structure**: Added `__init__.py` files
6. **Clean code**: Removed ~200 lines of commented code

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

See `requirements.txt` for full dependencies.

## License

MIT License
