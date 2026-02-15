# ProtVL

**Proteome Virtual Labeling** — Generate predicted protein localization images from reference microscopy channels using a Diffusion Transformer (DiT) with EDM scheduling.

## Installation

```bash
git clone https://github.com/YOUR_ORG/protvl.git
cd protvl
pip install -e .
```

For training extras (TensorBoard/WandB logging):

```bash
pip install -e ".[train]"
```

## Quick Start

### 1. Download checkpoints (first time only)

```python
from protvl import ProtVL

ProtVL.download_checkpoints()
```

### 2. Predict

```python
from protvl import ProtVL
from tifffile import imread

model = ProtVL()

img = imread("my_cell.tiff")  # [H, W, 3] or [H, W, 4]
results = model.predict(
    images=[img],
    protein_names=["ACTB"],
    cell_line_names=["A-431"],
)

predicted = results[0]  # numpy [H, W] float32
```

### 3. Fine-tune

```python
import os

model = ProtVL()
model.fit(
    image_dir="./data/train",
    image_files=os.listdir("./data/train"),
    protein_names=["CDT1", "CD8", "CTNNB1", ...],
    cell_line_names=["U-2 OS", "U-2 OS", "A-431", ...],
    output_dir="./finetuned",
    num_epochs=50,
)
```

---

## Detailed API Reference

### `ProtVL(...)` — Constructor

```python
model = ProtVL(
    checkpoint_dir=None,    # str or Path. Default: protvl/checkpoint/
    vae_dir=None,           # str or Path. Default: protvl/vae/
    device=None,            # str. Default: "cuda" if available, else "cpu"
    dtype="float32",        # str. One of "float32", "float16", "bfloat16"
    protein_map=None,       # str, Path, or dict. Default: protvl/data/antibody_map.pkl
    cellline_map=None,      # str, Path, or dict. Default: protvl/data/cell_line_map.pkl
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | `str`, `Path`, or `None` | `protvl/checkpoint/` | Path to the DiT model checkpoint directory. |
| `vae_dir` | `str`, `Path`, or `None` | `protvl/vae/` | Path to the VAE checkpoint directory. |
| `device` | `str` or `None` | `"cuda"` / `"cpu"` | Device to run on. Auto-detects GPU if available. |
| `dtype` | `str` | `"float32"` | Weight precision. Use `"float16"` or `"bfloat16"` to reduce memory. |
| `protein_map` | `str`, `Path`, `dict`, or `None` | `antibody_map.pkl` | Protein-to-label-index mapping. |
| `cellline_map` | `str`, `Path`, `dict`, or `None` | `cell_line_map.pkl` | Cell-line-to-label-index mapping. |

Models are lazy-loaded — weights are only loaded into GPU memory on the first call to `predict()` or `fit()`.

---

### `model.predict(...)` — Inference

```python
results = model.predict(
    images=[img1, img2, img3],          # Required. List of reference images.
    protein_names=["ACTB", "TUBB", "LMNA"],  # Required. One per image.
    cell_line_names=["A-431", "A-431", "U-2 OS"],  # Optional. Defaults to index 0.
    num_inference_steps=50,             # Default: 50
    batch_size=4,                       # Default: 4
    seed=42,                            # Default: None (random)
    return_latents=False,               # Default: False
    show_progress=True,                 # Default: True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `list[np.ndarray]` | *required* | Reference channel images. See [Input Format](#input-format). |
| `protein_names` | `list[str]` | *required* | Target protein/antibody name for each image. Must exist in the model vocabulary. |
| `cell_line_names` | `list[str]` or `None` | `None` | Cell line name for each image. If `None`, uses default (unconditioned). |
| `num_inference_steps` | `int` | `50` | Number of EDM denoising steps. Higher values improve quality but slow down generation. |
| `batch_size` | `int` | `4` | Number of images to process simultaneously. Increase for faster throughput if GPU memory allows. |
| `seed` | `int` or `None` | `None` | Random seed for reproducible results. |
| `return_latents` | `bool` | `False` | If `True`, includes raw latent tensors in the result object. |
| `show_progress` | `bool` | `True` | Show a progress bar during generation. |

**Returns:** `PredictionResult` with:
- `.images` — list of `[H, W]` float32 numpy arrays
- `.latents` — list of latent arrays (if `return_latents=True`)
- `.metadata` — list of dicts with `protein_name` and `cell_line_name` per sample

**Note:** Inference uses the `unet` (ordinary) checkpoint weights.

---

### `model.fit(...)` — Fine-tuning

```python
model.fit(
    image_dir="./data/train",           # Required.
    image_files=["cell_0.tiff", ...],   # Required.
    protein_names=["CDT1", "CD8", ...], # Required.
    cell_line_names=["U-2 OS", ...],    # Optional.
    output_dir="./protvl_finetune",     # Default: "./protvl_finetune"
    num_epochs=100,                     # Default: 100
    batch_size=16,                      # Default: 16
    learning_rate=1e-4,                 # Default: 1e-4
    resume_from=None,                   # Default: None
    # Additional kwargs passed to run_finetuning:
    label_dropout_prob=0.2,
    lr_scheduler_type="cosine",
    lr_warmup_steps=500,
    gradient_accumulation_steps=1,
    checkpointing_steps=500,
    save_model_epochs=10,
    max_grad_norm=1.0,
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_weight_decay=1e-6,
    adam_epsilon=1e-8,
    use_ema=False,
    mixed_precision="no",
    num_workers=4,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_dir` | `str` or `Path` | *required* | Directory containing training TIFF images. |
| `image_files` | `list[str]` | *required* | Filenames within `image_dir`. |
| `protein_names` | `list[str]` | *required* | Target protein name per image. Must match length of `image_files`. |
| `cell_line_names` | `list[str]` or `None` | `None` | Cell line name per image. If `None`, defaults to label index 0. |
| `output_dir` | `str` | `"./protvl_finetune"` | Directory to save fine-tuned checkpoints. |
| `num_epochs` | `int` | `100` | Total number of training epochs. |
| `batch_size` | `int` | `16` | Training batch size per device. |
| `learning_rate` | `float` | `1e-4` | Peak learning rate. |
| `resume_from` | `str` or `None` | `None` | Path to a checkpoint directory to resume training from. |
| `label_dropout_prob` | `float` | `0.2` | Probability of dropping protein/cell line labels during training (classifier-free guidance). |
| `lr_scheduler_type` | `str` | `"cosine"` | Learning rate scheduler. Options: `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"constant"`, `"constant_with_warmup"`. |
| `lr_warmup_steps` | `int` | `500` | Number of warmup steps for the learning rate scheduler. |
| `gradient_accumulation_steps` | `int` | `1` | Number of gradient accumulation steps before each optimizer update. |
| `checkpointing_steps` | `int` | `500` | Save a training checkpoint every N optimizer steps. |
| `save_model_epochs` | `int` | `10` | Save the model every N epochs. |
| `max_grad_norm` | `float` | `1.0` | Maximum gradient norm for gradient clipping. |
| `adam_beta1` | `float` | `0.95` | Adam optimizer beta1. |
| `adam_beta2` | `float` | `0.999` | Adam optimizer beta2. |
| `adam_weight_decay` | `float` | `1e-6` | Adam weight decay. |
| `adam_epsilon` | `float` | `1e-8` | Adam epsilon. |
| `use_ema` | `bool` | `False` | Whether to use Exponential Moving Average during fine-tuning. |
| `mixed_precision` | `str` | `"no"` | Mixed precision mode. Options: `"no"`, `"fp16"`, `"bf16"`. |
| `num_workers` | `int` | `4` | DataLoader workers (automatically set to 0 on Windows). |

**Returns:** `self` (for method chaining).

**Note:** Fine-tuning loads the `unet_ema` (Exponential Moving Average) checkpoint weights as the starting point.

After fine-tuning, load the fine-tuned model in a new session:

```python
model = ProtVL(checkpoint_dir="./finetuned")
```

---

### `model.save(path)` — Save Model

```python
model.save("./my_model")
```

Saves the DiT weights, protein map, and cell line map to the specified directory.

---

### `ProtVL.download_checkpoints(...)` — Download Weights

```python
ProtVL.download_checkpoints(
    dest_dir=None,          # Default: protvl/ package directory
    checkpoint_url="...",   # Default: Stanford ELL vault URL
    vae_url="...",          # Default: Stanford ELL vault URL
)
```

Downloads and extracts pre-trained model weights. Only needed once.

---

### Utility Properties

```python
model.available_proteins    # list[str] — all protein names the model can predict
model.available_cell_lines  # list[str] — all cell line names the model recognizes
model.summary()             # str — human-readable model summary (params, vocab sizes, device)
```

---

## Input Format

**For prediction:** `[H, W, 3]` float32 array with 3 reference channels (nucleus, ER, microtubules) in `[-1, 1]`, or `[H, W, 4]` TIFF where channel 1 is ignored and channels 0, 2, 3 are used.

**For training:** `[H, W, 4]` TIFF where:
- Channel 0 = microtubules
- Channel 1 = protein (ground truth target)
- Channel 2 = nucleus
- Channel 3 = ER

---

## Project Structure

```
protvl-repo/
├── pyproject.toml
├── README.md
├── protvl/
│   ├── __init__.py
│   ├── model.py              # Main ProtVL class (predict, fit, save)
│   ├── _sampling.py           # EDM sampling loop
│   ├── _training.py           # Fine-tuning loop
│   ├── config/
│   │   ├── config.py          # EDMConfig dataclass
│   │   └── default_config.py  # Training argparse config & EDM constants
│   ├── data/
│   │   ├── antibody_map.pkl   # Protein label vocabulary
│   │   └── cell_line_map.pkl  # Cell line label vocabulary
│   ├── models/
│   │   ├── dit.py             # DiT Transformer architecture
│   │   └── basic_transformer_block.py
│   ├── schedulers/
│   │   └── edm_scheduler.py   # EDM noise scheduler
│   └── utils/
│       ├── checkpoint_utils.py
│       ├── download.py
│       ├── edm_utils.py
│       └── logging_utils.py
├── checkpoint/                # Downloaded model weights
│   ├── unet/                  # Ordinary weights (used for inference)
│   └── unet_ema/              # EMA weights (used for fine-tuning)
└── vae/                       # Downloaded VAE weights
```

---

## EDM Configuration

The diffusion process uses Elucidating Diffusion Models (EDM) with these default constants:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SIGMA_MIN` | `0.002` | Minimum noise level |
| `SIGMA_MAX` | `80.0` | Maximum noise level |
| `SIGMA_DATA` | `0.5` | Standard deviation of the data distribution |
| `RHO` | `7` | EDM time step discretization parameter |

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- diffusers >= 0.25.0
- CUDA-capable GPU (recommended)

## License

MIT
