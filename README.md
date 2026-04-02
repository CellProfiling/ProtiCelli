# ProtiCelli

**ProtiCelli establishes a foundation for spatial virtual cell modeling** — it generates virtual microscopy images of nearly proteome-wide human protein staining patterns in single cells from input images containing three cellular landmark channels: nucleus, endoplasmic reticulum (ER), and microtubules.

<p align="center">
  <img src="assets/image.png" height="300px" />
  &nbsp;&nbsp;
  <img src="assets/all_cell_lines_protein_tour.gif" height="300px" />
</p>

## Installation

```bash
git clone https://github.com/CellProfiling/proticelli.git
cd proticelli
pip install -e .
```

For training extras (TensorBoard/WandB logging):

```bash
pip install -e ".[train]"
```

---

## Quick Start

### 1. Download checkpoints (first time only)

```python
from proticelli import Model

Model.download_checkpoints()
```

### 2. Assemble channels from separate files

If your channels are stored as individual files, use `ChannelAssembler` to build a single stack:

```python
from proticelli.data import ChannelAssembler

# Inference — no protein channel needed
stack = ChannelAssembler(has_protein=False).transform({
    "microtubules": "mt.tif",
    "nucleus":      "nucleus.tif",
    "er":           "er.tif",
})
# stack.shape → (H, W, 4), channel 1 (protein) filled with zeros

# Training — include the ground-truth protein channel
stack = ChannelAssembler(has_protein=True).transform({
    "microtubules": "mt.tif",
    "nucleus":      "nucleus.tif",
    "er":           "er.tif",
    "protein":      "protein.tif",
})
```

### 3. Normalize images

All inputs to the model must be normalized to `[-1, 1]`. Use `ImageNormalizer` on any stack, whether assembled from separate files or loaded directly:

```python
from proticelli.data import ImageNormalizer

norm = ImageNormalizer(bit_depth=16).transform(stack, save_path="cell_norm.tif")
# norm.shape → (H, W, 4), float32, values in [-1, 1]
# also written to cell_norm.tif
```

Each image is normalized independently — no fitting step is required. The same normalizer instance can be reused across a dataset:

```python
normalizer = ImageNormalizer(bit_depth=16)
norm_train = normalizer.transform(train_stack, save_path="train_norm.tif")
norm_test  = normalizer.transform(test_stack,  save_path="test_norm.tif")
```

### 4. Predict a single protein

```python
from proticelli import Model
from tifffile import imread

model = Model()

img = imread("my_cell.tiff")  # [H, W, 3] or [H, W, 4], normalized to [-1, 1]
results = model.predict(
    images=[img],
    protein_names=["ACTB"],
    cell_line_names=["A-431"],
)

predicted = results[0]  # numpy [H, W] float32
```

### 5. Predict a batch

```python
results = model.predict(
    images=[img1, img2, img3],
    protein_names=["ACTB", "TUBB", "LMNA"],
    cell_line_names=["A-431", "A-431", "U-2 OS"],
)

results.show_prediction()                                        # visualize in matplotlib
results.save_prediction(prefix="exp1", directory="./outputs")   # save as TIFFs
```

### 6. Fine-tune on new data

```python
import os

model = Model()
model.fit(
    image_dir="./data/train",
    image_files=os.listdir("./data/train"),
    protein_names=["CDT1", "CD8", "CTNNB1"],
    cell_line_names=["U-2 OS", "U-2 OS", "A-431"],
    output_dir="./finetuned",
    num_epochs=50,
)
```

Load the fine-tuned model in a new session:

```python
model = Model(checkpoint_dir="./finetuned")
```

---

## API Reference

### `Model.download_checkpoints(...)` — Download Weights

Downloads and extracts pre-trained model weights. Only needed once.

```python
Model.download_checkpoints(
    dest_dir=None,          # Default: proticelli/ package directory
    checkpoint_url="...",   # Default: Stanford ELL vault URL
    vae_url="...",          # Default: Stanford ELL vault URL
)
```

---

### `Model(...)` — Constructor

```python
model = Model(
    checkpoint_dir=None,    # str or Path. Default: proticelli/checkpoint/
    vae_dir=None,           # str or Path. Default: proticelli/vae/
    device=None,            # str. Default: "cuda" if available, else "cpu"
    dtype="float32",        # str. One of "float32", "float16", "bfloat16"
    protein_map=None,       # str, Path, or dict. Default: proticelli/data/antibody_map.pkl
    cellline_map=None,      # str, Path, or dict. Default: proticelli/data/cell_line_map.pkl
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `checkpoint_dir` | `str`, `Path`, or `None` | `proticelli/checkpoint/` | Path to the DiT model checkpoint directory. |
| `vae_dir` | `str`, `Path`, or `None` | `proticelli/vae/` | Path to the VAE checkpoint directory. |
| `device` | `str` or `None` | `"cuda"` / `"cpu"` | Device to run on. Auto-detects GPU if available. |
| `dtype` | `str` | `"float32"` | Weight precision. Use `"float16"` or `"bfloat16"` to reduce memory. |
| `protein_map` | `str`, `Path`, `dict`, or `None` | `antibody_map.pkl` | Protein-to-label-index mapping. |
| `cellline_map` | `str`, `Path`, `dict`, or `None` | `cell_line_map.pkl` | Cell-line-to-label-index mapping. |

Models are lazy-loaded — weights are only loaded into GPU memory on the first call to `predict()` or `fit()`.

#### Utility Properties

```python
model.available_proteins    # list[str] — all protein names the model can predict
model.available_cell_lines  # list[str] — all cell line names the model recognizes
model.summary()             # str — human-readable model summary (params, vocab sizes, device)
```

---

### `model.predict(...)` — Inference

Uses the `unet` (ordinary) checkpoint weights.

```python
results = model.predict(
    images=[img1, img2, img3],
    protein_names=["ACTB", "TUBB", "LMNA"],
    cell_line_names=["A-431", "A-431", "U2OS"],
    num_inference_steps=50,
    batch_size=4,
    seed=42,
    return_latents=False,
    show_progress=True,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
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

---

### `PredictionResult` — Methods

#### `results.show_prediction()`

Display all predicted images in a matplotlib figure with cell line / protein titles.

```python
results.show_prediction()
```

#### `results.save_prediction(prefix="", directory="./")`

Save predicted images as 8-bit TIFF files.

```python
results.save_prediction(prefix="exp1", directory="./outputs")
# Saves: outputs/exp1_0_U-251MG_cell_COL12A1.tif, ...
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefix` | `str` | `""` | Filename prefix. If empty, files are named `{index}_{cell_line}_cell_{protein}.tif`. |
| `directory` | `str` | `"./"` | Output directory. Created automatically if it does not exist. |

Filenames follow the pattern `{prefix}_{index}_{cell_line}_cell_{protein}.tif`.

---

### `model.fit(...)` — Fine-tuning

Uses the `unet_ema` (Exponential Moving Average) checkpoint weights as the starting point.

```python
model.fit(
    image_dir="./data/train",
    image_files=["cell_0.tiff", ...],
    protein_names=["CDT1", "CD8", ...],
    cell_line_names=["U2OS", ...],
    output_dir="./proticelli_finetune",
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    resume_from=None,
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
| --- | --- | --- | --- |
| `image_dir` | `str` or `Path` | *required* | Directory containing training TIFF images. |
| `image_files` | `list[str]` | *required* | Filenames within `image_dir`. |
| `protein_names` | `list[str]` | *required* | Target protein name per image. Must match length of `image_files`. |
| `cell_line_names` | `list[str]` or `None` | `None` | Cell line name per image. If `None`, defaults to label index 0. |
| `output_dir` | `str` | `"./proticelli_finetune"` | Directory to save fine-tuned checkpoints. |
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

---

### `model.save(path)` — Save Model

```python
model.save("./my_model")
```

Saves the DiT weights, protein map, and cell line map to the specified directory.

---

### `ChannelAssembler` — Build a channel stack from separate files

```python
from proticelli.data import ChannelAssembler

# Inference (no protein channel)
assembler = ChannelAssembler(has_protein=False)
stack = assembler.transform({
    "microtubules": "mt.tif",
    "nucleus":      "nucleus.tif",
    "er":           "er.tif",
})
# stack.shape → (H, W, 4), channel 1 is zeros

# Training (include ground-truth protein channel)
assembler = ChannelAssembler(has_protein=True)
stack = assembler.transform({
    "microtubules": "mt.tif",
    "nucleus":      "nucleus.tif",
    "er":           "er.tif",
    "protein":      "protein.tif",
})
```

Each dict value accepts a file path **or** a numpy array. Files saved as `(1, H, W)` or `(H, W, 1)` are automatically squeezed to `(H, W)`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `has_protein` | `bool` | `True` | Whether to expect a `"protein"` key. If `False`, channel 1 is filled with zeros. |

---

### `ImageNormalizer` — Normalize to `[-1, 1]`

```python
from proticelli.data import ImageNormalizer

normalizer = ImageNormalizer(bit_depth=16)
norm = normalizer.transform(stack, save_path="cell_norm.tif")
# norm.shape → (H, W, C), float32, values in [-1, 1]
```

**Algorithm:**

1. Compute a clip threshold from the **MT channel** (channel 0) at `percentile` (default 99.95), capped at the bit-depth maximum (255 for 8-bit, 65535 for 16-bit).
2. Apply that single clip value to **all channels** (preserves relative scale). Set `clip_channel=None` to clip each channel independently.
3. **Global normalization** — divide all channels by the clipped MT-channel max.
4. **Per-channel fallback** — if any channel's max is less than `scale_threshold × MT_max`, normalize each channel by its own max instead.
5. Rescale `[0, 1] → [-1, 1]`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `bit_depth` | `int` | `8` | Input bit depth (`8` or `16`). Caps the clip threshold at 255 or 65535. |
| `percentile` | `float` | `99.95` | Percentile of the reference channel used to compute the clip threshold. |
| `clip_channel` | `int \| None` | `0` | Channel whose percentile sets the clip for all channels. `None` clips each channel independently. |
| `scale_threshold` | `float` | `0.1` | Fraction of MT max below which per-channel normalization replaces global normalization. |

`transform(X, save_path=None)` — `save_path` optionally writes the normalized result as a float32 TIFF. For a batch `[N, H, W, C]`, one file per image is written as `{stem}_{i}.tif`.

Each image is normalized independently using its own MT-channel statistics. The same normalizer instance can be reused across a dataset:

```python
normalizer = ImageNormalizer(bit_depth=16)
norm_train = normalizer.transform(train_stack, save_path="train_norm.tif")
norm_test  = normalizer.transform(test_stack,  save_path="test_norm.tif")
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

## EDM Configuration

The diffusion process uses Elucidating Diffusion Models (EDM) with these default constants:

| Parameter | Value | Description |
| --- | --- | --- |
| `SIGMA_MIN` | `0.002` | Minimum noise level |
| `SIGMA_MAX` | `80.0` | Maximum noise level |
| `SIGMA_DATA` | `0.5` | Standard deviation of the data distribution |
| `RHO` | `7` | EDM time step discretization parameter |

---

## Project Structure

```text
proticelli-repo/
├── pyproject.toml
├── README.md
├── proticelli/
│   ├── __init__.py
│   ├── model.py              # Main Model class (predict, fit, save)
│   ├── _sampling.py          # EDM sampling loop
│   ├── _training.py          # Fine-tuning loop
│   ├── config/
│   │   ├── config.py         # EDMConfig dataclass
│   │   └── default_config.py # Training argparse config & EDM constants
│   ├── data/
│   │   ├── preprocessing.py  # ChannelAssembler, ImageNormalizer
│   │   ├── antibody_map.pkl  # Protein label vocabulary
│   │   └── cell_line_map.pkl # Cell line label vocabulary
│   ├── models/
│   │   ├── dit.py            # DiT Transformer architecture
│   │   └── basic_transformer_block.py
│   ├── schedulers/
│   │   └── edm_scheduler.py  # EDM noise scheduler
│   └── utils/
│       ├── checkpoint_utils.py
│       ├── download.py
│       ├── edm_utils.py
│       └── logging_utils.py
├── checkpoint/               # Downloaded model weights
│   ├── unet/                 # Ordinary weights (used for inference)
│   └── unet_ema/             # EMA weights (used for fine-tuning)
└── vae/                      # Downloaded VAE weights
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- diffusers >= 0.25.0
- CUDA-capable GPU (recommended)

---

## License

MIT
