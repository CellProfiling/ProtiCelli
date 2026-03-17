import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

# from skimage.filters import threshold_otsu, gaussian
from sklearn.cluster import KMeans
from tifffile import imread, imwrite
from tqdm.auto import tqdm

from protvs import ProtVS
import pickle

gene_name_map = pickle.load(open("protvs/data/antibody_map.pkl", "rb"))
all_gene_names = set(gene_name_map.keys())

model = ProtVS()



img1 = r"D:\protVL_standalone\example_images\cell_1.tiff"
img1 = imread(img1)
img2 = r"D:\protVL_standalone\example_images\cell_2.tiff"
img2 = imread(img2)

results = model.predict(
    images=[img1, img2],          # Required. List of reference images.
    protein_names=["COL12A1", "COL12A1"],  # Required. One per image.
    cell_line_names=["U-251MG", "U-251MG"],  # Optional. Defaults to index 0.
    num_inference_steps=50,             # Default: 50
    batch_size=4,                       # Default: 4
    seed=42,                            # Default: None (random)
    return_latents=False,               # Default: False
    show_progress=True,                 # Default: True
    )


for i, image in enumerate(results['images']):
    # convert to uint8
    image = (image * 255).astype(np.uint8)
    imwrite(f"cell_{i}.tiff", image)