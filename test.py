import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

# from skimage.filters import threshold_otsu, gaussian
from sklearn.cluster import KMeans
from tifffile import imread, imwrite
from tqdm.auto import tqdm

from protvl import ProtVL
import pickle

gene_name_map = pickle.load(open("protvl/data/antibody_map.pkl", "rb"))
all_gene_names = set(gene_name_map.keys())

model = ProtVL()
image_dir = "./A-431"  # directory containing the TIFF images
image_files = os.listdir(image_dir)  # ["cell_0.tiff", "cell_1.tiff", ...]

# randomly generate protein names for each image (replace with actual mapping if available)
protein_names = [np.random.choice(list(all_gene_names)) for _ in image_files]  # e.g., ["EGFR", "HER2", ...]
cell_line_names = ["A-431"] * len(image_files)   # optional

model.fit(
    image_dir=image_dir,
    image_files=image_files,
    protein_names=protein_names,
    cell_line_names=cell_line_names,
    output_dir="./finetuned",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
)