"""Cluster superplexed cells using the v3, B2AI, or 3D publication workflow.

One shared pipeline retains profile-specific input names, preprocessing, marker
rules, and colors. Full KMeans fitting uses disk staging; sampling is opt-in.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import re
import tempfile
import warnings

import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening
from skimage.filters import gaussian, threshold_otsu
from tifffile import imread


GENES = (
    'TPR', 'LMNB1', 'SUN2', 'TMPO', 'LEMD2', 'TOR1AIP1', 'LBR', 'LMNB2', 'CASC3', 'NUP153',
    'DDX47', 'RPF1', 'UTP6', 'NOL10', 'FTSJ3', 'UBTF', 'NAA50', 'NOP53', 'NOP56', 'NOLC1',
    'POLR2K', 'NOP16', 'UTP4', 'NOP2', 'TAF15', 'SMARCAD1', 'SRRM2', 'RBM25', 'PML',
    'MKI67', 'RSL1D1', 'CENPC', 'RPS19', 'RAN', 'HNRNPA1', 'H2AZ1', 'HNRNPC', 'HMGB1',
    'HNRNPK', 'HNRNPA3', 'ATF4', 'MORF4L1', 'SEPTIN9', 'FGD4', 'ZYX', 'VCL', 'GSN',
    'MTSS2', 'TNS3', 'LAD1', 'LIMCH1', 'SPECC1L', 'TNS1', 'ARSJ', 'TUBA1A',
    'TUBA1A,TUBA1B,TUBA1C', 'CS', 'LRPPRC', 'SLC25A24', 'TIMM44', 'GCDH', 'TRAP1',
    'MT-CYB', 'ATP5F1B', 'HSPD1', 'CHCHD2', 'SLC25A3', 'COX4I1', 'ATP5ME', 'ATP5MK',
    'TOMM6', 'TOMM20', 'TOMM40', 'NDUFA13', 'KRT19', 'KRT4', 'DES', 'NES', 'KRT17',
    'KRT13', 'VIM', 'KRT8', 'KRT17', 'KRT14', 'NEFM', 'GFAP', 'KRT80', 'ADSL', 'ATXN2',
    'G3BP2', 'AIMP1', 'SERBP1', 'CCDC43', 'ATXN2L', 'AMPD2', 'RABGAP1', 'GOLGB1',
    'GOLGA5', 'GALNT2', 'ZFPL1', 'GORASP2', 'GOLM1', 'GOLIM4', 'SPP1', 'SRGN', 'CD74',
    'RER1', 'LMAN2', 'COPE', 'SDF4', 'TMED3', 'NUCB2', 'GALNT2', 'STX4', 'SLC16A1',
    'EZR', 'EPB41L3', 'CTNNB1', 'ANK3', 'SLC41A3', 'AP2M1', 'MSN', 'GNB2', 'CD9',
    'SLC38A2', 'EZR', 'ATP1B3', 'ITGA3', 'BASP1', 'HTRA1', 'HSP90B1', 'CANX', 'KTN1',
    'PDIA3', 'RRBP1', 'SEC61B', 'RPL41', 'CALR', 'P4HB', 'CALU', 'COL1A2', 'RTN4',
    'RPN2', 'BCAP31', 'SSR4',
)
COLORS = np.array([
    (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 128),
    (128, 0, 0), (0, 0, 128), (128, 128, 0), (255, 192, 203), (165, 42, 42),
    (0, 255, 0), (135, 206, 235), (75, 0, 130), (255, 215, 0), (192, 192, 192),
], dtype=np.uint8)
V3_COLORS = np.concatenate([COLORS[:10], COLORS[[16, 10]]])
PROFILES = {
    'v3': ('/scratch/users/sunh/cytoVL', None),
    'b2ai': ('latent_diffusion_edm/b2ai_YWHAB_organelle_examples', 'B2AI_YWHAB_clustered_cell'),
    '3d': ('MDA_3D_organelle', 'MDA_3d_clustering'),
}


def _profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _names(values, name):
    if isinstance(values, str):
        raise ValueError(f"{name} must be a nonempty sequence of names")
    result = list(values)
    if not result or any(not isinstance(value, str) or not value or '/' in value or '\\' in value for value in result):
        raise ValueError(f"{name} contains an empty or invalid name")
    return result


def match_genes(genes, gene_order):
    """Retain the original first case-insensitive substring match and duplicates."""
    genes, gene_order = _names(genes, 'genes'), _names(gene_order, 'gene_order')
    matched, indices = [], []
    for gene in genes:
        index = next((i for i, name in enumerate(gene_order) if gene.lower() in name.lower()), None)
        if index is None:
            warnings.warn(f"Gene {gene} not found in gene_order", RuntimeWarning, stacklevel=2)
        else:
            matched.append(gene)
            indices.append(index)
    if not indices:
        raise ValueError("No selected genes match gene_order")
    return matched, indices


def discover_cell_keys(input_dir, profile, *, cell_lines=('A-431',), cell_line='MDA',
                       n_images=200, max_images=800):
    """Discover cells; 3D selection retains whole plate/cell z-plane groups."""
    _profile(profile)
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {root}")
    _integer('n_images', n_images, 1)
    _integer('max_images', max_images, 1)
    if profile == 'v3':
        return [(line, index) for line in _names(cell_lines, 'cell_lines')
                for index in range(n_images)
                if (root / f'{line}_superplexed_{index:03d}_stacked_images.npy').is_file()]
    _names([cell_line], 'cell_line')
    if profile == 'b2ai':
        pattern = re.compile(r'B2AI_1_(.+)_C11_R(\d+)_cell_(\d+)_' + re.escape(cell_line)
                             + r'_(.+)_pred\.tiff?', re.IGNORECASE)
    else:
        pattern = re.compile(r'P(\d+)\.ome_cell_(\d+)_z(\d+)_' + re.escape(cell_line)
                             + r'_(.+)_pred\.tiff?', re.IGNORECASE)
    keys = set()
    for path in sorted(root.iterdir()):
        match = pattern.fullmatch(path.name)
        if path.is_file() and match:
            keys.add(tuple(match.group(i) for i in (1, 2, 3)))
    if profile == 'b2ai':
        return sorted(keys)
    groups = defaultdict(list)
    for plate, cell, z in sorted(keys):
        groups[(plate, cell)].append(z)
    selected = []
    for (plate, cell), planes in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if len(selected) + len(planes) <= max_images:
            selected.extend((plate, cell, z) for z in planes)
    return sorted(selected)


def _cell_paths(key, genes, root, profile, cell_line):
    if profile == 'v3':
        return [root / f'{key[0]}_superplexed_{key[1]:03d}_stacked_images.npy']
    if profile == 'b2ai':
        prefix = f'B2AI_1_{key[0]}_C11_R{key[1]}_cell_{key[2]}_{cell_line}_'
    else:
        prefix = f'P{key[0]}.ome_cell_{key[1]}_z{key[2]}_{cell_line}_'
    paths = []
    for gene in genes:
        path = root / f'{prefix}{gene}_pred.tif'
        alternative = path.with_suffix('.tiff')
        paths.append(alternative if not path.exists() and alternative.exists() else path)
    return paths


def _finite_stack(stack):
    array = np.asarray(stack)
    if array.ndim != 3 or min(array.shape) < 1 or array.dtype.kind not in 'buif':
        raise ValueError("Expected a nonempty, real numeric [H, W, genes] stack")
    if not np.isfinite(array).all():
        raise ValueError("Stack contains non-finite pixel values")
    if np.any(array > np.finfo(np.float32).max) or np.any(array < -np.finfo(np.float32).max):
        raise ValueError("Pixel values exceed float32 range")
    with np.errstate(under='ignore'):
        converted = array.astype(np.float32, copy=False)
    if np.any((array != 0) & (converted == 0)):
        raise ValueError("Pixel values underflow float32 precision")
    return converted


def load_cell_stack(key, genes, input_dir, *, profile='v3', gene_indices=None,
                    cell_line='MDA', channel=1):
    _profile(profile)
    _integer('channel', channel, 0)
    genes = _names(genes, 'genes')
    paths = _cell_paths(key, genes, Path(input_dir), profile, cell_line)
    if profile == 'v3':
        source = np.load(paths[0], mmap_mode='r', allow_pickle=False)
        if source.ndim != 3 or min(source.shape) < 1:
            raise ValueError("NPY image must have shape [H, W, proteins]")
        if (gene_indices is None or len(gene_indices) != len(genes)
                or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= source.shape[2]
                       for i in gene_indices)):
            raise ValueError("Gene indices do not match the NPY protein axis")
        return _finite_stack(source[:, :, gene_indices])
    stack = None
    for index, path in enumerate(paths):
        image = np.asarray(imread(path))
        if image.ndim == 3:
            if channel >= image.shape[2]:
                raise ValueError(f"Channel {channel} unavailable in {path.name}")
            image = image[:, :, channel]
        if image.ndim != 2 or not image.size:
            raise ValueError(f"Expected a 2D or HWC TIFF: {path.name}")
        if stack is not None and stack.shape[:2] != image.shape:
            raise ValueError(f"Gene images have mismatched shapes: {path.name}")
        if stack is None:
            stack = np.empty((*image.shape, len(genes)), dtype=np.float32)
        stack[:, :, index] = _finite_stack(image[:, :, None])[:, :, 0]
        # A selected channel view otherwise retains the entire source TIFF.
        del image
    return stack


def preprocess_stack(stack, profile='v3'):
    """Apply historical per-profile thresholds, returning a new binary stack."""
    _profile(profile)
    stack = _finite_stack(stack)
    output = np.empty(stack.shape, dtype=bool)
    sigma, multiplier = (0.5, 1.25) if profile == 'v3' else (1.0, 2.0)
    for index in range(stack.shape[2]):
        # Float32 Otsu variance can overflow/underflow on finite images. Work
        # in float64 so positive intensity scaling preserves the segmentation.
        blurred = gaussian(stack[:, :, index].astype(np.float64), sigma=sigma)
        binary = blurred > multiplier * threshold_otsu(blurred)
        if profile != 'v3':
            binary = binary_opening(binary, structure=np.ones((2, 2)))
        output[:, :, index] = binary
    return output


def reorder_clusters(model):
    """Sort labels by training population, retaining all requested cluster slots."""
    centers = np.asarray(getattr(model, 'binary_centers_', model.cluster_centers_))
    labels = np.asarray(model.labels_)
    if (centers.ndim != 2 or not centers.size or labels.ndim != 1 or not labels.size
            or labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= len(centers)
            or not np.isfinite(centers).all()):
        raise ValueError("Invalid fitted KMeans labels or centers")
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts, kind='stable')
    return order, counts[order], centers[order]


def expression_matrices(centers):
    """Historical gene-by-cluster expression, specificity, and enrichment."""
    centers = np.asarray(centers)
    if (centers.ndim != 2 or min(centers.shape) < 1 or centers.shape[0] < 2
            or centers.dtype.kind not in 'buif' or not np.isfinite(centers).all()):
        raise ValueError("Expected finite real centers for at least two clusters")
    expression = centers.astype(np.float64).T
    mean = expression.mean(axis=1, keepdims=True)
    std = expression.std(axis=1, keepdims=True)
    total = expression.sum(axis=1, keepdims=True)
    other = (total - expression) / (expression.shape[1] - 1)
    z_score = np.divide(expression - mean, std, out=np.zeros_like(expression), where=std > 1e-6)
    specificity = expression / (other + 1e-6)
    fold = expression / (mean + 1e-6)
    fraction = expression / (total + 1e-6)
    return {'expression': expression, 'fold_change': specificity, 'z_score': z_score,
            'specificity': specificity, 'fraction': fraction, 'fold': fold,
            'enrichment': (2 * specificity + z_score + fold) / 4}


def select_marker_genes(centers, genes, *, profile='3d', fallback_top_n=10,
                        allow_multi=True, max_secondary=1):
    if profile not in ('b2ai', '3d'):
        raise ValueError("Marker selection requires the b2ai or 3d profile")
    genes = _names(genes, 'genes')
    _integer('fallback_top_n', fallback_top_n, 0)
    _integer('max_secondary', max_secondary, 0)
    matrices = expression_matrices(centers)
    expression = matrices['expression']
    if len(genes) != len(expression):
        raise ValueError("Gene names do not match the centers")
    count = expression.shape[1]
    result = {cluster: [] for cluster in range(count)}

    def entry(gene, cluster, assignment):
        return {'gene': genes[gene], 'gene_idx': gene, 'assignment': assignment,
                'expr': float(expression[gene, cluster]),
                'fold': float(matrices['fold'][gene, cluster]),
                'z': float(matrices['z_score'][gene, cluster]),
                'spec': float(matrices['specificity'][gene, cluster]),
                'frac': float(matrices['fraction'][gene, cluster]),
                'enrich': float(matrices['enrichment'][gene, cluster])}

    for gene in range(len(genes)):
        if profile == 'b2ai':
            cluster = int(np.argmax(expression[gene]))
            expr = expression[gene, cluster]
            mean_other = np.delete(expression[gene], cluster).mean()
            if expr > 0.3 and expr > 1.5 * mean_other:
                record = entry(gene, cluster, 'primary')
                record['score'] = record['expr'] * record['spec']
                result[cluster].append(record)
            continue
        if expression[gene].sum() < 1e-6:
            continue
        ranked = np.argsort(-matrices['enrichment'][gene], kind='stable')
        primary = int(ranked[0])
        if not (matrices['z_score'][gene, primary] >= 1
                and matrices['specificity'][gene, primary] >= 2
                and matrices['fraction'][gene, primary] >= 0.4):
            continue
        result[primary].append(entry(gene, primary, 'primary'))
        if allow_multi:
            assigned = 0
            for cluster in ranked[1:]:
                if assigned >= max_secondary:
                    break
                if (expression[gene, cluster] >= 0.9 * expression[gene, primary]
                        and matrices['specificity'][gene, cluster] >= 2
                        and matrices['fraction'][gene, cluster] >= 0.35):
                    result[int(cluster)].append(entry(gene, int(cluster), 'secondary'))
                    assigned += 1
                if matrices['enrichment'][gene, cluster] < 0.6 * matrices['enrichment'][gene, primary]:
                    break
    for cluster, records in result.items():
        field = 'expr' if profile == 'b2ai' else 'enrich'
        records.sort(key=lambda record: record[field], reverse=True)
        if not records and profile == '3d':
            top = np.argsort(-expression[:, cluster], kind='stable')[:fallback_top_n]
            records.extend(entry(int(gene), cluster, 'fallback') for gene in top)
        for rank, record in enumerate(records, 1):
            record['rank'] = rank
    return result


def save_matrix(path, matrix, gene_names, col_names):
    order = np.argsort(-matrix.mean(axis=1), kind='stable')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Gene', *col_names])
        for gene in order:
            writer.writerow([gene_names[gene], *[f'{value:.6f}' for value in matrix[gene]]])


def _write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _write_markers(output, markers, counts, colors, profile):
    fields = ['Cluster_ID', 'RGB_Color', 'Cluster_Size', 'Percentage', 'Rank', 'Gene_Name', 'Expression']
    fields += (['Specificity', 'Combined_Score'] if profile == 'b2ai' else
               ['Z_Score', 'Specificity', 'Fraction', 'Enrichment', 'Assignment', 'Fallback_Cluster'])
    with (output / 'cluster_gene_analysis.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for cluster, records in markers.items():
            for record in records:
                row = [cluster, str(tuple(int(v) for v in colors[cluster])), int(counts[cluster]),
                       f'{100 * counts[cluster] / counts.sum():.2f}', record['rank'], record['gene'],
                       f"{record['expr']:.4f}"]
                if profile == 'b2ai':
                    row += [f"{record['spec']:.4f}", f"{record['score']:.4f}"]
                else:
                    row += [f"{record['z']:.4f}", f"{record['spec']:.4f}", f"{record['frac']:.4f}",
                            f"{record['enrich']:.4f}", record['assignment'],
                            'YES' if record['assignment'] == 'fallback' else 'NO']
                writer.writerow(row)
    with (output / 'cluster_gene_analysis.txt').open('w', encoding='utf-8') as handle:
        handle.write('CLUSTER MARKER GENE ANALYSIS\n' + '=' * 80 + '\n')
        handle.write('Specificity = expression / (mean expression in other clusters + 1e-6)\n')
        handle.write('Enrichment = (2*specificity + z_score + fold_change) / 4\n')
        for cluster, records in markers.items():
            handle.write(f'\nCluster {cluster}: {counts[cluster]} training pixels\n')
            for record in records:
                handle.write(f"  {record['rank']}. {record['gene']}: expression={record['expr']:.4f}, "
                             f"specificity={record['spec']:.4f}, {record['assignment']}\n")


def _output_name(key, profile):
    if profile == 'v3':
        return f'{key[0]}_clustered_cell_{key[1]:03d}.png'
    if profile == 'b2ai':
        return f'clustered_{key[0]}_R{key[1]}_cell{key[2]}.png'
    return f'clustered_P{key[0]}_cell{key[1]}_z{key[2]}.png'


def _fingerprint(config, fit_keys, genes, root, profile, cell_line):
    files = {}
    for key in fit_keys:
        for path in _cell_paths(key, genes, root, profile, cell_line):
            if path.is_file():
                stat = path.stat()
                files[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns]
            else:
                files[str(path.resolve())] = None
    payload = {'config': config, 'files': files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
    return digest


def _fit_model(chunks, output, n_genes, n_clusters, seed, n_init, max_fit_pixels,
               save_pixels, fit_dtype=np.float32):
    """Stage full data on disk; optional priority reservoir samples pixels uniformly."""
    from sklearn.cluster import KMeans

    total, sampled, priorities = 0, None, None
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix='.cluster_pixels_', dir=output) as temporary:
        pixel_path = Path(temporary) / 'pixels.bin'
        with pixel_path.open('wb') as handle:
            for chunk in chunks:
                pixels = chunk.reshape(-1, n_genes)
                total += len(pixels)
                if max_fit_pixels is None:
                    pixels.astype(fit_dtype).tofile(handle)
                else:
                    scores = rng.random(len(pixels))
                    if sampled is not None:
                        pixels = np.concatenate([sampled, pixels])
                        scores = np.concatenate([priorities, scores])
                    if len(pixels) > max_fit_pixels:
                        selected = np.argpartition(scores, max_fit_pixels - 1)[:max_fit_pixels]
                        pixels, scores = pixels[selected], scores[selected]
                    sampled, priorities = pixels, scores
            if sampled is not None:
                sampled.astype(fit_dtype).tofile(handle)
        used = total if max_fit_pixels is None else min(total, max_fit_pixels)
        if used < n_clusters:
            raise ValueError(f'Only {used} valid training pixels; need at least {n_clusters}')
        data = np.memmap(pixel_path, dtype=fit_dtype, mode='r+', shape=(used, n_genes))
        try:
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init,
                           copy_x=False, algorithm='lloyd').fit(data)
            _, counts, _ = reorder_clusters(model)
            if np.any(counts == 0):
                raise ValueError('Fewer distinct pixel clusters than requested; reduce --n-clusters')
            # Derive reporting centroids from exact binary sufficient statistics.
            # Small positive KMeans residuals at zero get amplified by the 1e-6
            # denominator in enrichment ratios. Restore the 0/1 staging rows
            # after KMeans' in-place centering before accumulating integer sums.
            sums = np.zeros((n_clusters, n_genes), dtype=np.int64)
            block_rows = max(1, (8 * 1024 * 1024) // (n_genes * data.dtype.itemsize))
            for start in range(0, used, block_rows):
                block = data[start:start + block_rows]
                np.rint(block, out=block)
                labels = model.labels_[start:start + block_rows]
                for cluster in np.unique(labels):
                    sums[cluster] += np.count_nonzero(block[labels == cluster], axis=0)
            label_counts = np.bincount(model.labels_, minlength=n_clusters)
            model.binary_centers_ = sums / label_counts[:, None]
            if save_pixels:
                np.save(output / 'all_pixels.npy', data)
        finally:
            data._mmap.close()
    return model, total, used


def run_clustering(input_dir=None, output_dir=None, *, profile='v3', genes=None, gene_order=None,
                   cell_lines=('A-431',), cell_line='MDA', n_clusters=12, seed=42,
                   n_images_fit=100, n_images=200, max_images=800, max_fit_pixels=None,
                   n_init='auto', reuse_model=False, channel=1, fallback_top_n=10,
                   allow_multi=True, save_pixels=False):
    """Fit exact global KMeans and export population-ranked maps and marker tables."""
    _profile(profile)
    for name, value, minimum in [('n_clusters', n_clusters, 2), ('seed', seed, 0),
                                  ('n_images_fit', n_images_fit, 1), ('n_images', n_images, 1),
                                  ('max_images', max_images, 1), ('channel', channel, 0),
                                  ('fallback_top_n', fallback_top_n, 0)]:
        _integer(name, value, minimum)
    if max_fit_pixels is not None:
        _integer('max_fit_pixels', max_fit_pixels, n_clusters)
    if n_init != 'auto':
        _integer('n_init', n_init, 1)
        n_init = int(n_init)
    palette = V3_COLORS if profile == 'v3' else COLORS
    if n_clusters > len(palette):
        raise ValueError(f'This profile supports at most {len(palette)} cluster colors')
    cell_lines = _names(cell_lines, 'cell_lines')
    _names([cell_line], 'cell_line')
    selected_genes = list(GENES) if genes is None else _names(genes, 'genes')
    if genes is None and profile == 'b2ai':
        selected_genes = [gene.replace('TUBA1A,TUBA1B,TUBA1C', 'TUBA1A,TUBA1B,TUBA1C,TUBA3C,TU')
                          for gene in selected_genes]
    indices = None
    if profile == 'v3':
        if gene_order is None or isinstance(gene_order, (str, Path)):
            with Path('protein_order.pkl' if gene_order is None else gene_order).open('rb') as handle:
                gene_order = pickle.load(handle)
        selected_genes, indices = match_genes(selected_genes, gene_order)
    root = Path(PROFILES[profile][0] if input_dir is None else input_dir)
    default_output = PROFILES[profile][1] or f"spatial_clusters_v3/{'_'.join(cell_lines)}_organelle_markers_clustered_cell"
    output = Path(default_output if output_dir is None else output_dir)
    keys = discover_cell_keys(root, profile, cell_lines=cell_lines, cell_line=cell_line,
                              n_images=max(n_images, n_images_fit), max_images=max_images)
    fit_keys = [key for key in keys if profile != 'v3' or key[1] < n_images_fit]
    render_keys = [key for key in keys if profile != 'v3' or key[1] < n_images]
    if not fit_keys:
        raise ValueError('No training cells found for the selected profile and limits')
    output.mkdir(parents=True, exist_ok=True)
    config = {'profile': profile, 'genes': selected_genes, 'gene_indices': indices,
              'n_clusters': int(n_clusters), 'seed': int(seed), 'n_init': n_init,
              'channel': int(channel), 'cell_line': cell_line, 'cell_lines': cell_lines,
              'n_images_fit': int(n_images_fit), 'max_images': int(max_images),
              'max_fit_pixels': int(max_fit_pixels) if max_fit_pixels is not None else None,
              'save_pixels': bool(save_pixels), 'sklearn_version': version('scikit-learn'),
              'fit_dtype': 'float64' if profile == 'v3' else 'float32',
              'preprocessing_version': 2}
    digest = _fingerprint(config, fit_keys, selected_genes, root, profile, cell_line)
    model_path, model_metadata_path = output / 'kmeans_model.pkl', output / 'model_metadata.json'
    training_images, skipped = [], {'fit': {}, 'render': {}}

    def load(key):
        stack = load_cell_stack(key, selected_genes, root, profile=profile, gene_indices=indices,
                                cell_line=cell_line, channel=channel)
        if profile == '3d' and stack.min() == stack.max():
            raise ValueError('Constant 3D image stack')
        return preprocess_stack(stack, profile)

    def chunks():
        for key in fit_keys:
            try:
                stack = load(key)
            except (ValueError, OSError, EOFError) as error:
                skipped['fit'][str(key)] = str(error)
                warnings.warn(f'Skipping training cell {key}: {error}', RuntimeWarning, stacklevel=2)
                continue
            training_images.append(list(key))
            yield stack

    reused = bool(reuse_model and model_path.exists())
    if reused:
        if not model_metadata_path.exists():
            raise ValueError('Cached model has no metadata; rerun without --reuse-model to refit')
        with model_metadata_path.open(encoding='utf-8') as handle:
            cached = json.load(handle)
        if cached.get('fingerprint') != digest:
            raise ValueError('Cached model does not match inputs/configuration; rerun without --reuse-model to refit')
        if save_pixels and not (output / 'all_pixels.npy').is_file():
            raise ValueError('Cached training pixels are missing; rerun without --reuse-model to refit')
        with model_path.open('rb') as handle:
            model = pickle.load(handle)
        training_images, total, used = cached['training_images'], cached['available_pixels'], cached['fit_pixels']
        skipped['fit'] = cached['skipped_fit']
    else:
        model, total, used = _fit_model(chunks(), output, len(selected_genes), n_clusters, seed,
                                        n_init, max_fit_pixels, save_pixels,
                                        np.float64 if profile == 'v3' else np.float32)
        with model_path.open('wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(model_metadata_path, {'fingerprint': digest, 'configuration': config,
                                        'training_images': training_images, 'available_pixels': total,
                                        'fit_pixels': used, 'skipped_fit': skipped['fit']})
    order, counts, centers = reorder_clusters(model)
    if centers.shape != (n_clusters, len(selected_genes)):
        raise ValueError('Cached model shape does not match the selected genes and clusters; refit')
    # Binary pixel means are probabilities; remove floating roundoff outside [0,1].
    matrices = expression_matrices(np.clip(centers, 0, 1))
    columns = [f'Cluster_{index}' for index in range(n_clusters)]
    save_matrix(output / 'expression_global.csv', matrices['expression'], selected_genes, columns)
    save_matrix(output / 'fold_change_global.csv', matrices['fold_change'], selected_genes, columns)
    if profile != 'v3':
        markers = select_marker_genes(np.clip(centers, 0, 1), selected_genes, profile=profile,
                                      fallback_top_n=fallback_top_n, allow_multi=allow_multi)
        _write_markers(output, markers, counts, palette, profile)
    inverse = np.empty(n_clusters, dtype=np.intp)
    inverse[order] = np.arange(n_clusters)
    rendered = []
    for key in render_keys:
        try:
            stack = load(key)
        except (ValueError, OSError, EOFError) as error:
            skipped['render'][str(key)] = str(error)
            warnings.warn(f'Skipping render cell {key}: {error}', RuntimeWarning, stacklevel=2)
            continue
        pixels = stack.reshape(-1, len(selected_genes)).astype(model.cluster_centers_.dtype)
        labels = inverse[model.predict(pixels)].reshape(stack.shape[:2])
        name = _output_name(key, profile)
        Image.fromarray(palette[labels]).save(output / name)
        rendered.append(name)
    summary = {'profile': profile, 'genes': selected_genes, 'cluster_counts': counts.tolist(),
               'training_images': training_images, 'rendered_images': rendered, 'skip_reasons': skipped,
               'fit_pixels': int(used), 'available_pixels': int(total), 'reused_model': reused,
               'configuration': {**config, 'n_images': int(n_images), 'fallback_top_n': int(fallback_top_n),
                                 'allow_multi': bool(allow_multi)},
               'provenance': {'input_dir': str(root.resolve()), 'model_fingerprint': digest,
                              'package_versions': {name: version(name) for name in
                                                   ('numpy', 'scipy', 'scikit-image', 'scikit-learn', 'tifffile')}}}
    _write_json(output / 'run_metadata.json', summary)
    print(f"Saved {len(rendered)} cluster maps to {output} ({used:,}/{total:,} training pixels)")
    return summary


def _n_init(value):
    if value == 'auto':
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('n-init must be auto or a positive integer') from error
    if parsed < 1:
        raise argparse.ArgumentTypeError('n-init must be auto or a positive integer')
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='v3')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--gene-order', default='protein_order.pkl')
    parser.add_argument('--genes', nargs='+')
    parser.add_argument('--cell-lines', nargs='+', default=['A-431'])
    parser.add_argument('--cell-line', default='MDA')
    parser.add_argument('--n-clusters', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-images-fit', type=int, default=100)
    parser.add_argument('--n-images', type=int, default=200)
    parser.add_argument('--max-images', type=int, default=800)
    parser.add_argument('--max-fit-pixels', type=int)
    parser.add_argument('--n-init', type=_n_init, default='auto')
    parser.add_argument('--reuse-model', action='store_true')
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--fallback-top-n', type=int, default=10)
    parser.add_argument('--allow-multi', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-pixels', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run_clustering(**vars(args))
    except (ValueError, OSError, EOFError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
