from skimage.transform import rotate
import numpy as np
from math import sqrt, floor, ceil
import os
import pandas as pd
from skimage.io import imread


def MidPoint(mask):
    row_summed = np.sum(mask, axis=1)
    col_summed = np.sum(mask, axis=0)
    half_row = np.sum(row_summed) / 2
    half_col = np.sum(col_summed) / 2
    row_mid = next(i for i, n in enumerate(np.add.accumulate(row_summed)) if n > half_row)
    col_mid = next(i for i, n in enumerate(np.add.accumulate(col_summed)) if n > half_col)
    return row_mid, col_mid


def cut_mask(mask):
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)
    active_cols = [i for i, v in enumerate(col_sums) if v != 0]
    active_rows = [i for i, v in enumerate(row_sums) if v != 0]
    if not active_rows or not active_cols:
        return None
    return mask[active_rows[0]:active_rows[-1]+1,
                active_cols[0]:active_cols[-1]+1]


def asymmetry(mask):
    row_mid, col_mid = MidPoint(mask)
    upper = mask[:ceil(row_mid), :]
    lower = mask[floor(row_mid):, :]
    left  = mask[:, :ceil(col_mid)]
    right = mask[:, floor(col_mid):]

    fl = np.flip(lower, axis=0)
    fr = np.flip(right, axis=1)

    min_r = min(upper.shape[0], fl.shape[0])
    min_c = min(left.shape[1],  fr.shape[1])

    hxor = np.logical_xor(upper[:min_r, :], fl[:min_r, :])
    vxor = np.logical_xor(left[:, :min_c], fr[:, :min_c])

    total = np.sum(mask)
    score = (np.sum(hxor) + np.sum(vxor)) / (total * 2)
    return round(float(score), 4)


def feature_asymmetry(mask, n: int):
    '''Rotate mask n times, compute asymmetry at each rotation,
    return the mean score across all valid rotations.

    Args:
        mask (numpy.ndarray): binary mask as float64 (values 0.0 / 1.0)
        n (int): number of rotations between 0 and 90 degrees

    Returns:
        mean_score (float | None): mean asymmetry score, or None if all rotations failed
        valid_n (int): number of rotations that produced a valid score
    '''
    scores = []
    for i in range(n):
        degrees = 90 * i / n
        rotated   = rotate(mask.astype(np.float64), degrees, cval=0.0)  # float64 keeps 0/1 scale
        binarized = (rotated > 0.5).astype(np.uint8)
        cut       = cut_mask(binarized)

        if cut is None:
            print(f"  Warning: empty mask at {degrees:.1f}°, skipping")
            continue

        scores.append(asymmetry(cut))

    if not scores:
        return None, 0

    return round(float(np.mean(scores)), 4), len(scores)


# ── OPTION 1: folder of mask images ───────────────────────────────────────
def process_folder(folder_path: str, n: int, output_path: str = "features_asymmetry.csv"):
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    rows = []
    for filename in sorted(os.listdir(folder_path)):
        if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:
            continue

        img = imread(os.path.join(folder_path, filename), as_gray=True)

        # Normalize to 0-1 if loaded as uint8 (max > 1 means it wasn't auto-normalized)
        if img.max() > 1.0:
            img = img / 255.0

        # Guard: skip nearly empty masks (fewer than 50 white pixels — likely corrupt/empty)
        binary = (img > 0.5)
        if binary.sum() < 50:
            print(f"  Skipped {filename}: only {binary.sum()} white pixels after binarization")
            continue

        mask = binary.astype(np.float64)
        mean_score, valid_n = feature_asymmetry(mask, n)

        if mean_score is None:
            print(f"  Skipped {filename}: all rotations failed")
            continue

        # Sanity check — score must be 0-1, flag anything outside
        if mean_score > 1.0:
            print(f"  WARNING: {filename} produced out-of-bounds score {mean_score}, skipping")
            continue

        rows.append({
            "filename":        filename,
            "asymmetry_score": mean_score,
            "rotations_used":  valid_n,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} masks to {output_path}")


# ── OPTION 2: metadata CSV with paths to mask files ───────────────────────
def process_metadata_csv(metadata_path: str, n: int,
                         path_column: str = "mask_path",
                         output_path: str = "features_asymmetry.csv"):
    metadata = pd.read_csv(metadata_path)
    if path_column not in metadata.columns:
        raise ValueError(f"Column '{path_column}' not found. Available: {list(metadata.columns)}")

    rows = []
    for _, row in metadata.iterrows():
        mask = imread(row[path_column], as_gray=True)
        mask = (mask > 0.5).astype(np.float64)  # float64 — NOT uint8

        mean_score, valid_n = feature_asymmetry(mask, n)
        if mean_score is None:
            print(f"  Skipped {row[path_column]}: all rotations failed")
            continue

        base = row.to_dict()
        base.update({"asymmetry_score": mean_score, "rotations_used": valid_n})
        rows.append(base)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} masks to {output_path}")


# ── OPTION 3: metadata CSV with inline flattened mask arrays ──────────────
def process_metadata_csv_inline(metadata_path: str, n: int,
                                mask_column: str = "mask",
                                output_path: str = "features_asymmetry.csv"):
    metadata = pd.read_csv(metadata_path)
    if mask_column not in metadata.columns:
        raise ValueError(f"Column '{mask_column}' not found. Available: {list(metadata.columns)}")

    rows = []
    for _, row in metadata.iterrows():
        flat = np.fromstring(row[mask_column], dtype=np.uint8, sep=" ")
        size = int(sqrt(len(flat)))
        mask = flat.reshape((size, size)).astype(np.float64)  # float64 — NOT uint8

        mean_score, valid_n = feature_asymmetry(mask, n)
        if mean_score is None:
            print(f"  Skipped a row: all rotations failed")
            continue

        base = row.to_dict()
        base.update({"asymmetry_score": mean_score, "rotations_used": valid_n})
        rows.append(base)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} masks to {output_path}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODE = "folder"   # "folder" | "metadata_csv" | "metadata_csv_inline"
    N    = 8

    if MODE == "folder":
        process_folder("/Users/edvinasgross/Documents/GitHub/PersonalExercisesSem2/05_Feature_Extraction/masks", n=N)

    elif MODE == "metadata_csv":
        process_metadata_csv("metadata.csv", n=N, path_column="mask_path")

    elif MODE == "metadata_csv_inline":
        process_metadata_csv_inline("metadata.csv", n=N, mask_column="mask")