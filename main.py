from pathlib import Path

script = r'''"""
Unified feature extraction script for the skin lesion project.
"""

from pathlib import Path
from math import floor, ceil
import argparse

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage import img_as_float, transform, morphology, measure
from skimage.segmentation import find_boundaries, slic
from skimage.transform import rotate
from skimage.color import rgb2hsv, rgb2lab

from scipy.stats import circvar
from sklearn.cluster import KMeans


# ============================================================
# 1. General settings
# ============================================================

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

ITA_THRESHOLDS = [55.0, 41.0, 28.0, 10.0, -30.0]


# ============================================================
# 2. Project paths - GitHub friendly
# ============================================================

def find_project_paths():
    """
    Finds the project folders without using personal computer paths.

    Works if masks are stored in either:

    data/masks

    or:

    data/masks/masks
    """

    start = Path(__file__).resolve().parent

    for parent in [start] + list(start.parents):
        img_dir = parent / "data" / "imgs"

        mask_candidates = [
            parent / "data" / "masks" / "masks",
            parent / "data" / "masks",
        ]

        output_dir = parent / "data" / "Separate csv of features"

        if img_dir.exists():
            for mask_dir in mask_candidates:
                if mask_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                    return {
                        "project_root": parent,
                        "img_dir": img_dir,
                        "mask_dir": mask_dir,
                        "output_dir": output_dir,
                    }

    raise FileNotFoundError(
        "Could not find project folders. Make sure the repo contains "
        "'data/imgs' and either 'data/masks' or 'data/masks/masks'."
    )


# ============================================================
# 3. Load image and mask
# ============================================================

def load_image(path, max_size=512):
    """
    Loads an image as RGB float values.
    If the image is larger than max_size, it is resized.
    """

    img = img_as_float(imread(path))

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if max_size is not None:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)

        if scale < 1.0:
            img = transform.resize(
                img,
                (int(h * scale), int(w * scale)),
                anti_aliasing=True,
            )

    return img


def load_mask(mask_path):
    """
    Reads a mask image and converts it into a binary mask.
    """

    mask = imread(mask_path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.max() > 1:
        mask = mask / 255.0

    return mask > 0.5


def find_mask_for_image(image_path, mask_dir):
    """
    Finds the mask corresponding to an image.

    Example:
    image: PAT_1000_31_620.png
    mask:  PAT_1000_31_620_mask.png
    """

    image_stem = image_path.stem

    possible_mask_names = []

    for ext in VALID_EXTENSIONS:
        possible_mask_names.append(f"{image_stem}_mask{ext}")
        possible_mask_names.append(f"{image_stem}{ext}")

    for mask_name in possible_mask_names:
        candidate = mask_dir / mask_name

        if candidate.exists():
            return candidate

    return None


# ============================================================
# 4. Border and shape features
# ============================================================

def preprocess_mask(mask_bin):
    """
    Cleans the binary mask by removing small noise and filling small holes.
    """

    mask_bin = morphology.remove_small_objects(mask_bin, min_size=100)
    mask_bin = morphology.remove_small_holes(mask_bin, area_threshold=100)

    return mask_bin


def get_main_lesion(mask_bin):
    """
    Keeps only the largest connected lesion region.
    """

    labels = measure.label(mask_bin)
    regions = measure.regionprops(labels)

    if len(regions) == 0:
        return None, None

    largest = max(regions, key=lambda r: r.area)
    lesion = labels == largest.label

    return lesion, largest


def extract_border_shape_features(mask_bin):
    """
    Extracts border and shape features from the lesion mask.
    """

    cleaned_mask = preprocess_mask(mask_bin)
    lesion, region = get_main_lesion(cleaned_mask)

    if lesion is None:
        return None

    border = find_boundaries(lesion, mode="inner")

    area = region.area
    perimeter = region.perimeter
    total_pixels = lesion.shape[0] * lesion.shape[1]

    features = {
        "total_pixels": total_pixels,
        "area": area,
        "lesion_percentage": area / total_pixels if total_pixels > 0 else np.nan,
        "perimeter": perimeter,
        "compactness": (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.nan,
        "border_pixels": int(np.sum(border)),
    }

    return features


# ============================================================
# 5. Asymmetry features
# ============================================================

def midpoint(mask):
    """
    Finds the midpoint of the mask based on the cumulative lesion area.
    """

    row_summed = np.sum(mask, axis=1)
    col_summed = np.sum(mask, axis=0)

    half_row = np.sum(row_summed) / 2
    half_col = np.sum(col_summed) / 2

    row_mid = next(
        i for i, n in enumerate(np.add.accumulate(row_summed))
        if n > half_row
    )

    col_mid = next(
        i for i, n in enumerate(np.add.accumulate(col_summed))
        if n > half_col
    )

    return row_mid, col_mid


def cut_mask(mask):
    """
    Crops the mask to the smallest rectangle containing the lesion.
    """

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = [i for i, v in enumerate(col_sums) if v != 0]
    active_rows = [i for i, v in enumerate(row_sums) if v != 0]

    if not active_rows or not active_cols:
        return None

    return mask[
        active_rows[0]:active_rows[-1] + 1,
        active_cols[0]:active_cols[-1] + 1,
    ]


def asymmetry(mask):
    """
    Computes asymmetry by comparing the lesion with its horizontal and vertical flips.
    """

    row_mid, col_mid = midpoint(mask)

    upper = mask[:ceil(row_mid), :]
    lower = mask[floor(row_mid):, :]

    left = mask[:, :ceil(col_mid)]
    right = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower, axis=0)
    flipped_right = np.flip(right, axis=1)

    min_r = min(upper.shape[0], flipped_lower.shape[0])
    min_c = min(left.shape[1], flipped_right.shape[1])

    hxor = np.logical_xor(
        upper[:min_r, :],
        flipped_lower[:min_r, :],
    )

    vxor = np.logical_xor(
        left[:, :min_c],
        flipped_right[:, :min_c],
    )

    total = np.sum(mask)

    if total == 0:
        return None

    score = (np.sum(hxor) + np.sum(vxor)) / (total * 2)

    return round(float(score), 4)


def feature_asymmetry(mask, n_rotations=8):
    """
    Rotates the mask and computes average asymmetry across rotations.
    """

    scores = []

    for i in range(n_rotations):
        degrees = 90 * i / n_rotations

        rotated = rotate(
            mask.astype(np.float64),
            degrees,
            cval=0.0,
        )

        binarized = (rotated > 0.5).astype(np.uint8)
        cut = cut_mask(binarized)

        if cut is None:
            continue

        score = asymmetry(cut)

        if score is not None and 0 <= score <= 1:
            scores.append(score)

    if not scores:
        return None, 0

    return round(float(np.mean(scores)), 4), len(scores)


# ============================================================
# 6. Color features
# ============================================================

def ita_to_fst(ita):
    """
    Converts ITA value into predicted Fitzpatrick Skin Type.
    """

    for fst, threshold in enumerate(ITA_THRESHOLDS, start=1):
        if ita > threshold:
            return fst

    return 6


def predict_fst(image, segments):
    """
    Predicts Fitzpatrick Skin Type using ITA.
    """

    lab = rgb2lab(image)

    seg_ids = np.unique(segments)
    n = len(seg_ids)

    lut = np.zeros(segments.max() + 1, dtype=np.intp)
    lut[seg_ids] = np.arange(n)
    flat_idx = lut[segments.ravel()]

    counts = np.bincount(flat_idx, minlength=n).astype(np.float64)

    lab_flat = lab.reshape(-1, 3)
    lab_sum = np.zeros((n, 3))

    np.add.at(lab_sum, flat_idx, lab_flat)

    lab_means = lab_sum / counts[:, None]

    L = lab_means[:, 0]
    b = lab_means[:, 2]

    skin_mask = (L >= 30) & (L <= 90)

    if skin_mask.sum() == 0:
        skin_mask = np.ones(n, dtype=bool)

    L_skin = L[skin_mask]
    b_skin = b[skin_mask]

    b_safe = np.where(np.abs(b_skin) < 1e-6, 1e-6, b_skin)

    ita_values = np.degrees(np.arctan((L_skin - 50) / b_safe))

    ita_mean = float(np.mean(ita_values))
    fst = ita_to_fst(ita_mean)

    return ita_mean, fst


def get_segment_features(image, hsv, segments):
    """
    Computes mean RGB and HSV values for each SLIC segment.
    """

    seg_ids = np.unique(segments)
    n = len(seg_ids)

    lut = np.zeros(segments.max() + 1, dtype=np.intp)
    lut[seg_ids] = np.arange(n)
    flat_idx = lut[segments.ravel()]

    counts = np.bincount(flat_idx, minlength=n).astype(np.float64)

    rgb_flat = image.reshape(-1, 3)
    rgb_sum = np.zeros((n, 3))

    np.add.at(rgb_sum, flat_idx, rgb_flat)

    rgb_means = rgb_sum / counts[:, None]

    hsv_flat = hsv.reshape(-1, 3)

    sv_sum = np.zeros((n, 2))
    np.add.at(sv_sum, flat_idx, hsv_flat[:, 1:])

    sv_means = sv_sum / counts[:, None]

    angles = hsv_flat[:, 0] * 2 * np.pi

    sin_sum = np.zeros(n)
    cos_sum = np.zeros(n)

    np.add.at(sin_sum, flat_idx, np.sin(angles))
    np.add.at(cos_sum, flat_idx, np.cos(angles))

    h_means = (
        np.arctan2(sin_sum / counts, cos_sum / counts) % (2 * np.pi)
    ) / (2 * np.pi)

    hsv_means = np.column_stack([h_means, sv_means])

    return rgb_means, hsv_means


def calculate_variances(means):
    """
    Variance of RGB values across segments.
    """

    if len(means) < 2:
        return (0.0, 0.0, 0.0)

    return tuple(np.var(means, axis=0))


def calculate_hsv_variances(means):
    """
    Variance of HSV values across segments.
    Hue uses circular variance.
    """

    if len(means) < 2:
        return (0.0, 0.0, 0.0)

    return (
        circvar(means[:, 0], high=1, low=0),
        np.var(means[:, 1]),
        np.var(means[:, 2]),
    )


def color_dominance(hsv_flat, clusters=5, pixel_step=4):
    """
    Finds dominant colors using KMeans in HSV space.
    """

    sample = hsv_flat[::pixel_step]

    if len(sample) < clusters:
        clusters = len(sample)

    if clusters <= 0:
        return []

    kmeans = KMeans(
        n_clusters=clusters,
        n_init=10,
        random_state=0,
    )

    kmeans.fit(sample)

    _, counts = np.unique(kmeans.labels_, return_counts=True)
    ratios = counts / len(kmeans.labels_)

    dominant = sorted(
        zip(ratios, kmeans.cluster_centers_),
        key=lambda x: x[0],
        reverse=True,
    )

    return dominant


def extract_color_features(
    image,
    n_segments=50,
    n_dominant_colors=5,
    pixel_step=4,
):
    """
    Extracts color variation, dominant color, ITA, and predicted FST.
    """

    hsv = rgb2hsv(image)
    hsv_flat = hsv.reshape(-1, 3)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
        start_label=1,
    )

    rgb_means, hsv_means = get_segment_features(image, hsv, segments)

    rgb_var = calculate_variances(rgb_means)
    hsv_var = calculate_hsv_variances(hsv_means)

    dominant_colors = color_dominance(
        hsv_flat,
        clusters=n_dominant_colors,
        pixel_step=pixel_step,
    )

    ita_mean, fst_predicted = predict_fst(image, segments)

    features = {
        "rgb_var_r": rgb_var[0],
        "rgb_var_g": rgb_var[1],
        "rgb_var_b": rgb_var[2],
        "hsv_var_h": hsv_var[0],
        "hsv_var_s": hsv_var[1],
        "hsv_var_v": hsv_var[2],
        "ita_mean": round(ita_mean, 4),
        "fst_predicted": fst_predicted,
    }

    for i in range(n_dominant_colors):
        if i < len(dominant_colors):
            ratio, color = dominant_colors[i]

            features[f"dom_color_{i+1}_ratio"] = ratio
            features[f"dom_color_{i+1}_h"] = color[0]
            features[f"dom_color_{i+1}_s"] = color[1]
            features[f"dom_color_{i+1}_v"] = color[2]

        else:
            features[f"dom_color_{i+1}_ratio"] = np.nan
            features[f"dom_color_{i+1}_h"] = np.nan
            features[f"dom_color_{i+1}_s"] = np.nan
            features[f"dom_color_{i+1}_v"] = np.nan

    return features


# ============================================================
# 7. Main processing loop
# ============================================================

def process_all_images(
    img_dir,
    mask_dir,
    output_csv,
    max_size=512,
    n_rotations=8,
    n_segments=50,
    pixel_step=4,
    n_dominant_colors=5,
):
    """
    Processes all images and saves one CSV with all extracted features.
    """

    rows = []

    image_files = [
        file for file in sorted(img_dir.iterdir())
        if file.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Images found: {len(image_files)}")

    for image_path in image_files:
        mask_path = find_mask_for_image(image_path, mask_dir)

        if mask_path is None:
            print(f"Skipped {image_path.name}: no matching mask found")
            continue

        try:
            image = load_image(image_path, max_size=max_size)
            mask_bin = load_mask(mask_path)

            if np.sum(mask_bin) < 50:
                print(f"Skipped {image_path.name}: mask has fewer than 50 white pixels")
                continue

            border_shape_features = extract_border_shape_features(mask_bin)

            if border_shape_features is None:
                print(f"Skipped {image_path.name}: no valid lesion found")
                continue

            asymmetry_score, rotations_used = feature_asymmetry(
                mask_bin,
                n_rotations=n_rotations,
            )

            color_features = extract_color_features(
                image,
                n_segments=n_segments,
                n_dominant_colors=n_dominant_colors,
                pixel_step=pixel_step,
            )

            row = {
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                **border_shape_features,
                "asymmetry_score": asymmetry_score,
                "rotations_used": rotations_used,
                **color_features,
            }

            rows.append(row)

        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("image_name").reset_index(drop=True)

    df.to_csv(output_csv, index=False)

    print(f"\nSaved {len(df)} rows to:")
    print(output_csv)

    return df


# ============================================================
# 8. Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract border, asymmetry, color, ITA, and FST features from skin lesion images."
    )

    parser.add_argument(
        "--output",
        default="features_all.csv",
        help="Output CSV filename. It will be saved inside 'data/Separate csv of features'.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Maximum image size used for color features.",
    )

    parser.add_argument(
        "--rotations",
        type=int,
        default=8,
        help="Number of rotations used for asymmetry calculation.",
    )

    parser.add_argument(
        "--segments",
        type=int,
        default=50,
        help="Number of SLIC segments used for color features.",
    )

    parser.add_argument(
        "--pixel-step",
        type=int,
        default=4,
        help="Pixel step used when sampling pixels for KMeans dominant colors.",
    )

    parser.add_argument(
        "--dominant-colors",
        type=int,
        default=5,
        help="Number of dominant color clusters to save.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    paths = find_project_paths()

    project_root = paths["project_root"]
    img_dir = paths["img_dir"]
    mask_dir = paths["mask_dir"]
    output_dir = paths["output_dir"]

    output_csv = output_dir / args.output

    print("Project root:", project_root)
    print("Image folder:", img_dir)
    print("Mask folder:", mask_dir)
    print("Output CSV:", output_csv)

    features_df = process_all_images(
        img_dir=img_dir,
        mask_dir=mask_dir,
        output_csv=output_csv,
        max_size=args.max_size,
        n_rotations=args.rotations,
        n_segments=args.segments,
        pixel_step=args.pixel_step,
        n_dominant_colors=args.dominant_colors,
    )

    print("\nPreview:")
    print(features_df.head())


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/extract_all_features.py")
path.write_text(script, encoding="utf-8")

print(f"Created: {path}")
print(f"Size: {path.stat().st_size:,} bytes")