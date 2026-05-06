import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IMG_DIR         = Path("data/imgs")
MASK_DIR        = Path("data/masks")
OUTPUT_DIR      = Path("data/pen_masks")
ANNOTATIONS_CSV = Path("data/annotations_combined.csv")

DARK_SKIN_THRESHOLD = 120   # skin gray mean below this → dark skin mode

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def remove_lesion(img, lesion_mask):
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
    median_color = np.median(img[lm == 0], axis=0).astype(np.uint8)
    clean = img.copy()
    clean[lm == 255] = median_color
    return clean


def skin_gray_mean(img, lesion_mask):
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
    px = img[lm == 0].astype(float)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).mean()


def directional_dilation(mask):
    """
    Dilate in 4 directions and keep pixels agreed on by at least 2 directions.
    This follows pen stroke trends without bloating uniformly.
    """
    h_kernel = np.ones((1, 15), np.uint8)   # horizontal
    v_kernel = np.ones((15, 1), np.uint8)   # vertical
    d1       = np.eye(11, dtype=np.uint8)   # 45°
    d2       = np.fliplr(d1)                # 135°

    dil_h  = cv2.dilate(mask, h_kernel, iterations=1)
    dil_v  = cv2.dilate(mask, v_kernel, iterations=1)
    dil_d1 = cv2.dilate(mask, d1,       iterations=1)
    dil_d2 = cv2.dilate(mask, d2,       iterations=1)

    vote = (dil_h.astype(int) + dil_v.astype(int) +
            dil_d1.astype(int) + dil_d2.astype(int)) // 255
    return (vote >= 2).astype(np.uint8) * 255


def detect_pen_dark_skin(img, lm):
    """For images with skin gray mean < DARK_SKIN_THRESHOLD."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]

    dark_pen = (gray.astype(int) < 90) & (s < 110)

    s_thresh = np.percentile(s[lm == 0], 65)
    hue_pen  = (h >= 95) & (h <= 175) & (s.astype(int) > s_thresh)

    pen = (dark_pen | hue_pen).astype(np.uint8) * 255
    pen[lm == 255] = 0
    return pen


def detect_pen_light_skin(img, lm):
    """For images with skin gray mean >= DARK_SKIN_THRESHOLD."""
    b_ch = img[:, :, 0].astype(float)
    g_ch = img[:, :, 1].astype(float)
    r_ch = img[:, :, 2].astype(float)
    gray = 0.114 * b_ch + 0.587 * g_ch + 0.299 * r_ch

    skin  = lm == 0
    sg    = gray[skin]
    gmed  = np.median(sg)
    gstd  = max(np.median(np.abs(sg - gmed)) * 1.4826, 8.0)

    br        = b_ch[skin] - r_ch[skin]
    br_med    = np.median(br)
    br_std    = max(np.median(np.abs(br - br_med)) * 1.4826, 5.0)

    dark_pen = gray < (gmed - 1.6 * gstd)
    blue_pen = (b_ch - r_ch) > (br_med + 2.0 * br_std)

    pen = (dark_pen | blue_pen).astype(np.uint8) * 255
    pen[lm == 255] = 0
    return pen


def generate_pen_mask(img, lesion_mask):
    """
    Full pipeline: lesion fill → adaptive pen detection → directional dilation.
    Returns binary pen mask (255 = pen, 0 = clean).
    """
    lesion_mask = cv2.resize(lesion_mask, (img.shape[1], img.shape[0]))
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)

    clean_img = remove_lesion(img, lm)
    gm        = skin_gray_mean(img, lm)

    if gm < DARK_SKIN_THRESHOLD:
        pen = detect_pen_dark_skin(clean_img, lm)
    else:
        pen = detect_pen_light_skin(clean_img, lm)

    # Base morphological cleanup
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN,  k3)
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, k5)

    # Directional dilation to follow stroke trends
    pen = directional_dilation(pen)

    # Final close to merge remaining gaps
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, k5)

    # Ensure lesion area is not included (dilation may have crept in)
    pen[lm == 255] = 0

    # Drop small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pen)
    cleaned = np.zeros_like(pen)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 80:
            cleaned[labels == i] = 255

    return cleaned


# ── Load and filter annotations ───────────────────────────────────────────────
df = pd.read_csv(ANNOTATIONS_CSV)

pen_cols     = [c for c in ["pen_1", "pen_2", "pen_3", "pen_4"] if c in df.columns]
has_pen      = df[pen_cols].eq(1).sum(axis=1) >= 2
pen_df       = df[has_pen]
image_ids    = pen_df["img_id"].tolist()

print(f"Found {len(image_ids)} images with pen marks (checked: {pen_cols})")

# ── Main loop ─────────────────────────────────────────────────────────────────
processed = skipped = 0

for img_id in image_ids:
    # Support img_id with or without .png extension
    stem      = img_id.replace(".png", "")
    img_path  = IMG_DIR  / f"{stem}.png"
    mask_path = MASK_DIR / f"{stem}_mask.png"

    if not img_path.exists():
        print(f"  [MISSING IMG]  {stem}")
        skipped += 1
        continue

    if not mask_path.exists():
        print(f"  [MISSING MASK] {stem}")
        skipped += 1
        continue

    img         = cv2.imread(str(img_path))
    lesion_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or lesion_mask is None:
        print(f"  [UNREADABLE]   {stem}")
        skipped += 1
        continue

    try:
        pen_mask = generate_pen_mask(img, lesion_mask)
        out_path = OUTPUT_DIR / f"{stem}_penmask.png"
        cv2.imwrite(str(out_path), pen_mask)
        processed += 1
        print(f"  [OK] {stem}  coverage={((pen_mask>0).sum()/pen_mask.size*100):.1f}%")
    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        skipped += 1

print(f"\nDone — saved: {processed} | skipped: {skipped} → {OUTPUT_DIR}")