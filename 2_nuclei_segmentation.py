import os
import cv2
import numpy as np
from skimage import io, exposure
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes, erosion

MIN_OBJ     = 100       # min nucleus area (px)
MIN_HOLE    = 30        # fill small holes (px)
NUCLEI_CORE_ALPHA = 0.2 # strictness of core seed threshold for watershed

OPEN_R  = 4             # opening radius
CLOSE_R = 1             # closing radius
ERODE_R = 1             # erosion radius

DATA_FOLDER      = '/Nuclei_Segmentation/data/original_images'
GLOM_MASKS_ROOT  = '/Nuclei_Segmentation/data/glomeruli_masks'
RESULTS_ROOT     = '/Nuclei_Segmentation/results'
STEP_DIR         = os.path.join(RESULTS_ROOT, 'step_results')
OVERLAY_DIR      = os.path.join(RESULTS_ROOT, 'segmentation_results')
NUCLEI_MASK_DIR  = os.path.join(RESULTS_ROOT, 'Nuclei_segmented')

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def save_image(image, out_dir, filename, normalize=True):
    ensure_dir(out_dir)
    arr = image
    if normalize and image.dtype != np.uint8:
        rng = np.ptp(image)
        arr = np.zeros_like(image, dtype=np.uint8) if rng == 0 else (255 * (image - np.min(image)) / rng).astype(np.uint8)
    io.imsave(os.path.join(out_dir, filename), arr)

def load_glom_mask_for_image(base_filename, masks_root):
    """Load {stem}_mask.png as 0/255 uint8; return None if missing."""
    p = os.path.join(masks_root, f"{base_filename}_mask.png")
    if not os.path.isfile(p):
        return None
    m = io.imread(p)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    return ((m > 0).astype(np.uint8) * 255)

def otsu_and_core(h_norm, glom_mask=None):
    """Otsu on normalized H. Returns (bin_otsu 0/255, core_seeds bool)."""
    if glom_mask is not None:
        m = (glom_mask > 0)
        h_m = h_norm[m] if m.any() else h_norm
    else:
        m = None
        h_m = h_norm

    t = threshold_otsu(h_m)
    p95 = np.percentile(h_m, 95)
    t_core = t + NUCLEI_CORE_ALPHA * (p95 - t)

    bin_otsu_bool = (h_norm > t)
    core_bool     = (h_norm > t_core)

    if m is not None:
        bin_otsu_bool &= m
        core_bool     &= m

    return (bin_otsu_bool.astype(np.uint8) * 255), core_bool

def morph_clean(binary_mask_uint8, open_r, close_r):
    """opening -> closing -> remove small holes."""
    m = (binary_mask_uint8 > 0)
    if open_r > 0:
        m = opening(m, disk(open_r))
    if close_r > 0:
        m = closing(m, disk(close_r))
    m = remove_small_objects(m, min_size=MIN_OBJ)
    m = remove_small_holes(m, area_threshold=MIN_HOLE)
    return (m.astype(np.uint8) * 255)

def process_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]

    # read RGB
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] cannot read {image_path}")
        return 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    save_image(rgb, STEP_DIR, f"{base}_01_original.jpg", normalize=False)

    # glomerulus mask
    glom_mask = load_glom_mask_for_image(base, GLOM_MASKS_ROOT)
    if glom_mask is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        glom_mask = (gray > 10).astype(np.uint8) * 255
    save_image(glom_mask, STEP_DIR, f"{base}_02_glom_mask.jpg", normalize=False)

    # Hematoxylin channel + normalize
    h = rgb2hed(rgb)[:, :, 0]
    h_norm = exposure.rescale_intensity(h, in_range='image', out_range=(0, 1))
    save_image(h_norm, STEP_DIR, f"{base}_02b_h_norm.jpg")

    # Otsu + core seeds
    bin_nuclei, core_seeds = otsu_and_core(h_norm, glom_mask)
    save_image(bin_nuclei, STEP_DIR, f"{base}_03_binary_nuclei.jpg", normalize=False)

    # Morph cleaning
    cleaned = morph_clean(bin_nuclei, OPEN_R, CLOSE_R)
    save_image(cleaned, STEP_DIR, f"{base}_04_cleaned_o{OPEN_R}_c{CLOSE_R}.jpg", normalize=False)

    # Distance + markers
    bin_bool = (cleaned > 0)
    dist = distance_transform_edt(bin_bool)
    save_image(dist, STEP_DIR, f"{base}_05_distance.jpg")

    core_seeds = remove_small_objects(core_seeds.copy(), min_size=20)
    markers = label(core_seeds)

    # Watershed
    labels_ws = watershed(-dist, markers, mask=bin_bool)
    save_image(labels_ws, STEP_DIR, f"{base}_06_labels.jpg")

    # Final mask within glom + optional contraction
    labels_glom = (labels_ws > 0).astype(np.uint8) * ((glom_mask > 0).astype(np.uint8))
    final_mask = labels_glom > 0
    if ERODE_R > 0:
        final_mask = erosion(final_mask, disk(ERODE_R))
    final_mask_u8 = final_mask.astype(np.uint8) * 255

    # Overlay
    overlay = rgb.copy()
    overlay[final_mask_u8 > 0] = [255, 0, 0]
    save_image(overlay, STEP_DIR, f"{base}_07_overlay_nuclei.jpg", normalize=False)

    comp_count = label(final_mask).max()
    return comp_count

def main():
    counts_log = os.path.join(OVERLAY_DIR, "nuclei_counts.txt")
    with open(counts_log, "w"):
        pass

    for fn in sorted(os.listdir(DATA_FOLDER)):
        if not fn.lower().endswith('.tif'):
            continue
        img_path = os.path.join(DATA_FOLDER, fn)
        ncount = process_image(img_path)
        with open(counts_log, "a") as f:
            f.write(f"{fn}: {ncount}\n")
    print(f"[DONE] Outputs -> overlays:{OVERLAY_DIR}  masks:{NUCLEI_MASK_DIR}")

if __name__ == "__main__":
    main()
