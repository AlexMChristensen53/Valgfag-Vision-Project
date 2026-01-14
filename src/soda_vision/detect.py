from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class RefEntry:
    brand: str
    path: Path
    gray: np.ndarray
    kp: List[cv2.KeyPoint]
    desc: np.ndarray
    shape: Tuple[int, int]  # (h, w)



# Paths
ROOT = Path(__file__).resolve().parents[2]  # repo root
REFS_DIR = ROOT / "data" / "refs"
_REFS_CACHE: Optional[List[RefEntry]] = None



def _sift():
    return cv2.SIFT_create()


def _resize_max(img_gray: np.ndarray, max_dim: int = 900) -> np.ndarray:
    """Resize grayscale image so max(width,height) <= max_dim, keeping aspect."""
    h, w = img_gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img_gray
    scale = max_dim / m
    return cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _cap_features(kp: List[cv2.KeyPoint], desc: np.ndarray, max_kp: int = 800):
    """Keep top-N keypoints by response; keep matching descriptors rows."""
    if desc is None or kp is None:
        return kp, desc
    if len(kp) <= max_kp:
        return kp, desc
    idx = np.argsort([-k.response for k in kp])[:max_kp]
    kp2 = [kp[i] for i in idx]
    desc2 = desc[idx, :]
    return kp2, desc2


def _load_refs(max_dim: int = 900, max_kp: int = 800) -> List[RefEntry]:
    if not REFS_DIR.exists():
        raise FileNotFoundError(f"Mangler refs-mappe: {REFS_DIR}")

    sift = _sift()
    refs: List[RefEntry] = []

    brand_dirs = [p for p in REFS_DIR.iterdir() if p.is_dir()]
    if not brand_dirs:
        raise RuntimeError(f"Ingen brand-mapper fundet i {REFS_DIR}")

    for brand_dir in sorted(brand_dirs):
        brand = brand_dir.name
        img_paths = sorted(list(brand_dir.glob("*.jpg")) + list(brand_dir.glob("*.png")))
        for p in img_paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            gray = _resize_max(gray, max_dim=max_dim)
            kp, desc = sift.detectAndCompute(gray, None)
            kp, desc = _cap_features(kp, desc, max_kp=max_kp)

            if desc is None or kp is None or len(kp) < 10:
                continue

            refs.append(
                RefEntry(
                    brand=brand,
                    path=p,
                    gray=gray,
                    kp=kp,
                    desc=desc,
                    shape=gray.shape[:2],
                )
            )

    if not refs:
        raise RuntimeError(
            "Ingen gyldige reference-features fundet. "
        )

    return refs


def _get_refs() -> List[RefEntry]:
    global _REFS_CACHE
    if _REFS_CACHE is None:
        _REFS_CACHE = _load_refs()
    return _REFS_CACHE


def _mutual_good_matches(desc_ref: np.ndarray, desc_img: np.ndarray, ratio: float = 0.72) -> List[cv2.DMatch]:
    """
    Mutual KNN ratio test:
      - ref -> img (ratio)
      - img -> ref (ratio)
      - keep only pairs that agree (mutual)
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_ri = bf.knnMatch(desc_ref, desc_img, k=2)
    good_ri = []
    for pair in knn_ri:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ri.append(m)

    knn_ir = bf.knnMatch(desc_img, desc_ref, k=2)
    good_ir = []
    for pair in knn_ir:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ir.append(m)

    # mutual consistency
    map_ir = {(m.trainIdx, m.queryIdx): m for m in good_ir}
    mutual = [m for m in good_ri if (m.queryIdx, m.trainIdx) in map_ir]
    return mutual


def _homography_inliers(
    ref_kp: List[cv2.KeyPoint],
    img_kp: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_thresh: float = 5.0,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      inlier_count, H (3x3), inlier_mask (Nx1)
    """
    if len(matches) < 4:
        return 0, None, None

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([img_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
    if H is None or mask is None:
        return 0, None, None

    inliers = int(mask.sum())
    return inliers, H, mask


def _project_ref_corners(ref_shape_hw: Tuple[int, int], H: np.ndarray) -> np.ndarray:
    """Project reference image corners into input image using homography H."""
    h, w = ref_shape_hw
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)
    return proj



# Main API
def detect(image_bgr: np.ndarray) -> Tuple[str, float, dict]:
    """
    Returnerer:
        brand (str): bedste brand eller "unknown"
        score (float): inlier-count (RANSAC homografi) (højere=bedre)
        extra (dict): debug data til visualisering
    """
    refs = _get_refs()
    sift = _sift()

    # Preprocess input
    img_gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = _resize_max(img_gray_full, max_dim=1200)

    kp_img, desc_img = sift.detectAndCompute(img_gray, None)
    kp_img, desc_img = _cap_features(kp_img, desc_img, max_kp=800)

    if desc_img is None or kp_img is None or len(kp_img) < 10:
        return "unknown", 0.0, {"reason": "no_features"}

    ratio = 0.77
    ransac_thresh = 5.0

    # Track best per brand by inliers
    best_inliers_per_brand: Dict[str, int] = {}

    # Track global best
    best_brand = "unknown"
    best_inliers = 0
    best_ref: Optional[RefEntry] = None
    best_matches: List[cv2.DMatch] = []
    best_inlier_mask: Optional[np.ndarray] = None
    best_H: Optional[np.ndarray] = None
    best_poly: Optional[np.ndarray] = None

    for r in refs:
        matches = _mutual_good_matches(r.desc, desc_img, ratio=ratio)

        # Homography filtering => inliers
        inliers, H, mask = _homography_inliers(r.kp, kp_img, matches, ransac_reproj_thresh=ransac_thresh)

        # Update per-brand max
        best_inliers_per_brand[r.brand] = max(best_inliers_per_brand.get(r.brand, 0), inliers)

        # Update global best
        if inliers > best_inliers:
            best_inliers = inliers
            best_brand = r.brand
            best_ref = r
            best_matches = matches
            best_inlier_mask = mask
            best_H = H
            best_poly = _project_ref_corners(r.shape, H) if H is not None else None

    # Threshold: how many inliers to accept as a confident detection
    min_inliers = 12
    if best_inliers < min_inliers:
        result_brand = "unknown"
    else:
        result_brand = best_brand

    extra = {
        "ratio": ratio,
        "ransac_thresh": ransac_thresh,
        "min_inliers": min_inliers,
        "best_inliers_per_brand": best_inliers_per_brand,

        "img_gray": img_gray,
        "kp_img": kp_img,

        "best_ref_brand": best_ref.brand if best_ref else None,
        "best_ref_path": str(best_ref.path) if best_ref else None,
        "best_ref_gray": best_ref.gray if best_ref else None,
        "best_ref_kp": best_ref.kp if best_ref else None,

        "best_matches": best_matches,
        "best_inlier_mask": best_inlier_mask,
        "best_H": best_H,
        "best_poly": best_poly, 
        "best_inliers": best_inliers,
    }

    return result_brand, float(best_inliers), extra


if __name__ == "__main__":
    refs = _get_refs()
    print("Loaded refs:", len(refs))
    brands = sorted(set(r.brand for r in refs))
    print("Brands:", brands)
