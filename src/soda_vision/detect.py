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


@dataclass
class Detection:
    brand: str
    inliers: int
    poly: np.ndarray                 # shape (4,1,2) float32 in scene_gray coords
    ref_path: str
    H: np.ndarray                    # 3x3
    matches: List[cv2.DMatch]
    inlier_mask: Optional[np.ndarray]


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
        raise RuntimeError("Ingen gyldige reference-features fundet.")

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


def _poly_bbox(poly: np.ndarray) -> Tuple[int, int, int, int]:
    """Axis-aligned bbox from poly (4,1,2). Returns x,y,w,h as ints."""
    pts = poly.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    return int(x), int(y), int(w), int(h)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _mask_polygon(scene_gray: np.ndarray, poly: np.ndarray, pad: int = 8) -> np.ndarray:
    """
    Mask polygon area (plus small padding) to prevent re-detecting same instance.
    Works in-place on a copy and returns it.
    """
    out = scene_gray.copy()
    pts = poly.reshape(-1, 2).astype(np.int32)

    # draw filled poly
    cv2.fillConvexPoly(out, pts, 0)

    # optional padding via dilation of mask
    mask = np.zeros_like(scene_gray, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    if pad > 0:
        k = max(1, pad // 2) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
        out[mask > 0] = 0

    return out


def _find_one_for_brand(
    brand: str,
    refs: List[RefEntry],
    sift: cv2.SIFT,
    scene_gray: np.ndarray,
    ratio: float,
    ransac_thresh: float,
    min_inliers: int,
) -> Tuple[Optional[Detection], Dict]:
    """
    Find best single instance for a given brand in the provided scene_gray.
    Returns (Detection or None, debug dict).
    """
    #scene features
    kp_img, desc_img = sift.detectAndCompute(scene_gray, None)
    kp_img, desc_img = _cap_features(kp_img, desc_img, max_kp=800)

    if desc_img is None or kp_img is None or len(kp_img) < 10:
        return None, {"reason": "no_features_in_scene"}

    best_inliers = 0
    best_ref: Optional[RefEntry] = None
    best_matches: List[cv2.DMatch] = []
    best_mask: Optional[np.ndarray] = None
    best_H: Optional[np.ndarray] = None
    best_poly: Optional[np.ndarray] = None

    for r in refs:
        if r.brand != brand:
            continue

        matches = _mutual_good_matches(r.desc, desc_img, ratio=ratio)
        inliers, H, mask = _homography_inliers(r.kp, kp_img, matches, ransac_reproj_thresh=ransac_thresh)

        if inliers > best_inliers and H is not None:
            best_inliers = inliers
            best_ref = r
            best_matches = matches
            best_mask = mask
            best_H = H
            best_poly = _project_ref_corners(r.shape, H)

    if best_ref is None or best_H is None or best_poly is None:
        return None, {"reason": "no_homography"}

    if best_inliers < min_inliers:
        return None, {
            "reason": "below_threshold",
            "best_inliers": best_inliers,
            "min_inliers": min_inliers,
            "best_ref_path": str(best_ref.path),
        }

    det = Detection(
        brand=brand,
        inliers=int(best_inliers),
        poly=best_poly,
        ref_path=str(best_ref.path),
        H=best_H,
        matches=best_matches,
        inlier_mask=best_mask,
    )

    dbg = {
        "brand": brand,
        "best_inliers": best_inliers,
        "best_ref_path": str(best_ref.path),
        "kp_img": kp_img,
        "desc_img": desc_img,
        "scene_gray": scene_gray,
        "ref_gray": best_ref.gray,
        "ref_kp": best_ref.kp,
        "matches": best_matches,
        "inlier_mask": best_mask,
        "H": best_H,
        "poly": best_poly,
    }
    return det, dbg



# multi-detection
def detect_all(image_bgr: np.ndarray) -> Tuple[List[Detection], dict]:
    """
    Finder 0..N forekomster af hvert brand i samme billede.
    Returnerer:
      detections: list[Detection]
      extra: debug/metrics
    """
    all_refs = _get_refs()
    brands = sorted(set(r.brand for r in all_refs))
    sift = _sift()

    # Preprocess input (work in resized gray coordinate space)
    img_gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = _resize_max(img_gray_full, max_dim=1200)

    # Parameters
    ratio = 0.77
    ransac_thresh = 5.0
    min_inliers = 12
    max_per_brand = 5
    mask_pad = 10
    iou_dedupe = 0.45  # bbox IoU threshold for considering duplicate

    detections: List[Detection] = []
    debug_per_brand: Dict[str, List[dict]] = {b: [] for b in brands}

    # We do per brand iterative detection with masking
    for brand in brands:
        work = img_gray.copy()

        for _ in range(max_per_brand):
            det, dbg = _find_one_for_brand(
                brand=brand,
                refs=all_refs,
                sift=sift,
                scene_gray=work,
                ratio=ratio,
                ransac_thresh=ransac_thresh,
                min_inliers=min_inliers,
            )

            debug_per_brand[brand].append(dbg)

            if det is None:
                break

            # dedupe against existing detections of same brand
            det_bb = _poly_bbox(det.poly)
            is_dup = False
            for prev in detections:
                if prev.brand != brand:
                    continue
                if _bbox_iou(det_bb, _poly_bbox(prev.poly)) >= iou_dedupe:
                    is_dup = True
                    break

            if is_dup:
                # If duplicate, still mask it out to avoid infinite loop, then continue
                work = _mask_polygon(work, det.poly, pad=mask_pad)
                continue

            detections.append(det)

            # mask away this instance and search again
            work = _mask_polygon(work, det.poly, pad=mask_pad)

    extra = {
        "ratio": ratio,
        "ransac_thresh": ransac_thresh,
        "min_inliers": min_inliers,
        "max_per_brand": max_per_brand,
        "mask_pad": mask_pad,
        "iou_dedupe": iou_dedupe,
        "brands": brands,
        "img_gray": img_gray,
        "debug_per_brand": debug_per_brand,
    }

    return detections, extra


# Single Best
def detect(image_bgr: np.ndarray) -> Tuple[str, float, dict]:
    """
    Backwards compatible wrapper:
    returns best brand among all detections (or 'unknown').

    score = inliers for best detection
    """
    detections, extra_all = detect_all(image_bgr)
    if not detections:
        return "unknown", 0.0, {"reason": "no_detections", **extra_all}

    best = max(detections, key=lambda d: d.inliers)
    extra = {
        **extra_all,
        "best_poly": best.poly,
        "best_inliers": best.inliers,
        "best_ref_path": best.ref_path,
        "best_ref_brand": best.brand,
    }
    return best.brand, float(best.inliers), extra


if __name__ == "__main__":
    refs = _get_refs()
    print("Loaded refs:", len(refs))
    brands = sorted(set(r.brand for r in refs))
    print("Brands:", brands)
