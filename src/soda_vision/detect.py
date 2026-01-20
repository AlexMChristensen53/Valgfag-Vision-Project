from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RefEntry:
    brand: str
    path: Path
    gray: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    shape_hw: Tuple[int, int]


@dataclass
class Detection:
    brand: str
    inliers: int
    polygon: np.ndarray
    ref_path: str
    homography: np.ndarray
    matches: List[cv2.DMatch]
    inlier_mask: Optional[np.ndarray]
    debug: Optional[dict] = None

# Paths
ROOT = Path(__file__).resolve().parents[2]
REFS_DIR = ROOT / "data" / "refs"
_REFS_CACHE: Optional[List[RefEntry]] = None


# Feature helpers
def _sift() -> cv2.SIFT:
    return cv2.SIFT_create()


def _resize_max(img_gray: np.ndarray, max_dim: int = 900) -> np.ndarray:
    """Resize grayscale image so max(width,height) <= max_dim, keeping aspect."""
    h, w = img_gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img_gray
    scale = max_dim / m
    return cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _cap_features(keypoints: List[cv2.KeyPoint], descriptors: np.ndarray, max_keypoints: int = 800):
    """Behold top-N keypoints efter response; behold tilsvarende descriptor-rækker."""
    if descriptors is None or keypoints is None:
        return keypoints, descriptors
    if len(keypoints) <= max_keypoints:
        return keypoints, descriptors
    idx = np.argsort([-k.response for k in keypoints])[:max_keypoints]
    keypoints2 = [keypoints[i] for i in idx]
    descriptors2 = descriptors[idx, :]
    return keypoints2, descriptors2


def _load_refs(max_dim: int = 900, max_keypoints: int = 800) -> List[RefEntry]:
    if not REFS_DIR.exists():
        raise FileNotFoundError(f"Mangler refs-mappe: {REFS_DIR}")

    sift = _sift()
    refs: List[RefEntry] = []

    brand_dirs = [p for p in REFS_DIR.iterdir() if p.is_dir()]
    if not brand_dirs:
        raise RuntimeError(f"Ingen brand-mapper fundet i {REFS_DIR}")

    for brand_dir in sorted(brand_dirs):
        brand = brand_dir.name
        image_paths = sorted(list(brand_dir.glob("*.jpg")) + list(brand_dir.glob("*.png")))

        for p in image_paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            gray = _resize_max(gray, max_dim=max_dim)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            keypoints, descriptors = _cap_features(keypoints, descriptors, max_keypoints=max_keypoints)

            if descriptors is None or keypoints is None or len(keypoints) < 10:
                continue

            refs.append(
                RefEntry(
                    brand=brand,
                    path=p,
                    gray=gray,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    shape_hw=gray.shape[:2],
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
      - behold kun matches som er gensidige
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_ref_to_img = matcher.knnMatch(desc_ref, desc_img, k=2)
    good_ref_to_img: List[cv2.DMatch] = []
    for pair in knn_ref_to_img:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ref_to_img.append(m)

    knn_img_to_ref = matcher.knnMatch(desc_img, desc_ref, k=2)
    good_img_to_ref: List[cv2.DMatch] = []
    for pair in knn_img_to_ref:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_img_to_ref.append(m)

    reverse_map = {(m.trainIdx, m.queryIdx): m for m in good_img_to_ref}
    mutual = [m for m in good_ref_to_img if (m.queryIdx, m.trainIdx) in reverse_map]
    return mutual


def _homography_inliers(
    ref_keypoints: List[cv2.KeyPoint],
    scene_keypoints: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_thresh: float = 5.0,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Returnerer: (inlier_count, H, inlier_mask)."""
    if len(matches) < 4:
        return 0, None, None

    src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
    if H is None or mask is None:
        return 0, None, None

    inliers = int(mask.sum())
    return inliers, H, mask



def _polygon_area(polygon: np.ndarray) -> float:
    pts = polygon.reshape(-1, 2).astype(np.float32)
    return float(cv2.contourArea(pts))


def _polygon_intersection_area(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """
    Robust overlap mellem to konvekse polygoner.
    MinAreaRect giver altid konvekst output, så intersectConvexConvex passer fint her.
    """
    a = polygon_a.reshape(-1, 2).astype(np.float32)
    b = polygon_b.reshape(-1, 2).astype(np.float32)

    if len(a) < 3 or len(b) < 3:
        return 0.0

    area, _ = cv2.intersectConvexConvex(a, b)
    return float(area) if area is not None else 0.0


def _overlap_ratio_min_area(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """
    overlap = intersection_area / min(areaA, areaB)
    God til dublet-detektion, også når den ene polygon er "lille i stor".
    """
    area_a = _polygon_area(polygon_a)
    area_b = _polygon_area(polygon_b)
    min_area = min(area_a, area_b)
    if min_area <= 1e-6:
        return 0.0

    inter = _polygon_intersection_area(polygon_a, polygon_b)
    return inter / min_area


def _min_area_rect_from_inliers(
    scene_keypoints: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Byg en polygon via minAreaRect baseret på INLIER keypoints i scenen.
    Det giver typisk en "strammere" og mere stabil boks end at projicere ref-hjørner.
    """
    mask_flat = np.asarray(inlier_mask).reshape(-1)
    if len(mask_flat) != len(matches):
        return None

    inlier_points: List[Tuple[float, float]] = []
    for m, keep in zip(matches, mask_flat):
        if int(keep) != 1:
            continue
        x, y = scene_keypoints[m.trainIdx].pt
        inlier_points.append((x, y))

    if len(inlier_points) < 4:
        return None

    pts = np.array(inlier_points, dtype=np.float32).reshape(-1, 2)

    rect = cv2.minAreaRect(pts)  # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)    # (4,2)
    box = np.array(box, dtype=np.float32).reshape(-1, 1, 2)

    # Filtrér helt degenererede bokse væk
    if _polygon_area(box) < 50.0:
        return None

    return box


def _mask_polygon(scene_gray: np.ndarray, polygon: np.ndarray, pad: int = 12) -> np.ndarray:
    """
    Mask polygon-området (plus lidt padding) så vi ikke finder samme objekt igen.
    """
    out = scene_gray.copy()
    pts = polygon.reshape(-1, 2).astype(np.int32)

    # Fyld polygonen
    cv2.fillConvexPoly(out, pts, 0)

    if pad > 0:
        mask = np.zeros_like(scene_gray, dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)

        kernel_size = max(1, pad // 2) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

        out[mask > 0] = 0

    return out



# single-instance search for ét brand
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
    Find bedste ENKELTE instans for et brand i den givne scene_gray.
    Returnerer (Detection eller None, debug dict).
    """

    # Udtræk scene-features
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_gray, None)
    scene_keypoints, scene_descriptors = _cap_features(
        scene_keypoints, scene_descriptors, max_keypoints=800
    )

    if scene_descriptors is None or scene_keypoints is None or len(scene_keypoints) < 10:
        return None, {"reason": "no_features_in_scene"}

    best_inliers = 0
    best_ref: Optional[RefEntry] = None
    best_matches: List[cv2.DMatch] = []
    best_inlier_mask: Optional[np.ndarray] = None
    best_homography: Optional[np.ndarray] = None
    best_polygon: Optional[np.ndarray] = None

    for ref in refs:
        if ref.brand != brand:
            continue

        matches = _mutual_good_matches(ref.descriptors, scene_descriptors, ratio=ratio)
        inliers, homography, inlier_mask = _homography_inliers(
            ref.keypoints, scene_keypoints, matches, ransac_reproj_thresh=ransac_thresh
        )

        if homography is None or inlier_mask is None:
            continue

        if inliers > best_inliers:
            # Byg polygon fra inlier-punkter (minAreaRect)
            polygon = _min_area_rect_from_inliers(scene_keypoints, matches, inlier_mask)
            if polygon is None:
                continue

            best_inliers = int(inliers)
            best_ref = ref
            best_matches = matches
            best_inlier_mask = inlier_mask
            best_homography = homography
            best_polygon = polygon

    if best_ref is None or best_homography is None or best_polygon is None:
        return None, {"reason": "no_homography"}

    if best_inliers < min_inliers:
        return None, {
            "reason": "below_threshold",
            "best_inliers": int(best_inliers),
            "min_inliers": int(min_inliers),
            "best_ref_path": str(best_ref.path),
        }
        
    debug_data = {
        "brand": brand,
        "best_inliers": int(best_inliers),
        "best_ref_path": str(best_ref.path),

        "scene_gray": scene_gray,
        "scene_keypoints": scene_keypoints,

        "ref_gray": best_ref.gray,
        "ref_keypoints": best_ref.keypoints,

        "matches": best_matches,
        "inlier_mask": best_inlier_mask,
        "homography": best_homography,
        "polygon": best_polygon,
    }

    detection = Detection(
        brand=brand,
        inliers=int(best_inliers),
        polygon=best_polygon,
        ref_path=str(best_ref.path),
        homography=best_homography,
        matches=best_matches,
        inlier_mask=best_inlier_mask,
        debug=debug_data,
    )

    return detection, debug_data



# Multi-detect
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

    # Preprocess input
    img_gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scene_gray = _resize_max(img_gray_full, max_dim=1200)

    # Parametre
    ratio = 0.77
    ransac_thresh = 5.0
    min_inliers = 12

    max_per_brand = 6
    mask_pad = 25

    # Dedupe (indenfor samme brand)
    overlap_min_area_threshold = 0.95  # hvis overlap/min(area) er høj => samme objekt

    detections: List[Detection] = []
    debug_per_brand: Dict[str, List[dict]] = {b: [] for b in brands}

    for brand in brands:
        # Vi starter frisk fra original scene for hvert brand (masker kun brand-instanser)
        work_scene = scene_gray.copy()

        for _ in range(max_per_brand):
            det, dbg = _find_one_for_brand(
                brand=brand,
                refs=all_refs,
                sift=sift,
                scene_gray=work_scene,
                ratio=ratio,
                ransac_thresh=ransac_thresh,
                min_inliers=min_inliers,
            )
            debug_per_brand[brand].append(dbg)

            if det is None:
                break

            # Dedupe: undgå at samme flaske giver 2 detections
            is_duplicate = False
            for prev in detections:
                if prev.brand != brand:
                    continue
                overlap_ratio = _overlap_ratio_min_area(det.polygon, prev.polygon)
                if overlap_ratio >= overlap_min_area_threshold:
                    is_duplicate = True
                    break

            # Mask altid det fundne område væk, så vi ikke ender i loop på samme label
            work_scene = _mask_polygon(work_scene, det.polygon, pad=mask_pad)

            if is_duplicate:
                continue

            detections.append(det)

    extra = {
        "ratio": ratio,
        "ransac_thresh": ransac_thresh,
        "min_inliers": min_inliers,
        "max_per_brand": max_per_brand,
        "mask_pad": mask_pad,
        "overlap_min_area_threshold": overlap_min_area_threshold,
        "brands": brands,
        "scene_gray": scene_gray,
        "debug_per_brand": debug_per_brand,
    }
    return detections, extra


def detect(image_bgr: np.ndarray) -> Tuple[str, float, dict]:
    """
    Backwards compatible wrapper:
      - returnerer bedste brand blandt alle detections (eller 'unknown')
      - score = inliers for bedste detection
    """
    detections, extra_all = detect_all(image_bgr)
    if not detections:
        return "unknown", 0.0, {"reason": "no_detections", **extra_all}

    best = max(detections, key=lambda d: d.inliers)
    extra = {
        **extra_all,
        "best_polygon": best.polygon,
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
