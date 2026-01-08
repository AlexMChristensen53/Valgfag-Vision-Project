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
    kp: List[cv2.KeyPoint]
    desc: np.ndarray



ROOT = Path(__file__).resolve().parents[2]
REFS_DIR = ROOT / "data" / "refs"

_REFS_CACHE: Optional[List[RefEntry]] = None


def _sift():
    # Hvis denne fejler: installer opencv-contrib-python
    return cv2.SIFT_create()


def _load_refs() -> List[RefEntry]:
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

            kp, desc = sift.detectAndCompute(gray, None)
            if desc is None or len(kp) < 10:
                continue

            refs.append(RefEntry(brand=brand, path=p, gray=gray, kp=kp, desc=desc))

    if not refs:
        raise RuntimeError(
            "Ingen gyldige reference-features fundet. "
            "Brug skarpere/tættere referencebilleder af label/logo."
        )

    return refs


def _get_refs() -> List[RefEntry]:
    global _REFS_CACHE
    if _REFS_CACHE is None:
        _REFS_CACHE = _load_refs()
    return _REFS_CACHE


def _good_matches(desc_ref: np.ndarray, desc_img: np.ndarray, ratio: float = 0.75) -> List[cv2.DMatch]:
    # SIFT -> L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_ref, desc_img, k=2)

    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def detect(image_bgr: np.ndarray) -> Tuple[str, float, dict]:
    """
    Returnerer:
      brand (str): bedste brand eller "unknown"
      score (float): antal good matches (højere=bedre)
      extra (dict): debug data til visualisering
    """
    refs = _get_refs()
    sift = _sift()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kp_img, desc_img = sift.detectAndCompute(gray, None)

    if desc_img is None or len(kp_img) < 10:
        return "unknown", 0.0, {"reason": "no_features"}

    best_per_brand: Dict[str, int] = {}

    # Global "bedste reference" (til at tegne matches)
    best_ref: Optional[RefEntry] = None
    best_good: List[cv2.DMatch] = []
    best_score: int = 0

    ratio = 0.75

    for r in refs:
        good = _good_matches(r.desc, desc_img, ratio=ratio)
        score = len(good)

        # Gem bedste score per brand (max over refs)
        best_per_brand[r.brand] = max(best_per_brand.get(r.brand, 0), score)

        # Gem global bedste ref til visualisering
        if score > best_score:
            best_score = score
            best_ref = r
            best_good = good

    # Find bedst brand ud fra best_per_brand
    best_brand = max(best_per_brand, key=best_per_brand.get)
    best_brand_score = float(best_per_brand[best_brand])

    # Minimumskrav (tunes)
    min_good = 20

    extra = {
        "best_per_brand": best_per_brand,
        "min_good": min_good,
        "ratio": ratio,
        # til visualisering
        "kp_img": kp_img,
        "img_gray": gray,
        "best_ref_path": str(best_ref.path) if best_ref else None,
        "best_ref_brand": best_ref.brand if best_ref else None,
        "best_ref_gray": best_ref.gray if best_ref else None,
        "best_ref_kp": best_ref.kp if best_ref else None,
        "best_good_matches": best_good,
        "best_ref_score": best_score,
    }

    # Hvis bedste brand ikke når threshold -> unknown, men stadig med debug
    if best_brand_score < min_good:
        return "unknown", best_brand_score, extra

    return best_brand, best_brand_score, extra


if __name__ == "__main__":
    print("Loaded refs from:", REFS_DIR)
    print("Count refs:", len(_get_refs()))
