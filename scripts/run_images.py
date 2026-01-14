from pathlib import Path
import sys
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.soda_vision.detect import detect  # noqa: E402

TEST_DIR = ROOT / "data" / "test_images"


def _resize_for_screen(img, max_w=1400, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def show_matches_window(img_name: str, brand: str, score: float, extra: dict, max_draw: int = 60):
    """
    Viser et vindue med match-linjer mellem best_ref og test-image.
    Inliers (RANSAC) bliver grønne, outliers røde.
    """
    ref_gray = extra.get("best_ref_gray")
    ref_kp = extra.get("best_ref_kp")
    img_gray = extra.get("img_gray")
    img_kp = extra.get("kp_img")

    matches = extra.get("best_matches", [])
    inlier_mask = extra.get("best_inlier_mask", None)

    if ref_gray is None or ref_kp is None or img_gray is None or img_kp is None or not matches:
        print("  (Ingen debug-data til at tegne matches)")
        return

    # Sortér matches efter distance og begræns antal
    matches = sorted(matches, key=lambda m: m.distance)[:max_draw]

    # Hvis vi har inlier-mask, så farv dem i output
    if inlier_mask is not None and len(inlier_mask) >= len(extra["best_matches"]):
        # Vi skal lave en mask for de matches vi har valgt (top max_draw)
        # inlier_mask matcher original match-liste, så vi mapper via index.
        # Derfor: find index i original-listen.
        original = extra["best_matches"]
        idxs = []
        for m in matches:
            # find first exact object match.
            try:
                i = original.index(m)
            except ValueError:
                i = next((j for j, mm in enumerate(original)
                          if (mm.queryIdx == m.queryIdx and mm.trainIdx == m.trainIdx and mm.distance == m.distance)), -1)
            idxs.append(i)

        inliers = []
        outliers = []
        inlier_mask = np.asarray(inlier_mask).reshape(-1)
        for m, i in zip(matches, idxs):
            if i >= 0 and int(inlier_mask[i]) == 1:
                inliers.append(m)
            else:
                outliers.append(m)

        vis_in = cv2.drawMatches(
            ref_gray, ref_kp,
            img_gray, img_kp,
            inliers, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        vis_out = cv2.drawMatches(
            ref_gray, ref_kp,
            img_gray, img_kp,
            outliers, None,
            matchColor=(0, 0, 255),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        vis = cv2.addWeighted(vis_in, 0.8, vis_out, 0.8, 0)
    else:
        vis = cv2.drawMatches(
            ref_gray, ref_kp,
            img_gray, img_kp,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    title = f"{img_name} -> {brand} (inliers={score:.0f})"
    vis = _resize_for_screen(vis)
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_polygon_on_scene(img_name: str, brand: str, score: float, extra: dict):
    """
    Viser scenebilledet med projiceret polygon (ref-corners) hvis homografi findes.
    NB: polygon coords er i detect.py's resized grayscale koordinater, ikke nødvendigvis i original BGR.
    For at holde det enkelt, tegner vi på en resized version af originalen med samme max_dim som detect.
    """
    poly = extra.get("best_poly", None)
    if brand == "unknown":
        return

    if poly is None or len(poly) != 4:
        return


    # Brug samme resize som detect: max_dim=1200
    img_bgr = extra.get("orig_bgr_for_poly", None)
    if img_bgr is None:
        return

    out = img_bgr.copy()
    pts = poly.reshape(-1, 2).astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.putText(out, f"{brand} inliers={score:.0f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, f"{brand} inliers={score:.0f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    out = _resize_for_screen(out)
    cv2.imshow(f"POLY: {img_name}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if not TEST_DIR.exists():
        raise SystemExit(f"Mangler mappe: {TEST_DIR}")

    images = sorted(list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png")))
    if not images:
        raise SystemExit(f"Ingen billeder fundet i: {TEST_DIR}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Kunne ikke læse: {img_path.name}")
            continue

        brand, score, extra = detect(img)

        print(
            f"{img_path.name:25s} -> {brand:10s} "
            f"inliers={score:.0f} "
            f"best_inliers_per_brand={extra.get('best_inliers_per_brand')}"
        )

        # For polygon-visning skal vi tegne i samme resize-space som detect.
        # Vi laver derfor en resized BGR her med samme max_dim=1200.
        h, w = img.shape[:2]
        m = max(h, w)
        if m > 1200:
            scale = 1200 / m
            img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img

        extra["orig_bgr_for_poly"] = img_resized

        show_matches_window(img_path.name, brand, score, extra, max_draw=60)
        show_polygon_on_scene(img_path.name, brand, score, extra)


if __name__ == "__main__":
    main()
