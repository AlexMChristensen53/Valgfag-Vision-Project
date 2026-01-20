from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.soda_vision.detect import detect_all, Detection  # noqa: E402

TEST_DIR = ROOT / "data" / "test_images"


def _resize_for_screen(img, max_w=1400, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _to_int_pts(poly: np.ndarray) -> np.ndarray:
    """poly (4,1,2) -> (4,2) int"""
    return poly.reshape(-1, 2).astype(int)


def draw_all_polygons(img_name: str, img_bgr_resized: np.ndarray, detections: list[Detection]):
    """
    Tegner alle detections' polygons på samme scene.
    """
    if not detections:
        out = img_bgr_resized.copy()
        cv2.putText(out, "NO DETECTIONS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, "NO DETECTIONS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        out = _resize_for_screen(out)
        cv2.imshow(f"POLY: {img_name}", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    out = img_bgr_resized.copy()

    # Sortér så den stærkeste tegnes først
    dets = sorted(detections, key=lambda d: d.inliers, reverse=True)

    for det in dets:
        pts = _to_int_pts(det.poly)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # label ved første hjørne
        x, y = pts[0]
        cv2.putText(out, f"{det.brand} ({det.inliers})", (max(0, x), max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"{det.brand} ({det.inliers})", (max(0, x), max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out = _resize_for_screen(out)
    cv2.imshow(f"POLY: {img_name}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_matches_for_detection(
    img_name: str,
    det: Detection,
    extra: dict,
    max_draw: int = 60,
):
    """
    Viser match-linjer for EN detection:
    - ref vs scene
    - inliers grøn, outliers rød
    Kræver at detect_all's extra indeholder debug_per_brand (som i mit forslag).
    """
    dbg_list = extra.get("debug_per_brand", {}).get(det.brand, [])
    if not dbg_list:
        print("  (Ingen debug_per_brand for dette brand)")
        return

    # Find debug entry der matcher det.ref_path (bedste ref for denne detection)
    dbg = next((d for d in dbg_list if d.get("best_ref_path") == det.ref_path), None)
    if dbg is None:
        # fallback: tag sidste debug entry der havde en homografi
        dbg = next((d for d in reversed(dbg_list) if d.get("ref_gray") is not None and d.get("matches")), None)

    if dbg is None:
        print("  (Ingen debug-data fundet til at tegne matches for denne detection)")
        return

    ref_gray = dbg.get("ref_gray")
    ref_kp = dbg.get("ref_kp")
    scene_gray = dbg.get("scene_gray")
    img_kp = dbg.get("kp_img")
    matches = dbg.get("matches", [])
    inlier_mask = dbg.get("inlier_mask", None)

    if ref_gray is None or ref_kp is None or scene_gray is None or img_kp is None or not matches:
        print("  (Manglende data for match-visning)")
        return

    matches = sorted(matches, key=lambda m: m.distance)[:max_draw]

    if inlier_mask is not None:
        inlier_mask = np.asarray(inlier_mask).reshape(-1)

        original = dbg.get("matches", [])
        idxs = []
        for m in matches:
            try:
                i = original.index(m)
            except ValueError:
                i = next((j for j, mm in enumerate(original)
                          if (mm.queryIdx == m.queryIdx and mm.trainIdx == m.trainIdx and mm.distance == m.distance)), -1)
            idxs.append(i)

        inliers = []
        outliers = []
        for m, i in zip(matches, idxs):
            if i >= 0 and i < len(inlier_mask) and int(inlier_mask[i]) == 1:
                inliers.append(m)
            else:
                outliers.append(m)

        vis_in = cv2.drawMatches(
            ref_gray, ref_kp,
            scene_gray, img_kp,
            inliers, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        vis_out = cv2.drawMatches(
            ref_gray, ref_kp,
            scene_gray, img_kp,
            outliers, None,
            matchColor=(0, 0, 255),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        vis = cv2.addWeighted(vis_in, 0.8, vis_out, 0.8, 0)
    else:
        vis = cv2.drawMatches(
            ref_gray, ref_kp,
            scene_gray, img_kp,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    title = f"{img_name} | {det.brand} inliers={det.inliers} | ref={Path(det.ref_path).name}"
    vis = _resize_for_screen(vis)
    cv2.imshow(title, vis)
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

        detections, extra = detect_all(img)

        # Lav samme resized BGR til polygon-visning
        h, w = img.shape[:2]
        m = max(h, w)
        if m > 1200:
            scale = 1200 / m
            img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img

        # Print summary
        per_brand = {}
        for d in detections:
            per_brand[d.brand] = per_brand.get(d.brand, 0) + 1

        print(
            f"{img_path.name:25s} -> detections={len(detections):2d} "
            f"per_brand={per_brand}"
        )

        # Vis alle ROIs på scenen
        draw_all_polygons(img_path.name, img_resized, detections)

        # vis matches pr detection
        for d in sorted(detections, key=lambda x: x.inliers, reverse=True):
            show_matches_for_detection(img_path.name, d, extra, max_draw=60)


if __name__ == "__main__":
    main()
