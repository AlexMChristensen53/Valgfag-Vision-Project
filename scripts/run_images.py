from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.soda_vision.detect import detect_all

TEST_DIR = ROOT / "data" / "test_images"


def choose_mode() -> str:
    print("\nVælg mode:")
    print("  1) Normal (kun detections)")
    print("  2) Debug (SIFT matches + detections)")
    choice = input("Valg [1/2]: ").strip()
    return "debug" if choice == "2" else "normal"


def nav_action(key: int) -> str | None:
    """
    Returnér en navigation-handling ud fra en tast.
    Bruges både i normal- og debug-visning. Bruger waitKeyEx-keycodes.
    """
    # ESC / q
    if key in (27, ord("q"), ord("Q")):
        return "quit"

    # Venstre (Windows / Linux-X11) + fallback (A/J)
    if key in (2424832, 81, 65361, ord("a"), ord("A"), ord("j"), ord("J")):
        return "prev"

    # Højre (Windows / Linux-X11) + fallback (D/L)
    if key in (2555904, 83, 65363, ord("d"), ord("D"), ord("l"), ord("L")):
        return "next"

    return None


def _resize_for_screen(img, max_w=1400, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _polygon_top_left(polygon: np.ndarray) -> tuple[int, int]:
    pts = polygon.reshape(-1, 2)
    x = int(np.min(pts[:, 0]))
    y = int(np.min(pts[:, 1]))
    return x, y


def _place_label_without_overlap(
    label_text: str,
    anchor_xy: tuple[int, int],
    used_label_rectangles: list,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.9,
    thickness=2,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """
    Placer tekst så den ikke overlapper tidligere labels.
    (Heuristik: prøv flere offsets og tag første uden overlap)
    """
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    x0, y0 = anchor_xy

    candidate_offsets = [
        (0, -10), (0, -30), (0, -50),
        (10, -10), (10, -30), (10, -50),
        (0, 20), (0, 40),
        (20, 20), (20, 40),
        (-text_w - 10, -10), (-text_w - 10, -30),
    ]

    def rect_for_position(x, y):
        rect_x = x
        rect_y = y - text_h
        rect_w = text_w
        rect_h = text_h + baseline
        return (rect_x, rect_y, rect_w, rect_h)

    def intersects(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)

    for dx, dy in candidate_offsets:
        x = x0 + dx
        y = y0 + dy
        rect = rect_for_position(x, y)

        collision = any(intersects(rect, r) for r in used_label_rectangles)
        if not collision:
            used_label_rectangles.append(rect)
            return x, y, rect

    rect = rect_for_position(x0, y0 - 10)
    used_label_rectangles.append(rect)
    return x0, y0 - 10, rect


def draw_detections_on_image(scene_bgr_resized: np.ndarray, detections: list) -> np.ndarray:
    """
    Tegner alle detections på samme billede og forsøger at undgå label-overlap.
    """
    output = scene_bgr_resized.copy()
    used_label_rectangles = []

    detections_sorted = sorted(detections, key=lambda d: d.inliers, reverse=True)

    for det in detections_sorted:
        polygon = det.polygon
        pts = polygon.reshape(-1, 2).astype(int)

        # Polygon
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Label-tekst
        label = f"{det.brand} ({det.inliers})"
        anchor = _polygon_top_left(polygon)
        text_x, text_y, _ = _place_label_without_overlap(label, anchor, used_label_rectangles)

        cv2.putText(output, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return output


def _draw_sift_matches_window(
    image_title: str,
    det,
    max_draw: int = 80,
) -> str | None:
    """
    Debug-visning: viser SIFT matches for EN detection.
    Inliers = grøn, outliers = rød.

    Navigation:
      - Venstre/Højre piletast (eller A/D, J/L) => skift billede
      - ESC/q => stop debug-visning og gå videre til scene-visning
    """
    dbg = getattr(det, "debug", None)
    if not dbg:
        print(f"  [DEBUG] Mangler det.debug for {det.brand} ({det.inliers}). "
              f"Ret detect.py så Detection(debug=debug_data) sættes.")
        return None

    ref_gray = dbg.get("ref_gray")
    ref_kp = dbg.get("ref_keypoints")
    scene_gray = dbg.get("scene_gray")
    scene_kp = dbg.get("scene_keypoints")
    matches = dbg.get("matches", [])
    inlier_mask = dbg.get("inlier_mask", None)

    if ref_gray is None or ref_kp is None or scene_gray is None or scene_kp is None or not matches:
        print(f"  [DEBUG] Ufuldstændig debug-data for {det.brand}.")
        return None

    # Begræns antal matches vi tegner, sortér efter distance
    matches_sorted = sorted(matches, key=lambda m: m.distance)[:max_draw]

    # Split i inliers/outliers hvis vi har maske
    if inlier_mask is not None:
        mask_flat = np.asarray(inlier_mask).reshape(-1).astype(int)
        if len(mask_flat) == len(matches):
            inliers = []
            outliers = []
            for m in matches_sorted:
                idx = next(
                    (i for i, mm in enumerate(matches)
                     if (mm.queryIdx == m.queryIdx and mm.trainIdx == m.trainIdx)),
                    -1
                )
                if idx >= 0 and mask_flat[idx] == 1:
                    inliers.append(m)
                else:
                    outliers.append(m)

            vis_in = cv2.drawMatches(
                ref_gray, ref_kp,
                scene_gray, scene_kp,
                inliers, None,
                matchColor=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            vis_out = cv2.drawMatches(
                ref_gray, ref_kp,
                scene_gray, scene_kp,
                outliers, None,
                matchColor=(0, 0, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            vis = cv2.addWeighted(vis_in, 0.8, vis_out, 0.8, 0)
        else:
            vis = cv2.drawMatches(
                ref_gray, ref_kp,
                scene_gray, scene_kp,
                matches_sorted, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
    else:
        vis = cv2.drawMatches(
            ref_gray, ref_kp,
            scene_gray, scene_kp,
            matches_sorted, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    
    title = f"{image_title} | SIFT: {det.brand} (inliers={det.inliers})"
    vis = _resize_for_screen(vis)
    cv2.imshow(title, vis)

    key = cv2.waitKeyEx(0)
    cv2.destroyWindow(title)

    action = nav_action(key)
    if action == "quit":
        raise KeyboardInterrupt
    return action


def main():
    mode = choose_mode()

    if not TEST_DIR.exists():
        raise SystemExit(f"Mangler mappe: {TEST_DIR}")

    images = sorted(list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png")))
    if not images:
        raise SystemExit(f"Ingen billeder fundet i: {TEST_DIR}")

    idx = 0
    num_images = len(images)

    while True:
        img_path = images[idx]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Kunne ikke læse: {img_path.name}")
            idx = (idx + 1) % num_images
            continue

        # Kør multi-detection
        detections, extra = detect_all(img_bgr)

        # Udskriv kort status
        per_brand = {}
        for d in detections:
            per_brand[d.brand] = per_brand.get(d.brand, 0) + 1
        print(f"{img_path.name:25s} -> detections={len(detections)} per_brand={per_brand}")

        # Byg resized BGR i samme space som detect_all
        h, w = img_bgr.shape[:2]
        m = max(h, w)
        if m > 1200:
            scale = 1200 / m
            scene_bgr_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scene_bgr_resized = img_bgr

        # Debug-mode: vis først SIFT matches pr detection
        if mode == "debug" and detections:
            detections_sorted = sorted(detections, key=lambda d: d.inliers, reverse=True)
            jumped = False
            try:
                for det in detections_sorted:
                    action = _draw_sift_matches_window(img_path.name, det, max_draw=80)
                    if action == "prev":
                        idx = (idx - 1) % num_images
                        jumped = True
                        break
                    if action == "next":
                        idx = (idx + 1) % num_images
                        jumped = True
                        break
            except KeyboardInterrupt:
                print("  [DEBUG] Afbrød debug-visning (ESC/q). Fortsætter til scene-billedet.")

            if jumped:
                # Skift billede med det samme
                continue

        # Tegn alle detections på samme billede
        out = draw_detections_on_image(scene_bgr_resized, detections)

        # Hvis ingen detections, vis tydelig tekst
        if not detections:
            cv2.putText(out, "NO DETECTIONS", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(out, "NO DETECTIONS", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        out_show = _resize_for_screen(out)

        cv2.imshow(f"POLY: {img_path.name}  [{idx+1}/{num_images}]", out_show)

        key = cv2.waitKeyEx(0)
        cv2.destroyAllWindows()

        action = nav_action(key)
        if action == "quit":
            break
        elif action == "prev":
            idx = (idx - 1) % num_images
        elif action == "next":
            idx = (idx + 1) % num_images
        else:
            # Ukendt tast: bliv på samme billede
            continue


if __name__ == "__main__":
    main()
