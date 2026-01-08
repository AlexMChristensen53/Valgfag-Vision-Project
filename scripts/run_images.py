from pathlib import Path
import sys
import cv2

# --- FIX PYTHON PATH (så "src." imports virker) ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.soda_vision.detect import detect  # noqa: E402


TEST_DIR = ROOT / "data" / "test_images"


def show_matches_window(img_name: str, brand: str, score: float, extra: dict, max_draw: int = 60):
    """
    Viser et vindue med match-linjer mellem best_ref og test-image.
    """
    ref_gray = extra.get("best_ref_gray", None)
    ref_kp = extra.get("best_ref_kp", None)
    img_gray = extra.get("img_gray", None)
    img_kp = extra.get("kp_img", None)
    good = extra.get("best_good_matches", [])

    if ref_gray is None or ref_kp is None or img_gray is None or img_kp is None or not good:
        print("  (Ingen debug-data til at tegne matches)")
        return

    good = sorted(good, key=lambda m: m.distance)[:max_draw]

    vis = cv2.drawMatches(
        ref_gray, ref_kp,
        img_gray, img_kp,
        good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    title = f"{img_name} -> {brand} (score={score:.1f})"
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

        brand, score, extra = detect(img)

        print(
            f"{img_path.name:25s} -> {brand:10s} "
            f"score={score:.1f} "
            f"best_per_brand={extra.get('best_per_brand')}"
        )

        # VIS matches (hver gang)
        show_matches_window(img_path.name, brand, score, extra, max_draw=60)


if __name__ == "__main__":
    main()
