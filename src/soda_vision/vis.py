import cv2
import numpy as np

def draw_matches(ref_gray, ref_kp, img_gray, img_kp, good_matches, max_draw=60):
    """Returnerer et billede hvor matches er tegnet mellem ref og input."""
    good_matches = sorted(good_matches, key=lambda m: m.distance)[:max_draw]
    out = cv2.drawMatches(
        ref_gray, ref_kp,
        img_gray, img_kp,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return out

def put_label(img_bgr, text, org=(20, 40)):
    out = img_bgr.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return out
