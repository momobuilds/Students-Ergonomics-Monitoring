"""
overlay_drawing.py - Shared, good-looking overlays for the camera/video monitors.

Used by both main.realtime.py and main.TestVideo.py so the on-screen graphics
stay identical:

  - draw_posture_skeleton: a body skeleton that follows the person, with the
    three measured posture parameters drawn as highlighted lines + value chips
    (shoulder asymmetry, head lateral tilt, forward-head proxy), colored green
    when within the literature threshold and red when not.
  - draw_eye_boxes: clean corner-bracket boxes around both eyes (green when
    open, red when closed) with the eye landmark outline.
  - draw_state_banner: bottom banner showing the model's predicted state.

All functions take the already-imported ``cv2`` module as the first argument so
this file has no heavy import of its own at module load time.
"""

import numpy as np


# --- colors (BGR) ------------------------------------------------------------
GOOD = (90, 220, 90)        # green
BAD = (60, 60, 255)         # red
SKELETON = (235, 206, 135)  # light blue
REF = (120, 120, 120)       # dim grey (reference axes)
FHP = (255, 178, 80)        # orange-ish
WHITE = (255, 255, 255)
DIM = (160, 160, 160)

STATE_COLORS = {
    "normal": (90, 220, 90),
    "sleepy": (0, 200, 255),
    "stress": (60, 60, 255),
    "abnormal": (60, 60, 255),
    "anomalous": (60, 60, 255),
}

# MediaPipe Pose landmark indices we care about.
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH = 11, 12
L_ELB, R_ELB, L_WR, R_WR = 13, 14, 15, 16
L_HIP, R_HIP = 23, 24

# Upper-body skeleton connections (drawn only when both ends are visible).
SKELETON_LINKS = [
    (L_SH, R_SH), (L_SH, L_ELB), (L_ELB, L_WR), (R_SH, R_ELB), (R_ELB, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (NOSE, L_EAR), (NOSE, R_EAR), (L_EAR, L_SH), (R_EAR, R_SH),
]


# --- small drawing helpers ---------------------------------------------------

def _font(cv2):
    return cv2.FONT_HERSHEY_SIMPLEX


def _text(cv2, frame, text, x, y, scale, color, thickness=1):
    """Text with a dark outline for readability over any background."""
    cv2.putText(frame, text, (x, y), _font(cv2), scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), _font(cv2), scale, color,
                thickness, cv2.LINE_AA)


def _glow_line(cv2, frame, p1, p2, color, thickness):
    """A line with a soft dark halo, so it reads on busy frames."""
    cv2.line(frame, p1, p2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _dashed_line(cv2, frame, p1, p2, color, dash=10):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    length = float(np.hypot(*(p2 - p1)))
    if length < 1:
        return
    steps = max(int(length // dash), 1)
    for i in range(0, steps, 2):
        a = p1 + (p2 - p1) * (i / steps)
        b = p1 + (p2 - p1) * (min(i + 1, steps) / steps)
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)),
                 color, 1, cv2.LINE_AA)


def _chip(cv2, frame, text, org, color):
    """A small translucent label with a colored left border."""
    h, w = frame.shape[:2]
    (tw, th), bl = cv2.getTextSize(text, _font(cv2), 0.5, 1)
    pad = 5
    x = int(np.clip(org[0], 6, w - tw - 10))
    y = int(np.clip(org[1], th + 8, h - 10))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + bl), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x - pad, y - th - pad), (x - pad + 3, y + bl), color, -1)
    cv2.putText(frame, text, (x, y), _font(cv2), 0.5, color, 1, cv2.LINE_AA)


def _corner_box(cv2, frame, x1, y1, x2, y2, color, thickness=2):
    """Draw only the 4 corner brackets of a box (a clean 'tech' look)."""
    arm = max(8, int((x2 - x1) * 0.28))
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)


def _vpx(landmarks, idx, w, h, min_vis):
    lm = landmarks[idx]
    if lm["visibility"] < min_vis:
        return None
    return (int(lm["x"] * w), int(lm["y"] * h))


def _mid(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


# --- public overlays ---------------------------------------------------------

def draw_posture_skeleton(cv2, frame, landmarks, values, cfg):
    """Draw the body skeleton + the three measured posture parameters.

    Args:
        landmarks: list of 33 MediaPipe dicts (x, y, z, visibility) or None.
        values: (shoulder_asym, head_tilt, fhp) or None (used for the labels).
        cfg: the Posture config module (thresholds + MIN_VISIBILITY).
    """
    if landmarks is None:
        return

    h, w = frame.shape[:2]
    mv = cfg.MIN_VISIBILITY
    pts = {i: _vpx(landmarks, i, w, h, mv)
           for i in (NOSE, L_EAR, R_EAR, L_SH, R_SH,
                     L_ELB, R_ELB, L_WR, R_WR, L_HIP, R_HIP)}

    # 1) skeleton links
    for a, b in SKELETON_LINKS:
        if pts.get(a) and pts.get(b):
            _glow_line(cv2, frame, pts[a], pts[b], SKELETON, 2)

    # 2) joints
    for point in pts.values():
        if point:
            cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 4, (0, 0, 0), 1, cv2.LINE_AA)

    shoulder_val, tilt_val, fhp_val = values if values else (None, None, None)
    ls, rs = pts.get(L_SH), pts.get(R_SH)
    le, re = pts.get(L_EAR), pts.get(R_EAR)
    nose = pts.get(NOSE)
    sh_mid = _mid(ls, rs) if ls and rs else None
    ear_mid = _mid(le, re) if le and re else None

    # 3) shoulder asymmetry: the shoulder line itself
    if ls and rs:
        ok = shoulder_val is None or abs(shoulder_val) <= cfg.SHOULDER_ASYMMETRY_THRESHOLD
        color = GOOD if ok else BAD
        _glow_line(cv2, frame, ls, rs, color, 4)
        if shoulder_val is not None:
            _chip(cv2, frame, f"Shoulder {shoulder_val:+.1f}deg",
                  (min(ls[0], rs[0]), min(ls[1], rs[1]) - 14), color)

    # 4) head lateral tilt: ear-mid -> shoulder-mid vs the vertical
    if ear_mid and sh_mid:
        ok = tilt_val is None or abs(tilt_val) <= cfg.HEAD_LATERAL_TILT_THRESHOLD
        color = GOOD if ok else BAD
        _dashed_line(cv2, frame, sh_mid, (sh_mid[0], sh_mid[1] - 170), REF)
        _glow_line(cv2, frame, ear_mid, sh_mid, color, 3)
        if tilt_val is not None:
            _chip(cv2, frame, f"Head tilt {tilt_val:+.1f}deg",
                  (ear_mid[0] + 12, ear_mid[1]), color)

    # 5) forward-head proxy: nose -> shoulder-mid vertical span
    if nose and sh_mid:
        ok = fhp_val is None or fhp_val >= cfg.FORWARD_HEAD_PROXY_THRESHOLD
        color = FHP if ok else BAD
        _glow_line(cv2, frame, nose, sh_mid, color, 2)
        if fhp_val is not None:
            _chip(cv2, frame, f"FHP {fhp_val:.2f}",
                  (sh_mid[0] + 12, _mid(nose, sh_mid)[1]), color)


def draw_eye_boxes(cv2, frame, left_eye, right_eye, is_closed):
    """Draw corner-bracket boxes + outline around both eyes."""
    color = BAD if is_closed else GOOD
    for points in (left_eye, right_eye):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pad_x = max(8, int((max(xs) - min(xs)) * 0.35))
        pad_y = max(8, int((max(ys) - min(ys)) * 0.9))
        x1, y1 = min(xs) - pad_x, min(ys) - pad_y
        x2, y2 = max(xs) + pad_x, max(ys) + pad_y
        _corner_box(cv2, frame, x1, y1, x2, y2, color)
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True,
                      color, 1, cv2.LINE_AA)


def draw_state_banner(cv2, frame, prediction, waiting_text="analyzing..."):
    """Bottom banner with the predicted state (+ probabilities if available)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if prediction is None:
        _text(cv2, frame, f"STATE: {waiting_text}", 12, h - 38, 0.7, DIM, 2)
        return

    label = prediction["label"]
    color = STATE_COLORS.get(label, WHITE)
    _text(cv2, frame, f"STATE: {label.upper()}", 12, h - 38, 0.9, color, 2)
    if prediction.get("proba"):
        proba_txt = "   ".join(
            f"{cls}:{prediction['proba'][cls]:.0%}"
            for cls in sorted(prediction["proba"])
        )
        _text(cv2, frame, proba_txt, 12, h - 12, 0.55, WHITE)
