#!/usr/bin/env python3
import cv2
import numpy as np
import time

# -------------------- Tunables --------------------
DISPLAY_W, DISPLAY_H = 960, 540     # display/processing size (downscale improves robustness + speed)
PATCH_RADIUS = 50                  # click -> template patch radius (patch size = 2R+1)
KLT_MAX_CORNERS = 200
KLT_QUALITY = 0.01
KLT_MIN_DIST = 5

MIN_AFFINE_INLIERS = 18            # inliers needed to stay in TRACKING
MIN_H_INLIERS = 20                 # inliers needed to reacquire
REACQUIRE_EVERY_N_FRAMES = 1        # try reacquire every N frames when LOST (1 = every frame)

# Template update (only when very confident)
ALLOW_TEMPLATE_UPDATE = True
TEMPLATE_UPDATE_COOLDOWN_SEC = 1.5
TEMPLATE_UPDATE_MIN_H_INLIERS = 35  # higher threshold for "safe" update

# Reacquire uses a search pyramid (handles scale change)
REACQ_SCALES = [1.0, 0.85, 0.7]     # resize frame for detection; add more if needed

# -------------------- Feature choice --------------------
def make_detector():
    # Prefer SIFT (best robustness to scale/rotation), else AKAZE, else ORB.
    if hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=1200)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        det_name = "SIFT"
    elif hasattr(cv2, "AKAZE_create"):
        det = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        det_name = "AKAZE"
    else:
        det = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        det_name = "ORB"
    return det, matcher, det_name

detector, matcher, DET_NAME = make_detector()

# -------------------- Helpers --------------------
def resize_to(frame, w, h):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

def clamp_roi(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return x0, y0, x1, y1

def extract_patch(gray, center, radius):
    h, w = gray.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0, x1, y1 = clamp_roi(cx - radius, cy - radius, cx + radius + 1, cy + radius + 1, w, h)
    patch = gray[y0:y1, x0:x1]
    return patch, (x0, y0, x1, y1)

def good_corners(gray, center):
    patch, (x0, y0, x1, y1) = extract_patch(gray, center, PATCH_RADIUS)
    pts = cv2.goodFeaturesToTrack(
        patch, maxCorners=KLT_MAX_CORNERS, qualityLevel=KLT_QUALITY,
        minDistance=KLT_MIN_DIST, blockSize=7, useHarrisDetector=False
    )
    if pts is None:
        return None
    pts[:, 0, 0] += x0
    pts[:, 0, 1] += y0
    return pts.astype(np.float32)

def median_center(pts):
    pts2 = pts.reshape(-1, 2)
    return float(np.median(pts2[:, 0])), float(np.median(pts2[:, 1]))

def knn_ratio_test(knn, ratio=0.75):
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def perspective_point(H, pt):
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])

# -------------------- Tracking state --------------------
state = "IDLE"  # IDLE, TRACKING, LOST
clicked = False
click_center = (0.0, 0.0)

template_gray = None
template_kp = None
template_des = None
template_center = None   # in template coords
template_last_update = 0.0

prev_gray = None
klt_pts = None
center = None

lost_frame_count = 0
frame_count = 0

# -------------------- Mouse --------------------
def on_mouse(event, x, y, flags, param):
    global clicked, click_center
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        click_center = (float(x), float(y))

# -------------------- Main --------------------
def init_target(gray, center_xy):
    global template_gray, template_kp, template_des, template_center
    global prev_gray, klt_pts, center, state, lost_frame_count, template_last_update

    # Build template patch around click
    patch, (x0, y0, x1, y1) = extract_patch(gray, center_xy, PATCH_RADIUS)
    if patch.size < 20 * 20:
        return False

    template_gray = patch.copy()
    template_center = (template_gray.shape[1] / 2.0, template_gray.shape[0] / 2.0)

    # Features on template
    template_kp, template_des = detector.detectAndCompute(template_gray, None)
    if template_des is None or len(template_kp) < 12:
        # Still can try KLT-only, but reacquire will be weak
        template_kp, template_des = None, None

    # Init KLT points around center
    klt_pts = good_corners(gray, center_xy)
    if klt_pts is None or len(klt_pts) < 8:
        # No corners => hard to track
        return False

    center = center_xy
    prev_gray = gray
    state = "TRACKING"
    lost_frame_count = 0
    template_last_update = time.time()
    return True

def try_template_update(gray, H, h_inliers):
    global template_gray, template_kp, template_des, template_center, template_last_update

    if not ALLOW_TEMPLATE_UPDATE:
        return
    now = time.time()
    if now - template_last_update < TEMPLATE_UPDATE_COOLDOWN_SEC:
        return
    if h_inliers < TEMPLATE_UPDATE_MIN_H_INLIERS:
        return

    # Update template around current center in the frame (refresh appearance/scale)
    # We assume center is already updated outside.
    # We'll rebuild template patch, features.
    patch, _ = extract_patch(gray, center, PATCH_RADIUS)
    if patch.size < 20 * 20:
        return
    kp, des = detector.detectAndCompute(patch, None)
    if des is None or len(kp) < 12:
        return

    template_gray = patch
    template_center = (template_gray.shape[1] / 2.0, template_gray.shape[0] / 2.0)
    template_kp, template_des = kp, des
    template_last_update = now

def reacquire(gray):
    """
    Global reacquire:
    - detect features in frame (possibly at multiple scales)
    - match to template descriptors
    - homography RANSAC -> get center
    """
    global center, state, klt_pts, prev_gray, lost_frame_count

    if template_des is None or template_kp is None:
        return False

    best = None  # (h_inliers, H, scale, kp_frame)
    for scale in REACQ_SCALES:
        if scale != 1.0:
            small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        kp2, des2 = detector.detectAndCompute(small, None)
        if des2 is None or len(kp2) < 30:
            continue

        knn = matcher.knnMatch(template_des, des2, k=2)
        good = knn_ratio_test(knn, ratio=0.75 if DET_NAME != "ORB" else 0.8)
        if len(good) < 16:
            continue

        src = np.float32([template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            continue
        inliers = int(mask.sum())
        if inliers >= MIN_H_INLIERS:
            if best is None or inliers > best[0]:
                best = (inliers, H, scale, kp2)

    if best is None:
        return False

    h_inliers, H, scale, _ = best
    # Template center -> frame coords (in the scaled image coords)
    c_small = perspective_point(H, template_center)
    # Convert back to full coords if scaled
    c_full = (c_small[0] / scale, c_small[1] / scale)

    center = c_full
    # Re-init KLT near new center
    klt_pts_new = good_corners(gray, center)
    if klt_pts_new is None or len(klt_pts_new) < 8:
        # Still accept reacquire, but KLT needs another attempt next frame
        klt_pts = None
    else:
        klt_pts = klt_pts_new
    prev_gray = gray
    state = "TRACKING"
    lost_frame_count = 0

    # Optional: refresh template when super-confident
    try_template_update(gray, H, h_inliers)
    return True

def track(gray):
    """KLT + affine RANSAC filtering. Updates center and klt_pts."""
    global prev_gray, klt_pts, center, state, lost_frame_count

    if prev_gray is None or klt_pts is None or len(klt_pts) < 8:
        state = "LOST"
        lost_frame_count += 1
        prev_gray = gray
        return

    lk_params = dict(
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    )
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, klt_pts, None, **lk_params)
    if p1 is None or st is None:
        state = "LOST"
        lost_frame_count += 1
        prev_gray = gray
        return

    good_new = p1[st == 1].reshape(-1, 2)
    good_old = klt_pts[st == 1].reshape(-1, 2)

    if len(good_new) < MIN_AFFINE_INLIERS:
        # Not enough points -> lost
        if len(good_new) >= 1:
            center = (float(np.median(good_new[:, 0])), float(np.median(good_new[:, 1])))
        state = "LOST"
        lost_frame_count += 1
        prev_gray = gray
        return

    # Affine RANSAC to filter outliers (background points)
    M, inliers = cv2.estimateAffinePartial2D(
        good_old, good_new,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )

    if M is None or inliers is None:
        state = "LOST"
        lost_frame_count += 1
        prev_gray = gray
        return

    inl = inliers.ravel().astype(bool)
    inlier_count = int(inl.sum())
    if inlier_count < MIN_AFFINE_INLIERS:
        state = "LOST"
        lost_frame_count += 1
        prev_gray = gray
        return

    inlier_new = good_new[inl]
    # Update center robustly:
    # 1) transform old center by affine (good for rotation/scale)
    cx, cy = center
    new_center = (M[0, 0] * cx + M[0, 1] * cy + M[0, 2],
                  M[1, 0] * cx + M[1, 1] * cy + M[1, 2])
    # 2) blend with median of inlier points (stabilizes drift)
    mx, my = float(np.median(inlier_new[:, 0])), float(np.median(inlier_new[:, 1]))
    center = (0.7 * new_center[0] + 0.3 * mx, 0.7 * new_center[1] + 0.3 * my)

    # Keep only inlier points for next iteration
    klt_pts = inlier_new.reshape(-1, 1, 2).astype(np.float32)
    prev_gray = gray
    lost_frame_count = 0

def main():
    global clicked, state, prev_gray, klt_pts, center, frame_count

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    win = f"Robust Click-to-Track ({DET_NAME}) | click target | r reset | q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    fps_t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_to(frame, DISPLAY_W, DISPLAY_H)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Click init
        if clicked:
            clicked = False
            # Reset and init new target
            state_reset()
            ok_init = init_target(gray, click_center)
            if not ok_init:
                cv2.putText(frame, "Init failed: click on textured part of object",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # State update
        if state == "TRACKING":
            track(gray)
        elif state == "LOST":
            if (frame_count % REACQUIRE_EVERY_N_FRAMES) == 0:
                reacquire(gray)

        # Draw UI (no box)
        if state in ("TRACKING", "LOST") and center is not None:
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            if klt_pts is not None:
                for pt in klt_pts.reshape(-1, 2):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # FPS
        frames += 1
        dt = time.time() - fps_t0
        if dt >= 0.5:
            fps = frames / dt
            frames = 0
            fps_t0 = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('r'):
            state_reset()

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def state_reset():
    global state, template_gray, template_kp, template_des, template_center
    global prev_gray, klt_pts, center, lost_frame_count
    state = "IDLE"
    template_gray = None
    template_kp = None
    template_des = None
    template_center = None
    prev_gray = None
    klt_pts = None
    center = None
    lost_frame_count = 0

if __name__ == "__main__":
    main()
