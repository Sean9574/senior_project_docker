#!/usr/bin/env python3
"""
Auto-guided camera calibration (checkerboard) -> camera_calib.npz

What it does:
- Detects checkerboard continuously
- Auto-captures frames when the board pose/position is "new enough"
- Tracks coverage across the image (3x3 grid)
- Tracks tilt variety (based on apparent perspective)
- Displays guidance: where to move the board next + progress

Controls:
    q   quit + calibrate (if enough samples)
    r   reset captured data
    s   save snapshot (forces capture if board detected)
"""

import time
import cv2
import numpy as np

# ---------------- USER SETTINGS ----------------
CHECKERBOARD = (9, 6)   # inner corners (cols, rows)
SQUARE_SIZE = 0.015     # meters (15mm squares -> 0.015)
DEVICE = 0
TARGET_CAPTURES = 30
MIN_CAPTURES = 20

# Auto-capture behavior
COOLDOWN_SEC = 0.35          # min time between accepted captures
MIN_MOVE_NORM = 0.020        # normalized move threshold to count as "new"
MIN_SCALE_CHANGE = 0.050     # relative size change threshold (distance change)
MIN_TILT_CHANGE = 0.10       # tilt metric change threshold

# Coverage grid (3x3)
GRID_N = 3

# Camera resolution (match your pipeline)
FRAME_W, FRAME_H = 1280, 720
# ----------------------------------------------

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def build_objp(checkerboard, square_size):
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def corners_metrics(corners, img_w, img_h):
    """
    Returns:
      - center normalized (cx, cy) in [0,1]
      - scale: normalized board size proxy
      - tilt: perspective proxy (0=fronto-parallel, higher=more tilted)
      - bbox normalized (w, h)
    """
    pts = corners.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    cx = (x_min + x_max) * 0.5 / img_w
    cy = (y_min + y_max) * 0.5 / img_h

    # scale proxy: fraction of image occupied
    scale = np.sqrt((bw * bh) / (img_w * img_h))

    # tilt proxy: difference between opposite edges lengths
    # Take 4 extreme corners by bbox approx:
    # We approximate using nearest to corners of bbox.
    # For a checkerboard, corners[0] is one corner and corners[-1] opposite,
    # but ordering depends; we use bbox corners approximation:
    tl = pts[np.argmin((pts[:,0]-x_min)**2 + (pts[:,1]-y_min)**2)]
    tr = pts[np.argmin((pts[:,0]-x_max)**2 + (pts[:,1]-y_min)**2)]
    bl = pts[np.argmin((pts[:,0]-x_min)**2 + (pts[:,1]-y_max)**2)]
    br = pts[np.argmin((pts[:,0]-x_max)**2 + (pts[:,1]-y_max)**2)]

    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    # perspective distortion proxy:
    # if fronto-parallel, top~bot and left~right
    tilt = abs(top - bot) / max(1.0, (top + bot)) + abs(left - right) / max(1.0, (left + right))

    return (cx, cy), scale, tilt, (bw / img_w, bh / img_h)

def grid_cell(cx, cy, n):
    gx = int(np.clip(cx * n, 0, n - 1))
    gy = int(np.clip(cy * n, 0, n - 1))
    return gx, gy

def draw_grid(vis, n):
    h, w = vis.shape[:2]
    for i in range(1, n):
        x = int(w * i / n)
        y = int(h * i / n)
        cv2.line(vis, (x, 0), (x, h), (50, 50, 50), 1)
        cv2.line(vis, (0, y), (w, y), (50, 50, 50), 1)

def guidance_text(coverage, tilt_bins_filled, captures, target):
    """
    Build a simple instruction string:
    - prefer filling uncovered grid cells
    - then prefer more tilt variety
    """
    # Find an uncovered cell (prefer corners first)
    n = coverage.shape[0]
    priorities = [(0,0),(n-1,0),(0,n-1),(n-1,n-1),(1,0),(0,1),(n-2,0),(0,n-2),(n-1,1),(1,n-1),(1,1),(n-2,1),(1,n-2),(n-2,n-2)]
    for gx, gy in priorities:
        if gx < n and gy < n and coverage[gy, gx] == 0:
            pos_name = {
                (0,0):"top-left", (n-1,0):"top-right", (0,n-1):"bottom-left", (n-1,n-1):"bottom-right",
                (1,0):"top-center", (0,1):"left-center", (n-2,0):"top-center", (0,n-2):"left-center",
                (n-1,1):"right-center",(1,n-1):"bottom-center",(1,1):"center",(n-2,1):"center-right",
                (1,n-2):"center-bottom",(n-2,n-2):"center"
            }.get((gx,gy), "that area")
            return f"Move board to {pos_name} (fill coverage)."

    if tilt_bins_filled < 3:
        return "Tilt board more (rotate it in 3D)."

    if captures < target:
        return "Good—keep moving/tilting slowly for more samples."

    return "Coverage looks good. Press q to calibrate & save."

def tilt_bin(tilt):
    # crude bins: low / medium / high tilt
    if tilt < 0.03:
        return 0
    if tilt < 0.08:
        return 1
    return 2

def should_accept(new_center, new_scale, new_tilt, last_center, last_scale, last_tilt, cov, cxcy):
    """
    Accept if:
    - new coverage cell not yet filled, OR
    - sufficiently moved / scale changed / tilt changed
    """
    (cx, cy) = cxcy
    gx, gy = grid_cell(cx, cy, cov.shape[0])
    new_cell = (cov[gy, gx] == 0)

    if last_center is None:
        return True, True  # accept first

    dc = np.linalg.norm(np.array(new_center) - np.array(last_center))
    ds = abs(new_scale - last_scale) / max(1e-6, last_scale)
    dt = abs(new_tilt - last_tilt)

    changed = (dc > MIN_MOVE_NORM) or (ds > MIN_SCALE_CHANGE) or (dt > MIN_TILT_CHANGE)
    return (new_cell or changed), new_cell

def main():
    objp = build_objp(CHECKERBOARD, SQUARE_SIZE)
    objpoints = []
    imgpoints = []

    coverage = np.zeros((GRID_N, GRID_N), dtype=np.uint8)
    tilt_bins = np.zeros(3, dtype=np.uint8)

    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    last_accept_time = 0.0
    last_center = None
    last_scale = None
    last_tilt = None
    last_gray_shape = None

    print("\nAuto-guided calibration running...")
    print("Hold checkerboard in view; it will auto-capture good frames.")
    print("Keys: q=finish+calibrate  r=reset  s=force-save-capture\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        vis = frame.copy()
        draw_grid(vis, GRID_N)

        status = "No board detected"
        hint = ""

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)

            (cx, cy), scale, tilt, (bw_n, bh_n) = corners_metrics(corners2, vis.shape[1], vis.shape[0])
            tb = tilt_bin(tilt)

            hint = guidance_text(coverage, int(tilt_bins.sum()), len(objpoints), TARGET_CAPTURES)

            now = time.time()
            can_try = (now - last_accept_time) > COOLDOWN_SEC

            accept, filled_new_cell = (False, False)
            if can_try:
                accept, filled_new_cell = should_accept(
                    (cx, cy), scale, tilt,
                    last_center, last_scale, last_tilt,
                    coverage, (cx, cy)
                )

            if accept and len(objpoints) < TARGET_CAPTURES:
                gx, gy = grid_cell(cx, cy, GRID_N)
                coverage[gy, gx] = 1
                tilt_bins[tb] = 1

                objpoints.append(objp.copy())
                imgpoints.append(corners2)

                last_accept_time = now
                last_center, last_scale, last_tilt = (cx, cy), scale, tilt

                status = f"AUTO CAPTURED #{len(objpoints)}  cell=({gx},{gy})  tiltbin={tb}"
            else:
                gx, gy = grid_cell(cx, cy, GRID_N)
                status = f"Detected  cell=({gx},{gy})  scale={scale:.3f}  tilt={tilt:.3f}"

            # Draw center marker
            cv2.circle(vis, (int(cx * vis.shape[1]), int(cy * vis.shape[0])), 6, (0, 255, 255), -1)

        # HUD
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 85), (0, 0, 0), -1)
        cv2.putText(vis, f"{status}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cov_count = int(coverage.sum())
        tilt_count = int(tilt_bins.sum())
        cv2.putText(vis, f"captures: {len(objpoints)}/{TARGET_CAPTURES}  coverage:{cov_count}/{GRID_N*GRID_N}  tilt_bins:{tilt_count}/3",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if hint:
            cv2.putText(vis, f"hint: {hint}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)

        cv2.imshow("Auto Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            objpoints.clear()
            imgpoints.clear()
            coverage[:] = 0
            tilt_bins[:] = 0
            last_center = last_scale = last_tilt = None
            last_accept_time = 0.0
            print("Reset captures.")
        elif key == ord("s"):
            if found:
                # force capture even if not "new"
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                (cx, cy), scale, tilt, _ = corners_metrics(corners2, vis.shape[1], vis.shape[0])
                gx, gy = grid_cell(cx, cy, GRID_N)
                coverage[gy, gx] = 1
                tilt_bins[tilt_bin(tilt)] = 1
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                last_center, last_scale, last_tilt = (cx, cy), scale, tilt
                last_accept_time = time.time()
                print(f"Forced capture #{len(objpoints)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < MIN_CAPTURES:
        print(f"\nNot enough captures ({len(objpoints)}). Need at least {MIN_CAPTURES}.")
        print("Run again and let it auto-capture more variety (edges + tilts).")
        return

    print("\nCalibrating...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, last_gray_shape, None, None
    )

    # reprojection error
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print("\nCamera matrix:\n", camera_matrix)
    print("\nDist coeffs:\n", dist_coeffs)
    print(f"\nMean reprojection error: {mean_error:.4f} px")

    np.savez("camera_calib.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("\nSaved: camera_calib.npz")

if __name__ == "__main__":
    main()
