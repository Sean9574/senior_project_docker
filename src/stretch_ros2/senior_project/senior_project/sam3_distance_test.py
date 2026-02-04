#!/usr/bin/env python3
"""
SAM3 Webcam with Distance Estimation (Undistorted + Calibrated)

Key change:
- Uses real camera intrinsics (fx) and undistorts frames before measuring width.
- This fixes the "distance undershoots near edges" issue caused by lens distortion.

Formula: distance = (real_width * fx) / pixel_width

Controls:
    q       - quit
    f       - flip camera
    +/-     - adjust inference resolution
    [/]     - adjust confidence
    UP/DOWN - adjust fx scale (fine calibration tweak)
    1-9     - quick select object

Calibration file (optional but recommended):
    camera_calib.npz containing:
        - camera_matrix
        - dist_coeffs
"""

import base64
import threading
import time
import os

import cv2
import numpy as np
import requests

SERVER = "http://localhost:8100"
FLIP_CAMERA = True

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

# Object sizes in meters (width, height)
OBJECT_SIZES = {
    "person": (0.5, 1.7),
    "cup": (0.08, 0.12),
    "bottle": (0.08, 0.25),
    "phone": (0.075, 0.15),
    "book": (0.2, 0.25),
    "laptop": (0.35, 0.25),
    "chair": (0.5, 0.9),
    "keyboard": (0.45, 0.15),
    "mouse": (0.06, 0.1),
    "remote": (0.05, 0.2),
    "banana": (0.2, 0.04),
    "apple": (0.08, 0.08),
    "bowl": (0.15, 0.08),
    "clock": (0.3, 0.3),
    "vase": (0.15, 0.3),
    "teddy bear": (0.3, 0.4),
    "backpack": (0.35, 0.5),
}

QUICK_SELECT = {
    ord('1'): "cup",
    ord('2'): "bottle",
    ord('3'): "phone",
    ord('4'): "book",
    ord('5'): "laptop",
    ord('6'): "person",
    ord('7'): "chair",
    ord('8'): "banana",
    ord('9'): "bowl",
}


class SAM3DistanceTest:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {self.frame_width}x{self.frame_height}")

        self.current_frame = None
        self.display_frame = None
        self.result = {"prompt": "loading...", "num_objects": 0, "boxes": [], "scores": [], "masks_base64": []}
        self.running = True
        self.lock = threading.Lock()
        self.inference_fps = 0

        self.inference_width = 640
        self.confidence = 0.2

        # --- Calibration / Undistortion ---
        self.calib_path = "camera_calib.npz"
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.undistort_map1 = None
        self.undistort_map2 = None

        self._load_calibration_if_available()

        # Focal length:
        # - If calibrated: use fx from camera matrix
        # - Else: fallback guess (works only near center)
        if self.camera_matrix is not None:
            self.fx = float(self.camera_matrix[0, 0])
            print(f"Using calibrated fx={self.fx:.2f}px")
        else:
            self.fx = self.frame_width * 0.9
            print(f"No calibration file found. Using fallback fx={self.fx:.2f}px (center-only)")

        # Optional fine-tune multiplier (lets you “calibrate” against a tape measure even after calibration)
        self.fx_scale = 1.0

        self.current_object = "cup"

    def _load_calibration_if_available(self):
        if not os.path.exists(self.calib_path):
            return
        try:
            data = np.load(self.calib_path)
            self.camera_matrix = data["camera_matrix"].astype(np.float32)
            self.dist_coeffs = data["dist_coeffs"].astype(np.float32)

            # Precompute undistortion maps for speed
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs,
                (self.frame_width, self.frame_height), alpha=0
            )
            self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix,
                (self.frame_width, self.frame_height), cv2.CV_16SC2
            )

            print(f"Loaded calibration from {self.calib_path}")
        except Exception as e:
            print(f"Failed to load calibration file '{self.calib_path}': {e}")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.new_camera_matrix = None
            self.undistort_map1 = None
            self.undistort_map2 = None

    def undistort_frame(self, frame):
        if self.undistort_map1 is None or self.undistort_map2 is None:
            return frame
        return cv2.remap(frame, self.undistort_map1, self.undistort_map2, interpolation=cv2.INTER_LINEAR)

    def estimate_distance_from_size(self, pixel_width, pixel_height, object_name):
        """
        Distance using WIDTH only.
        NOTE: Width-only removes pitch sensitivity, but NOT lens distortion.
              That's why we undistort first.
        """
        import math

        if pixel_width <= 1 or pixel_height <= 1:
            return None

        obj_key = object_name.lower()
        if obj_key in OBJECT_SIZES:
            real_width, real_height = OBJECT_SIZES[obj_key]
        else:
            for key, size in OBJECT_SIZES.items():
                if key in obj_key or obj_key in key:
                    real_width, real_height = size
                    break
            else:
                real_width, real_height = 0.2, 0.2

        fx_eff = self.fx * self.fx_scale
        distance = (real_width * fx_eff) / float(pixel_width)

        # Pitch inference (kept from your original idea)
        expected_ratio = real_width / real_height
        measured_ratio = float(pixel_width) / float(pixel_height)

        ratio = expected_ratio / measured_ratio
        ratio = max(0.0, min(1.0, ratio))

        if ratio < 1.0:
            pitch_rad = math.acos(ratio)
            pitch_deg = math.degrees(pitch_rad)
        else:
            pitch_deg = 0.0

        return distance, pitch_deg, real_width

    def decode_mask(self, mask_b64):
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            return cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        except:
            return None

    def get_mask_dimensions(self, mask):
        """
        Get width/height from mask pixels.
        Adds +1 so a single-pixel-wide object doesn't return 0.
        """
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None, None, None, None, None

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        width = (x2 - x1) + 1
        height = (y2 - y1) + 1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        area = float(len(xs))  # mask area in pixels (useful for debugging)

        return width, height, cx, cy, area

    def overlay_results(self, frame, masks_b64, scores, prompt):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for i, (mask_b64, score) in enumerate(zip(masks_b64, scores)):
            mask = self.decode_mask(mask_b64)
            if mask is None:
                continue

            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            mask_w, mask_h, cx, cy, area = self.get_mask_dimensions(mask)
            if mask_w is None:
                continue

            # Overlay mask
            alpha = (mask.astype(np.float32) / 255.0) * 0.5
            color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
            overlay = (overlay.astype(np.float32) * (1.0 - alpha[..., None]) +
                       color[None, None, :] * alpha[..., None]).astype(np.uint8)

            dist_result = self.estimate_distance_from_size(mask_w, mask_h, prompt)
            if dist_result:
                dist, pitch_deg, real_w = dist_result

                # Contour
                contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, COLORS[i % len(COLORS)], 2)

                # Big distance at center
                dist_text = f"{dist:.2f}m"
                cv2.putText(overlay, dist_text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                cv2.putText(overlay, dist_text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                ys, xs = np.where(mask > 127)
                y1 = int(ys.min())
                x1 = int(xs.min())

                label = f"{prompt}: {score:.2f} | pitch: {pitch_deg:.1f}°"
                cv2.putText(overlay, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i % len(COLORS)], 2)

                cv2.putText(overlay, f"mask: {mask_w}x{mask_h}px | area:{area:.0f}px | ref_w:{real_w*100:.0f}cm",
                            (x1, min(h - 5, int(ys.max()) + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return overlay

    def capture_thread(self):
        global FLIP_CAMERA
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            # --- CRITICAL FIX: undistort BEFORE inference and BEFORE measuring ---
            frame = self.undistort_frame(frame)

            h, w = frame.shape[:2]
            inference_frame = cv2.resize(frame, (self.inference_width, int(h * self.inference_width / w)))

            with self.lock:
                self.display_frame = frame
                self.current_frame = inference_frame

    def inference_thread(self):
        while self.running:
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
                conf = self.confidence

            try:
                start = time.time()
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_b64 = base64.b64encode(buf).decode()

                r = requests.post(
                    f"{SERVER}/segment",
                    json={"image_base64": img_b64, "confidence_threshold": conf},
                    timeout=10
                )
                self.inference_fps = 1.0 / max(1e-6, (time.time() - start))
                with self.lock:
                    self.result = r.json()
            except Exception as e:
                print(f"Inference error: {e}")

    def set_prompt(self, prompt):
        try:
            requests.post(f"{SERVER}/prompt/{prompt}", timeout=5)
            print(f"Prompt: {prompt}")
        except Exception as e:
            print(f"Failed to set prompt: {e}")

    def run(self):
        global FLIP_CAMERA
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.inference_thread, daemon=True).start()
        self.set_prompt(self.current_object)

        print("\n" + "=" * 60)
        print("SAM3 Distance Test (UNDISTORTED + CALIBRATED FX)")
        print("=" * 60)
        print("Fixes edge undershoot by undistorting frames.")
        print("UP/DOWN adjusts fx_scale (fine-tune against tape measure).")
        print("1-9 select object | q quit | f flip | +/- resolution | [/] confidence")
        print("=" * 60 + "\n")

        while self.running:
            with self.lock:
                if self.display_frame is None:
                    continue
                display = self.display_frame.copy()
                result = self.result.copy()
                fps = self.inference_fps

            prompt = result.get("prompt", "?")
            masks_b64 = result.get("masks_base64", [])
            scores = result.get("scores", [])

            if masks_b64:
                display = self.overlay_results(display, masks_b64, scores, prompt)

            # Info bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
            fx_eff = self.fx * self.fx_scale
            calib_status = "CALIB" if self.camera_matrix is not None else "NO-CALIB"
            cv2.putText(
                display,
                f"'{prompt}' | fx:{fx_eff:.0f}px (scale:{self.fx_scale:.3f}) | {fps:.1f}Hz | {calib_status} | UNDISTORT",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            if prompt.lower() in OBJECT_SIZES:
                ow, oh = OBJECT_SIZES[prompt.lower()]
                cv2.putText(display, f"Size: {ow*100:.0f}x{oh*100:.0f}cm", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

            cv2.putText(display, "UP/DOWN:fx_scale  1-9:object  q:quit  f:flip",
                        (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("SAM3 Distance Test (Undistorted)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
            elif key == ord("f"):
                FLIP_CAMERA = not FLIP_CAMERA
            elif key == ord("+") or key == ord("="):
                self.inference_width = min(1280, self.inference_width + 80)
            elif key == ord("-"):
                self.inference_width = max(320, self.inference_width - 80)
            elif key == ord("]"):
                self.confidence = min(0.9, self.confidence + 0.05)
            elif key == ord("["):
                self.confidence = max(0.05, self.confidence - 0.05)
            elif key == 82:  # UP
                self.fx_scale *= 1.01
                print(f"fx_scale: {self.fx_scale:.4f}")
            elif key == 84:  # DOWN
                self.fx_scale /= 1.01
                print(f"fx_scale: {self.fx_scale:.4f}")
            elif key in QUICK_SELECT:
                self.current_object = QUICK_SELECT[key]
                self.set_prompt(self.current_object)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SAM3DistanceTest().run()
