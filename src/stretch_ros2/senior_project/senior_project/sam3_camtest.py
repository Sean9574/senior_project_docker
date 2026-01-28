#!/usr/bin/env python3
"""SAM3 Webcam - Adjustable resolution (smooth mask rendering) - Color Fixed"""

import base64
import threading
import time

import cv2
import numpy as np
import requests

SERVER = "http://localhost:8100"
FLIP_CAMERA = True

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


class SAM3Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Force proper color and format settings
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Debug info
        print(f"Backend: {self.cap.getBackendName()}")
        print(f"Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(f"Format: {fourcc.to_bytes(4, 'little').decode()}")
        
        self.current_frame = None
        self.display_frame = None  # Full-res frame for display
        self.result = {"prompt": "loading...", "num_objects": 0, "boxes": [], "scores": [], "masks_base64": []}
        self.running = True
        self.lock = threading.Lock()
        self.inference_fps = 0

        # Adjustable settings
        self.inference_width = 640  # Smaller for faster inference
        self.confidence = 0.2

        # Smooth mask rendering settings
        self.smooth_masks = True
        self.smooth_sigma = 1.2
        self.alpha_strength = 0.60
        self.contour_threshold = 0.50
        self.morph_kernel = 3

    def decode_mask(self, mask_b64):
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            return mask
        except Exception:
            return None

    def overlay_masks(self, frame, masks_b64, boxes, scores, prompt, scale_x=1.0, scale_y=1.0):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for i, (mask_b64, box, score) in enumerate(zip(masks_b64, boxes, scores)):
            mask = self.decode_mask(mask_b64)
            if mask is None:
                continue

            # Resize mask to frame size
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Convert to alpha in [0,1]
            alpha = (mask.astype(np.float32) / 255.0)

            # Optional blur to smooth edges
            if self.smooth_masks and self.smooth_sigma > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), self.smooth_sigma)

            # Soft overlay
            color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
            a = np.clip(alpha * float(self.alpha_strength), 0.0, 1.0)[..., None]
            overlay = (overlay.astype(np.float32) * (1.0 - a) + color[None, None, :] * a).astype(np.uint8)

            # Draw contour
            contour_bin = (alpha >= float(self.contour_threshold)).astype(np.uint8) * 255

            if self.morph_kernel and self.morph_kernel > 0:
                k = int(self.morph_kernel)
                k = k if (k % 2 == 1) else (k + 1)
                kernel = np.ones((k, k), np.uint8)
                contour_bin = cv2.morphologyEx(contour_bin, cv2.MORPH_OPEN, kernel, iterations=1)
                contour_bin = cv2.morphologyEx(contour_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(contour_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, COLORS[i % len(COLORS)], 2)

            # Scale box coordinates if display is different size than inference
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            label = f"{prompt}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS[i % len(COLORS)], -1)
            cv2.putText(
                overlay, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return overlay

    def capture_thread(self):
        global FLIP_CAMERA
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if FLIP_CAMERA:
                    frame = cv2.flip(frame, 1)
                
                # Keep full resolution for display
                h, w = frame.shape[:2]
                
                # Create smaller version for inference
                inference_frame = cv2.resize(
                    frame, 
                    (self.inference_width, int(h * self.inference_width / w))
                )
                
                with self.lock:
                    self.display_frame = frame.copy()
                    self.current_frame = inference_frame.copy()

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
                    timeout=10,
                )

                self.inference_fps = 1.0 / max(1e-6, (time.time() - start))

                with self.lock:
                    self.result = r.json()

            except Exception as e:
                print(f"Inference error: {e}")

    def run(self):
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.inference_thread, daemon=True).start()

        print("Controls:")
        print("  q - quit")
        print("  f - flip camera")
        print("  +/- - adjust inference resolution")
        print("  [/] - adjust confidence")

        global FLIP_CAMERA

        while self.running:
            with self.lock:
                if self.display_frame is None:
                    continue
                display = self.display_frame.copy()
                inference_frame = self.current_frame
                result = self.result.copy()
                fps = self.inference_fps

            prompt = result.get("prompt", "?")
            num = result.get("num_objects", 0)
            masks_b64 = result.get("masks_base64", [])
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])

            # Calculate scale factors between inference and display
            if inference_frame is not None:
                scale_x = display.shape[1] / inference_frame.shape[1]
                scale_y = display.shape[0] / inference_frame.shape[0]
            else:
                scale_x, scale_y = 1.0, 1.0

            if masks_b64 and boxes:
                display = self.overlay_masks(display, masks_b64, boxes, scores, prompt, scale_x, scale_y)

            cv2.putText(
                display,
                f"'{prompt}' | Found: {num} | {fps:.1f}Hz | InfRes:{self.inference_width} | Conf:{self.confidence:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "q:quit f:flip +/-:res [/]:conf",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            cv2.imshow("SAM3 Segmentation", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
                break
            elif key == ord("f"):
                FLIP_CAMERA = not FLIP_CAMERA
            elif key == ord("+") or key == ord("="):
                self.inference_width = min(1280, self.inference_width + 80)
                print(f"Inference resolution: {self.inference_width}")
            elif key == ord("-"):
                self.inference_width = max(320, self.inference_width - 80)
                print(f"Inference resolution: {self.inference_width}")
            elif key == ord("]"):
                self.confidence = min(0.9, self.confidence + 0.05)
                print(f"Confidence: {self.confidence:.2f}")
            elif key == ord("["):
                self.confidence = max(0.05, self.confidence - 0.05)
                print(f"Confidence: {self.confidence:.2f}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SAM3Webcam().run()