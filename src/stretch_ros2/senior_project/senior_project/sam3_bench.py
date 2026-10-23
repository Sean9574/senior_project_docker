#!/usr/bin/env python3
"""
SAM3 server throughput with N robots sharing it (no ROS).

Starts N client threads that call /segment back to back with a camera-sized
image, like N goal generators would, and reports what each robot would get.

  python3 -m senior_project.sam3_server --port 8100 &      # if not running
  python3 -m senior_project.sam3_bench --robots 8 --seconds 30
"""

import argparse
import base64
import threading
import time

import cv2
import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--prompt", default="chair")
    # The goal generator sends the 1280x720 head-camera frame turned upright
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    args = parser.parse_args()

    requests.post(f"{args.url}/prompt/{args.prompt}", timeout=10).raise_for_status()
    # A different frame every request, as from moving robots: the server skips
    # the encoder for a repeated frame, which made one frame sent over and
    # over look ~5x faster than SAM3 is. Blurred noise compresses about like a
    # rendered frame (raw noise JPEGs are ~20x bigger).
    payloads = []
    for seed in range(2 * args.robots):
        noise = (np.random.default_rng(seed).random((args.height, args.width, 3)) * 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", cv2.GaussianBlur(noise, (0, 0), 8), [cv2.IMWRITE_JPEG_QUALITY, 60])
        payloads.append({"image_base64": base64.b64encode(buf).decode(), "confidence_threshold": 0.2})

    latencies, errors, sent, lock = [], [0], [0], threading.Lock()
    stop_at = time.time() + args.seconds

    def client():
        session = requests.Session()
        while time.time() < stop_at:
            with lock:
                payload = payloads[sent[0] % len(payloads)]
                sent[0] += 1
            t0 = time.perf_counter()
            try:
                session.post(f"{args.url}/segment", json=payload, timeout=10).raise_for_status()
                with lock:
                    latencies.append(time.perf_counter() - t0)
            except Exception:
                with lock:
                    errors[0] += 1

    threads = [threading.Thread(target=client) for _ in range(args.robots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = len(latencies)
    total_rate = done / args.seconds
    lat = np.array(latencies) if latencies else np.array([float("nan")])
    print(f"sam3       {total_rate:.1f} req/s total with {args.robots} clients, errors {errors[0]}")
    print(f"latency    p50 {np.percentile(lat, 50) * 1000:.0f} ms   p90 {np.percentile(lat, 90) * 1000:.0f} ms")
    per_robot = total_rate / args.robots
    planned = 2.0 * min(1.0, 4.0 / args.robots)
    print(f"verdict    each robot can get {per_robot:.2f} Hz; grid_navigation plans {planned:.2f} Hz "
          f"-> {'OK' if per_robot >= planned else 'TOO SLOW (lower num_robots or add batching)'}")


if __name__ == "__main__":
    main()
