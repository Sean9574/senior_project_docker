#!/usr/bin/env python3
"""\
SAM3 ROS2 Client Node
Run this in your ROS2 environment (Python 3.10)

This node:
  - Subscribes to camera images
  - Sends them to SAM3 server for segmentation
  - Publishes segmentation masks
  - Optionally publishes detections (vision_msgs) and viz images

Video-speed optimizations (still true segmentation):
  - Requests "combined" mask mode (one merged mask per frame)
  - Requests a downscaled mask from the server (e.g., 256px long-side), then upsamples
    to camera size before publishing
  - Optional resize of the input frame before sending to the server
  - Uses a persistent HTTP session (keep-alive)

Usage:
    # Make sure sam3_server.py is running first!
    ros2 run senior_project sam3_ros_node --ros-args -p prompt:="chair"

    # Or launch with parameters
    ros2 run senior_project sam3_ros_node --ros-args \
        -p prompt:="furniture" \
        -p camera_topic:="/camera/color/image_raw" \
        -p rate:=10.0
"""

import base64
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool

# For bounding box / detection publishing
try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    HAS_VISION_MSGS = True
except ImportError:
    HAS_VISION_MSGS = False
    print("[WARN] vision_msgs not installed, detection publishing disabled")


class SAM3ROSNode(Node):
    """ROS2 node that interfaces with SAM3 segmentation server"""

    def __init__(self):
        super().__init__("sam3_segmentation_node")

        # -------------------- Parameters --------------------
        self.declare_parameter("server_url", "http://localhost:8100")
        self.declare_parameter("prompt", "object")
        self.declare_parameter("camera_topic", "/camera/color/image_raw")

        # Inference rate limiting (Hz)
        self.declare_parameter("rate", 5.0)

        # Server-side filtering
        self.declare_parameter("confidence_threshold", 0.30)
        self.declare_parameter("max_objects", 50)
        self.declare_parameter("mask_threshold", 0.50)
        self.declare_parameter("min_mask_area_frac", 0.0)

        # Mask output mode ("combined" is fastest; still a segmentation mask)
        self.declare_parameter("mask_mode", "combined")
        self.declare_parameter("mask_size", 256)  # server downscale long-side

        # Client-side video bandwidth/latency knobs
        self.declare_parameter("resize_width", 640)  # 0 disables; otherwise downscale before sending
        self.declare_parameter("jpeg_quality", 70)

        # Misc
        self.declare_parameter("publish_visualization", False)
        self.declare_parameter("enabled", True)

        self.server_url = self.get_parameter("server_url").value
        self.prompt = self.get_parameter("prompt").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.rate_hz = float(self.get_parameter("rate").value)

        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_objects = int(self.get_parameter("max_objects").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.min_mask_area_frac = float(self.get_parameter("min_mask_area_frac").value)

        self.mask_mode = str(self.get_parameter("mask_mode").value)
        self.mask_size = int(self.get_parameter("mask_size").value)

        self.resize_width = int(self.get_parameter("resize_width").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.publish_viz = bool(self.get_parameter("publish_visualization").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        # Parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

        # -------------------- HTTP Session --------------------
        self.session = requests.Session()

        # -------------------- CV Bridge --------------------
        self.bridge = CvBridge()

        # Latest image storage
        self.latest_image: Optional[np.ndarray] = None
        self.latest_header = None
        self.image_lock = threading.Lock()

        # Rate limiting
        self.min_interval = 1.0 / max(self.rate_hz, 1e-6)
        self.last_inference_time = 0.0

        # -------------------- Publishers --------------------
        self.mask_pub = self.create_publisher(Image, "~/segmentation_mask", 10)
        self.viz_pub = self.create_publisher(Image, "~/visualization", 10)
        self.prompt_pub = self.create_publisher(String, "~/current_prompt", 10)
        self.status_pub = self.create_publisher(String, "~/status", 10)

        if HAS_VISION_MSGS:
            self.detection_pub = self.create_publisher(Detection2DArray, "~/detections", 10)

        # QoS profile to match camera topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -------------------- Subscribers --------------------
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile,
        )

        # Compressed images (common on real robots)
        self.compressed_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic + "/compressed",
            self.compressed_callback,
            qos_profile,
        )

        # Prompt subscriber (change prompt via topic)
        self.prompt_sub = self.create_subscription(
            String,
            "~/set_prompt",
            self.prompt_callback,
            10,
        )

        # Services
        self.enable_srv = self.create_service(SetBool, "~/enable", self.enable_callback)

        # Timer for inference (decoupled from image callback)
        self.inference_timer = self.create_timer(self.min_interval, self.inference_loop)

        # Check server connection
        self.check_server_connection()

        self.get_logger().info("SAM3 ROS Node started")
        self.get_logger().info(f"  Server: {self.server_url}")
        self.get_logger().info(f"  Prompt: '{self.prompt}'")
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Rate: {self.rate_hz} Hz")
        self.get_logger().info(f"  Mask mode: {self.mask_mode} | mask_size: {self.mask_size}")
        self.get_logger().info(f"  Send resize_width: {self.resize_width} | jpeg_quality: {self.jpeg_quality}")

    # -------------------- Helpers --------------------

    def check_server_connection(self):
        """Check if SAM3 server is reachable"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.get_logger().info("✓ SAM3 server connected")
                self.get_logger().info(f"  CUDA: {data.get('cuda_device', 'N/A')}")
                self.publish_status("connected")
            else:
                self.get_logger().warn(f"SAM3 server returned status {response.status_code}")
                self.publish_status("error")
        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f"Cannot connect to SAM3 server at {self.server_url}\n"
                "Make sure sam3_server.py is running in your sam3 conda environment!"
            )
            self.publish_status("disconnected")
        except Exception as e:
            self.get_logger().error(f"Server check failed: {e}")
            self.publish_status("error")

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _resize_for_send(self, rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Optionally resize RGB image before sending; returns (image_to_send, (orig_w, orig_h))."""
        h, w = rgb.shape[:2]
        orig = (w, h)

        if self.resize_width and self.resize_width > 0 and w > self.resize_width:
            new_w = int(self.resize_width)
            new_h = int(round(h * (new_w / float(w))))
            rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return rgb_small, orig

        return rgb, orig

    # -------------------- Parameter updates --------------------

    def parameter_callback(self, params):
        for param in params:
            if param.name == "prompt":
                self.prompt = str(param.value)
                self.get_logger().info(f"Prompt changed to: '{self.prompt}'")
            elif param.name == "rate":
                self.rate_hz = float(param.value)
                self.min_interval = 1.0 / max(self.rate_hz, 1e-6)
                try:
                    self.inference_timer.cancel()
                except Exception:
                    pass
                self.inference_timer = self.create_timer(self.min_interval, self.inference_loop)
            elif param.name == "confidence_threshold":
                self.confidence_threshold = float(param.value)
            elif param.name == "max_objects":
                self.max_objects = int(param.value)
            elif param.name == "mask_threshold":
                self.mask_threshold = float(param.value)
            elif param.name == "min_mask_area_frac":
                self.min_mask_area_frac = float(param.value)
            elif param.name == "mask_mode":
                self.mask_mode = str(param.value)
            elif param.name == "mask_size":
                self.mask_size = int(param.value)
            elif param.name == "resize_width":
                self.resize_width = int(param.value)
            elif param.name == "jpeg_quality":
                self.jpeg_quality = int(param.value)
            elif param.name == "enabled":
                self.enabled = bool(param.value)
            elif param.name == "publish_visualization":
                self.publish_viz = bool(param.value)

        return SetParametersResult(successful=True)

    # -------------------- Image callbacks --------------------

    def image_callback(self, msg: Image):
        with self.image_lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_header = msg.header

    def compressed_callback(self, msg: CompressedImage):
        with self.image_lock:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            self.latest_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.latest_header = msg.header

    def prompt_callback(self, msg: String):
        self.prompt = msg.data
        self.get_logger().info(f"Prompt updated to: '{self.prompt}'")

    def enable_callback(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f"Segmentation {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)
        return response

    # -------------------- Main inference loop --------------------

    def inference_loop(self):
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_inference_time < self.min_interval:
            return

        with self.image_lock:
            if self.latest_image is None:
                return
            rgb = self.latest_image.copy()
            header = self.latest_header

        self.last_inference_time = now

        # Publish current prompt
        prompt_msg = String()
        prompt_msg.data = self.prompt
        self.prompt_pub.publish(prompt_msg)

        # Resize before sending (optional)
        rgb_send, (orig_w, orig_h) = self._resize_for_send(rgb)
        send_h, send_w = rgb_send.shape[:2]

        # JPEG encode
        bgr_send = cv2.cvtColor(rgb_send, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(
            ".jpg",
            bgr_send,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            self.get_logger().warn("Failed to encode JPEG")
            return

        image_b64 = base64.b64encode(buffer).decode("utf-8")

        # Call SAM3 server
        try:
            response = self.session.post(
                f"{self.server_url}/segment",
                json={
                    "image_base64": image_b64,
                    "prompt": self.prompt,
                    "confidence_threshold": float(self.confidence_threshold),
                    "max_objects": int(self.max_objects),
                    "mask_mode": self.mask_mode,
                    "mask_threshold": float(self.mask_threshold),
                    "mask_size": int(self.mask_size),
                    "min_mask_area_frac": float(self.min_mask_area_frac),
                    "return_visualization": bool(self.publish_viz),
                },
                timeout=10,
            )

            if response.status_code != 200:
                self.get_logger().warn(f"Server error: {response.status_code}")
                self.publish_status("error")
                return

            result = response.json()

            if not result.get("success", False):
                self.get_logger().warn(f"Segmentation failed: {result.get('error')}")
                self.publish_status("error")
                return

            num_objects = int(result.get("num_objects", 0))
            inference_ms = float(result.get("inference_time_ms", 0.0))

            self.get_logger().debug(f"Found {num_objects} '{self.prompt}' objects in {inference_ms:.0f}ms")

            # Publish segmentation mask (upsampled to original camera resolution)
            masks_b64 = result.get("masks_base64", []) or []
            if len(masks_b64) > 0:
                combined_mask = None

                for mask_b64 in masks_b64:
                    try:
                        mask_bytes = base64.b64decode(mask_b64)
                        mask_np = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if mask_np is None:
                            continue

                        if combined_mask is None:
                            combined_mask = mask_np
                        else:
                            combined_mask = np.maximum(combined_mask, mask_np)
                    except Exception:
                        continue

                if combined_mask is not None:
                    # Upsample to original camera size for publishing
                    if combined_mask.shape[0] != orig_h or combined_mask.shape[1] != orig_w:
                        combined_mask = cv2.resize(combined_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, "mono8")
                    if header is not None:
                        mask_msg.header = header
                    self.mask_pub.publish(mask_msg)

            # Publish detections (rescale boxes back to original frame if we resized before sending)
            if HAS_VISION_MSGS and num_objects > 0:
                self.publish_detections(result, header, orig_w, orig_h, send_w, send_h)

            self.publish_status("running")

        except requests.exceptions.Timeout:
            self.get_logger().warn("SAM3 server timeout")
            self.publish_status("timeout")
        except requests.exceptions.ConnectionError:
            self.get_logger().error("Lost connection to SAM3 server")
            self.publish_status("disconnected")
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            self.publish_status("error")

    def publish_detections(self, result, header, orig_w: int, orig_h: int, send_w: int, send_h: int):
        """Publish detection results as Detection2DArray (boxes scaled to original camera frame)."""
        if not HAS_VISION_MSGS:
            return

        det_array = Detection2DArray()
        if header is not None:
            det_array.header = header

        boxes = result.get("boxes", []) or []
        scores = result.get("scores", []) or []

        # Scale boxes if we resized before sending
        sx = float(orig_w) / float(send_w) if send_w > 0 else 1.0
        sy = float(orig_h) / float(send_h) if send_h > 0 else 1.0

        for box, score in zip(boxes, scores):
            try:
                x1, y1, x2, y2 = box
                x1 *= sx
                x2 *= sx
                y1 *= sy
                y2 *= sy

                det = Detection2D()
                det.bbox.center.position.x = float((x1 + x2) / 2.0)
                det.bbox.center.position.y = float((y1 + y2) / 2.0)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.prompt
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)

                det_array.detections.append(det)
            except Exception:
                continue

        self.detection_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3ROSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
