#!/usr/bin/env python3
"""
SAM3 Goal Generator V2 (Updated)

Finds objects with SAM3, estimates distance using multiple methods:
  1) Depth camera (if available) - most accurate
  2) Monocular depth estimation (Depth Anything server) - fallback
  3) Width-only calibrated size method from mask (like your distance test) - fallback
     distance = (real_width * fx) / pixel_width
     - uses mask width (not bbox)
     - can use camera_calib.npz for undistortion + fx
  4) Bounding box size heuristic (legacy fallback, optional last resort)

Key change requested:
- If program detects a depth camera stream, AUTO mode defaults to depth (best accuracy).
- If depth is not available, use the same width-only mask method as your distance test.

Usage:
    ros2 run senior_project sam3_goal_generator
"""

import base64
import math
import threading
import json
import os
import time
from enum import Enum
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion, quaternion_matrix
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker


# =============================================================================
# EDIT THIS DEFAULT (no terminal args needed)
# =============================================================================
DEFAULT_SHOW_GUI = False          # <--- set True/False here
DEFAULT_GUI_SCALE = 1.0          # e.g. 0.75 smaller
DEFAULT_GUI_FPS_OVERLAY = True   # show FPS overlay text
DEFAULT_GUI_SAVE_DIR = "/tmp"    # screenshots go here
DEFAULT_GUI_WINDOW_NAME = "SAM3 Goal Generator"
# =============================================================================


class DepthMode(Enum):
    """Depth estimation mode"""
    AUTO = "auto"           # Try depth camera first, fallback to mono/size
    DEPTH_CAMERA = "depth"  # Depth camera only
    MONOCULAR = "mono"      # Monocular depth estimation
    SIZE_WIDTH = "size"     # Width-only size method (mask-based)
    BBOX_SIZE = "bbox"      # BBox size heuristic (legacy)


DEFAULT_OBJECT_SIZES: Dict[str, Tuple[float, float]] = {
    "person": (0.5, 1.7),
    "chair": (0.5, 0.9),
    "couch": (2.0, 0.9),
    "table": (1.2, 0.75),
    "cup": (0.08, 0.12),
    "bottle": (0.08, 0.25),
    "laptop": (0.35, 0.25),
    "phone": (0.075, 0.15),
    "book": (0.2, 0.25),
    "door": (0.9, 2.0),
    "tv": (1.0, 0.6),
    "monitor": (0.5, 0.35),
    "keyboard": (0.45, 0.15),
    "mouse": (0.06, 0.1),
    "plant": (0.3, 0.5),
    "vase": (0.15, 0.3),
    "clock": (0.3, 0.3),
    "lamp": (0.3, 0.5),
    "bed": (2.0, 0.6),
    "toilet": (0.4, 0.45),
    "sink": (0.5, 0.3),
    "refrigerator": (0.8, 1.8),
    "oven": (0.6, 0.9),
    "microwave": (0.5, 0.3),
    "toaster": (0.25, 0.2),
    "dog": (0.6, 0.5),
    "cat": (0.35, 0.3),
    "ball": (0.22, 0.22),
    "backpack": (0.35, 0.5),
    "umbrella": (0.1, 0.8),
    "handbag": (0.3, 0.25),
    "suitcase": (0.45, 0.7),
    "shoe": (0.3, 0.1),
    "bowl": (0.15, 0.08),
    "banana": (0.2, 0.04),
    "apple": (0.08, 0.08),
    "sandwich": (0.12, 0.06),
    "pizza": (0.35, 0.05),
    "cake": (0.25, 0.1),
    "potted plant": (0.3, 0.5),
    "teddy bear": (0.3, 0.4),
    "object": (0.3, 0.3),
    "furniture": (0.8, 0.8),
    "food": (0.15, 0.1),
    "animal": (0.5, 0.4),
}


class SAM3GoalGeneratorV2(Node):
    def __init__(self):
        super().__init__('sam3_goal_generator')

        # ==================== Parameters ====================
        self.declare_parameter('server_url', 'http://localhost:8100')
        self.declare_parameter('mono_depth_url', 'http://localhost:8101')
        self.declare_parameter('target', 'cup')
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('rate', 0.5)
        self.declare_parameter('ns', 'stretch')
        self.declare_parameter('min_distance', 0.5)
        self.declare_parameter('max_distance', 5.0)
        self.declare_parameter('goal_offset', 0.5)
        self.declare_parameter('auto_publish_goal', True)

        # Depth selection
        self.declare_parameter('depth_mode', 'auto')  # auto | depth | mono | size | bbox
        self.declare_parameter('custom_object_sizes', '{}')

        # Intrinsics fallback
        self.declare_parameter('default_focal_length', 600.0)

        # Topics
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        # Simple geometry fallback if TF off
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_height', 1.0)
        self.declare_parameter('camera_forward_offset', 0.0)

        # TF
        self.declare_parameter('use_tf', True)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')

        # Calibration file (for width-only size fallback)
        # Put your file in the package dir and set to "camera_calib.npz" or an absolute path.
        self.declare_parameter('camera_calib_path', 'camera_calib.npz')
        self.declare_parameter('use_undistort_for_size_method', True)

        # ==================== GUI Parameters ====================
        self.declare_parameter('show_gui', DEFAULT_SHOW_GUI)
        self.declare_parameter('gui_window_name', DEFAULT_GUI_WINDOW_NAME)
        self.declare_parameter('gui_scale', DEFAULT_GUI_SCALE)
        self.declare_parameter('gui_fps', DEFAULT_GUI_FPS_OVERLAY)
        self.declare_parameter('gui_save_dir', DEFAULT_GUI_SAVE_DIR)

        # Get parameters
        self.server_url = self.get_parameter('server_url').value
        self.mono_depth_url = self.get_parameter('mono_depth_url').value
        self.target = self.get_parameter('target').value
        self.confidence = float(self.get_parameter('confidence_threshold').value)
        self.rate = float(self.get_parameter('rate').value)
        self.ns = self.get_parameter('ns').value
        self.min_dist = float(self.get_parameter('min_distance').value)
        self.max_dist = float(self.get_parameter('max_distance').value)
        self.goal_offset = float(self.get_parameter('goal_offset').value)
        self.auto_publish = bool(self.get_parameter('auto_publish_goal').value)
        self.default_focal_length = float(self.get_parameter('default_focal_length').value)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self.camera_pitch_rad = math.radians(float(self.get_parameter('camera_pitch_deg').value))
        self.camera_height = float(self.get_parameter('camera_height').value)
        self.camera_forward_offset = float(self.get_parameter('camera_forward_offset').value)

        self.use_tf = bool(self.get_parameter('use_tf').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        # Calibration config
        self.camera_calib_path = str(self.get_parameter('camera_calib_path').value)
        self.use_undistort_for_size_method = bool(self.get_parameter('use_undistort_for_size_method').value)

        # GUI config
        self.show_gui = bool(self.get_parameter('show_gui').value)
        self.gui_window_name = str(self.get_parameter('gui_window_name').value)
        self.gui_scale = float(self.get_parameter('gui_scale').value)
        self.gui_show_fps = bool(self.get_parameter('gui_fps').value)
        self.gui_save_dir = str(self.get_parameter('gui_save_dir').value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_available = False

        # Depth mode
        mode_str = str(self.get_parameter('depth_mode').value).lower()
        if mode_str in [m.value for m in DepthMode]:
            self.depth_mode = DepthMode(mode_str)
        else:
            self.depth_mode = DepthMode.AUTO

        # Custom object sizes
        try:
            custom_sizes = json.loads(self.get_parameter('custom_object_sizes').value)
            self.object_sizes = {**DEFAULT_OBJECT_SIZES, **custom_sizes}
        except json.JSONDecodeError:
            self.object_sizes = DEFAULT_OBJECT_SIZES.copy()

        # ==================== State ====================
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.goal_sent = False

        # Depth camera availability should be "fresh"
        self.depth_camera_available = False
        self._last_depth_msg_time = 0.0
        self.depth_stale_sec = 1.0  # if no depth msg within this window, treat as unavailable

        # Mono depth server availability
        self.mono_depth_available = False
        self.last_depth_check = 0.0
        self.depth_check_interval = 5.0

        self.session = requests.Session()

        # GUI runtime state
        self.gui_ready = False
        self.gui_paused = False
        self._last_viz_for_gui = None
        self._last_gui_time = time.time()
        self._gui_fps = 0.0

        # Calibration (intrinsics + distortion)
        self._calib_loaded = False
        self._K = None
        self._D = None
        self._undistort_maps = None  # (map1, map2, newK, size)
        self._load_camera_calibration()

        # ==================== QoS ====================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ==================== Subscribers ====================
        self.create_subscription(Image, rgb_topic, self.rgb_callback, qos)
        self.create_subscription(Image, depth_topic, self.depth_callback, qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos)
        self.create_subscription(Odometry, f'/{self.ns}/odom', self.odom_callback, 10)
        self.create_subscription(String, '~/set_target', self.target_callback, 10)

        # ==================== Publishers ====================
        self.goal_pub = self.create_publisher(PointStamped, f'/{self.ns}/goal', 10)
        self.marker_pub = self.create_publisher(Marker, f'/{self.ns}/goal_marker', 10)
        self.viz_pub = self.create_publisher(Image, '~/visualization', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.depth_viz_pub = self.create_publisher(Image, '~/depth_visualization', 10)

        # ==================== Timer ====================
        self.timer = self.create_timer(1.0 / max(self.rate, 0.1), self.process_callback)
        self._sam3_lock = threading.Lock()
        self._sam3_busy = False
        
        # ==================== Initialize ====================
        self.set_sam3_prompt(self.target)
        self.check_mono_depth_server()
        self._init_gui()

        self.get_logger().info('SAM3 Goal Generator V2 started')
        self.get_logger().info(f'  Target: "{self.target}"')
        self.get_logger().info(f'  SAM3 Server: {self.server_url}')
        self.get_logger().info(f'  Mono Depth Server: {self.mono_depth_url}')
        self.get_logger().info(f'  Depth Mode: {self.depth_mode.value} (AUTO prefers depth if detected)')
        self.get_logger().info(f'  Calibration: {"LOADED" if self._calib_loaded else "not loaded"} | file={self.camera_calib_path}')
        self.get_logger().info(f'  GUI: {"ON" if self.gui_ready else "OFF"} (default={DEFAULT_SHOW_GUI})')

    # ==================== Calibration Helpers ====================

    def _load_camera_calibration(self):
        """
        Loads camera_calib.npz if available (camera_matrix, dist_coeffs).
        Works with absolute path or relative to current working dir.
        """
        path = self.camera_calib_path
        try_paths = [path]
        if not os.path.isabs(path):
            try_paths.append(os.path.join(os.getcwd(), path))

        for p in try_paths:
            if os.path.exists(p):
                try:
                    d = np.load(p)
                    K = d["camera_matrix"]
                    D = d["dist_coeffs"]
                    if K.shape != (3, 3):
                        raise ValueError(f"camera_matrix shape is {K.shape}, expected (3,3)")
                    self._K = K.astype(np.float64)
                    self._D = D.astype(np.float64)
                    self._calib_loaded = True
                    self.get_logger().info(f"Loaded calibration from: {p}")
                    return
                except Exception as e:
                    self.get_logger().warn(f"Found calib file but failed to load ({p}): {e}")
                    self._calib_loaded = False
                    return

        self._calib_loaded = False

    def _ensure_undistort_maps(self, w: int, h: int):
        """
        Builds undistort maps for the given image size, if calibration is loaded.
        """
        if not self._calib_loaded or not self.use_undistort_for_size_method:
            return

        if self._undistort_maps is not None:
            _, _, _, size = self._undistort_maps
            if size == (w, h):
                return

        try:
            # alpha=0 crops to valid pixels (stable geometry)
            newK, _ = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w, h), alpha=0)
            map1, map2 = cv2.initUndistortRectifyMap(self._K, self._D, None, newK, (w, h), cv2.CV_16SC2)
            self._undistort_maps = (map1, map2, newK, (w, h))
            self.get_logger().info(f"Undistort maps ready for size {w}x{h}")
        except Exception as e:
            self.get_logger().warn(f"Failed to create undistort maps: {e}")
            self._undistort_maps = None

    def _undistort_image(self, img: np.ndarray) -> np.ndarray:
        if self._undistort_maps is None:
            return img
        map1, map2, _, _ = self._undistort_maps
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def _get_fx_for_size_method(self) -> float:
        """
        For width-only size method:
        Prefer fx from camera_calib.npz (if loaded),
        else CameraInfo K,
        else default_focal_length.
        """
        if self._calib_loaded and self._undistort_maps is not None:
            # use newK fx if we undistort (geometry is in that space)
            _, _, newK, _ = self._undistort_maps
            return float(newK[0, 0])

        if self._calib_loaded and self._K is not None:
            return float(self._K[0, 0])

        if self.camera_info is not None:
            return float(self.camera_info.k[0])

        return float(self.default_focal_length)

    # ==================== GUI Helpers ====================

    def _init_gui(self):
        if not self.show_gui:
            self.gui_ready = False
            return

        if os.name != "nt" and "DISPLAY" not in os.environ:
            self.get_logger().warn("GUI enabled in code, but DISPLAY is not set (headless). Disabling GUI.")
            self.gui_ready = False
            return

        try:
            cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.gui_window_name, 960, 540)
            self.gui_ready = True
            self.get_logger().info("OpenCV window opened. Keys: q/ESC=quit, p=pause, s=screenshot")
        except Exception as e:
            self.get_logger().warn(f"Could not open OpenCV window (disabling GUI): {e}")
            self.gui_ready = False

    def destroy_node(self):
        try:
            if self.gui_ready:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

    # ==================== Server Communication ====================

    def set_sam3_prompt(self, target: str):
        try:
            self.session.post(
                f"{self.server_url}/prompt/{target.replace(' ', '%20')}",
                timeout=5
            )
            self.get_logger().info(f'SAM3 prompt set to: "{target}"')
        except Exception as e:
            self.get_logger().warn(f'Could not set SAM3 prompt: {e}')

    def check_mono_depth_server(self):
        try:
            r = self.session.get(f"{self.mono_depth_url}/health", timeout=2)
            self.mono_depth_available = (r.status_code == 200)
            if self.mono_depth_available:
                self.get_logger().info('âœ“ Monocular depth server available')
        except Exception:
            self.mono_depth_available = False
            self.get_logger().warn('Monocular depth server not available')

    def get_mono_depth(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.mono_depth_available:
            return None

        try:
            _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf).decode()

            r = self.session.post(
                f"{self.mono_depth_url}/estimate",
                json={"image_base64": img_b64},
                timeout=5
            )

            if r.status_code != 200:
                return None

            result = r.json()
            if not result.get('success'):
                return None

            depth_b64 = result.get('depth_base64')
            if not depth_b64:
                return None

            depth_bytes = base64.b64decode(depth_b64)
            depth_np = np.frombuffer(depth_bytes, dtype=np.float32)

            h = result.get('height', rgb.shape[0])
            w = result.get('width', rgb.shape[1])
            depth = depth_np.reshape((h, w))
            return depth

        except Exception as e:
            self.get_logger().debug(f'Mono depth request failed: {e}')
            return None

    # ==================== Callbacks ====================

    def target_callback(self, msg: String):
        self.target = msg.data
        self.goal_sent = False
        self.set_sam3_prompt(self.target)
        self.get_logger().info(f'Target changed to: "{self.target}"')

    def rgb_callback(self, msg: Image):
        with self.lock:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg: Image):
        with self.lock:
            if msg.encoding == '16UC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            self.depth_camera_available = True
            self._last_depth_msg_time = time.time()

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ==================== Distance helpers (NEW width-only mask method) ====================

    def _lookup_object_size(self, object_class: str) -> Tuple[float, float]:
        object_class_lower = object_class.lower().strip()

        if object_class_lower in self.object_sizes:
            return self.object_sizes[object_class_lower]

        for key, size in self.object_sizes.items():
            k = key.lower()
            if k in object_class_lower or object_class_lower in k:
                return size

        return self.object_sizes.get("object", (0.3, 0.3))

    def _mask_dimensions(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Returns (w, h, x1, y1, x2, y2) for mask foreground bbox.
        """
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return (w, h, x1, y1, x2, y2)

    def estimate_distance_from_mask_width(self, mask: np.ndarray, object_class: str, rgb_shape: Tuple[int, int, int]) -> Optional[float]:
        """
        WIDTH-ONLY distance estimate (pitch-independent) like your distance test:
          distance = (real_width * fx) / pixel_width

        Uses mask width (not bbox width). Optionally undistorts mask before measuring width.
        """
        if mask is None:
            return None

        img_h, img_w = rgb_shape[0], rgb_shape[1]
        self._ensure_undistort_maps(img_w, img_h)

        mask_use = mask
        if mask_use.shape[:2] != (img_h, img_w):
            mask_use = cv2.resize(mask_use, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        if self._undistort_maps is not None:
            # undistort mask for more stable width near edges
            mask_use = self._undistort_image(mask_use)

        dims = self._mask_dimensions(mask_use)
        if dims is None:
            return None

        mask_w, mask_h, _, _, _, _ = dims
        if mask_w <= 0 or mask_h <= 0:
            return None

        real_w, _real_h = self._lookup_object_size(object_class)
        fx = self._get_fx_for_size_method()

        distance = (real_w * fx) / float(mask_w)

        if distance < 0.1 or distance > 20.0:
            return None
        return float(distance)

    # ==================== Legacy bbox method (kept as last resort) ====================

    def estimate_distance_from_bbox(self, box: List[float], object_class: str,
                                    img_width: int, img_height: int) -> Optional[float]:
        x1, y1, x2, y2 = box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0 or bbox_height <= 0:
            return None

        real_width, real_height = self._lookup_object_size(object_class)

        if self.camera_info is not None:
            focal_length = (self.camera_info.k[0] + self.camera_info.k[4]) / 2.0
        else:
            focal_length = self.default_focal_length

        dist_from_width = (real_width * focal_length) / bbox_width
        dist_from_height = (real_height * focal_length) / bbox_height
        distance = max(dist_from_width, dist_from_height)

        if distance < 0.1 or distance > 20.0:
            return None
        return float(distance)

    # ==================== Depth helpers ====================

    def get_depth_at_point(self, u: int, v: int, depth_img: np.ndarray,
                           window_size: int = 5) -> Optional[float]:
        h, w = depth_img.shape[:2]
        u = max(window_size, min(w - window_size - 1, u))
        v = max(window_size, min(h - window_size - 1, v))

        window = depth_img[v-window_size:v+window_size+1, u-window_size:u+window_size+1]
        valid = (window > 0.1) & (window < 20.0) & (~np.isnan(window))
        if not np.any(valid):
            return None
        return float(np.median(window[valid]))

    def _get_depth_from_mask_or_point(self, mask: Optional[np.ndarray],
                                      depth_img: np.ndarray,
                                      center_x: int, center_y: int) -> Optional[float]:
        if mask is not None:
            if mask.shape != depth_img.shape:
                mask_rs = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_rs = mask
            mask_bool = mask_rs > 127
            depths = depth_img[mask_bool]
            valid = (depths > 0.1) & (depths < 20.0) & (~np.isnan(depths))
            if np.any(valid):
                return float(np.median(depths[valid]))
        return self.get_depth_at_point(center_x, center_y, depth_img)

    def get_mask_center(self, mask: Optional[np.ndarray], box: List[float], rgb: np.ndarray) -> Tuple[int, int]:
        if mask is not None:
            mask_rs = mask
            if mask_rs.shape[:2] != rgb.shape[:2]:
                mask_rs = cv2.resize(mask_rs, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            ys, xs = np.where(mask_rs > 127)
            if len(xs) > 0:
                return int(np.mean(xs)), int(np.mean(ys))

        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _depth_is_fresh(self) -> bool:
        if not self.depth_camera_available:
            return False
        return (time.time() - self._last_depth_msg_time) <= self.depth_stale_sec

    def get_distance_auto(self, mask: Optional[np.ndarray], depth_img: Optional[np.ndarray],
                          box: List[float], object_class: str, rgb: np.ndarray) -> Tuple[Optional[float], str]:
        """
        Implements requested behavior:
        - AUTO: if depth camera detected -> use it
        - else mono if available
        - else size(width-only from mask) (same as distance test)
        - else bbox (last resort)
        """
        # keep depth availability fresh
        depth_ok = depth_img is not None and self._depth_is_fresh()

        cx, cy = self.get_mask_center(mask, box, rgb)

        if self.depth_mode == DepthMode.DEPTH_CAMERA:
            if depth_ok:
                d = self._get_depth_from_mask_or_point(mask, depth_img, cx, cy)
                return d, "depth_camera"
            return None, "depth_camera"

        if self.depth_mode == DepthMode.MONOCULAR:
            mono = self.get_mono_depth(rgb)
            if mono is not None:
                d = self._get_depth_from_mask_or_point(mask, mono, cx, cy)
                return d, "mono_depth"
            return None, "mono_depth"

        if self.depth_mode == DepthMode.SIZE_WIDTH:
            if mask is not None:
                d = self.estimate_distance_from_mask_width(mask, object_class, rgb.shape)
                return d, "size_width"
            return None, "size_width"

        if self.depth_mode == DepthMode.BBOX_SIZE:
            d = self.estimate_distance_from_bbox(box, object_class, rgb.shape[1], rgb.shape[0])
            return d, "bbox_size"

        # AUTO
        if depth_ok:
            d = self._get_depth_from_mask_or_point(mask, depth_img, cx, cy)
            if d is not None:
                return d, "depth_camera"

        if self.mono_depth_available:
            mono = self.get_mono_depth(rgb)
            if mono is not None:
                d = self._get_depth_from_mask_or_point(mask, mono, cx, cy)
                if d is not None:
                    return d, "mono_depth"

        if mask is not None:
            d = self.estimate_distance_from_mask_width(mask, object_class, rgb.shape)
            if d is not None:
                return d, "size_width"

        d = self.estimate_distance_from_bbox(box, object_class, rgb.shape[1], rgb.shape[0])
        return d, "bbox_size" if d is not None else "none"

    # ==================== Geometry / TF ====================

    def pixel_to_3d(self, u: int, v: int, depth: float) -> Optional[Tuple[float, float, float]]:
        if depth <= 0 or np.isnan(depth):
            return None

        if self.camera_info is not None:
            fx = float(self.camera_info.k[0])
            fy = float(self.camera_info.k[4])
            cx = float(self.camera_info.k[2])
            cy = float(self.camera_info.k[5])
        else:
            fx = fy = float(self.default_focal_length)
            cx = 320.0
            cy = 240.0

        z = depth
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return (x, y, z)

    def get_camera_to_odom_transform(self) -> Optional[np.ndarray]:
        if not self.use_tf:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            t = transform.transform.translation
            q = transform.transform.rotation
            rot_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
            rot_matrix[0, 3] = t.x
            rot_matrix[1, 3] = t.y
            rot_matrix[2, 3] = t.z

            self.tf_available = True
            return rot_matrix

        except TransformException as e:
            if self.tf_available:
                self.get_logger().debug(f'TF lookup failed: {e}')
            self.tf_available = False
            return None

    def camera_to_odom(self, cam_x: float, cam_y: float, cam_z: float) -> Tuple[float, float]:
        tf_matrix = self.get_camera_to_odom_transform()
        if tf_matrix is not None:
            point_camera = np.array([cam_x, cam_y, cam_z, 1.0])
            point_odom = tf_matrix @ point_camera
            return (float(point_odom[0]), float(point_odom[1]))

        cos_pitch = math.cos(self.camera_pitch_rad)
        sin_pitch = math.sin(self.camera_pitch_rad)

        robot_forward = cam_z * cos_pitch - cam_y * sin_pitch
        robot_left = -cam_x
        robot_up = cam_z * sin_pitch + cam_y * cos_pitch

        robot_forward += self.camera_forward_offset

        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)

        odom_x = self.robot_x + robot_forward * cos_yaw - robot_left * sin_yaw
        odom_y = self.robot_y + robot_forward * sin_yaw + robot_left * cos_yaw
        return (odom_x, odom_y)

    # ==================== Mask decode ====================

    def decode_mask(self, mask_b64: str) -> Optional[np.ndarray]:
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            return cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

    # ==================== Goal publish ====================

    def publish_goal(self, x: float, y: float):
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.point.x = x
        goal.point.y = y
        goal.point.z = 0.0
        self.goal_pub.publish(goal)

        marker = Marker()
        marker.header = goal.header
        marker.ns = 'sam3_goal'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.25
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

        self.get_logger().info(f'Goal published: ({x:.2f}, {y:.2f})')

    # ==================== Main loop ====================

    def process_callback(self):
        current_time = time.time()

        # Keep depth freshness updated
        if self.depth_camera_available and (current_time - self._last_depth_msg_time) > self.depth_stale_sec:
            self.depth_camera_available = False

        # Skip if previous request still running - CHECK FIRST before expensive operations
        if self._sam3_busy:
            return

        if current_time - self.last_depth_check > self.depth_check_interval:
            self.check_mono_depth_server()
            self.last_depth_check = current_time

        with self.lock:
            if self.latest_rgb is None:
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy() if self.latest_depth is not None else None

        # Only encode after we know we're not busy (saves CPU when SAM3 is processing)
        _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(buf).decode()

        def run_sam3():
            self._sam3_busy = True
            try:
                r = self.session.post(
                    f"{self.server_url}/segment",
                    json={"image_base64": img_b64, "confidence_threshold": self.confidence},
                    timeout=10
                )
                with self._sam3_lock:
                    self._last_sam3_result = r.json()
            except Exception as e:
                self.get_logger().warn(f'SAM3 request failed: {e}')
            finally:
                self._sam3_busy = False

        threading.Thread(target=run_sam3, daemon=True).start()

        # Use cached result
        with self._sam3_lock:
            result = getattr(self, '_last_sam3_result', None)
        if result is None or not result.get('success'):
            return


        boxes = result.get('boxes', [])
        scores = result.get('scores', [])
        masks_b64 = result.get('masks_base64', [])
        prompt = result.get('prompt', self.target)

        best_detection = None
        best_score = 0.0
        viz = rgb.copy()

        for box, score, mask_b64 in zip(boxes, scores, masks_b64):
            mask = self.decode_mask(mask_b64)

            # center for projection
            cx, cy = self.get_mask_center(mask, box, rgb)

            dist, method = self.get_distance_auto(mask, depth, box, prompt, rgb)
            if dist is None or dist < self.min_dist or dist > self.max_dist:
                continue

            cam_point = self.pixel_to_3d(cx, cy, dist)
            if cam_point is None:
                continue

            odom_x, odom_y = self.camera_to_odom(*cam_point)

            x1, y1, x2, y2 = [int(v) for v in box]
            color = (0, 255, 0) if float(score) > best_score else (255, 255, 0)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

            method_short = {
                "depth_camera": "D",
                "mono_depth": "M",
                "size_width": "S",
                "bbox_size": "B",
                "none": "?"
            }.get(method, "?")

            label = f"{prompt}: {float(score):.2f} | {float(dist):.2f}m [{method_short}]"
            cv2.putText(viz, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(viz, (cx, cy), 5, color, -1)

            if float(score) > best_score:
                best_score = float(score)
                best_detection = {
                    'distance': float(dist),
                    'odom_x': float(odom_x),
                    'odom_y': float(odom_y),
                    'score': float(score),
                    'method': method
                }

        depth_status = []
        if self._depth_is_fresh():
            depth_status.append("depth_cam")
        if self.mono_depth_available:
            depth_status.append("mono")
        depth_status.append("size")
        depth_status.append("bbox")

        if best_detection:
            dx = best_detection['odom_x'] - self.robot_x
            dy = best_detection['odom_y'] - self.robot_y
            dist_to_obj = math.sqrt(dx * dx + dy * dy)

            if dist_to_obj > self.goal_offset:
                scale = (dist_to_obj - self.goal_offset) / dist_to_obj
                goal_x = self.robot_x + dx * scale
                goal_y = self.robot_y + dy * scale
            else:
                goal_x = best_detection['odom_x']
                goal_y = best_detection['odom_y']

            status = {
                'found': True,
                'target': prompt,
                'distance': best_detection['distance'],
                'confidence': best_detection['score'],
                'depth_method': best_detection['method'],
                'available_methods': depth_status,
                'object_position': [best_detection['odom_x'], best_detection['odom_y']],
                'goal_position': [float(goal_x), float(goal_y)]
            }

            method_name = best_detection['method'].replace('_', ' ').title()
            cv2.putText(viz, f"GOAL: ({goal_x:.1f}, {goal_y:.1f}) via {method_name}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.auto_publish and not self.goal_sent:
                self.publish_goal(goal_x, goal_y)
                self.goal_sent = True
        else:
            status = {
                'found': False,
                'target': prompt,
                'available_methods': depth_status,
                'message': 'Object not found in valid range'
            }

        self.status_pub.publish(String(data=json.dumps(status)))

        mode_str = f"Mode: {self.depth_mode.value} | Available: {', '.join(depth_status)}"
        cv2.putText(viz, f"Target: {prompt} | Robot: ({self.robot_x:.1f}, {self.robot_y:.1f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, mode_str, (10, viz.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz, 'bgr8'))

        self._last_viz_for_gui = viz
        self._update_gui(viz)

    def _update_gui(self, viz_bgr: np.ndarray):
        if not self.gui_ready or viz_bgr is None:
            return

        # Rotate 90 degrees clockwise to correct camera orientation
        viz_bgr = cv2.rotate(viz_bgr, cv2.ROTATE_90_CLOCKWISE)

        now = time.time()
        dt = max(now - self._last_gui_time, 1e-6)
        self._last_gui_time = now
        inst = 1.0 / dt
        self._gui_fps = 0.9 * self._gui_fps + 0.1 * inst

        frame = viz_bgr
        if self.gui_scale and abs(self.gui_scale - 1.0) > 1e-3:
            new_w = max(1, int(frame.shape[1] * self.gui_scale))
            new_h = max(1, int(frame.shape[0] * self.gui_scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.gui_show_fps:
            txt = f"FPS: {self._gui_fps:.1f}"
            if self.gui_paused:
                txt += " | PAUSED"
            cv2.putText(frame, txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        try:
            cv2.imshow(self.gui_window_name, frame)
            key = cv2.waitKey(1) & 0xFF
        except Exception as e:
            self.get_logger().warn(f"GUI error; disabling GUI: {e}")
            self.gui_ready = False
            return

        if key in (ord('q'), 27):
            self.get_logger().info("GUI quit key pressed. Shutting down.")
            rclpy.shutdown()
        elif key == ord('p'):
            self.gui_paused = not self.gui_paused
        elif key == ord('s'):
            try:
                os.makedirs(self.gui_save_dir, exist_ok=True)
                fname = os.path.join(self.gui_save_dir, f"sam3_viz_{int(time.time())}.png")
                cv2.imwrite(fname, viz_bgr)
                self.get_logger().info(f"Saved screenshot: {fname}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save screenshot: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SAM3GoalGeneratorV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()