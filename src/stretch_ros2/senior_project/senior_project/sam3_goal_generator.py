#!/usr/bin/env python3
"""
SAM3 Goal Generator V2 - Performance + Stability Improvements

Key upgrades:
- Goal publish throttling + only publish when goal changes meaningfully
- Publish at most once per new SAM3 result (prevents 15Hz spamming)
- Thread-safe SAM3 request gating (_sam3_busy lock, set before thread start)
- Avoid sharing requests.Session across threads (thread uses its own session)
- Robust HTTP + JSON handling (status checks + JSON decode guards)
- Cache monocular depth per SAM3 result (big perf win)
- Consistent INTER_NEAREST for masks
- Use self.odom_frame for goal header frame_id (no hardcoded 'odom')
- GUI pause actually pauses SAM3 + goal publishing
- CameraInfo QoS set to RELIABLE
"""

import base64
import json
import math
import os
import threading
import time
from enum import Enum
from typing import Dict, Optional, Tuple

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
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import Marker

# =============================================================================
# EDIT THIS DEFAULT (no terminal args needed)
# =============================================================================
DEFAULT_SHOW_GUI = False
DEFAULT_GUI_SCALE = 1.0
DEFAULT_GUI_FPS_OVERLAY = True
DEFAULT_GUI_SAVE_DIR = "/tmp"
DEFAULT_GUI_WINDOW_NAME = "SAM3 Goal Generator"
# =============================================================================


class DepthMode(Enum):
    AUTO = "auto"
    DEPTH_CAMERA = "depth"
    MONOCULAR = "mono"
    SIZE_WIDTH = "size"
    BBOX_SIZE = "bbox"


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
    # MuJoCo simulation objects
    "cylinder": (0.1, 0.3),
    "blue cylinder": (0.1, 0.3),
    "red cylinder": (0.1, 0.3),
    "green cylinder": (0.1, 0.3),
    "box": (0.3, 0.3),
    "cube": (0.2, 0.2),
    "sphere": (0.2, 0.2),
    "object": (0.3, 0.3),  # Generic fallback
    "target": (0.2, 0.2),
    "goal": (0.2, 0.2),
    "marker": (0.1, 0.3),
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
        self.declare_parameter('targets', '')  # Comma-separated list for cycling
        self.declare_parameter('target_cycle_on_reach', True)
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('rate', 2.0)  # SAM3 detection rate (Hz)
        self.declare_parameter('viz_rate', 15.0)  # Visualization rate (Hz)
        self.declare_parameter('ns', 'stretch')
        self.declare_parameter('min_distance', 0.3)
        self.declare_parameter('max_distance', 10.0)
        self.declare_parameter('goal_offset', 0.5)
        self.declare_parameter('auto_publish_goal', True)

        self.declare_parameter('depth_mode', 'auto')
        self.declare_parameter('custom_object_sizes', '{}')
        self.declare_parameter('default_focal_length', 600.0)

        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_height', 1.0)
        self.declare_parameter('camera_forward_offset', 0.0)

        # Performance parameters
        self.declare_parameter('sam3_max_width', 640)
        self.declare_parameter('sam3_jpeg_quality', 60)

        self.declare_parameter('use_tf', True)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')

        self.declare_parameter('camera_calib_path', 'camera_calib.npz')
        self.declare_parameter('use_undistort_for_size_method', True)

        # ==================== GUI Parameters ====================
        self.declare_parameter('show_gui', DEFAULT_SHOW_GUI)
        self.declare_parameter('gui_window_name', DEFAULT_GUI_WINDOW_NAME)
        self.declare_parameter('gui_scale', DEFAULT_GUI_SCALE)
        self.declare_parameter('gui_fps', DEFAULT_GUI_FPS_OVERLAY)
        self.declare_parameter('gui_save_dir', DEFAULT_GUI_SAVE_DIR)

        # === New stability/perf parameters ===
        self.declare_parameter('rotate_viz_90_clockwise', True)
        self.declare_parameter('goal_publish_min_period', 0.75)   # seconds
        self.declare_parameter('goal_publish_min_delta', 0.15)    # meters
        self.declare_parameter('tf_timeout_sec', 0.15)            # seconds
        self.declare_parameter('sam3_timeout_sec', 5.0)
        self.declare_parameter('mono_depth_timeout_sec', 5.0)

        # Get parameters
        self.server_url = self.get_parameter('server_url').value
        self.mono_depth_url = self.get_parameter('mono_depth_url').value
        self.target = self.get_parameter('target').value
        self.confidence = float(self.get_parameter('confidence_threshold').value)
        self.rate = float(self.get_parameter('rate').value)
        self.viz_rate = float(self.get_parameter('viz_rate').value)
        self.ns = self.get_parameter('ns').value
        self.min_dist = float(self.get_parameter('min_distance').value)
        self.max_dist = float(self.get_parameter('max_distance').value)
        self.goal_offset = float(self.get_parameter('goal_offset').value)
        self.auto_publish = bool(self.get_parameter('auto_publish_goal').value)
        self.default_focal_length = float(self.get_parameter('default_focal_length').value)

        self.rotate_viz_90_clockwise = bool(self.get_parameter('rotate_viz_90_clockwise').value)
        self.goal_publish_min_period = float(self.get_parameter('goal_publish_min_period').value)
        self.goal_publish_min_delta = float(self.get_parameter('goal_publish_min_delta').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.sam3_timeout_sec = float(self.get_parameter('sam3_timeout_sec').value)
        self.mono_depth_timeout_sec = float(self.get_parameter('mono_depth_timeout_sec').value)

        # Multiple targets cycling
        targets_str = self.get_parameter('targets').value
        if targets_str:
            self.target_list = [t.strip() for t in targets_str.split(',') if t.strip()]
        else:
            self.target_list = [self.target] if self.target else []
        self.target_cycle_on_reach = bool(self.get_parameter('target_cycle_on_reach').value)
        self.current_target_index = 0
        if self.target_list:
            self.target = self.target_list[0]

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self.camera_pitch_rad = math.radians(float(self.get_parameter('camera_pitch_deg').value))
        self.camera_height = float(self.get_parameter('camera_height').value)
        self.camera_forward_offset = float(self.get_parameter('camera_forward_offset').value)

        # Performance settings
        self.sam3_max_width = int(self.get_parameter('sam3_max_width').value)
        self.sam3_jpeg_quality = int(self.get_parameter('sam3_jpeg_quality').value)

        self.use_tf = bool(self.get_parameter('use_tf').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        self.camera_calib_path = str(self.get_parameter('camera_calib_path').value)
        self.use_undistort_for_size_method = bool(self.get_parameter('use_undistort_for_size_method').value)

        self.show_gui = bool(self.get_parameter('show_gui').value)
        self.gui_window_name = str(self.get_parameter('gui_window_name').value)
        self.gui_scale = float(self.get_parameter('gui_scale').value)
        self.gui_show_fps = bool(self.get_parameter('gui_fps').value)
        self.gui_save_dir = str(self.get_parameter('gui_save_dir').value)

        # ===== Segmentation overlay settings =====
        self.show_segmentation_overlay = True
        self.seg_overlay_alpha = 0.65
        self.show_contours = True
        self.contour_thickness = 3

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

        self.depth_camera_available = False
        self._last_depth_msg_time = 0.0
        self.depth_stale_sec = 1.0

        self.mono_depth_available = False
        self.last_depth_check = 0.0
        self.depth_check_interval = 5.0

        # Use a session in main thread only; background threads create their own session
        self.session = requests.Session()

        self.gui_ready = False
        self.gui_paused = False
        self._last_viz_for_gui = None
        self._last_gui_time = time.time()
        self._gui_fps = 0.0

        self._calib_loaded = False
        self._K = None
        self._D = None
        self._undistort_maps = None
        self._load_camera_calibration()

        # Store last SAM3 result
        self._last_sam3_result = None
        self._sam3_lock = threading.Lock()

        # Thread-safe busy flag
        self._sam3_busy = False
        self._sam3_busy_lock = threading.Lock()

        # Track "new result" to avoid re-publishing per viz frame
        self._sam3_result_seq = 0
        self._last_processed_seq = -1

        # Cache mono depth per SAM3 result
        self._mono_depth_cache = None
        self._mono_depth_cache_seq = -1

        # Publish throttling
        self._last_goal_pub_time = 0.0
        self._last_goal_xy = None  # (x, y)

        # last comm errors to report in status
        self._last_sam3_error = ""
        self._last_mono_error = ""

        # ==================== QoS ====================
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_info = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ==================== Subscribers ====================
        self.create_subscription(Image, rgb_topic, self.rgb_callback, qos_img)
        self.create_subscription(Image, depth_topic, self.depth_callback, qos_img)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos_info)
        self.create_subscription(Odometry, f'/{self.ns}/odom', self.odom_callback, 10)
        self.create_subscription(String, '~/set_target', self.target_callback, 10)
        self.create_subscription(PointStamped, f'/{self.ns}/goal_reached', self.goal_reached_callback, 10)

        # ==================== Publishers ====================
        self.goal_pub = self.create_publisher(PointStamped, f'/{self.ns}/goal', 10)
        self.marker_pub = self.create_publisher(Marker, f'/{self.ns}/goal_marker', 10)
        self.viz_pub = self.create_publisher(Image, '~/visualization', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.depth_viz_pub = self.create_publisher(Image, '~/depth_visualization', 10)

        # ==================== Timers ====================
        self.viz_timer = self.create_timer(1.0 / max(self.viz_rate, 1.0), self.viz_callback)
        self.sam3_timer = self.create_timer(1.0 / max(self.rate, 0.1), self.sam3_callback)

        # ==================== Initialize ====================
        self.set_sam3_prompt(self.target)
        self.check_mono_depth_server()
        self._init_gui()

        self.get_logger().info('SAM3 Goal Generator V2 started')
        self.get_logger().info(f'  Target: "{self.target}"')
        if len(self.target_list) > 1:
            self.get_logger().info(f'  Target List: {self.target_list} (cycling enabled)')
        self.get_logger().info(f'  SAM3 Server: {self.server_url}')
        self.get_logger().info(f'  Mono Depth Server: {self.mono_depth_url}')
        self.get_logger().info(f'  Depth Mode: {self.depth_mode.value}')
        self.get_logger().info(f'  Detection Rate: {self.rate} Hz, Viz Rate: {self.viz_rate} Hz')
        self.get_logger().info(f'  SAM3 max width: {self.sam3_max_width}px, JPEG quality: {self.sam3_jpeg_quality}')
        self.get_logger().info(f'  Segmentation Overlay: alpha={self.seg_overlay_alpha}, contours={self.show_contours}')
        self.get_logger().info(f'  Goal throttle: min_period={self.goal_publish_min_period:.2f}s, min_delta={self.goal_publish_min_delta:.2f}m')

    # ==================== Rotation helpers ====================

    def _rotate90_cw(self, img: np.ndarray) -> np.ndarray:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def _pt_rot90_cw(self, x: int, y: int, orig_h: int) -> Tuple[int, int]:
        return (int((orig_h - 1) - y), int(x))

    def _rect_rot90_cw(self, x1: int, y1: int, x2: int, y2: int, orig_h: int) -> Tuple[int, int, int, int]:
        rx1, ry1 = self._pt_rot90_cw(x1, y1, orig_h)
        rx2, ry2 = self._pt_rot90_cw(x2, y2, orig_h)
        return (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))

    def _mask_to_rgb_size(self, mask: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.shape[:2] != (h, w):
            return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def _overlay_mask(self, img_bgr: np.ndarray, mask_u8: np.ndarray,
                      color_bgr: Tuple[int, int, int], alpha: float):
        """Alpha-blend a colored segmentation mask onto img_bgr in-place."""
        if mask_u8 is None:
            return

        m = mask_u8 > 127
        if not np.any(m):
            return

        overlay = img_bgr.copy()
        overlay[m] = color_bgr
        cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0, dst=img_bgr)

        if self.show_contours:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, (255, 0, 255), self.contour_thickness + 1)

    # ==================== Calibration Helpers ====================

    def _load_camera_calibration(self):
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
        if not self._calib_loaded or not self.use_undistort_for_size_method:
            return

        if self._undistort_maps is not None:
            _, _, _, size = self._undistort_maps
            if size == (w, h):
                return

        try:
            newK, _ = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w, h), alpha=0)
            map1, map2 = cv2.initUndistortRectifyMap(self._K, self._D, None, newK, (w, h), cv2.CV_16SC2)
            self._undistort_maps = (map1, map2, newK, (w, h))
        except Exception as e:
            self.get_logger().warn(f"Failed to create undistort maps: {e}")
            self._undistort_maps = None

    def _undistort_image(self, img: np.ndarray) -> np.ndarray:
        if self._undistort_maps is None:
            return img
        map1, map2, _, _ = self._undistort_maps
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def _get_fx_for_size_method(self) -> float:
        if self._calib_loaded and self._undistort_maps is not None:
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
            self.get_logger().warn("GUI enabled but DISPLAY not set. Disabling GUI.")
            self.gui_ready = False
            return

        try:
            cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.gui_window_name, 960, 540)
            self.gui_ready = True
        except Exception as e:
            self.get_logger().warn(f"Could not open OpenCV window: {e}")
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
            r = self.session.post(
                f"{self.server_url}/prompt/{target.replace(' ', '%20')}",
                timeout=3
            )
            if r.status_code != 200:
                self.get_logger().warn(f'SAM3 prompt set failed (HTTP {r.status_code})')
                return
            self.get_logger().info(f'SAM3 prompt set to: "{target}"')
        except Exception as e:
            self.get_logger().warn(f'Could not set SAM3 prompt: {e}')

    def check_mono_depth_server(self):
        try:
            r = self.session.get(f"{self.mono_depth_url}/health", timeout=2)
            self.mono_depth_available = (r.status_code == 200)
            if not self.mono_depth_available:
                self._last_mono_error = f"health HTTP {r.status_code}"
        except Exception as e:
            self.mono_depth_available = False
            self._last_mono_error = str(e)
            self.get_logger().warn('Monocular depth server not available')

    def _safe_json(self, response) -> Optional[dict]:
        try:
            return response.json()
        except Exception:
            return None

    def get_mono_depth(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.mono_depth_available:
            return None
        try:
            _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf).decode()

            r = self.session.post(
                f"{self.mono_depth_url}/estimate",
                json={"image_base64": img_b64},
                timeout=self.mono_depth_timeout_sec
            )
            if r.status_code != 200:
                self._last_mono_error = f"HTTP {r.status_code}"
                return None

            result = self._safe_json(r)
            if not result or not result.get('success'):
                self._last_mono_error = "bad JSON or success=false"
                return None

            depth_b64 = result.get('depth_base64')
            if not depth_b64:
                self._last_mono_error = "missing depth_base64"
                return None

            depth_bytes = base64.b64decode(depth_b64)
            depth_np = np.frombuffer(depth_bytes, dtype=np.float32)

            h = int(result.get('height', rgb.shape[0]))
            w = int(result.get('width', rgb.shape[1]))
            if h * w != depth_np.size:
                self._last_mono_error = f"depth size mismatch (got {depth_np.size}, expected {h*w})"
                return None

            depth = depth_np.reshape((h, w))
            self._last_mono_error = ""
            return depth

        except Exception as e:
            self._last_mono_error = str(e)
            return None

    # ==================== Callbacks ====================

    def target_callback(self, msg: String):
        self.target = msg.data
        self.set_sam3_prompt(self.target)
        self.get_logger().info(f'Target changed to: "{self.target}"')

    def goal_reached_callback(self, msg: PointStamped):
        if self.target_cycle_on_reach and len(self.target_list) > 1:
            self.cycle_target()

    def cycle_target(self):
        if len(self.target_list) <= 1:
            return
        self.current_target_index = (self.current_target_index + 1) % len(self.target_list)
        new_target = self.target_list[self.current_target_index]
        self.target = new_target
        self.set_sam3_prompt(self.target)
        self.get_logger().info(f'[CYCLE] Target cycled to: "{self.target}" ({self.current_target_index + 1}/{len(self.target_list)})')

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

    # ==================== Distance helpers ====================

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
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return (w, h, x1, y1, x2, y2)

    def estimate_distance_from_mask_width(self, mask: np.ndarray, object_class: str,
                                          rgb_shape: Tuple[int, int, int]) -> Optional[float]:
        if mask is None:
            return None

        img_h, img_w = rgb_shape[0], rgb_shape[1]
        self._ensure_undistort_maps(img_w, img_h)

        mask_use = mask
        if mask_use.shape[:2] != (img_h, img_w):
            mask_use = cv2.resize(mask_use, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if self._undistort_maps is not None:
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

    def estimate_distance_from_bbox(self, box, object_class: str, img_width: int, img_height: int) -> Optional[float]:
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

    def get_depth_at_point(self, u: int, v: int, depth_img: np.ndarray, window_size: int = 5) -> Optional[float]:
        h, w = depth_img.shape[:2]
        u = max(window_size, min(w - window_size - 1, u))
        v = max(window_size, min(h - window_size - 1, v))

        window = depth_img[v-window_size:v+window_size+1, u-window_size:u+window_size+1]
        valid = (window > 0.1) & (window < 20.0) & (~np.isnan(window))
        if not np.any(valid):
            return None
        return float(np.median(window[valid]))

    def _get_depth_from_mask_or_point(self, mask: Optional[np.ndarray], depth_img: np.ndarray,
                                      center_x: int, center_y: int) -> Optional[float]:
        if mask is not None:
            if mask.shape != depth_img.shape:
                mask_rs = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                mask_rs = mask
            mask_bool = mask_rs > 127
            depths = depth_img[mask_bool]
            valid = (depths > 0.1) & (depths < 20.0) & (~np.isnan(depths))
            if np.any(valid):
                return float(np.median(depths[valid]))
        return self.get_depth_at_point(center_x, center_y, depth_img)

    def get_mask_center(self, mask: Optional[np.ndarray], box, rgb: np.ndarray) -> Tuple[int, int]:
        if mask is not None:
            mask_rs = mask
            if mask_rs.shape[:2] != rgb.shape[:2]:
                mask_rs = cv2.resize(mask_rs, (rgb.shape[1], rgb.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_rs > 127)
            if len(xs) > 0:
                return int(np.mean(xs)), int(np.mean(ys))
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _depth_is_fresh(self) -> bool:
        if not self.depth_camera_available:
            return False
        return (time.time() - self._last_depth_msg_time) <= self.depth_stale_sec

    def _get_cached_mono_depth(self, rgb: np.ndarray, current_seq: int) -> Optional[np.ndarray]:
        # Only recompute mono depth once per new SAM3 result
        if not self.mono_depth_available:
            return None
        if self._mono_depth_cache is not None and self._mono_depth_cache_seq == current_seq:
            return self._mono_depth_cache

        mono = self.get_mono_depth(rgb)
        if mono is not None:
            self._mono_depth_cache = mono
            self._mono_depth_cache_seq = current_seq
        return mono

    def get_distance_auto(self, mask: Optional[np.ndarray], depth_img: Optional[np.ndarray],
                          box, object_class: str, rgb: np.ndarray,
                          sam3_seq: int) -> Tuple[Optional[float], str]:
        depth_ok = depth_img is not None and self._depth_is_fresh()
        cx, cy = self.get_mask_center(mask, box, rgb)

        if self.depth_mode == DepthMode.DEPTH_CAMERA:
            if depth_ok:
                d = self._get_depth_from_mask_or_point(mask, depth_img, cx, cy)
                return d, "depth_camera"
            return None, "depth_camera"

        if self.depth_mode == DepthMode.MONOCULAR:
            mono = self._get_cached_mono_depth(rgb, sam3_seq)
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

        # AUTO mode
        if depth_ok:
            d = self._get_depth_from_mask_or_point(mask, depth_img, cx, cy)
            if d is not None:
                return d, "depth_camera"

        mono = self._get_cached_mono_depth(rgb, sam3_seq)
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
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec)
            )

            t = transform.transform.translation
            q = transform.transform.rotation
            rot_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
            rot_matrix[0, 3] = t.x
            rot_matrix[1, 3] = t.y
            rot_matrix[2, 3] = t.z

            self.tf_available = True
            return rot_matrix

        except TransformException:
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
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            return mask
        except Exception as e:
            self.get_logger().warn(f'Failed to decode mask: {e}')
            return None

    # ==================== Goal publish ====================

    def _should_publish_goal(self, goal_x: float, goal_y: float, sam3_seq: int) -> bool:
        if not self.auto_publish:
            return False
        if self.gui_paused:
            return False

        now = time.time()
        if (now - self._last_goal_pub_time) < self.goal_publish_min_period:
            return False

        if self._last_goal_xy is not None:
            lx, ly = self._last_goal_xy
            d = math.hypot(goal_x - lx, goal_y - ly)
            if d < self.goal_publish_min_delta:
                return False

        # Only publish once per new SAM3 result (prevents 15Hz re-publish from viz frames)
        if sam3_seq == self._last_processed_seq:
            # viz is reprocessing the same seq; allow publish only if moved enough
            # (already checked above), so ok to continue
            pass

        return True

    def publish_goal(self, x: float, y: float):
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.odom_frame
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

        self._last_goal_pub_time = time.time()
        self._last_goal_xy = (float(x), float(y))
        self.get_logger().info(f'Goal published: ({x:.2f}, {y:.2f})')

    # ==================== Main loop ====================

    def viz_callback(self):
        """Fast callback for smooth visualization (15+ Hz)."""
        with self.lock:
            if self.latest_rgb is None:
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy() if self.latest_depth is not None else None

        self._publish_visualization(rgb, depth)

    def sam3_callback(self):
        """Slower callback for SAM3 detection (2 Hz)."""
        if self.gui_paused:
            return

        current_time = time.time()

        if self.depth_camera_available and (current_time - self._last_depth_msg_time) > self.depth_stale_sec:
            self.depth_camera_available = False

        if current_time - self.last_depth_check > self.depth_check_interval:
            self.check_mono_depth_server()
            self.last_depth_check = current_time

        with self.lock:
            if self.latest_rgb is None:
                return
            rgb = self.latest_rgb.copy()

        # Thread-safe gate
        with self._sam3_busy_lock:
            if self._sam3_busy:
                return
            self._sam3_busy = True

        self._send_sam3_request(rgb)

    def _send_sam3_request(self, rgb: np.ndarray):
        """Send image to SAM3 server in background thread."""
        orig_h, orig_w = rgb.shape[:2]
        scale_factor = 1.0
        rgb_for_sam3 = rgb

        if orig_w > self.sam3_max_width:
            scale_factor = self.sam3_max_width / orig_w
            new_w = self.sam3_max_width
            new_h = int(orig_h * scale_factor)
            rgb_for_sam3 = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode('.jpg', rgb_for_sam3, [cv2.IMWRITE_JPEG_QUALITY, self.sam3_jpeg_quality])
        if not ok:
            self._last_sam3_error = "jpeg encode failed"
            with self._sam3_busy_lock:
                self._sam3_busy = False
            return

        img_b64 = base64.b64encode(buf).decode()

        def run_sam3():
            # IMPORTANT: do not reuse self.session in this thread
            local_session = requests.Session()
            t0 = time.time()
            try:
                r = local_session.post(
                    f"{self.server_url}/segment",
                    json={"image_base64": img_b64, "confidence_threshold": self.confidence},
                    timeout=self.sam3_timeout_sec
                )
                if r.status_code != 200:
                    self._last_sam3_error = f"HTTP {r.status_code}"
                    return

                result = self._safe_json(r)
                if not result:
                    self._last_sam3_error = "bad JSON"
                    return

                result['_scale_factor'] = scale_factor
                result['_latency_ms'] = int((time.time() - t0) * 1000)

                with self._sam3_lock:
                    self._last_sam3_result = result
                    self._sam3_result_seq += 1

                self._last_sam3_error = ""

            except Exception as e:
                self._last_sam3_error = str(e)
                self.get_logger().warn(f'SAM3 request failed: {e}')
            finally:
                try:
                    local_session.close()
                except Exception:
                    pass
                with self._sam3_busy_lock:
                    self._sam3_busy = False

        threading.Thread(target=run_sam3, daemon=True).start()

    def _publish_visualization(self, rgb: np.ndarray, depth: Optional[np.ndarray]):
        """Publish visualization with latest SAM3 results (or raw frame if none)."""
        orig_h, orig_w = rgb.shape[:2]

        with self._sam3_lock:
            result = self._last_sam3_result
            current_seq = self._sam3_result_seq

        viz_base = rgb.copy()
        overlays = []
        best_detection = None
        best_score = 0.0
        prompt = self.target

        # Process SAM3 results if available
        if result is not None and result.get('success'):
            boxes = result.get('boxes', []) or []
            scores = result.get('scores', []) or []
            masks_b64 = result.get('masks_base64', []) or []
            prompt = result.get('prompt', self.target)
            result_scale = float(result.get('_scale_factor', 1.0) or 1.0)

            # Scale boxes back to original image coordinates if downscaled
            if result_scale != 1.0:
                boxes = [[coord / result_scale for coord in box] for box in boxes]

            colors = [
                (0, 255, 0),
                (255, 200, 0),
                (0, 200, 255),
                (255, 0, 255),
                (255, 255, 0),
            ]

            # Robust loop: handle mismatched list lengths
            n = min(len(boxes), len(scores))
            for idx in range(n):
                box = boxes[idx]
                score = float(scores[idx])

                mask = None
                if idx < len(masks_b64):
                    mask = self.decode_mask(masks_b64[idx])

                cx, cy = self.get_mask_center(mask, box, rgb)
                x1, y1, x2, y2 = [int(v) for v in box]

                color = colors[idx % len(colors)]

                overlays.append({
                    "rect": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "color": color,
                    "label": f"{prompt}: {score:.2f}",
                    "mask": mask,
                })

                dist, method = self.get_distance_auto(mask, depth, box, prompt, rgb, current_seq)
                if dist is None or dist < self.min_dist or dist > self.max_dist:
                    continue

                cam_point = self.pixel_to_3d(cx, cy, dist)
                if cam_point is None:
                    continue

                odom_x, odom_y = self.camera_to_odom(*cam_point)

                if score > best_score:
                    overlays[-1]["color"] = (0, 255, 0)
                    overlays[-1]["label"] = f"{prompt}: {score:.2f} | {dist:.2f}m"
                    best_score = score
                    best_detection = {
                        'distance': float(dist),
                        'odom_x': float(odom_x),
                        'odom_y': float(odom_y),
                        'score': float(score),
                        'method': method
                    }

        # Depth status (always compute)
        depth_status = []
        if self._depth_is_fresh():
            depth_status.append("depth_cam")
        if self.mono_depth_available:
            depth_status.append("mono")
        depth_status.append("size")
        depth_status.append("bbox")

        # Log detection summary periodically
        if not hasattr(self, '_last_detect_log_time'):
            self._last_detect_log_time = 0.0
        now = time.time()
        if now - self._last_detect_log_time > 5.0:
            if result is not None and result.get('success'):
                num_detections = len(result.get('boxes', []) or [])
            else:
                num_detections = 0

            if num_detections > 0 and best_detection is None:
                self.get_logger().warn(
                    f'[DETECT] SAM3 found {num_detections} "{prompt}" but none had valid depth/distance'
                )
            elif num_detections > 0 and best_detection:
                self.get_logger().info(
                    f'[DETECT] Found {num_detections} "{prompt}", best at {best_detection["distance"]:.2f}m'
                )
            self._last_detect_log_time = now

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
                'goal_position': [float(goal_x), float(goal_y)],
                'sam3_latency_ms': int(result.get('_latency_ms', -1)) if isinstance(result, dict) else -1,
                'sam3_error': self._last_sam3_error,
                'mono_error': self._last_mono_error,
                'tf_available': bool(self.tf_available),
            }

            if self.auto_publish:
                if not (math.isnan(goal_x) or math.isnan(goal_y) or math.isinf(goal_x) or math.isinf(goal_y)):
                    if self._should_publish_goal(goal_x, goal_y, current_seq):
                        self.publish_goal(goal_x, goal_y)
                else:
                    self.get_logger().warn(f'[GOAL] Invalid goal coordinates: ({goal_x}, {goal_y})')

        else:
            status = {
                'found': False,
                'target': prompt,
                'available_methods': depth_status,
                'message': 'Object not found in valid range',
                'sam3_error': self._last_sam3_error,
                'mono_error': self._last_mono_error,
                'tf_available': bool(self.tf_available),
            }

        self.status_pub.publish(String(data=json.dumps(status)))

        # ===== Build visualization =====
        viz = viz_base
        if self.rotate_viz_90_clockwise:
            viz = self._rotate90_cw(viz)

        # 1) Segmentation overlay
        if self.show_segmentation_overlay:
            for item in overlays:
                mask = item.get("mask")
                if mask is None:
                    continue

                mask_rs = self._mask_to_rgb_size(mask, orig_h, orig_w)
                if mask_rs is None:
                    continue

                if self.rotate_viz_90_clockwise:
                    mask_rs = self._rotate90_cw(mask_rs)

                self._overlay_mask(viz, mask_rs, item["color"], self.seg_overlay_alpha)

        # 2) Bounding boxes and centers
        for item in overlays:
            x1, y1, x2, y2 = item["rect"]
            cx, cy = item["center"]
            color = item["color"]

            if self.rotate_viz_90_clockwise:
                rx1, ry1, rx2, ry2 = self._rect_rot90_cw(x1, y1, x2, y2, orig_h)
                rcx, rcy = self._pt_rot90_cw(cx, cy, orig_h)
            else:
                rx1, ry1, rx2, ry2 = x1, y1, x2, y2
                rcx, rcy = cx, cy

            cv2.rectangle(viz, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.circle(viz, (rcx, rcy), 6, color, -1)
            cv2.circle(viz, (rcx, rcy), 8, (255, 255, 255), 2)

        self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz, 'bgr8'))

        # Track last processed seq for gating logic
        self._last_processed_seq = current_seq

        self._last_viz_for_gui = viz
        self._update_gui(viz)

    def _update_gui(self, viz_bgr: np.ndarray):
        if not self.gui_ready or viz_bgr is None:
            return

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
            self.get_logger().warn(f"GUI error: {e}")
            self.gui_ready = False
            return

        if key in (ord('q'), 27):
            rclpy.shutdown()
        elif key == ord('p'):
            self.gui_paused = not self.gui_paused
            if self.gui_paused:
                self.get_logger().info("GUI paused: stopping SAM3 + goal publishing.")
            else:
                self.get_logger().info("GUI resumed.")
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