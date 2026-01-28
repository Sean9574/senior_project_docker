#!/usr/bin/env python3
"""
SAM3 Room Scanner Node

This node accumulates segmentation results over time to build a semantic map
of detected objects in the room. Works with both simulation and real robot.

Features:
    - Scans room while robot moves
    - Tracks unique object instances
    - Publishes semantic point cloud
    - Saves room inventory to file

Usage:
    ros2 run senior_project sam3_room_scanner --ros-args \
        -p prompts:="['chair', 'table', 'couch', 'person', 'door']" \
        -p scan_duration:=60.0
"""

import base64
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import requests
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class DetectedObject:
    """Represents a detected object instance"""
    class_name: str
    confidence: float
    position: np.ndarray  # 3D position in world frame
    bbox_2d: List[float]  # [x1, y1, x2, y2] in image
    first_seen: float
    last_seen: float
    observation_count: int = 1
    
    def update(self, position: np.ndarray, confidence: float):
        """Update object with new observation"""
        # Running average of position
        self.position = (self.position * self.observation_count + position) / (self.observation_count + 1)
        self.confidence = max(self.confidence, confidence)
        self.last_seen = time.time()
        self.observation_count += 1


class RoomScanner(Node):
    """Scans room and builds semantic map using SAM3"""

    def __init__(self):
        super().__init__("sam3_room_scanner")

        # Parameters
        self.declare_parameter("server_url", "http://localhost:8100")
        self.declare_parameter("prompts", ["furniture", "door", "window", "person"])
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("scan_rate", 2.0)  # Hz
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("position_threshold", 0.5)  # meters - merge threshold
        self.declare_parameter("output_file", "room_inventory.json")

        self.server_url = self.get_parameter("server_url").value
        self.prompts = self.get_parameter("prompts").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.scan_rate = self.get_parameter("scan_rate").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.position_threshold = self.get_parameter("position_threshold").value
        self.output_file = self.get_parameter("output_file").value

        # State
        self.detected_objects: Dict[str, List[DetectedObject]] = defaultdict(list)
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.current_pose = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, self.camera_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, "~/object_markers", 10)
        self.status_pub = self.create_publisher(String, "~/scan_status", 10)
        self.inventory_pub = self.create_publisher(String, "~/inventory", 10)

        # Timer for scanning
        self.scan_timer = self.create_timer(1.0 / self.scan_rate, self.scan_callback)

        self.get_logger().info(f"Room Scanner started")
        self.get_logger().info(f"  Prompts: {self.prompts}")
        self.get_logger().info(f"  Rate: {self.scan_rate} Hz")

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.latest_rgb_header = msg.header

    def depth_callback(self, msg):
        # Handle different depth encodings
        if msg.encoding == "16UC1":
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        else:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def get_camera_to_world_transform(self) -> Optional[TransformStamped]:
        """Get transform from camera frame to world frame"""
        try:
            # Try common camera frames
            for frame in ["camera_color_optical_frame", "camera_link", "d435i_color_optical_frame"]:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "odom",  # or "map"
                        frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    return transform
                except TransformException:
                    continue
            return None
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")
            return None

    def pixel_to_3d(self, u: int, v: int, depth: float) -> Optional[np.ndarray]:
        """Convert pixel coordinates to 3D point in camera frame"""
        if self.camera_info is None or depth <= 0 or np.isnan(depth):
            return None

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def transform_point_to_world(self, point_camera: np.ndarray, transform: TransformStamped) -> np.ndarray:
        """Transform point from camera frame to world frame"""
        # Get translation
        t = transform.transform.translation
        # Get rotation as quaternion
        q = transform.transform.rotation

        # Convert quaternion to rotation matrix (simplified)
        # For proper implementation, use tf_transformations
        from tf_transformations import quaternion_matrix
        
        rot_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        translation = np.array([t.x, t.y, t.z])

        return rot_matrix @ point_camera + translation

    def find_or_create_object(self, class_name: str, position: np.ndarray, 
                               confidence: float, bbox: List[float]) -> DetectedObject:
        """Find existing object instance or create new one"""
        # Check for nearby existing object of same class
        for obj in self.detected_objects[class_name]:
            dist = np.linalg.norm(obj.position - position)
            if dist < self.position_threshold:
                obj.update(position, confidence)
                return obj

        # Create new object
        new_obj = DetectedObject(
            class_name=class_name,
            confidence=confidence,
            position=position,
            bbox_2d=bbox,
            first_seen=time.time(),
            last_seen=time.time(),
        )
        self.detected_objects[class_name].append(new_obj)
        self.get_logger().info(f"New object detected: {class_name} at {position}")
        return new_obj

    def scan_callback(self):
        """Main scanning loop"""
        if self.latest_rgb is None:
            return

        # Get transform
        transform = self.get_camera_to_world_transform()

        # Encode image
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        # Query SAM3 for each prompt
        try:
            response = requests.post(
                f"{self.server_url}/segment/batch",
                json={
                    "image_base64": image_b64,
                    "prompts": self.prompts,
                    "confidence_threshold": self.confidence_threshold,
                },
                timeout=15,
            )

            if response.status_code != 200:
                return

            result = response.json()
            if not result.get("success"):
                return

            # Process each prompt's results
            for prompt, data in result.get("results", {}).items():
                boxes = data.get("boxes", [])
                scores = data.get("scores", [])

                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    center_u = int((x1 + x2) / 2)
                    center_v = int((y1 + y2) / 2)

                    # Get depth at center
                    position = None
                    if self.latest_depth is not None:
                        h, w = self.latest_depth.shape
                        if 0 <= center_v < h and 0 <= center_u < w:
                            depth = self.latest_depth[center_v, center_u]
                            point_camera = self.pixel_to_3d(center_u, center_v, depth)

                            if point_camera is not None and transform is not None:
                                position = self.transform_point_to_world(point_camera, transform)
                            elif point_camera is not None:
                                position = point_camera  # Use camera frame if no transform

                    if position is not None:
                        self.find_or_create_object(prompt, position, score, box)

            # Publish markers and inventory
            self.publish_markers()
            self.publish_inventory()

        except Exception as e:
            self.get_logger().warn(f"Scan error: {e}")

    def publish_markers(self):
        """Publish visualization markers for detected objects"""
        marker_array = MarkerArray()
        marker_id = 0

        # Color map for different classes
        colors = {
            "chair": (1.0, 0.0, 0.0),
            "table": (0.0, 1.0, 0.0),
            "couch": (0.0, 0.0, 1.0),
            "person": (1.0, 1.0, 0.0),
            "door": (1.0, 0.5, 0.0),
            "window": (0.0, 1.0, 1.0),
            "furniture": (0.5, 0.5, 0.5),
        }

        for class_name, objects in self.detected_objects.items():
            color = colors.get(class_name, (0.5, 0.5, 0.5))

            for obj in objects:
                # Sphere marker
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = class_name
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(obj.position[0])
                marker.pose.position.y = float(obj.position[1])
                marker.pose.position.z = float(obj.position[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8
                marker_array.markers.append(marker)
                marker_id += 1

                # Text label
                text_marker = Marker()
                text_marker.header.frame_id = "odom"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = class_name + "_text"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(obj.position[0])
                text_marker.pose.position.y = float(obj.position[1])
                text_marker.pose.position.z = float(obj.position[2]) + 0.3
                text_marker.text = f"{class_name} ({obj.confidence:.2f})"
                text_marker.scale.z = 0.15
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                marker_array.markers.append(text_marker)
                marker_id += 1

        self.marker_pub.publish(marker_array)

    def publish_inventory(self):
        """Publish and save room inventory"""
        inventory = {}
        for class_name, objects in self.detected_objects.items():
            inventory[class_name] = [
                {
                    "position": obj.position.tolist(),
                    "confidence": obj.confidence,
                    "observation_count": obj.observation_count,
                }
                for obj in objects
            ]

        # Publish as JSON
        msg = String()
        msg.data = json.dumps(inventory, indent=2)
        self.inventory_pub.publish(msg)

        # Summary status
        total = sum(len(objs) for objs in self.detected_objects.values())
        status_msg = String()
        status_msg.data = f"Detected {total} objects: " + ", ".join(
            f"{len(objs)} {name}" for name, objs in self.detected_objects.items() if objs
        )
        self.status_pub.publish(status_msg)

    def save_inventory(self):
        """Save inventory to file"""
        inventory = {}
        for class_name, objects in self.detected_objects.items():
            inventory[class_name] = [
                {
                    "position": obj.position.tolist(),
                    "confidence": obj.confidence,
                    "observation_count": obj.observation_count,
                    "first_seen": obj.first_seen,
                    "last_seen": obj.last_seen,
                }
                for obj in objects
            ]

        with open(self.output_file, "w") as f:
            json.dump(inventory, f, indent=2)
        
        self.get_logger().info(f"Inventory saved to {self.output_file}")


def main(args=None):
    rclpy.init(args=args)
    node = RoomScanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_inventory()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()