#!/usr/bin/env python3
"""
SAM3 Goal Generator
Finds objects with SAM3 + depth camera, publishes goal for RL navigation.

Usage:
    ros2 run senior_project sam3_goal_generator --ros-args -p target:="cup"
    
Then the robot will navigate to the cup!
"""

import base64
import math
import threading
import json
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import requests
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker


class SAM3GoalGenerator(Node):
    def __init__(self):
        super().__init__('sam3_goal_generator')
        
        # Parameters
        self.declare_parameter('server_url', 'http://localhost:8100')
        self.declare_parameter('target', 'cup')  # What to find
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('ns', 'stretch')
        self.declare_parameter('min_distance', 0.5)
        self.declare_parameter('max_distance', 5.0)
        self.declare_parameter('goal_offset', 0.5)  # Stop this far from object
        self.declare_parameter('auto_publish_goal', True)
        
        self.server_url = self.get_parameter('server_url').value
        self.target = self.get_parameter('target').value
        self.confidence = self.get_parameter('confidence_threshold').value
        self.rate = self.get_parameter('rate').value
        self.ns = self.get_parameter('ns').value
        self.min_dist = self.get_parameter('min_distance').value
        self.max_dist = self.get_parameter('max_distance').value
        self.goal_offset = self.get_parameter('goal_offset').value
        self.auto_publish = self.get_parameter('auto_publish_goal').value
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # State
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.goal_sent = False
        
        # QoS for camera
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.create_subscription(
            Image, f'/camera/color/image_raw', 
            self.rgb_callback, qos)
        self.create_subscription(
            Image, f'/camera/depth/image_rect_raw',
            self.depth_callback, qos)
        self.create_subscription(
            CameraInfo, f'/camera/color/camera_info',
            self.camera_info_callback, qos)
        self.create_subscription(
            Odometry, f'/{self.ns}/odom',
            self.odom_callback, 10)
        self.create_subscription(
            String, '~/set_target',
            self.target_callback, 10)
        
        # Publishers
        self.goal_pub = self.create_publisher(
            PointStamped, f'/{self.ns}/goal', 10)
        self.marker_pub = self.create_publisher(
            Marker, f'/{self.ns}/goal_marker', 10)
        self.viz_pub = self.create_publisher(
            Image, '~/visualization', 10)
        self.status_pub = self.create_publisher(
            String, '~/status', 10)
        
        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self.process_callback)
        
        # Set SAM3 prompt
        self.set_sam3_prompt(self.target)
        
        self.get_logger().info(f'SAM3 Goal Generator started')
        self.get_logger().info(f'  Target: "{self.target}"')
        self.get_logger().info(f'  Server: {self.server_url}')
        self.get_logger().info(f'  Auto-publish goal: {self.auto_publish}')
    
    def set_sam3_prompt(self, target: str):
        """Set the SAM3 server prompt"""
        try:
            r = requests.post(
                f"{self.server_url}/prompt/{target.replace(' ', '%20')}",
                timeout=5
            )
            self.get_logger().info(f'SAM3 prompt set to: "{target}"')
        except Exception as e:
            self.get_logger().warn(f'Could not set SAM3 prompt: {e}')
    
    def target_callback(self, msg: String):
        """Change target object dynamically"""
        self.target = msg.data
        self.goal_sent = False  # Allow new goal
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
    
    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
    
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    
    def pixel_to_3d(self, u: int, v: int, depth: float) -> Optional[Tuple[float, float, float]]:
        """Convert pixel + depth to 3D point in camera frame"""
        if self.camera_info is None or depth <= 0 or np.isnan(depth):
            return None
        
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        # Camera frame: z forward, x right, y down
        z = depth
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        
        return (x, y, z)
    
    def camera_to_odom(self, cam_x: float, cam_y: float, cam_z: float) -> Tuple[float, float]:
        """
        Convert camera frame point to odom frame (2D ground position).
        Camera: z forward, x right, y down
        Robot: x forward, y left
        """
        # Camera z (forward) -> Robot x (forward)
        # Camera x (right) -> Robot -y (left)
        robot_frame_x = cam_z  # Forward distance
        robot_frame_y = -cam_x  # Left/right
        
        # Rotate by robot yaw and add robot position
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)
        
        odom_x = self.robot_x + robot_frame_x * cos_yaw - robot_frame_y * sin_yaw
        odom_y = self.robot_y + robot_frame_x * sin_yaw + robot_frame_y * cos_yaw
        
        return (odom_x, odom_y)
    
    def get_mask_center_depth(self, mask: np.ndarray, depth_img: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """Get center pixel and median depth of masked region"""
        if mask is None or depth_img is None:
            return None, None, None
        
        # Resize mask to match depth
        if mask.shape != depth_img.shape:
            mask = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]))
        
        mask_bool = mask > 127
        ys, xs = np.where(mask_bool)
        
        if len(xs) == 0:
            return None, None, None
        
        # Center of mask
        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))
        
        # Median depth in masked region
        depths = depth_img[mask_bool]
        valid = (depths > 0.1) & (depths < 10.0) & (~np.isnan(depths))
        
        if not np.any(valid):
            return center_x, center_y, None
        
        median_depth = np.median(depths[valid])
        return center_x, center_y, median_depth
    
    def decode_mask(self, mask_b64: str) -> Optional[np.ndarray]:
        """Decode base64 mask"""
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            return cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        except:
            return None
    
    def publish_goal(self, x: float, y: float):
        """Publish goal to RL system"""
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.point.x = x
        goal.point.y = y
        goal.point.z = 0.0
        self.goal_pub.publish(goal)
        
        # Also publish marker for RViz
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
    
    def process_callback(self):
        """Main processing loop"""
        with self.lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
        
        # Send to SAM3
        _, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(buf).decode()
        
        try:
            r = requests.post(
                f"{self.server_url}/segment",
                json={"image_base64": img_b64, "confidence_threshold": self.confidence},
                timeout=10
            )
            result = r.json()
        except Exception as e:
            self.get_logger().warn(f'SAM3 request failed: {e}')
            return
        
        if not result.get('success'):
            return
        
        boxes = result.get('boxes', [])
        scores = result.get('scores', [])
        masks_b64 = result.get('masks_base64', [])
        prompt = result.get('prompt', self.target)
        
        # Process detections
        best_detection = None
        best_score = 0
        
        viz = rgb.copy()
        
        for i, (box, score, mask_b64) in enumerate(zip(boxes, scores, masks_b64)):
            mask = self.decode_mask(mask_b64)
            cx, cy, dist = self.get_mask_center_depth(mask, depth)
            
            if dist is None:
                continue
            
            # Filter by distance
            if dist < self.min_dist or dist > self.max_dist:
                continue
            
            # Convert to odom frame
            cam_point = self.pixel_to_3d(cx, cy, dist)
            if cam_point is None:
                continue
            
            odom_x, odom_y = self.camera_to_odom(*cam_point)
            
            # Draw on visualization
            x1, y1, x2, y2 = [int(v) for v in box]
            color = (0, 255, 0) if score > best_score else (255, 255, 0)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            label = f"{prompt}: {score:.2f} | {dist:.2f}m"
            cv2.putText(viz, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Track best detection
            if score > best_score:
                best_score = score
                best_detection = {
                    'distance': dist,
                    'odom_x': odom_x,
                    'odom_y': odom_y,
                    'score': score
                }
        
        # Publish status
        if best_detection:
            # Calculate goal position (offset from object toward robot)
            dx = best_detection['odom_x'] - self.robot_x
            dy = best_detection['odom_y'] - self.robot_y
            dist_to_obj = math.sqrt(dx*dx + dy*dy)
            
            if dist_to_obj > self.goal_offset:
                # Goal is offset meters before the object
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
                'object_position': [best_detection['odom_x'], best_detection['odom_y']],
                'goal_position': [goal_x, goal_y]
            }
            
            # Draw goal on viz
            cv2.putText(viz, f"GOAL: ({goal_x:.1f}, {goal_y:.1f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Auto-publish goal
            if self.auto_publish and not self.goal_sent:
                self.publish_goal(goal_x, goal_y)
                self.goal_sent = True
        else:
            status = {
                'found': False,
                'target': prompt,
                'message': 'Object not found in valid range'
            }
        
        # Publish status
        self.status_pub.publish(String(data=json.dumps(status)))
        
        # Publish visualization
        cv2.putText(viz, f"Target: {prompt} | Robot: ({self.robot_x:.1f}, {self.robot_y:.1f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        viz_msg = self.bridge.cv2_to_imgmsg(viz, 'bgr8')
        self.viz_pub.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3GoalGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()