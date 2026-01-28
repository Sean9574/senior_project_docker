#!/usr/bin/env python3
"""
SAM3 + Depth Node
Detects objects with SAM3, calculates distance with depth camera,
and publishes goal to RL navigation system.

Subscribes:
  - /camera/color/image_raw (RGB)
  - /camera/depth/image_rect_raw (Depth)
  - /camera/color/camera_info (Camera intrinsics)
  - /stretch/odom (Robot odometry)
  - /tf (Transforms)

Publishes:
  - /stretch/goal (PointStamped) - Goal for RL navigation
  - ~/visualization (Image) - Debug visualization
  - ~/detections (String) - JSON detection info
"""

import base64
import json
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import requests
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker


@dataclass 
class Detection:
    label: str
    confidence: float
    bbox: List[int]
    distance: float
    position_camera: Optional[Tuple[float, float, float]]  # x,y,z in camera frame
    position_odom: Optional[Tuple[float, float]]  # x,y in odom frame


class SAM3DepthNode(Node):
    def __init__(self):
        super().__init__('sam3_depth_node')
        
        # Parameters
        self.declare_parameter('server_url', 'http://localhost:8100')
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('rate', 2.0)  # Hz
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('goal_topic', '/stretch/goal')
        self.declare_parameter('auto_send_goal', False)  # Set True to auto-navigate
        self.declare_parameter('min_distance', 0.3)  # Minimum distance to send goal (meters)
        self.declare_parameter('max_distance', 5.0)  # Maximum distance to send goal (meters)
        
        self.server_url = self.get_parameter('server_url').value
        self.confidence = self.get_parameter('confidence_threshold').value
        self.rate = self.get_parameter('rate').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_