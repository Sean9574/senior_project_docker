#!/usr/bin/env python3
"""
Webcam to ROS Publisher

Publishes laptop webcam to ROS topics for testing SAM3 goal generator.

Usage:
    # Terminal 1: SAM3 server
    python sam3_server.py
    
    # Terminal 2: Webcam publisher + fake TF
    ros2 run senior_project webcam_publisher
    
    # Terminal 3: Fake robot odometry/TF
    ros2 run senior_project test_camera_tf --ros-args -p camera_pitch_deg:=-10.0
    
    # Terminal 4: Goal generator (bbox mode since no depth camera)
    ros2 run senior_project sam3_goal_generator --ros-args \
        -p depth_mode:="bbox" \
        -p target:="cup" \
        -p auto_publish_goal:=false
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        
        # Parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('flip', True)  # Flip horizontally (mirror)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        
        camera_id = self.get_parameter('camera_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.flip = self.get_parameter('flip').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        # Open webcam
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            return
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.get_logger().info(f'Camera opened: {self.width}x{self.height} @ {fps}fps')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers - match RealSense topic names
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        
        # Build camera info (approximate for typical webcam)
        self.camera_info = self.build_camera_info()
        
        # Timer
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        
        self.get_logger().info('Webcam publisher started')
        self.get_logger().info(f'  Publishing to: /camera/color/image_raw')
        self.get_logger().info(f'  Frame: {self.camera_frame}')
        self.get_logger().info(f'  Flip: {self.flip}')
    
    def build_camera_info(self) -> CameraInfo:
        """Build approximate camera info for webcam"""
        info = CameraInfo()
        info.header.frame_id = self.camera_frame
        info.width = self.width
        info.height = self.height
        
        # Approximate focal length for typical webcam (60-70 degree FOV)
        # fx = width / (2 * tan(fov/2))
        fov_deg = 65.0
        fx = self.width / (2 * np.tan(np.radians(fov_deg / 2)))
        fy = fx  # Assume square pixels
        cx = self.width / 2
        cy = self.height / 2
        
        # Intrinsic matrix K
        info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]
        
        # Distortion (assume none)
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = 'plumb_bob'
        
        # Rectification matrix (identity)
        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # Projection matrix P
        info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        return info
    
    def publish_frame(self):
        """Capture and publish frame"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return
        
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        # Get timestamp
        now = self.get_clock().now().to_msg()
        
        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.camera_frame
        self.image_pub.publish(img_msg)
        
        # Publish camera info
        self.camera_info.header.stamp = now
        self.info_pub.publish(self.camera_info)
    
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
