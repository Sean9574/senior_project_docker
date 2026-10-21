#!/usr/bin/env python3
"""
ROS 2 front end for the grid arena: one MuJoCo world, N robots.

Per robot i (namespace r{i}):
  publishes  /r{i}/odom (room frame), /r{i}/scan (10 Hz, same conventions as
             stretch_mujoco_driver), /r{i}/camera/color/image_raw (bgr8),
             /r{i}/camera/depth/image_rect_raw (32FC1, m), /r{i}/camera/color/camera_info
  subscribes /r{i}/cmd_vel (clamped; stops after 0.5 s without a command)
  serves     /r{i}/reset (std_srvs/Trigger): respawn that robot only
  TF         r{i}/odom -> r{i}/camera_color_optical_frame (with each image), so
             the SAM3 goal generator's TF path places detections exactly

Physics runs in its own thread, paced to wall time; camera rendering happens
in that thread too (the GL context belongs to it).
"""

import math
import os
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from stretch_mujoco_driver.arena_core import Arena
from stretch_mujoco_driver.arena_world import build_arena

CMD_TIMEOUT_S = 0.5
TICK_S = 0.01  # physics loop period (wall time)


def _default_robot_xml() -> str:
    from stretch_mujoco import utils
    return os.path.join(os.path.dirname(utils.default_scene_xml_path), "stretch.xml")


def _default_room_xml() -> str:
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("senior_project"), "config", "scene.xml")


class ArenaSimNode(Node):
    def __init__(self):
        super().__init__("arena_sim")
        p = self.declare_parameter
        self.n = int(p("num_robots", 8).value)
        robot_xml = p("robot_xml", "").value or _default_robot_xml()
        room_xml = p("room_xml", "").value or _default_room_xml()
        self.scan_period = 1.0 / float(p("scan_hz", 10.0).value)
        self.odom_period = 1.0 / float(p("odom_hz", 30.0).value)
        camera_hz = float(p("camera_hz", 2.0).value)
        self.camera_period = 1.0 / camera_hz if camera_hz > 0 else float("inf")
        self.use_cameras = str(p("use_cameras", True).value).lower() == "true"
        random_spawn = str(p("random_spawn", True).value).lower() == "true"
        seed = int(p("seed", 0).value)
        # 4 ms (the model's default is 2 ms): measured identical driving speed
        # and twice the real-time headroom with 8 robots in one process
        timestep = float(p("physics_timestep", 0.004).value)

        t0 = time.time()
        world = build_arena(robot_xml, room_xml, self.n,
                            room_size=float(p("room_size", 10.0).value),
                            gap=float(p("room_gap", 2.0).value),
                            wall_height=float(p("wall_height", 2.5).value))
        world.model.opt.timestep = timestep
        self.arena = Arena(world, rng=np.random.default_rng(seed or None),
                           clearance=float(p("spawn_clearance", 1.0).value))
        self.random_spawn = random_spawn
        if random_spawn:
            for i in range(self.n):
                self.arena.respawn(i)
        self.get_logger().info(
            f"[ARENA] {self.n} robots in {self.n} rooms built in {time.time() - t0:.1f}s "
            f"(timestep {world.model.opt.timestep * 1000:.1f} ms, random_spawn={random_spawn})")

        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_cmd_t = [0.0] * self.n
        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_pub, self.scan_pub, self.rgb_pub, self.depth_pub, self.info_pub = [], [], [], [], []
        for i in range(self.n):
            ns = f"/r{i}"
            self.odom_pub.append(self.create_publisher(Odometry, f"{ns}/odom", 10))
            self.scan_pub.append(self.create_publisher(LaserScan, f"{ns}/scan", qos_profile_sensor_data))
            if self.use_cameras:
                self.rgb_pub.append(self.create_publisher(Image, f"{ns}/camera/color/image_raw", cam_qos))
                self.depth_pub.append(self.create_publisher(Image, f"{ns}/camera/depth/image_rect_raw", cam_qos))
                self.info_pub.append(self.create_publisher(CameraInfo, f"{ns}/camera/color/camera_info", cam_qos))
            self.create_subscription(Twist, f"{ns}/cmd_vel", lambda msg, i=i: self._on_cmd(i, msg), 10)
            self.create_service(Trigger, f"{ns}/reset", lambda req, res, i=i: self._on_reset(i, req, res))

        self._running = True
        self._thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ callbacks
    def _on_cmd(self, i: int, msg: Twist):
        with self.lock:
            self.arena.set_cmd(i, msg.linear.x, msg.angular.z)
            self.last_cmd_t[i] = time.time()

    def _on_reset(self, i: int, request, response):
        with self.lock:
            if self.random_spawn:
                x, y, yaw = self.arena.respawn(i)
            else:
                x, y, yaw = 0.0, 0.0, 0.0
                self.arena.respawn_at(i, x, y, yaw)
        self.get_logger().info(f"[SPAWN] r{i} x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.0f} deg")
        response.success = True
        response.message = f"r{i} respawned at ({x:.2f}, {y:.2f}, {yaw:.2f})"
        return response

    # ------------------------------------------------------------ physics thread
    def _physics_loop(self):
        model = self.arena.model
        substeps = max(int(round(TICK_S / model.opt.timestep)), 1)
        cameras = self._make_cameras() if self.use_cameras else None
        next_odom = next_scan = next_cam = next_rtf_log = time.time()
        sim_s, wall_start = 0.0, time.time()
        next_tick = time.time()
        while self._running and rclpy.ok():
            now = time.time()
            with self.lock:
                for i in range(self.n):
                    if self.last_cmd_t[i] and now - self.last_cmd_t[i] > CMD_TIMEOUT_S:
                        self.arena.set_cmd(i, 0.0, 0.0)
                        self.last_cmd_t[i] = 0.0
                self.arena.step(substeps)
                sim_s += substeps * model.opt.timestep
                if now >= next_odom:
                    self._publish_odom()
                    next_odom = now + self.odom_period
                if now >= next_scan:
                    self._publish_scans()
                    next_scan = now + self.scan_period
                if cameras is not None and now >= next_cam:
                    self._publish_cameras(cameras)
                    next_cam = now + self.camera_period
            if now >= next_rtf_log:
                wall = now - wall_start
                if wall > 5.0 and sim_s / wall < 0.9:
                    self.get_logger().warn(f"[ARENA] physics behind wall time: {sim_s / wall:.2f}x real time")
                next_rtf_log = now + 30.0
            next_tick += TICK_S
            delay = next_tick - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.time()  # behind: don't try to catch up in a burst

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _publish_odom(self):
        stamp = self._stamp()
        for i in range(self.n):
            x, y, yaw, v, w = self.arena.odom(i)
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = f"r{i}/odom"
            msg.child_frame_id = f"r{i}/base_link"
            msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.twist.twist.linear.x = v
            msg.twist.twist.angular.z = w
            self.odom_pub[i].publish(msg)

    def _publish_scans(self):
        stamp = self._stamp()
        for i in range(self.n):
            ranges = self.arena.scan(i)
            msg = LaserScan()  # same layout as stretch_mujoco_driver.create_laser_scan_msg
            msg.header.stamp = stamp
            msg.header.frame_id = f"r{i}/laser"
            msg.angle_min = 0.0
            msg.angle_max = 2.0 * math.pi
            msg.angle_increment = msg.angle_max / len(ranges)
            msg.range_min = 0.2
            msg.range_max = 20.0
            msg.ranges = ranges.astype(float).tolist()
            self.scan_pub[i].publish(msg)

    # ------------------------------------------------------------ cameras
    def _make_cameras(self):
        """Renderers + intrinsics exactly as stretch_mujoco's camera manager
        sets up the head D435i (fov, intrinsics, resolution, depth limit)."""
        import mujoco
        from stretch_mujoco import config, utils
        from stretch_mujoco.enums.stretch_cameras import StretchCameras

        model = self.arena.model
        settings = StretchCameras.cam_d435i_rgb.initial_camera_settings
        for room in self.arena.world.rooms:
            for cam in (room.camera_rgb, room.camera_depth):
                model.cam_fovy[cam] = settings.field_of_view_vertical_in_degrees
                model.cam_intrinsic[cam] = list(settings.focal) + [0, 0]
                model.cam_resolution[cam] = (settings.sensor_resolution if settings.sensor_resolution
                                             is not None else (settings.width, settings.height))
                if settings.sensor_size is not None:
                    model.cam_sensorsize[cam] = settings.sensor_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, settings.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, settings.height)
        rgb = mujoco.Renderer(model, width=settings.width, height=settings.height)
        depth = mujoco.Renderer(model, width=settings.width, height=settings.height)
        depth.enable_depth_rendering()
        info = CameraInfo()
        info.width, info.height = settings.width, settings.height
        info.distortion_model = "plumb_bob"
        info.d = settings.get_distortion_params_d()
        info.k = settings.get_intrinsic_params_k()
        info.p = settings.get_projection_matrix_p()
        limit = config.depth_limits["d435i"]
        return rgb, depth, info, (lambda d: utils.limit_depth_distance(d, limit))

    def _publish_cameras(self, cameras):
        rgb_r, depth_r, info, limit_depth = cameras
        stamp = self._stamp()
        transforms = []
        for i, room in enumerate(self.arena.world.rooms):
            frame = f"r{i}/camera_color_optical_frame"
            rgb_r.update_scene(self.arena.data, camera=room.camera_rgb)
            bgr = np.ascontiguousarray(rgb_r.render()[:, :, ::-1])
            depth_r.update_scene(self.arena.data, camera=room.camera_depth)
            depth = limit_depth(depth_r.render()).astype(np.float32)
            img = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            dmsg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            for m in (img, dmsg, info):
                m.header.stamp = stamp
                m.header.frame_id = frame
            self.rgb_pub[i].publish(img)
            self.depth_pub[i].publish(dmsg)
            self.info_pub[i].publish(info)
            transforms.append(self._camera_tf(i, stamp, frame))
        self.tf_broadcaster.sendTransform(transforms)

    def _camera_tf(self, i: int, stamp, frame: str) -> TransformStamped:
        pos, quat = self.arena.camera_pose(i)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = f"r{i}/odom"
        tf.child_frame_id = frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
        (tf.transform.rotation.x, tf.transform.rotation.y,
         tf.transform.rotation.z, tf.transform.rotation.w) = map(float, quat)
        return tf

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = ArenaSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
