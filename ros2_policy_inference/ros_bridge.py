"""
Subscribes to the real robot's camera + proprioception topics used as
policy observations, and publishes the policy's predicted EE-offset action.

Each topic is subscribed independently; there is NO time alignment.
get_latest() simply returns the most recent message seen on every topic —
the same approach ros2_data_collector's RosBridge uses, and for the same
reason: robust to topics running at different rates or with
un-synchronised header.stamps.

QoS — camera topics are typically published BEST_EFFORT (sensor-data QoS);
a RELIABLE subscriber silently receives nothing from them, so RGB/depth use
a BEST_EFFORT profile while joint_state/tcp_pose/F-T stay RELIABLE.

Gripper state is intentionally NOT subscribed here — see
PolicyInferenceNode, which feeds the policy's own previous gripper output
back in as the next step's gripper observation instead.
"""
import time

import numpy as np
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray


class RosBridge:
    def __init__(self, node, cfg, on_action_matched=None):
        """on_action_matched(info): called (from the node's executor) each time a subscriber on the
        action topic is matched or unmatched; info is rclpy's MatchedInfo."""
        self._node = node

        self._joint = None
        self._rgb = None
        self._depth = None
        self._pose = None
        self._ft = None

        # topic key -> topic name, for check_health(); last_seen tracked
        # separately (monotonic seconds) so health checks don't depend on
        # any particular clock source.
        self._topics = {
            "joint": cfg.joint_state_topic,
            "rgb": cfg.rgb_topic,
            "depth": cfg.depth_topic,
            "tcp_pose": cfg.tcp_pose_topic,
            "ft": cfg.ft_sensor_topic,
        }
        self._last_seen = {}

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        node.create_subscription(JointState, cfg.joint_state_topic, self._on_joint, 10)
        node.create_subscription(CompressedImage, cfg.rgb_topic, self._on_rgb, sensor_qos)
        node.create_subscription(Image, cfg.depth_topic, self._on_depth, sensor_qos)
        node.create_subscription(PoseStamped, cfg.tcp_pose_topic, self._on_pose, 10)
        node.create_subscription(WrenchStamped, cfg.ft_sensor_topic, self._on_ft, 10)

        self._action_pub = node.create_publisher(
            Float64MultiArray, cfg.action_topic, 10,
            event_callbacks=PublisherEventCallbacks(matched=on_action_matched) if on_action_matched else None,
        )

        log = node.get_logger()
        log.info("RosBridge topics:")
        log.info(f"  joint    = {cfg.joint_state_topic}")
        log.info(f"  rgb      = {cfg.rgb_topic}  (CompressedImage, BEST_EFFORT)")
        log.info(f"  depth    = {cfg.depth_topic}  (BEST_EFFORT)")
        log.info(f"  tcp_pose = {cfg.tcp_pose_topic}")
        log.info(f"  ft       = {cfg.ft_sensor_topic}")
        log.info(f"  action   = {cfg.action_topic}  (Float64MultiArray, published)")

    # ---------- per-topic callbacks ----------

    def _on_joint(self, msg):
        joint_pos = np.asarray(msg.position, dtype=np.float32)
        joint_vel = (
            np.asarray(msg.velocity, dtype=np.float32)
            if len(msg.velocity) == len(msg.position)
            else np.zeros_like(joint_pos)
        )
        self._joint = {"joint_pos": joint_pos, "joint_vel": joint_vel}
        self._last_seen["joint"] = time.monotonic()

    def _on_rgb(self, msg):
        self._rgb = {"rgb": self._compressed_msg_to_array(msg)}
        self._last_seen["rgb"] = time.monotonic()

    def _on_depth(self, msg):
        self._depth = {"depth": self._image_msg_to_array(msg)}
        self._last_seen["depth"] = time.monotonic()

    def _on_pose(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        self._pose = {
            "ee_pos": np.array([p.x, p.y, p.z], dtype=np.float64),
            "ee_quat": np.array([o.x, o.y, o.z, o.w], dtype=np.float64),  # xyzw
        }
        self._last_seen["tcp_pose"] = time.monotonic()

    def _on_ft(self, msg):
        f = msg.wrench.force
        t = msg.wrench.torque
        self._ft = {
            "ft_wrench": np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32),
        }
        self._last_seen["ft"] = time.monotonic()

    def publish_action(self, action: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in action]
        self._action_pub.publish(msg)

    @staticmethod
    def _image_msg_to_array(msg):
        h, w, enc = msg.height, msg.width, msg.encoding

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[..., :3]
            if enc == "bgra8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("16UC1", "mono16"):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()

        if enc == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()

        if enc == "mono8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()

        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1).copy()

    @staticmethod
    def _compressed_msg_to_array(msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)"
            )
        return np.ascontiguousarray(bgr[..., ::-1])

    def get_latest(self):
        """Most recent message from every topic, merged into one dict.
        Returns None until joint + rgb + depth + tcp_pose + ft have each
        been received at least once."""
        if (self._joint is None or self._rgb is None or self._depth is None
                or self._pose is None or self._ft is None):
            return None
        out = {}
        out.update(self._joint)
        out.update(self._rgb)
        out.update(self._depth)
        out.update(self._pose)
        out.update(self._ft)
        return out

    def check_health(self, timeout_sec):
        """{topic_name: status} for every subscribed topic, status one of:
          - "ok"           received a message within the last timeout_sec
          - "no_publisher" nothing on the ROS graph is publishing this topic
          - "no_data"      a publisher exists, but no message has arrived yet
                           (e.g. a QoS mismatch — publisher BEST_EFFORT vs a
                           RELIABLE subscription silently drops everything)
          - "stale"        was receiving messages, but not within timeout_sec
                           (e.g. the camera/driver died mid-session)
        """
        now = time.monotonic()
        result = {}
        for key, topic in self._topics.items():
            last = self._last_seen.get(key)
            if last is None:
                publishers = self._node.count_publishers(topic)
                status = "no_data" if publishers > 0 else "no_publisher"
            elif now - last > timeout_sec:
                status = "stale"
            else:
                status = "ok"
            result[topic] = status
        return result
