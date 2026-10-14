"""
Standalone ROS2 policy-inference node: subscribes to the real robot's
camera + proprioception topics, preprocesses them into the observation
format an IL/VLA policy was trained on, runs one prediction per timer
tick, and publishes the resulting EE-offset action.

Gripper state is not subscribed from the robot: the arm is always open
before an episode starts, and afterwards the policy's own previous
gripper output is fed back in as the next step's gripper observation
(see _gripper_state / _build_obs).

To deploy a trained model, add a PolicyModel subclass under policies/ (see
policies/base_policy.py for the interface), register it in
policies/__init__.py's create_policy(), and select it via the policy_name
parameter (config/params.yaml or `-p policy_name:=...`).
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Int32

from .policies import create_policy
from .policies.base_policy import ACTION_DIM
from .preprocessing import apply_ee_offset, resize_image
from .ros_bridge import RosBridge
from .state_range import load_state_range, range_normalize

_STATE_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]

_DEFAULT_PARAMS = dict(
    joint_state_topic="/left_arm/joint_states",
    rgb_topic="/camera_L/color/image_raw/compressed",
    depth_topic="/camera_L/depth/aligned_depth_to_color",
    tcp_pose_topic="/left_arm/tcp_pose_world",
    ft_sensor_topic="/left_ft/wrench_compensated",
    action_topic="/il/grasp/action",
    # Which policies/*.py implementation to run — see policies/__init__.py.
    policy_name="dummy",
    # Checkpoint override for act / depth_act / diffusion* / depth_inference*: "" = the policy's config.yaml
    # checkpoint_dir; otherwise a folder name such as diffusion_50k (policies.resolve_checkpoint).
    checkpoint="",
    inference_rate_hz=10.0,
    topic_timeout_sec=2.0,
    image_size=256,
    ee_pos_offset=[0.0, 0.0, 0.2204],
    ee_rot_offset_xyzw=[0.70711, 0.0, 0.70711, 0.0],
    initial_gripper_open=True,
    gripper_open_threshold=0.5,
    # A close (action[6] < gripper_open_threshold) is published only once it has been predicted
    # this many ticks in a row; until then action[6] is replaced by the open value. 1 = off.
    gripper_debounce_ticks=8,
    # Published by the sim (delta_extension) when the IL grasp step starts;
    # the node logs its state and the next handoff_log_actions actions.
    handoff_topic="/il/grasp/handoff",
    handoff_log_actions=10,
    # At a hand-off, reset the policy (action queue / chunks, observation
    # history) and the fed-back gripper state to initial_gripper_open, so the
    # policy starts like a fresh episode instead of carrying over what it
    # predicted while the scripted approach was moving the arm.
    reset_on_handoff=True,
    # Hand-off without any extension change: the extension's IL grasp step creates its subscription
    # on action_topic when it starts (and destroys it when it ends), so a NEW subscription from one of
    # these nodes (name or name prefix) counts as a hand-off, exactly like a message on handoff_topic.
    # Detected from the action publisher's matched event, not by polling.
    handoff_on_subscriber=True,
    handoff_subscriber_nodes=["delta_extension_node", "delta_il_sim_rollout_"],
    # Hand-off signals closer together than this are one hand-off (an extension that both creates its
    # subscription and publishes on handoff_topic at the IL start): reset once, not twice.
    handoff_dedupe_sec=1.0,
    # Grad-CAM (gradcam.py): which image regions drive the predicted actions, published as an
    # overlay image on gradcam_topic. Runs on its own thread at up to gradcam_rate_hz, so the
    # control loop never waits for it. Only for act, depth_act, diffusion*, depth_inference*.
    gradcam_enable=False,
    gradcam_rate_hz=5.0,
    gradcam_target="motion",
    gradcam_horizon=0,
    gradcam_topic="/policy/gradcam/image",
)


@dataclass
class InferenceConfig:
    joint_state_topic: str
    rgb_topic: str
    depth_topic: str
    tcp_pose_topic: str
    ft_sensor_topic: str
    action_topic: str
    policy_name: str
    checkpoint: str
    inference_rate_hz: float
    topic_timeout_sec: float
    image_size: int
    ee_pos_offset: list
    ee_rot_offset_xyzw: list
    initial_gripper_open: bool
    gripper_open_threshold: float
    gripper_debounce_ticks: int
    handoff_topic: str
    handoff_log_actions: int
    reset_on_handoff: bool
    handoff_on_subscriber: bool
    handoff_subscriber_nodes: list
    handoff_dedupe_sec: float
    gradcam_enable: bool
    gradcam_rate_hz: float
    gradcam_target: str
    gradcam_horizon: int
    gradcam_topic: str


class PolicyInferenceNode(Node):
    def __init__(self, policy=None):
        super().__init__("policy_inference_node")
        cfg = self._load_config()
        self.cfg = cfg

        self._ee_pos_offset = np.array(cfg.ee_pos_offset, dtype=np.float64)
        self._ee_rot_offset = Rotation.from_quat(cfg.ee_rot_offset_xyzw)

        # Subscriber-triggered hand-off state (see _on_action_matched); set before RosBridge creates
        # the action publisher, whose matched event may fire as soon as the node spins.
        self._handoff_pending = None
        self._handoff_last = None           # (id, monotonic time) of the last hand-off queued
        self._sub_handoff_count = 0
        self._known_action_subs = {}        # endpoint gid -> node name, every subscriber seen so far
        self._match_confirm_timer = None
        self._match_confirm_deadline = 0.0
        self.bridge = RosBridge(
            self, cfg, on_action_matched=self._on_action_matched if cfg.handoff_on_subscriber else None)
        self.policy = policy if policy is not None else create_policy(cfg.policy_name, cfg.checkpoint)
        self.policy.reset()

        if cfg.gripper_debounce_ticks < 1:
            raise ValueError(f"gripper_debounce_ticks must be >= 1, got {cfg.gripper_debounce_ticks}")
        self._gripper_state = 1.0 if cfg.initial_gripper_open else 0.0
        # Consecutive ticks whose raw action[6] was a close (see _debounce_gripper).
        self._close_ticks = 0
        # Published-action counter for the gripper trace below (tick 1 = first action).
        self._tick = 0

        rate_hz = cfg.inference_rate_hz
        policy_rate_hz = getattr(self.policy, "inference_rate_hz", None)
        if policy_rate_hz and policy_rate_hz != rate_hz:
            self.get_logger().info(
                f"policy '{cfg.policy_name}' requires {policy_rate_hz} Hz; "
                f"using it instead of inference_rate_hz={rate_hz}"
            )
            rate_hz = policy_rate_hz
        self.create_timer(1.0 / rate_hz, self._on_tick)
        self.create_timer(1.0, self._on_health_check)

        # IL hand-off diagnostics: training range of the state (None if the
        # policy has no lerobot checkpoint), pending hand-off id from the sim,
        # and the hand-off currently being logged.
        try:
            self._state_range = load_state_range(getattr(self.policy, "checkpoint_dir", None))
        except Exception as e:  # diagnostics only, never block inference
            self.get_logger().warn(f"could not read the checkpoint's state range: {e}")
            self._state_range = None
        self._handoff = None
        self.create_subscription(Int32, cfg.handoff_topic, self._on_handoff, 10)
        if cfg.handoff_on_subscriber:
            self.get_logger().info(
                f"hand-off on subscriber: a new {cfg.action_topic} subscription from "
                f"{', '.join(cfg.handoff_subscriber_nodes)}* starts an IL hand-off"
            )

        self.gradcam = self._start_gradcam() if cfg.gradcam_enable else None

    def _start_gradcam(self):
        """Grad-CAM worker publishing overlays on gradcam_topic; None (with a warning) if this
        policy can't be explained. Diagnostics only: a failure here never blocks inference."""
        from sensor_msgs.msg import Image
        from .gradcam import GradCAM, GradCAMWorker

        cfg, log = self.cfg, self.get_logger()
        try:
            gc = GradCAM(self.policy, target=cfg.gradcam_target, horizon=cfg.gradcam_horizon)
        except Exception as e:
            log.warn(f"Grad-CAM disabled: {e}")
            return None
        pub = self.create_publisher(Image, cfg.gradcam_topic, 1)

        def publish(rgb):
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height, msg.width = rgb.shape[:2]
            msg.encoding, msg.step = "rgb8", rgb.shape[1] * 3
            msg.data = rgb.tobytes()
            pub.publish(msg)

        log.info(
            f"Grad-CAM on: target {cfg.gradcam_target}, cameras {', '.join(gc.cameras)}, "
            f"up to {cfg.gradcam_rate_hz:g} Hz -> {cfg.gradcam_topic}"
        )
        return GradCAMWorker(gc, publish, cfg.gradcam_rate_hz, logger=log)

    def _load_config(self):
        for name, default in _DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        values = {name: self.get_parameter(name).value for name in _DEFAULT_PARAMS}
        return InferenceConfig(**values)

    def _on_tick(self):
        self._start_handoff_if_pending()
        ros_data = self.bridge.get_latest()
        if ros_data is None:
            self.get_logger().warn(
                "waiting for all observation topics before first inference...",
                throttle_duration_sec=5.0,
            )
            return

        obs = self._build_obs(ros_data)
        gripper_state_in = self._gripper_state
        action = np.asarray(self.policy.predict(obs), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"policy returned action of shape {action.shape}, "
                f"expected ({ACTION_DIM},)"
            )

        raw_gripper = float(action[6])
        action = self._debounce_gripper(action)

        if self.gradcam is not None:
            self.gradcam.submit(obs)
        self._log_handoff_tick(obs, action, raw_gripper)
        self.bridge.publish_action(action)
        self.get_logger().info(f"action: {np.array2string(action, precision=4)}")
        # Fed back from the PUBLISHED (debounced) value: it is what the sim/robot gripper does, and
        # in the demos the gripper state never flickers closed before the real grasp.
        self._gripper_state = (
            1.0 if action[6] >= self.cfg.gripper_open_threshold else 0.0
        )
        self._tick += 1
        self.get_logger().info(
            f"gripper trace: tick {self._tick} raw action[6]={raw_gripper:.4f} "
            f"close_ticks={self._close_ticks}/{self.cfg.gripper_debounce_ticks} "
            f"published={action[6]:.4f}"
            f"{' (stray close filtered)' if action[6] != raw_gripper else ''} "
            f"gripper_state fed in={gripper_state_in:.0f} next={self._gripper_state:.0f}"
        )

    def _debounce_gripper(self, action):
        """Count consecutive ticks whose raw action[6] is a close (< gripper_open_threshold); any
        open tick resets the count. Until the count reaches gripper_debounce_ticks, action[6] is
        replaced by the open value 1.0, so no consumer sees the close; from then on the raw close
        is published for as long as it lasts (the count only has to reach N once). With
        gripper_debounce_ticks=1 every action passes through unchanged."""
        if action[6] < self.cfg.gripper_open_threshold:
            self._close_ticks += 1
        else:
            self._close_ticks = 0
        if 0 < self._close_ticks < self.cfg.gripper_debounce_ticks:
            action = action.copy()
            action[6] = 1.0
        return action

    # ---------- IL hand-off diagnostics ----------

    def _on_handoff(self, msg):
        """The sim's IL grasp step started; acted on at the start of the next tick."""
        self._queue_handoff(msg.data, f"message on {self.cfg.handoff_topic}")

    def _on_action_matched(self, info):
        """Matched event of the action publisher: a subscriber appeared or went away. Look up which
        node it belongs to; the graph can learn the node name a little after the DDS match (more so
        across DDS vendors), so if a new subscriber isn't visible yet, re-check every 50 ms for up
        to 2 s. That timer only exists after an event and is destroyed once the check is done."""
        self.get_logger().info(
            f"{self.cfg.action_topic}: subscriber {'matched' if info.current_count_change > 0 else 'unmatched'} "
            f"(now {info.current_count})"
        )
        if info.current_count_change > 0:
            self._match_confirm_deadline = time.monotonic() + 2.0
        if self._check_action_subscribers() or info.current_count_change <= 0:
            return
        if self._match_confirm_timer is None:
            self._match_confirm_timer = self.create_timer(0.05, self._on_match_confirm)

    def _on_match_confirm(self):
        found = self._check_action_subscribers()
        if found or time.monotonic() > self._match_confirm_deadline:
            if not found:
                self.get_logger().warn(
                    f"{self.cfg.action_topic}: a subscriber matched but its node did not appear in the ROS "
                    "graph within 2 s; NOT treated as an IL hand-off"
                )
            self.destroy_timer(self._match_confirm_timer)
            self._match_confirm_timer = None

    def _check_action_subscribers(self):
        """Compare the action topic's subscribers with the ones already seen. A new one from a
        handoff_subscriber_nodes node queues a hand-off; others are only logged. Returns True if
        any new subscriber was found."""
        current = {
            bytes(i.endpoint_gid): i.node_name
            for i in self.get_subscriptions_info_by_topic(self.cfg.action_topic)
        }
        new = {gid: name for gid, name in current.items() if gid not in self._known_action_subs}
        self._known_action_subs = current
        for name in new.values():
            if any(name.startswith(n) for n in self.cfg.handoff_subscriber_nodes):
                self._sub_handoff_count += 1
                self._queue_handoff(f"s{self._sub_handoff_count}", f"new {self.cfg.action_topic} subscriber {name}")
            else:
                self.get_logger().info(
                    f"{self.cfg.action_topic}: new subscriber {name} is not in handoff_subscriber_nodes; "
                    "not an IL hand-off"
                )
        return bool(new)

    def _queue_handoff(self, handoff_id, source):
        """Queue a hand-off for the next tick, unless another one was queued within
        handoff_dedupe_sec: then it is the same hand-off, signalled twice."""
        now = time.monotonic()
        if self._handoff_last is not None and now - self._handoff_last[1] < self.cfg.handoff_dedupe_sec:
            self.get_logger().info(
                f"IL hand-off #{handoff_id} ({source}) came {1000 * (now - self._handoff_last[1]):.0f} ms after "
                f"#{self._handoff_last[0]}: same hand-off, not reset again"
            )
            return
        self._handoff_last = (handoff_id, now)
        self._handoff_pending = (handoff_id, source)

    def _start_handoff_if_pending(self):
        if self._handoff_pending is None:
            return
        (handoff_id, source), self._handoff_pending = self._handoff_pending, None
        self._handoff = {
            "id": handoff_id,
            "gripper_state_before": self._gripper_state,
            "actions_left": self.cfg.handoff_log_actions,
        }
        self.get_logger().info(
            f"IL hand-off #{handoff_id}: {source}; policy._gripper_state at this tick = {self._gripper_state:.0f}"
        )
        if self.cfg.reset_on_handoff:
            self.policy.reset()
            if self.gradcam is not None:
                self.gradcam.reset()
            self._gripper_state = 1.0 if self.cfg.initial_gripper_open else 0.0
            self._close_ticks = 0
            self.get_logger().info(
                f"IL hand-off #{handoff_id}: reset_on_handoff -> policy.reset(), "
                f"_gripper_state {self._handoff['gripper_state_before']:.0f} -> {self._gripper_state:.0f}, "
                f"gripper debounce count -> 0"
            )
        else:
            self.get_logger().info(f"IL hand-off #{handoff_id}: reset_on_handoff is off, policy state carried over")

    def _log_handoff_tick(self, obs, action, raw_gripper):
        """At the first tick after a hand-off, log the state the policy was given;
        then log each of the next handoff_log_actions actions."""
        h = self._handoff
        if h is None or h["actions_left"] <= 0:
            return
        log = self.get_logger()
        n = self.cfg.handoff_log_actions - h["actions_left"] + 1
        if n == 1:
            self._log_handoff_state(h, obs)
        gripper_in = float(obs["gripper_state"])
        is_open = action[6] >= self.cfg.gripper_open_threshold
        log.info(
            f"IL hand-off #{h['id']} action {n}/{self.cfg.handoff_log_actions}: "
            f"dpos(mm)={np.array2string(action[:3] * 1000, precision=3)} "
            f"drot(mrad)={np.array2string(action[3:6] * 1000, precision=3)} "
            f"gripper raw={raw_gripper:.3f} published={action[6]:.3f} -> {'open' if is_open else 'CLOSE'} "
            f"(gripper_state fed in {gripper_in:.0f})"
        )
        h.setdefault("actions", []).append([round(float(v), 6) for v in action])
        h["actions_left"] -= 1
        if h["actions_left"] == 0:
            log.info("HANDOFF_JSON " + json.dumps({"id": h["id"], "state": h["state"], "actions": h["actions"]}))

    def _log_handoff_state(self, h, obs):
        log = self.get_logger()
        state = np.r_[np.asarray(obs["joint_pos"], dtype=np.float64), float(obs["gripper_state"])]
        rng = self._state_range
        if rng is not None and len(rng[0]) != len(state):
            rng = None
        norm = range_normalize(state, rng) if rng is not None else None
        out = [] if norm is None else [i for i in range(len(state)) if abs(norm[i]) > 1.0 + 1e-6]
        log.info(
            f"IL hand-off #{h['id']} state given to the policy "
            f"(gripper_state before hand-off {h['gripper_state_before']:.0f}, fed now {state[-1]:.0f}); "
            + ("range-normalized: training data spans [-1, 1]" if rng is not None
               else "no training range available for this policy")
        )
        for i, name in enumerate(_STATE_NAMES[: len(state)]):
            if rng is None:
                log.info(f"    {name:8s} raw {state[i]:+.4f}")
            else:
                flag = "  <-- OUTSIDE TRAINING RANGE" if i in out else ""
                log.info(
                    f"    {name:8s} raw {state[i]:+.4f}  normalized {norm[i]:+7.2f}  "
                    f"training [{rng[0][i]:+.3f}, {rng[1][i]:+.3f}]{flag}"
                )
        if rng is not None:
            log.info(f"IL hand-off #{h['id']}: {len(out)} of {len(state)} state values outside the training range")
        h["state"] = {
            "raw": [round(float(v), 6) for v in state],
            "normalized": None if norm is None else [round(float(v), 3) for v in norm],
            "outside_training_range": [_STATE_NAMES[i] for i in out],
            "gripper_state_before": h["gripper_state_before"],
        }

    def _build_obs(self, ros_data):
        ee_pos, ee_quat = apply_ee_offset(
            ros_data["ee_pos"], ros_data["ee_quat"],
            self._ee_pos_offset, self._ee_rot_offset,
        )
        return {
            "rgb": resize_image(ros_data["rgb"], self.cfg.image_size, is_depth=False),
            "depth": resize_image(ros_data["depth"], self.cfg.image_size, is_depth=True),
            "joint_pos": ros_data["joint_pos"],
            "joint_vel": ros_data["joint_vel"],
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "gripper_state": np.float32(self._gripper_state),
            "ft_wrench": ros_data["ft_wrench"],
        }

    def _on_health_check(self):
        """Warn (throttled) about any subscribed topic that has no
        publisher, has never delivered a message, or has gone quiet."""
        messages = {
            "no_publisher": "no publisher found for {topic} — check the topic name / that its node is running",
            "no_data": "publisher exists for {topic} but no message received yet — check QoS (reliability) compatibility",
            "stale": "no message on {topic} in over {timeout:.1f}s — it may have died",
        }
        for topic, status in self.bridge.check_health(self.cfg.topic_timeout_sec).items():
            if status == "ok":
                continue
            self.get_logger().warn(
                messages[status].format(topic=topic, timeout=self.cfg.topic_timeout_sec),
                throttle_duration_sec=5.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.gradcam is not None:
            node.gradcam.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
