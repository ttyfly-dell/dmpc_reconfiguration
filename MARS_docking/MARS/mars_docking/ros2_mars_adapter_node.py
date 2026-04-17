#!/usr/bin/env python3
"""ROS2 适配节点: /aruco/center -> MARS 控制入口 -> relay 输入话题.

输入:
  - /aruco/center (geometry_msgs/msg/TwistStamped)
        - twist.linear.x: 相机坐标 x (右)
        - twist.linear.y: 相机坐标 y (上)
        - twist.linear.z: 相机坐标 z (前)
    - twist.angular.z: 相对航向 yaw (rad)

输出:
  - /d15019504/command/cmd_twist_relay_in (geometry_msgs/msg/Twist)
    - linear.x: 线速度 v
    - angular.z: 角速度 omega
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped

from .types import Pose2D, DockingConfig, DockingPhase
from .mpc_controller import MPCController
from .state_machine import DockingStateMachine
from .safety_layer import SafetyLayer


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ArucoToMarsNode(Node):
    """将 ArUco 相对位姿对接到 MARS 单机对接状态机."""

    def __init__(self):
        super().__init__("aruco_to_mars_adapter")

        # Topics
        self.declare_parameter("aruco_topic", "/aruco/center")
        self.declare_parameter(
            "cmd_vel_topic", "/d15019504/command/cmd_twist_relay_in"
        )

        # Control loop
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("aruco_timeout_sec", 5.0)

        # Relative pose mapping scales
        self.declare_parameter("x_scale", 1.0)
        self.declare_parameter("y_scale", 1.0)
        self.declare_parameter("yaw_scale", 1.0)
        self.declare_parameter("yaw_offset", 0.0)
        self.declare_parameter("output_omega_sign", 1.0)
        self.declare_parameter("aruco_lpf_enable", True)
        self.declare_parameter("aruco_lpf_tau_sec", 0.08)
        self.declare_parameter("aruco_outlier_reject_enable", True)#异常值剔除开关
        self.declare_parameter("aruco_outlier_max_dx", 0.03)#最大位移跳变阈值(米)
        self.declare_parameter("aruco_outlier_max_dy", 0.03)#最大位移跳变阈值(米)
        self.declare_parameter("aruco_outlier_max_dtheta", 0.12)#最大角度跳变阈值(弧度)

        # Measurement geometry compensation:
        # if input is "leader tail -> follower head" relative coordinate,
        # convert to "leader center -> follower center" by subtracting
        # (leader_tail_to_center + follower_head_to_center) on axial x.
        self.declare_parameter("input_is_tail_head", True)
        self.declare_parameter("leader_tail_to_center_m", 0.155)
        self.declare_parameter("follower_head_to_center_m", 0.150)

        # Static leader pose in relative frame (dynamic-to-static docking)
        self.declare_parameter("leader_x", 0.0)
        self.declare_parameter("leader_y", 0.0)
        self.declare_parameter("leader_theta", 0.0)

        # Safety behavior
        self.declare_parameter("auto_clear_estop_on_recover", True)
        self.declare_parameter("enable_estop", False)

        # Debug print
        self.declare_parameter("print_aruco_input", False)
        self.declare_parameter("print_every_n", 1)
        self.declare_parameter("print_control_output", False)

        aruco_topic = self.get_parameter("aruco_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        control_hz = float(self.get_parameter("control_hz").value)
        self._aruco_timeout = float(self.get_parameter("aruco_timeout_sec").value)

        self._x_scale = float(self.get_parameter("x_scale").value)
        self._y_scale = float(self.get_parameter("y_scale").value)
        self._yaw_scale = float(self.get_parameter("yaw_scale").value)
        self._yaw_offset = float(self.get_parameter("yaw_offset").value)
        self._output_omega_sign = float(self.get_parameter("output_omega_sign").value)
        self._aruco_lpf_enable = bool(self.get_parameter("aruco_lpf_enable").value)
        self._aruco_lpf_tau = max(0.0, float(self.get_parameter("aruco_lpf_tau_sec").value))
        # ArUco 跳变剔除参数
        self._aruco_outlier_reject_enable = bool(
            self.get_parameter("aruco_outlier_reject_enable").value
        )
        self._aruco_outlier_max_dx = max(
            0.0, float(self.get_parameter("aruco_outlier_max_dx").value)
        )
        self._aruco_outlier_max_dy = max(
            0.0, float(self.get_parameter("aruco_outlier_max_dy").value)
        )
        self._aruco_outlier_max_dtheta = max(
            0.0, float(self.get_parameter("aruco_outlier_max_dtheta").value)
        )
        self._input_is_tail_head = bool(self.get_parameter("input_is_tail_head").value)
        self._leader_tail_to_center = float(
            self.get_parameter("leader_tail_to_center_m").value
        )
        self._follower_head_to_center = float(
            self.get_parameter("follower_head_to_center_m").value
        )

        self._auto_clear_estop = bool(
            self.get_parameter("auto_clear_estop_on_recover").value
        )
        self._enable_estop = bool(self.get_parameter("enable_estop").value)
        self._print_aruco_input = bool(self.get_parameter("print_aruco_input").value)
        self._print_every_n = max(1, int(self.get_parameter("print_every_n").value))
        self._print_control_output = bool(self.get_parameter("print_control_output").value)
        self._aruco_msg_count = 0
        self._control_tick_count = 0

        leader_pose = Pose2D(
            float(self.get_parameter("leader_x").value),
            float(self.get_parameter("leader_y").value),
            float(self.get_parameter("leader_theta").value),
        )

        # MARS control entry
        self._config = DockingConfig()
        self._mpc = MPCController(self._config)
        self._sm = DockingStateMachine(self._config, self._mpc)
        self._sm.start(leader_pose)

        self._safety = SafetyLayer(
            v_limit=self._config.safety_v_limit,
            omega_limit=self._config.safety_omega_limit,
        )

        self._leader_pose = leader_pose
        self._latest_follower_pose: Pose2D | None = None
        self._filtered_follower_pose: Pose2D | None = None
        self._last_aruco_time = 0.0
        self._aruco_reject_count = 0

        self._aruco_sub = self.create_subscription(
            TwistStamped, aruco_topic, self._on_aruco, 10
        )
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        period = 1.0 / max(control_hz, 1e-3)
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Adapter started: sub={aruco_topic}, pub={cmd_vel_topic}, hz={control_hz:.1f}"
        )

    #跳变剔除
    def _reject_aruco_outlier(self, measured_pose: Pose2D) -> Pose2D:
        """剔除单帧明显跳变的观测, 避免错误位姿直接驱动控制器."""
        if not self._aruco_outlier_reject_enable:
            return measured_pose

        reference = self._filtered_follower_pose
        if reference is None:
            reference = self._latest_follower_pose
        if reference is None:
            return measured_pose

        dx = measured_pose.x - reference.x
        dy = measured_pose.y - reference.y
        dtheta = normalize_angle(measured_pose.theta - reference.theta)
        if (
            abs(dx) <= self._aruco_outlier_max_dx
            and abs(dy) <= self._aruco_outlier_max_dy
            and abs(dtheta) <= self._aruco_outlier_max_dtheta
        ):
            return measured_pose

        self._aruco_reject_count += 1
        self.get_logger().warn(
            (
                "Reject aruco outlier #%d: jump=(%.4f, %.4f, %.4f), "
                "threshold=(%.4f, %.4f, %.4f)"
            )
            % (
                self._aruco_reject_count,
                dx,
                dy,
                dtheta,
                self._aruco_outlier_max_dx,
                self._aruco_outlier_max_dy,
                self._aruco_outlier_max_dtheta,
            )
        )
        return reference

    #一阶低通滤波
    def _apply_aruco_lpf(self, measured_pose: Pose2D, now: float) -> Pose2D:
        """一阶低通滤波: y[k] = y[k-1] + alpha * (x[k] - y[k-1])."""
        if not self._aruco_lpf_enable or self._aruco_lpf_tau <= 0.0:
            self._filtered_follower_pose = measured_pose
            return measured_pose

        if self._filtered_follower_pose is None or self._last_aruco_time <= 0.0:
            self._filtered_follower_pose = measured_pose
            return measured_pose

        dt = max(0.0, now - self._last_aruco_time)
        if dt <= 1e-6:
            return self._filtered_follower_pose

        alpha = dt / (self._aruco_lpf_tau + dt)
        alpha = max(0.0, min(1.0, alpha))
        prev = self._filtered_follower_pose

        theta_delta = normalize_angle(measured_pose.theta - prev.theta)
        filtered = Pose2D(
            prev.x + alpha * (measured_pose.x - prev.x),
            prev.y + alpha * (measured_pose.y - prev.y),
            normalize_angle(prev.theta + alpha * theta_delta),
        )
        self._filtered_follower_pose = filtered
        return filtered

    def _on_aruco(self, msg: TwistStamped):
        # 相机中心坐标系: x前, y左, z上
        # 控制平面映射: x(前向)=x_cam, y(左向)=y_cam
        now = time.monotonic()
        rel_x = self._x_scale * float(msg.twist.linear.x)
        rel_y = self._y_scale * (float(msg.twist.linear.y))
        if self._input_is_tail_head:
            rel_x -= (self._leader_tail_to_center + self._follower_head_to_center)
        rel_yaw = normalize_angle(
            -self._yaw_scale * float(msg.twist.angular.z)- self._yaw_offset
        )

        measured_pose = Pose2D(rel_x, rel_y, rel_yaw)
        measured_pose = self._reject_aruco_outlier(measured_pose)
        self._latest_follower_pose = self._apply_aruco_lpf(measured_pose, now)
        self._last_aruco_time = now

        self._aruco_msg_count += 1
        if self._print_aruco_input and (self._aruco_msg_count % self._print_every_n == 0):
            self.get_logger().info(
                "aruco raw(x,y,z,yaw)=({:.4f}, {:.4f}, {:.4f}, {:.4f}) -> meas=({:.4f}, {:.4f}, {:.4f}) filt=({:.4f}, {:.4f}, {:.4f})".format(
                    float(msg.twist.linear.x),
                    float(msg.twist.linear.y),
                    float(msg.twist.linear.z),
                    float(msg.twist.angular.z),
                    rel_x,
                    rel_y,
                    rel_yaw,
                    self._latest_follower_pose.x,
                    self._latest_follower_pose.y,
                    self._latest_follower_pose.theta,
                )
            )

    def _publish_cmd(self, v: float, w: float):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(w)
        self._cmd_pub.publish(cmd)

    def _on_timer(self):
        now = time.monotonic()
        self._control_tick_count += 1

        # 测试模式: 关闭急停触发并清除锁存
        if not self._enable_estop and self._safety.estop_active:
            self._safety.clear_estop()

        if self._latest_follower_pose is None:
            if self._enable_estop and not self._safety.estop_active:
                self._safety.trigger_estop("waiting first /aruco/center")
            self._publish_cmd(0.0, 0.0)
            if self._print_control_output and (self._control_tick_count % self._print_every_n == 0):
                self.get_logger().info(
                    "ctrl phase=WAIT_INPUT estop={} cmd=(0.0000, 0.0000)".format(
                        self._safety.estop_active
                    )
                )
            return

        if (now - self._last_aruco_time) > self._aruco_timeout:
            if self._enable_estop and not self._safety.estop_active:
                self._safety.trigger_estop("/aruco/center timeout")
            self._publish_cmd(0.0, 0.0)
            if self._print_control_output and (self._control_tick_count % self._print_every_n == 0):
                self.get_logger().info(
                    "ctrl phase=TIMEOUT estop={} cmd=(0.0000, 0.0000)".format(
                        self._safety.estop_active
                    )
                )
            return

        if self._safety.estop_active and self._auto_clear_estop:
            self._safety.clear_estop()

        phase = self._sm.phase
        if phase in (DockingPhase.DOCKED, DockingPhase.FAILED):
            self._publish_cmd(0.0, 0.0)
            if self._print_control_output and (self._control_tick_count % self._print_every_n == 0):
                self.get_logger().info(
                    "ctrl phase={} estop={} cmd=(0.0000, 0.0000)".format(
                        phase.name, self._safety.estop_active
                    )
                )
            return

        v_cmd, w_cmd = self._sm.update(
            self._latest_follower_pose,
            leader_pose=self._leader_pose,
        )
        v_safe, w_safe = self._safety.apply(v_cmd, w_cmd)
        self._publish_cmd(v_safe, w_safe)

        if self._print_control_output and (self._control_tick_count % self._print_every_n == 0):
            self.get_logger().info(
                "ctrl phase={} estop={} raw=({:.4f}, {:.4f}) safe=({:.4f}, {:.4f})".format(
                    self._sm.phase.name,
                    self._safety.estop_active,
                    v_cmd,
                    w_cmd,
                    v_safe,
                    w_safe,
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoToMarsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # Ignore duplicate-shutdown or context errors during Ctrl+C teardown.
            pass


if __name__ == "__main__":
    main()
