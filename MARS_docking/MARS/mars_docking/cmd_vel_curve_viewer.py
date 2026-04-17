#!/usr/bin/env python3
"""实时可视化 MPC 从当前位置到目标点的预测轨迹.

默认订阅:
  /aruco/center (geometry_msgs/msg/TwistStamped)

说明:
  - 复用与 ros2_mars_adapter_node.py 一致的位姿映射与控制参数
  - 每个控制周期调用状态机 + MPC 进行一次求解
  - 显示: 当前位姿、目标点、MPC 预测轨迹(N+1)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

# 兼容两种运行方式:
# 1) python -m mars_docking.cmd_vel_curve_viewer
# 2) python /abs/path/cmd_vel_curve_viewer.py
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[1]  # .../MARS
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from mars_docking.mpc_controller import MPCController
from mars_docking.state_machine import DockingStateMachine
from mars_docking.types import DockingConfig, DockingPhase, Pose2D


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MpcTrajectoryViewer(Node):
    """实时显示 MPC 预测轨迹."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("mpc_trajectory_viewer")

        # 输入/控制参数（与 ros2_mars_adapter_node 保持一致）
        self._aruco_topic = args.aruco_topic
        self._control_hz = max(1.0, float(args.control_hz))
        self._aruco_timeout = max(0.1, float(args.aruco_timeout_sec))

        self._x_scale = float(args.x_scale)
        self._y_scale = float(args.y_scale)
        self._y_offset = float(args.y_offset)
        self._yaw_scale = float(args.yaw_scale)
        self._yaw_offset = float(args.yaw_offset)

        self._aruco_lpf_enable = bool(args.aruco_lpf_enable)
        self._aruco_lpf_tau = max(0.0, float(args.aruco_lpf_tau_sec))
        self._aruco_outlier_reject_enable = bool(args.aruco_outlier_reject_enable)
        self._aruco_outlier_max_dx = max(0.0, float(args.aruco_outlier_max_dx))
        self._aruco_outlier_max_dy = max(0.0, float(args.aruco_outlier_max_dy))
        self._aruco_outlier_max_dtheta = max(0.0, float(args.aruco_outlier_max_dtheta))

        self._input_is_tail_head = bool(args.input_is_tail_head)
        self._leader_tail_to_center = float(args.leader_tail_to_center_m)
        self._follower_head_to_center = float(args.follower_head_to_center_m)

        self._leader_pose = Pose2D(args.leader_x, args.leader_y, args.leader_theta)

        # MARS 控制入口
        self._config = DockingConfig()
        self._mpc = MPCController(self._config)
        self._sm = DockingStateMachine(self._config, self._mpc)
        self._sm.start(self._leader_pose)

        self._latest_follower_pose: Pose2D | None = None
        self._filtered_follower_pose: Pose2D | None = None
        self._last_aruco_time = 0.0
        self._aruco_reject_count = 0

        # 实际轨迹缓存（用于对比）
        self._history_sec = max(2.0, float(args.history_sec))
        self._start = time.monotonic()
        self._hist_t = deque()
        self._hist_x = deque()
        self._hist_y = deque()

        # 最近一次 MPC 结果
        self._last_v = 0.0
        self._last_w = 0.0
        self._pred_xy: np.ndarray | None = None  # shape=(N+1, 2)
        self._view_bounds: tuple[float, float, float, float] | None = None

        self._aruco_sub = self.create_subscription(
            TwistStamped, self._aruco_topic, self._on_aruco, 10
        )
        self._ctrl_timer = self.create_timer(1.0 / self._control_hz, self._on_control)
        self._draw_timer = self.create_timer(1.0 / max(1.0, float(args.redraw_hz)), self._on_draw)

        self._init_plot()

        self.get_logger().info(
            "MPC trajectory viewer started: sub=%s, control_hz=%.1f"
            % (self._aruco_topic, self._control_hz)
        )

    def _init_plot(self) -> None:
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8.5, 7.0))
        self._fig.canvas.manager.set_window_title("MARS MPC 预测轨迹")

        (self._line_hist,) = self._ax.plot([], [], color="#90caf9", lw=1.2, label="actual path")
        (self._line_pred,) = self._ax.plot([], [], "-o", color="#1e88e5", ms=3, lw=1.8, label="mpc prediction")
        (self._pt_current,) = self._ax.plot([], [], "o", color="#1565c0", ms=7, label="current")
        (self._pt_active_target,) = self._ax.plot(
            [], [], "*", color="#e53935", ms=11, label="active target"
        )
        (self._pt_approach,) = self._ax.plot([], [], "X", color="#8e24aa", ms=8, label="approach point")
        (self._pt_staging,) = self._ax.plot([], [], "D", color="#fb8c00", ms=7, label="staging point")
        (self._pt_dock,) = self._ax.plot([], [], "s", color="#2e7d32", ms=7, label="dock point")
        self._heading_arrow = None

        self._txt = self._ax.text(
            0.02,
            0.98,
            "",
            transform=self._ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )

        self._ax.set_title("MPC Predicted Trajectory")
        self._ax.set_xlabel("x (m)")
        self._ax.set_ylabel("y (m)")
        self._ax.grid(True, alpha=0.3)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.legend(loc="upper right")

    def _reject_aruco_outlier(self, measured_pose: Pose2D) -> Pose2D:
        """剔除单帧明显跳变的观测, 保持显示和控制输入一致稳定."""
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

    def _apply_aruco_lpf(self, measured_pose: Pose2D, now: float) -> Pose2D:
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

    def _on_aruco(self, msg: TwistStamped) -> None:
        now = time.monotonic()

        # 与 ros2_mars_adapter_node.py 同映射
        rel_x = self._x_scale * float(msg.twist.linear.x)
        rel_y = self._y_scale * float(msg.twist.linear.y)
        if self._input_is_tail_head:
            rel_x -= (self._leader_tail_to_center + self._follower_head_to_center)
        rel_yaw = normalize_angle(self._yaw_scale * float(msg.twist.angular.z) - self._yaw_offset)

        measured_pose = Pose2D(rel_x, rel_y, rel_yaw)
        measured_pose = self._reject_aruco_outlier(measured_pose)
        self._latest_follower_pose = self._apply_aruco_lpf(measured_pose, now)
        self._last_aruco_time = now

        t = now - self._start
        self._hist_t.append(t)
        self._hist_x.append(self._latest_follower_pose.x)
        self._hist_y.append(self._latest_follower_pose.y)
        self._trim_history(t)

    def _trim_history(self, t_now: float) -> None:
        min_t = t_now - self._history_sec
        while self._hist_t and self._hist_t[0] < min_t:
            self._hist_t.popleft()
            self._hist_x.popleft()
            self._hist_y.popleft()

    def _extract_pred_xy(self) -> np.ndarray | None:
        """从 MPC 内部解向量提取预测状态轨迹 (x,y)."""
        sol = self._mpc._prev_sol  # 只读使用, 不改现有框架
        if sol is None:
            return None

        n = self._mpc.N
        n_states = 3 * (n + 1)
        x_flat = np.asarray(sol[:n_states])
        x_traj = x_flat.reshape(n + 1, 3)
        return x_traj[:, :2].copy()

    def _current_target(self) -> Pose2D | None:
        dbg = self._sm.get_debug_info()
        phase_name = dbg.get("phase", "IDLE")

        if phase_name == DockingPhase.APPROACH.name:
            return dbg.get("approach_point")
        if phase_name == DockingPhase.ALIGN.name:
            return dbg.get("staging_point")
        if phase_name == DockingPhase.FINAL_APPROACH.name:
            return dbg.get("dock_point")

        # DOCKED/FAILED/IDLE 时默认展示 dock 点
        return dbg.get("dock_point")

    def _all_targets(self) -> tuple[Pose2D | None, Pose2D | None, Pose2D | None]:
        dbg = self._sm.get_debug_info()
        return dbg.get("approach_point"), dbg.get("staging_point"), dbg.get("dock_point")

    def _on_control(self) -> None:
        now = time.monotonic()

        if self._latest_follower_pose is None:
            return

        if (now - self._last_aruco_time) > self._aruco_timeout:
            return

        phase = self._sm.phase
        if phase in (DockingPhase.DOCKED, DockingPhase.FAILED):
            self._pred_xy = None
            self._last_v = 0.0
            self._last_w = 0.0
            return

        v_cmd, w_cmd = self._sm.update(
            self._latest_follower_pose,
            leader_pose=self._leader_pose,
        )
        self._last_v = v_cmd
        self._last_w = w_cmd
        self._pred_xy = self._extract_pred_xy()

    def _auto_bounds(self, xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dx = max(0.8, x_max - x_min)
        dy = max(0.8, y_max - y_min)
        pad_x = dx * 0.25
        pad_y = dy * 0.25
        return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y

    def _smooth_bounds(
        self, target_bounds: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """平滑坐标轴变化，避免视图抖动和剧烈缩放。"""
        if self._view_bounds is None:
            self._view_bounds = target_bounds
            return target_bounds

        # 越小越平滑（0~1），减小动态变化幅度
        alpha = 0.12
        prev = self._view_bounds
        smoothed = tuple(prev[i] + alpha * (target_bounds[i] - prev[i]) for i in range(4))
        self._view_bounds = smoothed
        return smoothed

    def _on_draw(self) -> None:
        if not plt.fignum_exists(self._fig.number):
            self.get_logger().info("Figure closed, shutting down.")
            rclpy.shutdown()
            return

        cur = self._latest_follower_pose
        tgt = self._current_target()
        approach_pt, staging_pt, dock_pt = self._all_targets()

        if cur is None or tgt is None:
            plt.pause(0.001)
            return

        hx = list(self._hist_x)
        hy = list(self._hist_y)

        px: list[float] = []
        py: list[float] = []
        if self._pred_xy is not None and len(self._pred_xy) > 0:
            px = self._pred_xy[:, 0].tolist()
            py = self._pred_xy[:, 1].tolist()

        self._line_hist.set_data(hx, hy)
        self._line_pred.set_data(px, py)
        self._pt_current.set_data([cur.x], [cur.y])
        self._pt_active_target.set_data([tgt.x], [tgt.y])
        if approach_pt is not None:
            self._pt_approach.set_data([approach_pt.x], [approach_pt.y])
        else:
            self._pt_approach.set_data([], [])
        if staging_pt is not None:
            self._pt_staging.set_data([staging_pt.x], [staging_pt.y])
        else:
            self._pt_staging.set_data([], [])
        if dock_pt is not None:
            self._pt_dock.set_data([dock_pt.x], [dock_pt.y])
        else:
            self._pt_dock.set_data([], [])

        # 当前机器人朝向箭头
        if self._heading_arrow is not None:
            self._heading_arrow.remove()
        arrow_len = 0.05
        dx = arrow_len * math.cos(cur.theta)
        dy = arrow_len * math.sin(cur.theta)
        self._heading_arrow = self._ax.arrow(
            cur.x,
            cur.y,
            dx,
            dy,
            width=0.003,
            head_width=0.01,
            head_length=0.015,
            fc="#1565c0",
            ec="#1565c0",
            alpha=0.9,
            length_includes_head=True,
            zorder=5,
        )

        xs = [cur.x, tgt.x] + hx + px
        ys = [cur.y, tgt.y] + hy + py
        if approach_pt is not None:
            xs.append(approach_pt.x)
            ys.append(approach_pt.y)
        if staging_pt is not None:
            xs.append(staging_pt.x)
            ys.append(staging_pt.y)
        if dock_pt is not None:
            xs.append(dock_pt.x)
            ys.append(dock_pt.y)
        target_bounds = self._auto_bounds(xs, ys)
        x0, x1, y0, y1 = self._smooth_bounds(target_bounds)
        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)

        self._txt.set_text(
            "phase={}\ncurrent=({:.3f}, {:.3f}, {:.1f} deg)\ntarget=({:.3f}, {:.3f}, {:.1f} deg)\ncmd=(v={:.3f} m/s, w={:.3f} rad/s)\nN={}, dt={:.2f}s".format(
                self._sm.phase.name,
                cur.x,
                cur.y,
                math.degrees(cur.theta),
                tgt.x,
                tgt.y,
                math.degrees(tgt.theta),
                self._last_v,
                self._last_w,
                self._config.mpc_horizon,
                self._config.mpc_dt,
            )
        )

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时显示 MPC 预测轨迹")
    p.add_argument("--aruco-topic", default="/aruco/center", help="输入 TwistStamped 话题")
    p.add_argument("--control-hz", type=float, default=20.0, help="控制求解频率")
    p.add_argument("--redraw-hz", type=float, default=20.0, help="画面刷新频率")
    p.add_argument("--aruco-timeout-sec", type=float, default=5.0, help="输入超时阈值")

    p.add_argument("--x-scale", type=float, default=1.0)
    p.add_argument("--y-scale", type=float, default=1.0)
    p.add_argument("--y-offset", type=float, default=0.0)
    p.add_argument("--yaw-scale", type=float, default=1.0)
    p.add_argument("--yaw-offset", type=float, default=0.0)

    p.add_argument("--aruco-lpf-enable", action="store_true", default=True)
    p.add_argument("--no-aruco-lpf", action="store_false", dest="aruco_lpf_enable")
    p.add_argument("--aruco-lpf-tau-sec", type=float, default=0.08)
    p.add_argument("--aruco-outlier-reject-enable", action="store_true", default=True)
    p.add_argument(
        "--no-aruco-outlier-reject",
        action="store_false",
        dest="aruco_outlier_reject_enable",
    )
    p.add_argument("--aruco-outlier-max-dx", type=float, default=0.03)
    p.add_argument("--aruco-outlier-max-dy", type=float, default=0.03)
    p.add_argument("--aruco-outlier-max-dtheta", type=float, default=0.12)

    p.add_argument("--input-is-tail-head", action="store_true", default=True)
    p.add_argument("--input-is-center-center", action="store_false", dest="input_is_tail_head")
    p.add_argument("--leader-tail-to-center-m", type=float, default=0.155)
    p.add_argument("--follower-head-to-center-m", type=float, default=0.150)

    p.add_argument("--leader-x", type=float, default=0.0)
    p.add_argument("--leader-y", type=float, default=0.0)
    p.add_argument("--leader-theta", type=float, default=0.0)

    p.add_argument("--history-sec", type=float, default=20.0, help="实际轨迹显示窗口")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()

    node = MpcTrajectoryViewer(args)
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
