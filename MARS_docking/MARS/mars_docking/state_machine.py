"""
对接状态机.

三点对接架构:
    approach_point ─(DMPC)─► staging_point ─(MPC ALIGN)─► dock_point
                                                 │
                                          FINAL_APPROACH (MPC 直推)

状态转移:
    IDLE → APPROACH → ALIGN → FINAL_APPROACH → DOCKED
                                    │
                                    ▼
                                  ALIGN  (横向超限回退)
    任何活动阶段 → FAILED (超时)

调用模式:
  1. 旧模式 (静态 Leader):  start(leader_pose) / update(follower, leader_pose)
  2. 编队模式 (动态):       start(approach_point=..., staging_point=..., dock_point=...)
                            update(follower, approach_point=..., staging_point=..., dock_point=...)
"""

import math

from .types import Pose2D, DockingPhase, DockingConfig
from .mpc_controller import MPCController
from .utils import (
    compute_approach_point,
    compute_staging_point,
    compute_dock_point,
    decompose_error,
    normalize_angle,
)


class DockingStateMachine:
    """对接状态机, 输入双方位姿, 输出 Follower 的速度指令."""

    def __init__(self, config: DockingConfig, mpc_controller: MPCController):
        self.config = config
        self.mpc = mpc_controller

        self._phase = DockingPhase.IDLE
        self._leader_pose = None

        # 仿真时间计步 (每次 update / check_and_advance 加 1)
        self._phase_step_count = 0

        # ALIGN 阶段对齐稳定计数 (连续满足条件的步数)
        self._align_dwell_count = 0


        # 对接目标缓存 (三点: approach → staging → dock)
        self._approach_point = None
        self._staging_point  = None
        self._dock_point     = None

    @property
    def phase(self) -> DockingPhase:
        return self._phase

    def start(self, leader_pose: Pose2D = None, *,
              approach_point: Pose2D = None,
              staging_point: Pose2D = None,
              dock_point: Pose2D = None):
        """
        启动对接流程.

        旧模式:   start(leader_pose)
        编队模式: start(approach_point=ap, staging_point=sp, dock_point=dp)
        """
        if dock_point is not None:
            self._approach_point = approach_point
            self._staging_point  = staging_point
            self._dock_point     = dock_point
        elif leader_pose is not None:
            self._leader_pose    = leader_pose
            self._approach_point = compute_approach_point(leader_pose, self.config)
            self._staging_point  = compute_staging_point(leader_pose, self.config)
            self._dock_point     = compute_dock_point(leader_pose, self.config)
        else:
            raise ValueError("Provide leader_pose or dock_point")

        self._set_phase(DockingPhase.APPROACH)
        self.mpc.reset()

    # ── Phase 1 专用: 仅做状态检查, 不计算速度 ──

    def check_and_advance_approach(self, follower_pose: Pose2D,
                                   pred_phase: DockingPhase = None,
                                   approach_point: Pose2D = None,
                                   staging_point: Pose2D = None,
                                   dock_point: Pose2D = None):
        """
        检查 APPROACH 阶段是否应切换到 ALIGN.

        由 ChainDockingCoordinator 在 Phase 1 每步调用,
        替代 update() 以避免不必要的 MPC 求解.

        切换条件: 到达 approach_point 附近 (dist < approach_capture_dist).
        """
        if self._phase != DockingPhase.APPROACH:
            return

        # 同步动态目标
        if approach_point is not None:
            self._approach_point = approach_point
        if staging_point is not None:
            self._staging_point = staging_point
        if dock_point is not None:
            self._dock_point = dock_point

        self._phase_step_count += 1

        if self._check_timeout():
            self._set_phase(DockingPhase.FAILED)
            return

        if follower_pose.distance_to(self._approach_point) < self.config.approach_capture_dist:
            # 门控: 前车必须已进入 FINAL_APPROACH 或 DOCKED 才允许切 ALIGN.
            # pred_phase=None 表示 anchor (永久 DOCKED), 直接放行.
            pred_ready = (pred_phase is None or
                          pred_phase in (DockingPhase.FINAL_APPROACH,
                                         DockingPhase.DOCKED))
            if pred_ready and not self.config.lock_approach:
                self._set_phase(DockingPhase.ALIGN)

    # ── Phase 2: 完整 MPC 控制 ──

    def update(self, follower_pose: Pose2D,
               leader_pose: Pose2D = None, *,
               approach_point: Pose2D = None,
               staging_point: Pose2D = None,
               dock_point: Pose2D = None,
               pred_phase: DockingPhase = None):
        """
        状态机主循环, 每个控制周期调用一次 (Phase 2 使用).

        编队模式: update(follower, approach_point=..., staging_point=..., dock_point=...)
        旧模式:   update(follower, leader_pose)

        Returns:
            (v, omega): Follower 的速度指令. 若已完成或失败, 返回 (0, 0).
        """
        # 更新目标
        if dock_point is not None:
            if approach_point is not None:
                self._approach_point = approach_point
            if staging_point is not None:
                self._staging_point = staging_point
            self._dock_point = dock_point
        elif leader_pose is not None:
            self._leader_pose    = leader_pose
            self._approach_point = compute_approach_point(leader_pose, self.config)
            self._staging_point  = compute_staging_point(leader_pose, self.config)
            self._dock_point     = compute_dock_point(leader_pose, self.config)

        # 终态不产生指令
        if self._phase in (DockingPhase.IDLE, DockingPhase.DOCKED,
                           DockingPhase.FAILED):
            return 0.0, 0.0

        # 计步
        self._phase_step_count += 1

        # 超时检测
        if self._check_timeout():
            self._set_phase(DockingPhase.FAILED)
            return 0.0, 0.0

        # 按阶段处理
        if self._phase == DockingPhase.APPROACH:
            return self._handle_approach(follower_pose)
        elif self._phase == DockingPhase.ALIGN:
            return self._handle_align(follower_pose, pred_phase)
        elif self._phase == DockingPhase.FINAL_APPROACH:
            return self._handle_final_approach(follower_pose)

        return 0.0, 0.0

    # ── 各阶段处理 ──

    def _handle_approach(self, follower_pose: Pose2D):
        """APPROACH 阶段: MPC bearing-blend 趋近 approach_point (单机模式).

        双条件触发 ALIGN: dist < approach_capture_dist AND |e_head| < approach_capture_heading.
        保证 MPC 接管时航向已基本对准 dock 轴, 避免大角度进入 ALIGN.
        """
        ap   = self._approach_point
        cfg  = self.config
        dist = follower_pose.distance_to(ap)

        if dist < cfg.approach_capture_dist and not cfg.lock_approach:
            self._set_phase(DockingPhase.ALIGN)

        return self.mpc.solve(follower_pose,
                              self._bearing_blend_target(follower_pose, ap),
                              DockingPhase.APPROACH)

    def _handle_align(self, follower_pose: Pose2D,
                      pred_phase: DockingPhase = None):
        """ALIGN 阶段: MPC bearing-blend 从 approach_point 对齐趋近 staging_point.

        以 staging_point 为目标, bearing-blend 全程保证 e_head ≈ 0.
        机器人到达 staging 时已完成横向 + 航向精修, 是 FINAL 推入的质量门.

        进入 FINAL 条件:
            到达 staging 附近 (dist < 0.08m) 且
            连续 align_dwell_steps 步满足 |e_lat| + |e_head| 在阈值内
            AND 前车已 DOCKED
        """
        staging = self._staging_point
        cfg     = self.config

        _, e_lat, e_head = decompose_error(follower_pose, staging)
        dist = follower_pose.distance_to(staging)

        pred_docked = (pred_phase is None or pred_phase == DockingPhase.DOCKED)

        step_aligned = (dist < 0.08 and
                        abs(e_lat)  < cfg.align_to_final_lat and
                        abs(e_head) < cfg.align_to_final_heading)
        if step_aligned:
            self._align_dwell_count += 1
        else:
            self._align_dwell_count = 0

        if self._align_dwell_count >= cfg.align_dwell_steps and pred_docked:
            self._set_phase(DockingPhase.FINAL_APPROACH)

        return self.mpc.solve(follower_pose,
                              self._bearing_blend_target(follower_pose, staging),
                              DockingPhase.ALIGN)

    def _handle_final_approach(self, follower_pose: Pose2D):
        """FINAL_APPROACH 阶段: 沿对接轴直线趋近, 三维度独立检查."""
        target = self._dock_point
        e_lon, e_lat, e_head = decompose_error(follower_pose, target)
        cfg = self.config

        # 对接完成
        if (abs(e_lon) < cfg.dock_lon_threshold and
                abs(e_lat) < cfg.dock_lat_threshold and
                abs(e_head) < cfg.dock_heading_threshold):
            self._set_phase(DockingPhase.DOCKED)
            return 0.0, 0.0

        # 横向超限 → 回退 ALIGN
        if abs(e_lat) > cfg.lateral_abort_threshold:
            self._set_phase(DockingPhase.ALIGN)

        return self.mpc.solve(follower_pose, target, self._phase)

    # ── 内部工具 ──

    def _bearing_blend_target(self, robot: Pose2D, target: Pose2D,
                               blend_dist: float = 1.2) -> Pose2D:
        """返回 bearing-blend 目标位姿.

        目标位置固定为 target.xy; 目标航向从"指向 target 的方位角"
        平滑插值到 target.theta, 插值比例由距离决定:
            dist >= blend_dist: 纯方位角 (机器人正对目标方向前进)
            dist -> 0         : 纯 target.theta (轴向对齐)

        这保证了 MPC 全程 e_head ≈ 0, 自然规划出弧线切入轨迹.
        """
        dist    = robot.distance_to(target)
        bearing = math.atan2(target.y - robot.y, target.x - robot.x)
        alpha   = min(dist / blend_dist, 1.0)
        heading = normalize_angle(
            target.theta + alpha * normalize_angle(bearing - target.theta)
        )
        return Pose2D(target.x, target.y, heading)

    def _set_phase(self, new_phase: DockingPhase):
        """切换阶段并重置计时和驻留计数. 保留 MPC warm-start 以保证速度连续."""
        self._phase = new_phase
        self._phase_step_count = 0
        self._align_dwell_count = 0

    def _check_timeout(self) -> bool:
        """检查当前阶段是否超时 (基于仿真步数)."""
        elapsed_sim = self._phase_step_count * self.config.mpc_dt
        cfg = self.config

        if self._phase == DockingPhase.APPROACH:
            return elapsed_sim > cfg.approach_timeout
        elif self._phase == DockingPhase.ALIGN:
            return elapsed_sim > cfg.align_timeout
        elif self._phase == DockingPhase.FINAL_APPROACH:
            return elapsed_sim > cfg.final_timeout

        return False

    def get_debug_info(self) -> dict:
        """返回调试信息字典."""
        return {
            "phase":          self._phase.name,
            "approach_point": self._approach_point,
            "staging_point":  self._staging_point,
            "dock_point":     self._dock_point,
            "sim_elapsed":    self._phase_step_count * self.config.mpc_dt,
        }
