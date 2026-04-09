"""
链式对接协调器 — DMPC 接近 + MPC 精确对接.

核心设计
---------
1. **动态目标 (incremental assembly)**
   每台 follower 的 dock/staging 点每步从**前车实时位姿**推算:
       dock_k  = pred_k.pos − unit·anchor_axis,   heading = anchor.θ
       stage_k = dock_k − staging_distance·anchor_axis
   效果: follower 的目标永远在前车后方, 几何上不可能超越前车.

2. **DMPC 导航 (Phase 1: APPROACH)**
   顺序式分布 MPC: 机器人按编号依次求解, 编号小的预测轨迹作为
   后续机器人的移动障碍物软排斥项, 保证路径平滑且避碰.

3. **MPC 精确对接 (Phase 2: ALIGN / FINAL_APPROACH)**
   状态机调用 MPC, 在目标坐标系分解代价, 保证轴向对齐.

4. **阶段优先级门控**
   ALIGN → FINAL_APPROACH 要求前车已 DOCKED.
   确保 FINAL 阶段前车静止, 彻底消除追尾碰撞.

前驱关系
    anchor (永久 DOCKED) → follower_0 → follower_1 → …
    pred_phase[0] = DOCKED (anchor)
    pred_phase[k] = sms[k-1].phase
"""

from .types import Pose2D, DockingPhase, DockingConfig
from .mpc_controller import MPCController
from .dmpc_controller import DMPCApproachController
from .state_machine import DockingStateMachine
from .utils import (compute_dynamic_approach_point, compute_dynamic_dock_point,
                    compute_dynamic_staging_point, compute_chain_approach_point,
                    compute_chain_dock_point, compute_chain_staging_point)


class ChainDockingCoordinator:
    """
    链式对接协调器.

    用法 (静态 anchor):
        coord = ChainDockingCoordinator(anchor, n_followers=2)
        while not coord.done:
            cmds = coord.update(follower_poses)

    用法 (动态 anchor / 遥控):
        cmds = coord.update(follower_poses, anchor_pose=new_anchor)
    """

    def __init__(self, anchor_pose: Pose2D, n_followers: int,
                 config: DockingConfig = None):
        self.config = config or DockingConfig()
        self._anchor = anchor_pose.copy()
        self._n = n_followers

        # 每台 follower 一个状态机 + 一个 DMPC 控制器
        self._sms: list[DockingStateMachine] = []
        self._dmpc: list[DMPCApproachController] = []
        for i in range(n_followers):
            mpc = MPCController(self.config)
            sm = DockingStateMachine(self.config, mpc)
            sm.start(
                approach_point=compute_chain_approach_point(anchor_pose, i + 1, self.config),
                staging_point =compute_chain_staging_point(anchor_pose, i + 1, self.config),
                dock_point    =compute_chain_dock_point(anchor_pose, i + 1, self.config),
            )
            self._sms.append(sm)
            self._dmpc.append(DMPCApproachController(self.config))

        # 存储各 follower 上一步的预测轨迹 (Sequential DMPC 障碍物源)
        # 形状: (N+1, 2), 初始化为当前位置重复
        self._pred_trajs: list = [None] * n_followers


    # ── 核心接口 ──

    def update(self, follower_poses: list,
               anchor_pose: Pose2D = None) -> list:
        """
        每个控制周期调用一次.

        APPROACH 阶段: 顺序 DMPC (机器人 0 先解, 其预测轨迹传给机器人 1, 以此类推).
        ALIGN/FINAL   阶段: 各自独立 MPC (状态机内部处理).

        Returns:
            [(v, omega), ...]: 每台 follower 的速度指令
        """
        if anchor_pose is not None and anchor_pose != self._anchor:
            self._anchor = anchor_pose.copy()

        # 前驱链: leaders[i] 是 follower i 的前车位姿
        leaders = [self._anchor] + list(follower_poses)

        # 前驱阶段: anchor 视为永久 DOCKED
        pred_phases = [DockingPhase.DOCKED] + [sm.phase for sm in self._sms]

        # Sequential DMPC: 每步重新收集本轮已解机器人的预测轨迹
        # (列表按 follower 编号增长, 后续机器人以此为移动障碍物)
        current_pred_trajs: list = []

        cmds = []
        for i, (sm, fp) in enumerate(zip(self._sms, follower_poses)):
            pred_phase = pred_phases[i]
            pred_ref = self._stable_pred_ref(i, leaders, pred_phase)
            ap = compute_dynamic_approach_point(pred_ref, self._anchor, self.config)
            sp = compute_dynamic_staging_point(pred_ref, self._anchor, self.config)
            dp = compute_dynamic_dock_point(pred_ref, self._anchor, self.config)

            if sm.phase == DockingPhase.APPROACH:
                # Phase 1: Sequential DMPC 导航至动态 approach_point
                # 以编号更小的机器人本轮预测轨迹作为移动障碍物
                sm.check_and_advance_approach(fp, pred_phase,
                                              approach_point=ap,
                                              staging_point=sp, dock_point=dp)
                if sm.phase == DockingPhase.APPROACH:
                    v, w, pred = self._dmpc[i].solve(fp, ap,
                                                      obs_trajs=current_pred_trajs)
                    current_pred_trajs.append(pred)
                    self._pred_trajs[i] = pred
                else:
                    # 刚切换到 ALIGN: MPC 接管, 桥接 DMPC 最后控制量
                    self._pred_trajs[i] = None
                    sm.mpc.set_prev_control(*self._dmpc[i].last_control)
                    self._dmpc[i].reset()
                    v, w = sm.update(fp, approach_point=ap,
                                     staging_point=sp, dock_point=dp,
                                     pred_phase=pred_phase)
            else:
                # Phase 2 (ALIGN / FINAL_APPROACH / DOCKED): MPC
                v, w = sm.update(fp, approach_point=ap,
                                 staging_point=sp, dock_point=dp,
                                 pred_phase=pred_phase)

            cmds.append((v, w))

        return cmds

    # ── 状态查询 ──

    @property
    def phases(self) -> list:
        return [sm.phase for sm in self._sms]

    @property
    def all_docked(self) -> bool:
        return all(p == DockingPhase.DOCKED for p in self.phases)

    @property
    def any_failed(self) -> bool:
        return any(p == DockingPhase.FAILED for p in self.phases)

    @property
    def done(self) -> bool:
        return self.all_docked or self.any_failed

    def dock_points(self, follower_poses: list) -> list:
        """各 follower 当前的动态 dock 目标 (基于前车实时位姿)."""
        leaders = [self._anchor] + list(follower_poses[:-1])
        return [compute_dynamic_dock_point(lp, self._anchor, self.config)
                for lp in leaders]

    def staging_points(self, follower_poses: list) -> list:
        """各 follower 当前的动态 staging 目标."""
        leaders = [self._anchor] + list(follower_poses[:-1])
        return [compute_dynamic_staging_point(lp, self._anchor, self.config)
                for lp in leaders]

    # ── 内部工具 ──

    def _stable_pred_ref(self, follower_idx: int,
                         leaders: list,
                         pred_phase: DockingPhase) -> Pose2D:
        """
        返回用于计算动态目标的"稳定前车参考位姿".

        前车在 ALIGN/FINAL 时用其 dock_point (固定) 而非实时位姿,
        避免前车小幅振荡时当前车目标也抖动.
        """
        pred_pose = leaders[follower_idx]

        if pred_phase in (DockingPhase.ALIGN, DockingPhase.FINAL_APPROACH):
            if follower_idx == 0:
                return self._anchor
            else:
                pred_pred_pose = leaders[follower_idx - 1]
                return compute_dynamic_dock_point(
                    pred_pred_pose, self._anchor, self.config)

        return pred_pose
