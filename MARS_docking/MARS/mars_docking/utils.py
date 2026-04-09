"""坐标变换、角度归一化等工具函数"""

import math
from .types import Pose2D, DockingConfig


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def pose_transform(base: Pose2D, local: Pose2D) -> Pose2D:
    """
    将 local 坐标系下的点转换到 base 坐标系的父坐标系 (全局).

    即: global = base ⊕ local
    """
    cos_b = math.cos(base.theta)
    sin_b = math.sin(base.theta)
    gx = base.x + cos_b * local.x - sin_b * local.y
    gy = base.y + sin_b * local.x + cos_b * local.y
    gtheta = normalize_angle(base.theta + local.theta)
    return Pose2D(gx, gy, gtheta)


def pose_inverse(pose: Pose2D) -> Pose2D:
    """
    计算位姿的逆变换.

    如果 pose 表示 A->B 的变换, 则返回 B->A 的变换.
    """
    cos_t = math.cos(pose.theta)
    sin_t = math.sin(pose.theta)
    inv_x = -(cos_t * pose.x + sin_t * pose.y)
    inv_y = -(-sin_t * pose.x + cos_t * pose.y)
    inv_theta = normalize_angle(-pose.theta)
    return Pose2D(inv_x, inv_y, inv_theta)


def compute_dock_point(leader_pose: Pose2D, config: DockingConfig) -> Pose2D:
    """
    计算 Follower 中心的最终对接停止位置.

    Follower 中心应位于 Leader 后方:
        offset = robot_length/2 (Leader后半) + dock_gap + robot_length/2 (Follower前半)
               = robot_length + dock_gap

    对接完成后, Follower 前肢跨越 dock_gap 插入 Leader 背面.

    返回的位姿航向与 Leader 一致.
    """
    offset = config.robot_length + config.dock_gap
    local = Pose2D(x=-offset, y=0.0, theta=0.0)
    return pose_transform(leader_pose, local)


def compute_staging_point(leader_pose: Pose2D, config: DockingConfig) -> Pose2D:
    """
    计算预对接点 (staging point) — Follower 中心的目标.

    位于 dock_point 再往后 staging_distance 处, 航向与 Leader 一致.
    """
    offset = config.robot_length + config.dock_gap + config.staging_distance
    local = Pose2D(x=-offset, y=0.0, theta=0.0)
    return pose_transform(leader_pose, local)


def compute_dynamic_dock_point(pred_pose: Pose2D, anchor_pose: Pose2D,
                               config: DockingConfig) -> Pose2D:
    """
    从**前车当前位姿**计算当前车的对接目标点.

    位置: 沿 anchor 轴向, 在前车中心后方 unit = robot_length + dock_gap.
    航向: 锁定为 anchor 的航向 (整条链共轴对齐).

    相比固定链式目标, 此函数保证:
    - 当前车的目标永远在前车后方 → 不会超越前车
    - 航向与 anchor 一致 → 轴向对齐精度不变
    """
    unit = config.robot_length + config.dock_gap
    cos_a = math.cos(anchor_pose.theta)
    sin_a = math.sin(anchor_pose.theta)
    return Pose2D(
        pred_pose.x - unit * cos_a,
        pred_pose.y - unit * sin_a,
        anchor_pose.theta,
    )


def compute_dynamic_staging_point(pred_pose: Pose2D, anchor_pose: Pose2D,
                                  config: DockingConfig) -> Pose2D:
    """
    从**前车当前位姿**计算当前车的预对接点 (staging point).

    = dynamic_dock_point 再往后 staging_distance.
    """
    offset = config.robot_length + config.dock_gap + config.staging_distance
    cos_a = math.cos(anchor_pose.theta)
    sin_a = math.sin(anchor_pose.theta)
    return Pose2D(
        pred_pose.x - offset * cos_a,
        pred_pose.y - offset * sin_a,
        anchor_pose.theta,
    )


def compute_chain_dock_point(anchor: Pose2D, rank: int,
                             config: DockingConfig) -> Pose2D:
    """
    计算链式对接中第 rank 台 follower 的固定对接点.

    rank=1 → 第 1 台 (紧贴 anchor 后方)
    rank=2 → 第 2 台 (紧贴第 1 台后方)
    以此类推.

    所有对接点固定从 anchor 位姿推算, 保证整条链共轴对齐.
    """
    unit = config.robot_length + config.dock_gap
    local = Pose2D(x=-rank * unit, y=0.0, theta=0.0)
    return pose_transform(anchor, local)


def compute_approach_point(leader_pose: Pose2D, config: DockingConfig) -> Pose2D:
    """
    计算 DMPC 导航的远点 (approach point).

    位于 dock_point 再往后 approach_distance 处 (= staging_point 再往后
    approach_distance - staging_distance), 航向与 Leader 一致.
    """
    offset = config.robot_length + config.dock_gap + config.approach_distance
    local = Pose2D(x=-offset, y=0.0, theta=0.0)
    return pose_transform(leader_pose, local)


def compute_dynamic_approach_point(pred_pose: Pose2D, anchor_pose: Pose2D,
                                   config: DockingConfig) -> Pose2D:
    """从**前车当前位姿**计算当前车的 approach_point (动态版)."""
    offset = config.robot_length + config.dock_gap + config.approach_distance
    cos_a = math.cos(anchor_pose.theta)
    sin_a = math.sin(anchor_pose.theta)
    return Pose2D(
        pred_pose.x - offset * cos_a,
        pred_pose.y - offset * sin_a,
        anchor_pose.theta,
    )


def compute_chain_approach_point(anchor: Pose2D, rank: int,
                                 config: DockingConfig) -> Pose2D:
    """计算链式对接中第 rank 台 follower 的固定 approach_point."""
    unit = config.robot_length + config.dock_gap
    offset = rank * unit + config.approach_distance
    local = Pose2D(x=-offset, y=0.0, theta=0.0)
    return pose_transform(anchor, local)


def compute_chain_staging_point(anchor: Pose2D, rank: int,
                                config: DockingConfig) -> Pose2D:
    """
    计算链式对接中第 rank 台 follower 的固定预对接点 (staging point).

    staging = dock + staging_distance (沿链轴方向再往后).
    """
    unit = config.robot_length + config.dock_gap
    offset = rank * unit + config.staging_distance
    local = Pose2D(x=-offset, y=0.0, theta=0.0)
    return pose_transform(anchor, local)


def decompose_error(follower_pose: Pose2D, target_pose: Pose2D):
    """
    将位姿误差分解为目标坐标系下的纵向/横向/航向分量.

    在目标坐标系中:
    - e_lon: 纵向误差 (沿目标朝向方向, 正值表示 follower 在目标后方)
    - e_lat: 横向误差 (垂直于目标朝向, 正值表示 follower 在目标左侧)
    - e_head: 航向误差 (follower 航向 - 目标航向, 归一化)

    Args:
        follower_pose: Follower 的全局位姿
        target_pose: 目标的全局位姿

    Returns:
        (e_lon, e_lat, e_head) 元组
    """
    # 全局坐标系下的位置差
    dx = follower_pose.x - target_pose.x
    dy = follower_pose.y - target_pose.y

    # 旋转到目标坐标系
    cos_t = math.cos(target_pose.theta)
    sin_t = math.sin(target_pose.theta)
    e_lon = cos_t * dx + sin_t * dy
    e_lat = -sin_t * dx + cos_t * dy

    # 航向误差
    e_head = normalize_angle(follower_pose.theta - target_pose.theta)

    return e_lon, e_lat, e_head
