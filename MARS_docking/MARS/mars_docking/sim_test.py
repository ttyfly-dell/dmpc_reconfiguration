#!/usr/bin/env python3
"""
MARS 对接仿真 + 可视化.

支持:
  1. 静态两车对接 (Leader 不动)
  2. 编队两车对接 (Leader 沿路径运动)
  3. 静态三车链式对接 (anchor 不动, 两台 follower 并行趋近)

用法:
    python -m mars_docking.sim_test                       # 两车静态 (弹窗)
    python -m mars_docking.sim_test --three-robot          # 三车链式 (弹窗)
    python -m mars_docking.sim_test --all --save-gif       # 全部保存 GIF
    python -m mars_docking.sim_test --scenario 7 --save-gif
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from .types import Pose2D, DockingPhase, DockingConfig
from .mpc_controller import MPCController
from .state_machine import DockingStateMachine
from .multi_robot import ChainDockingCoordinator
from .safety_layer import SafetyLayer
from .utils import (
    compute_approach_point,
    compute_staging_point,
    compute_dock_point,
    normalize_angle,
    compute_dynamic_approach_point,
    compute_dynamic_dock_point,
    compute_dynamic_staging_point,
)


# ========== 运动学仿真 ==========

def unicycle_step(pose: Pose2D, v: float, omega: float, dt: float) -> Pose2D:
    """单步差速驱动运动学 (Euler 积分)."""
    return Pose2D(
        pose.x + v * math.cos(pose.theta) * dt,
        pose.y + v * math.sin(pose.theta) * dt,
        normalize_angle(pose.theta + omega * dt),
    )


# ========== 绘图工具 ==========

# 颜色常量
ANCHOR_COLOR = "#E91E63"
FOLLOWER_COLORS = ["#2196F3", "#4CAF50", "#9C27B0", "#FF9800"]
FOLLOWER_LABELS = ["B", "C", "D", "E"]
PHASE_COLORS = {
    DockingPhase.APPROACH: "#2196F3",
    DockingPhase.ALIGN: "#FF9800",
    DockingPhase.FINAL_APPROACH: "#4CAF50",
    DockingPhase.DOCKED: "#8BC34A",
    DockingPhase.FAILED: "#F44336",
    DockingPhase.IDLE: "#9E9E9E",
}


def draw_robot(ax, pose: Pose2D, config: DockingConfig,
               color="blue", label=None, leg_raised=False):
    """绘制机器人: 圆角矩形主体 + 前肢."""
    x, y, theta = pose.x, pose.y, pose.theta
    L, W = config.robot_length, config.robot_width
    leg_L, leg_W = config.leg_length, config.leg_width
    deg = math.degrees(theta)
    t = Affine2D().rotate_deg(deg).translate(x, y) + ax.transData

    # 主体
    body = patches.FancyBboxPatch(
        (-L/2, -W/2), L, W, boxstyle="round,pad=0.01",
        lw=1.2, ec=color, fc=color, alpha=0.25)
    body.set_transform(t)
    ax.add_patch(body)
    outline = patches.Rectangle((-L/2, -W/2), L, W, lw=1.5, ec=color, fc="none")
    outline.set_transform(t)
    ax.add_patch(outline)

    # 前肢
    lc = "#FF5722" if leg_raised else color
    la = 0.7 if leg_raised else 0.4
    leg = patches.Rectangle((L/2, -leg_W/2), leg_L, leg_W, lw=1, ec=lc, fc=lc, alpha=la)
    leg.set_transform(t)
    ax.add_patch(leg)

    # 朝向箭头
    ts = W * 0.15
    fx, fy = x + L/2 * math.cos(theta), y + L/2 * math.sin(theta)
    ax.arrow(fx, fy, ts*math.cos(theta), ts*math.sin(theta),
             head_width=ts*1.5, head_length=ts, fc=color, ec=color, alpha=0.6)

    if label:
        ax.text(x, y + W*0.7, label, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")


def _draw_info_panel(ax, info_lines):
    """在 axes 上绘制信息面板."""
    y = 0.95
    for item in info_lines:
        if item[0] == "":
            y -= 0.02
            continue
        lbl, val = item[0], item[1] if len(item) > 1 else ""
        clr = item[2] if len(item) > 2 else "black"
        if "===" in lbl or "---" in lbl:
            ax.text(0, y, lbl, fontsize=10, transform=ax.transAxes,
                    va="top", fontweight="bold", color=clr)
        elif lbl == "Phase":
            ax.text(0, y, "Phase: ", fontsize=10, transform=ax.transAxes, va="top")
            ax.text(0.28, y, val, fontsize=11, transform=ax.transAxes,
                    va="top", color=clr, fontweight="bold")
        else:
            ax.text(0, y, f"{lbl}: {val}", fontsize=10,
                    transform=ax.transAxes, va="top", color=clr)
        y -= 0.038


def _compute_view_bounds(all_x, all_y, config):
    """计算绘图边界."""
    margin = 0.3 + config.robot_length / 2 + config.leg_length
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range < x_range * 0.5:
        mid = (y_min + y_max) / 2
        y_min, y_max = mid - x_range * 0.3, mid + x_range * 0.3
    return x_min, x_max, y_min, y_max


def _save_and_show(fig, anim, save_gif, show, num_frames):
    """保存 GIF 和/或显示窗口."""
    if save_gif:
        print(f"  Saving GIF to {save_gif} ({num_frames} frames)...")
        anim.save(save_gif, writer="pillow", fps=30, dpi=100)
        print(f"  GIF saved: {save_gif}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


# ========== 两车静态对接 ==========

def run_simulation(follower_init: Pose2D, leader_pose: Pose2D,
                   config: DockingConfig = None, max_steps: int = 3000,
                   animate: bool = True, save_gif: str = None,
                   estop_at_step: int = None):
    """两车对接仿真 (Leader 可匀速前进)."""
    config = config or DockingConfig()
    dt = config.mpc_dt

    mpc = MPCController(config)
    sm = DockingStateMachine(config, mpc)
    sm.start(leader_pose)
    safety = SafetyLayer(
        v_limit=config.safety_v_limit,
        omega_limit=config.safety_omega_limit,
    )

    leader = leader_pose.copy()
    fp = follower_init.copy()
    leader_traj = [leader.copy()]
    traj, phases = [fp.copy()], [sm.phase]
    vels_v, vels_w = [], []
    post_dock_steps = 0           # 对接完成后编队行驶计步
    post_dock_max = int(2.0 / dt) # 编队行驶展示 2 秒

    for step in range(max_steps):
        # 头车匀速前进
        if config.anchor_v > 0:
            leader.x += config.anchor_v * math.cos(leader.theta) * dt
            leader.y += config.anchor_v * math.sin(leader.theta) * dt
        leader_traj.append(leader.copy())

        v, w = sm.update(fp, leader)
        if estop_at_step is not None and step >= estop_at_step and not safety.estop_active:
            safety.trigger_estop(reason=f"sim estop at step={step}")
        v, w = safety.apply(v, w)
        vels_v.append(v)
        vels_w.append(w)

        if sm.phase == DockingPhase.DOCKED:
            # 刚性连接: follower 锁定到 leader 的 dock_point
            fp = compute_dock_point(leader, config)
            post_dock_steps += 1
            if post_dock_steps >= post_dock_max:
                traj.append(fp.copy())
                phases.append(sm.phase)
                break
        else:
            fp = unicycle_step(fp, v, w, dt)

        traj.append(fp.copy())
        phases.append(sm.phase)
        if sm.phase == DockingPhase.FAILED:
            break

    dock_pt = compute_dock_point(leader, config)
    result = {
        "success": sm.phase == DockingPhase.DOCKED,
        "total_time": (step + 1) * dt,
        "total_steps": step + 1,
        "final_distance": fp.distance_to(dock_pt),
        "final_heading_error_deg": math.degrees(
            abs(normalize_angle(fp.theta - leader.theta))),
        "trajectory": traj, "phases": phases,
        "leader_traj": leader_traj,
        "velocities_v": vels_v, "velocities_omega": vels_w,
        "final_pose": fp,
    }
    if animate or save_gif:
        show = animate if save_gif is None else False
        _animate_two_robot(result, follower_init, leader_pose, config,
                           save_gif=save_gif, show=show)
    return result


def _animate_two_robot(result, follower_init, leader_pose, config,
                       save_gif=None, show=True):
    """两车对接动画 (Leader 可运动)."""
    traj = result["trajectory"]
    leader_traj = result.get("leader_traj", [leader_pose] * len(traj))
    phases = result["phases"]
    vv, vw = result["velocities_v"], result["velocities_omega"]
    dt = config.mpc_dt

    # 视野包含 leader 全程轨迹
    all_x = [p.x for p in traj] + [p.x for p in leader_traj]
    all_y = [p.y for p in traj] + [p.y for p in leader_traj]
    x0, x1, y0, y1 = _compute_view_bounds(all_x, all_y, config)

    fig = plt.figure(figsize=(14, 8))
    ax_m = fig.add_axes([0.05, 0.25, 0.55, 0.7])
    ax_v = fig.add_axes([0.05, 0.05, 0.55, 0.18])
    ax_i = fig.add_axes([0.65, 0.05, 0.33, 0.9])
    ax_i.axis("off")

    total = len(traj)
    skip = max(1, total // 400)
    moving = config.anchor_v > 0

    def update(fi):
        i = min(fi * skip, total - 1)
        pi = traj[i]
        li = leader_traj[min(i, len(leader_traj) - 1)]
        ph = phases[min(i, len(phases)-1)]

        ax_m.cla()
        ax_m.set_xlim(x0, x1); ax_m.set_ylim(y0, y1)
        ax_m.set_aspect("equal"); ax_m.grid(True, alpha=0.3)
        title = "MARS Docking (Moving Leader)" if moving else "MARS Docking (Static Leader)"
        ax_m.set_title(title, fontsize=13)

        for j in range(1, i+1):
            c = PHASE_COLORS.get(phases[j], "#9E9E9E")
            ax_m.plot([traj[j-1].x, traj[j].x], [traj[j-1].y, traj[j].y],
                      color=c, lw=1.5, alpha=0.6)

        draw_robot(ax_m, li, config, color=ANCHOR_COLOR, label="A")
        draw_robot(ax_m, pi, config, color=FOLLOWER_COLORS[0], label="B",
                   leg_raised=ph in (DockingPhase.FINAL_APPROACH, DockingPhase.DOCKED))

        nv = min(i, len(vv))
        ax_v.cla()
        if nv > 0:
            ta = np.arange(nv) * dt
            ax_v.plot(ta, vv[:nv], color="#2196F3", label="v", lw=1)
            ax_v.plot(ta, vw[:nv], color="#FF9800", label="\u03c9", lw=1)
            ax_v.axvline(x=i*dt, color="gray", ls="--", alpha=0.5)
        ax_v.set_xlim(0, max(len(vv)*dt, 0.1))
        ax_v.set_ylabel("Velocity"); ax_v.set_xlabel("Time (s)")
        if nv > 0:
            ax_v.legend(loc="upper right", fontsize=8)
        ax_v.grid(True, alpha=0.3)

        dock_i = compute_dock_point(li, config)
        ax_i.cla(); ax_i.axis("off")
        info = [
            ("Time", f"{i*dt:.2f} s"),
            ("Phase", ph.name, PHASE_COLORS.get(ph, "#9E9E9E")),
            ("", ""),
            ("Dist to dock", f"{pi.distance_to(dock_i):.3f} m"),
            ("Heading err", f"{math.degrees(abs(normalize_angle(pi.theta - li.theta))):.1f}\u00b0"),
        ]
        if result["success"]:
            info += [("", ""), ("=== RESULT ===", ""), ("Status", "DOCKED"),
                     ("Total time", f"{result['total_time']:.2f} s"),
                     ("Pos error", f"{result['final_distance']*100:.1f} cm")]
        _draw_info_panel(ax_i, info)

    nf = total // skip + 1
    anim = FuncAnimation(fig, update, frames=nf, interval=20, repeat=False)
    return _save_and_show(fig, anim, save_gif, show, nf)


# ========== 三车链式对接 ==========

def run_chain_simulation(anchor_pose: Pose2D, follower_inits: list,
                         config: DockingConfig = None,
                         max_steps: int = 4000,
                         animate: bool = True, save_gif: str = None,
                         estop_at_step: int = None):
    """
    N+1 车链式对接仿真.

    anchor 固定, follower 逐个激活, 每台追踪前车实际位姿.
    """
    config = config or DockingConfig()
    dt = config.mpc_dt
    n = len(follower_inits)

    coord = ChainDockingCoordinator(anchor_pose, n, config)
    safety_layers = [
        SafetyLayer(
            v_limit=config.safety_v_limit,
            omega_limit=config.safety_omega_limit,
        )
        for _ in range(n)
    ]
    anchor = anchor_pose.copy()
    poses = [fp.copy() for fp in follower_inits]

    # 记录历史
    anchor_traj = [anchor.copy()]
    trajectories = [[fp.copy()] for fp in follower_inits]
    phase_hist = [list(coord.phases)]
    vel_hist = [[] for _ in range(n)]
    post_dock_steps = 0           # 全部对接完成后编队行驶计步
    post_dock_max = int(2.0 / dt) # 编队行驶展示 2 秒

    for step in range(max_steps):
        # 头车匀速前进
        if config.anchor_v > 0:
            anchor.x += config.anchor_v * math.cos(anchor.theta) * dt
            anchor.y += config.anchor_v * math.sin(anchor.theta) * dt
        anchor_traj.append(anchor.copy())

        cmds = coord.update(poses, anchor_pose=anchor)
        for i in range(n):
            v, w = cmds[i]
            if estop_at_step is not None and step >= estop_at_step and not safety_layers[i].estop_active:
                safety_layers[i].trigger_estop(reason=f"sim estop at step={step}")
            v, w = safety_layers[i].apply(v, w)
            vel_hist[i].append((v, w))

            if coord.phases[i] == DockingPhase.DOCKED:
                # 刚性连接: 已对接的 follower 锁定到前车 dock_point
                pred = anchor if i == 0 else poses[i - 1]
                dp = compute_dynamic_dock_point(pred, anchor, config)
                poses[i] = dp.copy()
            else:
                poses[i] = unicycle_step(poses[i], v, w, dt)

            trajectories[i].append(poses[i].copy())
        phase_hist.append(list(coord.phases))

        if coord.done:
            post_dock_steps += 1
            if post_dock_steps >= post_dock_max:
                break

    total_time = (step + 1) * dt

    # 最终 dock 点 (基于前车的实际最终位姿, 动态计算)
    dock_pts = coord.dock_points(poses)

    follower_results = []
    for i in range(n):
        dist = poses[i].distance_to(dock_pts[i])
        herr = abs(normalize_angle(poses[i].theta - dock_pts[i].theta))
        follower_results.append({
            "phase": coord.phases[i].name,
            "final_distance": dist,
            "final_heading_error_deg": math.degrees(herr),
        })

    result = {
        "success": coord.all_docked,
        "total_time": total_time,
        "total_steps": step + 1,
        "n_followers": n,
        "anchor_pose": anchor_pose,
        "anchor_traj": anchor_traj,
        "trajectories": trajectories,
        "phase_hist": phase_hist,
        "vel_hist": vel_hist,
        "final_poses": [p.copy() for p in poses],
        "follower_results": follower_results,
    }

    if animate or save_gif:
        show = animate if save_gif is None else False
        _animate_chain(result, follower_inits, config,
                       save_gif=save_gif, show=show)
    return result


def _animate_chain(result, follower_inits, config, save_gif=None, show=True):
    """三车链式对接动画."""
    anchor_traj = result.get("anchor_traj", [result["anchor_pose"]])
    trajs = result["trajectories"]
    ph_hist = result["phase_hist"]
    vel_hist = result["vel_hist"]
    n = result["n_followers"]
    dt = config.mpc_dt

    # 视野 (包含 anchor 全程轨迹)
    all_x = [p.x for p in anchor_traj] + [p.x for t in trajs for p in t]
    all_y = [p.y for p in anchor_traj] + [p.y for t in trajs for p in t]
    x0, x1, y0, y1 = _compute_view_bounds(all_x, all_y, config)

    fig = plt.figure(figsize=(14, 8))
    ax_m = fig.add_axes([0.05, 0.25, 0.55, 0.7])
    ax_v = fig.add_axes([0.05, 0.05, 0.55, 0.18])
    ax_i = fig.add_axes([0.65, 0.05, 0.33, 0.9])
    ax_i.axis("off")

    total = len(trajs[0])
    skip = max(1, total // 400)

    def update(fi):
        i = min(fi * skip, total - 1)
        phases_i = ph_hist[min(i, len(ph_hist) - 1)]

        # 当前帧 anchor 位姿
        anchor_i = anchor_traj[min(i, len(anchor_traj) - 1)]

        # 当前帧各车位姿
        cur_poses = [trajs[fi_idx][min(i, len(trajs[fi_idx])-1)]
                     for fi_idx in range(n)]

        # dock / staging: 从当前帧各前车实时位姿动态推算
        leaders_i = [anchor_i] + [trajs[k][min(i, len(trajs[k]) - 1)]
                                   for k in range(n - 1)]
        approach_pts_i = [compute_dynamic_approach_point(leaders_i[k], anchor_i, config)
                          for k in range(n)]
        dock_pts_i     = [compute_dynamic_dock_point(leaders_i[k], anchor_i, config)
                          for k in range(n)]
        staging_pts_i  = [compute_dynamic_staging_point(leaders_i[k], anchor_i, config)
                          for k in range(n)]

        # ---- 主视图 ----
        ax_m.cla()
        ax_m.set_xlim(x0, x1); ax_m.set_ylim(y0, y1)
        ax_m.set_aspect("equal"); ax_m.grid(True, alpha=0.3)
        ax_m.set_title(f"MARS Chain Docking ({n+1} robots)", fontsize=13)

        # 轨迹
        for fi_idx in range(n):
            tr = trajs[fi_idx]
            for j in range(1, min(i+1, len(tr))):
                ph_j = ph_hist[min(j, len(ph_hist)-1)][fi_idx]
                ax_m.plot([tr[j-1].x, tr[j].x], [tr[j-1].y, tr[j].y],
                          color=PHASE_COLORS.get(ph_j, "#9E9E9E"),
                          lw=1.5, alpha=0.5)

        # Anchor (动态位置)
        draw_robot(ax_m, anchor_i, config, color=ANCHOR_COLOR, label="A")

        # Followers
        for fi_idx in range(n):
            ph_fi = phases_i[fi_idx]
            leg_up = ph_fi in (DockingPhase.FINAL_APPROACH, DockingPhase.DOCKED)
            c = FOLLOWER_COLORS[fi_idx % len(FOLLOWER_COLORS)]
            lb = FOLLOWER_LABELS[fi_idx] if fi_idx < len(FOLLOWER_LABELS) else f"F{fi_idx}"
            draw_robot(ax_m, cur_poses[fi_idx], config, color=c, label=lb,
                       leg_raised=leg_up)

        # 起点
        for fi_idx in range(n):
            c = FOLLOWER_COLORS[fi_idx % len(FOLLOWER_COLORS)]
            ax_m.plot(follower_inits[fi_idx].x, follower_inits[fi_idx].y,
                      "s", color=c, ms=5, alpha=0.3)

        # ---- 速度曲线 ----
        ax_v.cla()
        nv = min(i, len(vel_hist[0]))
        if nv > 0:
            ta = np.arange(nv) * dt
            for fi_idx in range(n):
                c = FOLLOWER_COLORS[fi_idx % len(FOLLOWER_COLORS)]
                lb = FOLLOWER_LABELS[fi_idx] if fi_idx < len(FOLLOWER_LABELS) else f"F{fi_idx}"
                fv = [v[0] for v in vel_hist[fi_idx][:nv]]
                ax_v.plot(ta, fv, color=c, label=f"{lb}.v", lw=1)
            ax_v.axvline(x=i*dt, color="gray", ls="--", alpha=0.5)
        ax_v.set_xlim(0, max(len(vel_hist[0])*dt, 0.1))
        ax_v.set_ylabel("v (m/s)"); ax_v.set_xlabel("Time (s)")
        ax_v.legend(loc="upper right", fontsize=7); ax_v.grid(True, alpha=0.3)

        # ---- 信息面板 ----
        ax_i.cla(); ax_i.axis("off")
        info = [("Time", f"{i*dt:.2f} s")]
        for fi_idx in range(n):
            lb = FOLLOWER_LABELS[fi_idx] if fi_idx < len(FOLLOWER_LABELS) else f"F{fi_idx}"
            ph_fi = phases_i[fi_idx]
            dist = cur_poses[fi_idx].distance_to(dock_pts_i[fi_idx])
            herr = math.degrees(abs(normalize_angle(
                cur_poses[fi_idx].theta - anchor_i.theta)))
            pc = PHASE_COLORS.get(ph_fi, "#9E9E9E")
            info += [
                ("", ""),
                (f"--- {lb} ---", "", pc),
                ("Phase", ph_fi.name, pc),
                ("Dist to dock", f"{dist:.3f} m"),
                ("Heading err", f"{herr:.1f}\u00b0"),
            ]

        if result["success"] and i >= total - 2:
            info += [("", ""), ("=== ALL DOCKED ===", ""),
                     ("Total time", f"{result['total_time']:.2f} s")]
            for fi_idx in range(n):
                lb = FOLLOWER_LABELS[fi_idx]
                fr = result["follower_results"][fi_idx]
                info.append((f"{lb} err", f"{fr['final_distance']*100:.1f} cm"))
        _draw_info_panel(ax_i, info)

    nf = total // skip + 1
    anim = FuncAnimation(fig, update, frames=nf, interval=20, repeat=False)
    return _save_and_show(fig, anim, save_gif, show, nf)


# ========== 测试场景 ==========

# --- 两车静态 ---

def test_scenario_behind(save_gif=None, config=None, estop_at_step=None):
    """1: Follower 在 Leader 正后方."""
    _header("Scene 1: Follower behind Leader")
    r = run_simulation(
        Pose2D(-1.0, 0.0, 0.0), Pose2D(0, 0, 0),
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_result(r); return r

def test_scenario_side(save_gif=None, config=None, estop_at_step=None):
    """2: Follower 在侧后方."""
    _header("Scene 2: Follower at the side")
    r = run_simulation(
        Pose2D(-0.5, 0.6, -math.pi/3), Pose2D(0, 0, 0),
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_result(r); return r

def test_scenario_diagonal(save_gif=None, config=None, estop_at_step=None):
    """3: Follower 在斜后方."""
    _header("Scene 3: Follower diagonal-behind")
    r = run_simulation(
        Pose2D(-0.8, -0.5, math.pi*0.7), Pose2D(0, 0, 0),
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_result(r); return r

def test_scenario_angled_leader(save_gif=None, config=None, estop_at_step=None):
    """4: Leader 朝向非零."""
    _header("Scene 4: Leader rotated, Follower offset")
    r = run_simulation(
        Pose2D(-0.6, -0.3, math.pi/6), Pose2D(0, 0, math.pi/4),
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_result(r); return r

# --- 三车链式 ---

def test_chain_spread(save_gif=None, config=None, estop_at_step=None):
    """7: B 从左上方, C 从右下方远距离趋近 (路径交叉, DMPC 避障测试)."""
    _header("Scene 7: Chain 3-robot - far spread, crossing paths")
    r = run_chain_simulation(
        Pose2D(0, 0, 0),
        [Pose2D(-1.5, 0.8, -math.pi/6), Pose2D(-0.6, -1.5, math.pi/2)],
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_chain_result(r); return r

def test_chain_symmetric(save_gif=None, config=None, estop_at_step=None):
    """8: B 和 C 从较远的两侧对称趋近."""
    _header("Scene 8: Chain 3-robot - far symmetric approach")
    r = run_chain_simulation(
        Pose2D(0, 0, 0),
        [Pose2D(-1.0, 1.2, -math.pi/3), Pose2D(-1.8, -0.8, math.pi/4)],
        config=config, save_gif=save_gif, estop_at_step=estop_at_step)
    _print_chain_result(r); return r


# ========== 打印工具 ==========

def _header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)

def _print_result(result):
    s = "DOCKED" if result["success"] else "FAILED"
    print(f"  {s}  |  {result['total_time']:.2f}s  |  "
          f"pos_err={result['final_distance']*100:.1f}cm  |  "
          f"heading_err={result['final_heading_error_deg']:.2f}\u00b0")
    print()

def _print_chain_result(result):
    s = "ALL DOCKED" if result["success"] else "FAILED"
    print(f"  {s}  |  {result['total_time']:.2f}s  ({result['total_steps']} steps)")
    for i, fr in enumerate(result["follower_results"]):
        lb = FOLLOWER_LABELS[i] if i < len(FOLLOWER_LABELS) else f"F{i}"
        print(f"    {lb}: {fr['phase']:14s}  pos_err={fr['final_distance']*100:.1f}cm  "
              f"heading_err={fr['final_heading_error_deg']:.2f}\u00b0")
    print()


# ========== 入口 ==========

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description="MARS Docking Simulation")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Scenario number (1-4: 2-robot, 7-8: 3-robot)")
    parser.add_argument("--three-robot", action="store_true",
                        help="Run 3-robot chain scenarios")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--lock-approach", action="store_true",
                        help="Lock all followers in APPROACH phase (no ALIGN/FINAL)")
    parser.add_argument("--estop-at-step", type=int, default=None,
                        help="Trigger latched emergency stop at this simulation step")
    args = parser.parse_args()

    print("MARS Docking - Simulation Test\n")

    from mars_docking.types import DockingConfig
    run_config = DockingConfig(lock_approach=True) if args.lock_approach else None

    gif_dir = None
    if args.save_gif:
        gif_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(gif_dir, exist_ok=True)
        print(f"GIF output: {gif_dir}\n")

    scenarios = {
        1: ("behind", test_scenario_behind),
        2: ("side", test_scenario_side),
        3: ("diagonal", test_scenario_diagonal),
        4: ("angled_leader", test_scenario_angled_leader),
        7: ("chain_spread", test_chain_spread),
        8: ("chain_symmetric", test_chain_symmetric),
    }

    two_robot = {k: v for k, v in scenarios.items() if k <= 4}
    three_robot = {k: v for k, v in scenarios.items() if k in (7, 8)}

    if args.scenario > 0:
        to_run = {args.scenario: scenarios[args.scenario]}
    elif args.all:
        to_run = scenarios
    elif args.three_robot:
        to_run = three_robot
    else:
        to_run = two_robot

    for idx, (name, func) in to_run.items():
        gp = os.path.join(gif_dir, f"docking_{name}.gif") if gif_dir else None
        func(save_gif=gp, config=run_config, estop_at_step=args.estop_at_step)


if __name__ == "__main__":
    main()
