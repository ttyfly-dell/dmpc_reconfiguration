"""
DMPC Approach Controller — 带碰撞避免的分布式 MPC.

用于 APPROACH 阶段, 实现平滑路径规划 + 软排斥避障.

架构: Sequential DMPC
    机器人按编号顺序依次求解.
    机器人 i 把编号 < i 的机器人的预测轨迹作为移动障碍物软惩罚加入代价函数.

参数向量 P (19 + 2 + MAX_OBS × 2 × (N+1) 元素):
    P[0:3]   = x0            (当前状态)
    P[3:6]   = x_target      (目标位姿, approach_point)
    P[6:9]   = Q_diag        (阶段代价权重)
    P[9:11]  = R_diag        (控制代价权重)
    P[11:14] = Qf_diag       (终端代价权重)
    P[14]    = v_max         (线速度上限)
    P[15:17] = u_prev        (上一周期控制: [v_prev, omega_prev])
    P[17:19] = S_diag        (增量惩罚权重: [s_dv, s_domega])
    P[19]    = r_safe_sq     (碰撞半径的平方)
    P[20]    = lambda_rep    (排斥惩罚权重)
    P[21 :]  = 障碍物预测轨迹 (MAX_OBS × 2 × (N+1) 个浮点)

碰撞惩罚:
    J_rep = lambda_rep · Σ_k Σ_j max(0, r_safe² − ||p(k) − p_j(k)||²)²
"""

import numpy as np
import casadi as ca

from .types import Pose2D, DockingPhase, DockingConfig


class DMPCApproachController:
    """
    APPROACH 阶段的 DMPC 控制器.

    与 MPCController 并存, 仅用于 APPROACH 阶段多机避障.
    单机场景 (无障碍物) 退化为普通 MPC.
    """

    def __init__(self, config: DockingConfig):
        self.config = config
        self.N = config.mpc_horizon
        self.dt = config.mpc_dt
        self.MAX_OBS = config.dmpc_max_obstacles
        self._solver = None
        self._p_size = None
        self._prev_sol = None
        self._prev_u = np.zeros(2)
        self._build_solver()

    def _build_solver(self):
        N, dt, MAX_OBS = self.N, self.dt, self.MAX_OBS

        X = ca.MX.sym("X", 3, N + 1)
        U = ca.MX.sym("U", 2, N)

        obs_param_size = MAX_OBS * 2 * (N + 1)
        p_size = 19 + 2 + obs_param_size
        self._p_size = p_size

        P = ca.MX.sym("P", p_size)

        x0       = P[0:3]
        x_target = P[3:6]
        Q_diag   = P[6:9]
        R_diag   = P[9:11]
        Qf_diag  = P[11:14]
        v_max    = P[14]         # noqa: F841 (used in constraints)
        u_prev   = P[15:17]
        S_diag   = P[17:19]

        r_safe_sq  = P[19]
        lambda_rep = P[20]

        obs_params = P[21:]

        cos_ref = ca.cos(x_target[2])
        sin_ref = ca.sin(x_target[2])

        cost = 0.0

        # ---- 阶段代价 (追踪) ----
        # 倒车惩罚系数: 强烈抑制后退, 但不完全禁止 (避免三车死局)
        _backward_penalty = 30.0
        for k in range(N):
            dx = X[0, k] - x_target[0]
            dy = X[1, k] - x_target[1]
            e_lon  = cos_ref * dx + sin_ref * dy
            e_lat  = -sin_ref * dx + cos_ref * dy
            e_head = ca.atan2(
                ca.sin(X[2, k] - x_target[2]),
                ca.cos(X[2, k] - x_target[2]),
            )
            e_body = ca.vertcat(e_lon, e_lat, e_head)
            cost += ca.mtimes([e_body.T, ca.diag(Q_diag), e_body])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_diag), U[:, k]])
            cost += _backward_penalty * ca.fmax(0.0, -U[0, k]) ** 2

            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += ca.mtimes([du.T, ca.diag(S_diag), du])

        # ---- 终端代价 ----
        dx_f   = X[0, N] - x_target[0]
        dy_f   = X[1, N] - x_target[1]
        e_lon_f  = cos_ref * dx_f + sin_ref * dy_f
        e_lat_f  = -sin_ref * dx_f + cos_ref * dy_f
        e_head_f = ca.atan2(
            ca.sin(X[2, N] - x_target[2]),
            ca.cos(X[2, N] - x_target[2]),
        )
        e_body_f = ca.vertcat(e_lon_f, e_lat_f, e_head_f)
        cost += ca.mtimes([e_body_f.T, ca.diag(Qf_diag), e_body_f])

        # ---- 软排斥避障代价 ----
        # J_rep = lambda_rep · Σ_k Σ_j fmax(0, r_safe² − dist²)²
        for j in range(MAX_OBS):
            obs_x_base = j * 2 * (N + 1)        # obs_j 的 x 数据起点
            obs_y_base = obs_x_base + (N + 1)   # obs_j 的 y 数据起点
            for k in range(N + 1):
                obs_x_k = obs_params[obs_x_base + k]
                obs_y_k = obs_params[obs_y_base + k]
                dist_sq = (X[0, k] - obs_x_k) ** 2 + (X[1, k] - obs_y_k) ** 2
                repulsion = ca.fmax(0.0, r_safe_sq - dist_sq) ** 2
                cost += lambda_rep * repulsion

        # ---- 动力学等式约束 ----
        g = [X[:, 0] - x0]
        for k in range(N):
            x_next = ca.vertcat(
                X[0, k] + U[0, k] * ca.cos(X[2, k]) * dt,
                X[1, k] + U[0, k] * ca.sin(X[2, k]) * dt,
                X[2, k] + U[1, k] * dt,
            )
            g.append(X[:, k + 1] - x_next)
        g = ca.vertcat(*g)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        self._solver = ca.nlpsol("dmpc_approach", "ipopt", {
            "f": cost, "x": opt_vars, "p": P, "g": g,
        }, {
            "ipopt.max_iter":            150,
            "ipopt.print_level":         0,
            "ipopt.sb":                  "yes",
            "print_time":                0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol":                 1e-4,
            "ipopt.acceptable_tol":      1e-3,
        })

        n_states = 3 * (N + 1)
        n_vars   = n_states + 2 * N
        self._lbx_base = np.full(n_vars, -np.inf)
        self._ubx_base = np.full(n_vars,  np.inf)
        self._lbg = np.zeros(3 * (N + 1))
        self._ubg = np.zeros(3 * (N + 1))

    def solve(self,
              current: Pose2D,
              target: Pose2D,
              obs_trajs: list = None) -> tuple:
        """
        求解 DMPC, 返回 (v, omega, pred_traj).

        Args:
            current:    Follower 当前位姿
            target:     目标位姿 (approach_point)
            obs_trajs:  其他机器人的预测轨迹列表.
                        每项为 np.array shape=(N+1, 2), 列为 [x, y].
                        None 或 [] 表示无障碍物 (退化为普通 MPC).

        Returns:
            (v, omega): 控制指令
            pred_traj:  本机预测轨迹, shape=(N+1, 2), 供后续机器人使用
        """
        N   = self.N
        cfg = self.config
        Q   = cfg.approach_Q
        R   = cfg.approach_R
        Qf  = cfg.approach_Qf
        v_max     = cfg.approach_v_max
        omega_max = cfg.approach_omega_max

        # 构造障碍物参数 (MAX_OBS × 2 × (N+1), 未使用的槽填当前位置, lambda=0 时无影响)
        MAX_OBS = self.MAX_OBS
        obs_param = np.zeros(MAX_OBS * 2 * (N + 1))

        active_obs = obs_trajs or []
        n_active = min(len(active_obs), MAX_OBS)
        for j in range(n_active):
            traj = active_obs[j]   # shape (N+1, 2)
            obs_x_base = j * 2 * (N + 1)
            obs_y_base = obs_x_base + (N + 1)
            obs_param[obs_x_base: obs_x_base + (N + 1)] = traj[:, 0]
            obs_param[obs_y_base: obs_y_base + (N + 1)] = traj[:, 1]

        # 若无活跃障碍物, 排斥权重置 0 (完全退化为普通 MPC)
        lambda_rep = cfg.dmpc_lambda_rep if n_active > 0 else 0.0
        r_safe_sq  = cfg.dmpc_r_safe ** 2

        p = np.concatenate([
            [current.x, current.y, current.theta],
            [target.x,  target.y,  target.theta],
            Q, R, Qf,
            [v_max],
            self._prev_u,
            [cfg.smooth_dv, cfg.smooth_domega],
            [r_safe_sq, lambda_rep],
            obs_param,
        ])

        n_states = 3 * (N + 1)
        lbx = self._lbx_base.copy()
        ubx = self._ubx_base.copy()
        for k in range(N):
            lbx[n_states + 2 * k]     = -v_max      # 软惩罚替代硬约束, 极端情况允许微量后退
            ubx[n_states + 2 * k]     = v_max
            lbx[n_states + 2 * k + 1] = -omega_max
            ubx[n_states + 2 * k + 1] =  omega_max

        x0_guess = self._prev_sol if self._prev_sol is not None else (
            np.array([current.x, current.y, current.theta] * (N + 1) + [0.0] * (2 * N))
        )

        sol = self._solver(
            x0=x0_guess, p=p, lbx=lbx, ubx=ubx,
            lbg=self._lbg, ubg=self._ubg,
        )
        self._prev_sol = np.array(sol["x"]).flatten()

        opt = self._prev_sol
        v     = float(opt[n_states])
        omega = float(opt[n_states + 1])
        self._prev_u = np.array([v, omega])

        X_flat = opt[:n_states]
        X_traj = X_flat.reshape(N + 1, 3)
        pred_traj = X_traj[:, :2].copy()

        return v, omega, pred_traj

    @property
    def last_control(self):
        """上一步输出的控制量 (用于 DMPC→MPC 切换时桥接)."""
        return self._prev_u

    def reset(self):
        self._prev_sol = None
        self._prev_u = np.zeros(2)
