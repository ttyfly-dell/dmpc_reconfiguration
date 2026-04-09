"""
基于 CasADi 的非线性 MPC 控制器.

运动学模型: Unicycle (差速驱动)
    x'     = v * cos(theta)
    y'     = v * sin(theta)
    theta' = omega

状态: X = [x, y, theta]
控制: U = [v, omega]

代价函数在 **目标坐标系** 下分解:
    Q = diag(q_lon, q_lat, q_head)
即 Q[0] 控制纵向权重, Q[1] 控制横向权重, Q[2] 控制航向权重.
这保证了对任意 dock 轴朝向都旋转不变.

控制增量惩罚 ΔU:
    Σ_k (u_k − u_{k−1})^T S (u_k − u_{k−1})
    其中 u_{−1} 为上一周期实际输出, 保证步间平滑.

参数向量 P (19 元素):
    P[0:3]  = x0            (当前状态)
    P[3:6]  = x_target      (目标位姿)
    P[6:9]  = Q_diag        (阶段代价权重)
    P[9:11] = R_diag        (控制代价权重)
    P[11:14]= Qf_diag       (终端代价权重)
    P[14]   = v_max         (线速度上限, 用于约束参数化)
    P[15:17]= u_prev        (上一周期控制: [v_prev, omega_prev])
    P[17:19]= S_diag        (增量惩罚权重: [s_dv, s_domega])
"""

import numpy as np
import casadi as ca

from .types import Pose2D, DockingPhase, DockingConfig


class MPCController:
    """MPC 控制器, 支持按对接阶段切换权重和约束."""

    def __init__(self, config: DockingConfig):
        self.config = config
        self.N = config.mpc_horizon
        self.dt = config.mpc_dt
        self._solver = None
        self._lbx_base = None
        self._ubx_base = None
        self._lbg = None
        self._ubg = None
        self._prev_sol = None
        self._prev_u = np.zeros(2)
        self._build_solver()

    def _build_solver(self):
        N, dt = self.N, self.dt

        X = ca.MX.sym("X", 3, N + 1)
        U = ca.MX.sym("U", 2, N)

        P = ca.MX.sym("P", 19)
        x0       = P[0:3]
        x_target = P[3:6]
        Q_diag   = P[6:9]
        R_diag   = P[9:11]
        Qf_diag  = P[11:14]
        v_max    = P[14]
        u_prev   = P[15:17]
        S_diag   = P[17:19]

        cos_ref = ca.cos(x_target[2])
        sin_ref = ca.sin(x_target[2])

        cost = 0.0

        # ---- 阶段代价 + 控制代价 + 增量代价 ----
        for k in range(N):
            dx = X[0, k] - x_target[0]
            dy = X[1, k] - x_target[1]
            e_lon = cos_ref * dx + sin_ref * dy
            e_lat = -sin_ref * dx + cos_ref * dy
            e_head = ca.atan2(
                ca.sin(X[2, k] - x_target[2]),
                ca.cos(X[2, k] - x_target[2]),
            )
            e_body = ca.vertcat(e_lon, e_lat, e_head)
            cost += ca.mtimes([e_body.T, ca.diag(Q_diag), e_body])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_diag), U[:, k]])

            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += ca.mtimes([du.T, ca.diag(S_diag), du])

        # ---- 终端代价 ----
        dx_f = X[0, N] - x_target[0]
        dy_f = X[1, N] - x_target[1]
        e_lon_f = cos_ref * dx_f + sin_ref * dy_f
        e_lat_f = -sin_ref * dx_f + cos_ref * dy_f
        e_head_f = ca.atan2(
            ca.sin(X[2, N] - x_target[2]),
            ca.cos(X[2, N] - x_target[2]),
        )
        e_body_f = ca.vertcat(e_lon_f, e_lat_f, e_head_f)
        cost += ca.mtimes([e_body_f.T, ca.diag(Qf_diag), e_body_f])

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

        self._solver = ca.nlpsol("mpc", "ipopt", {
            "f": cost, "x": opt_vars, "p": P, "g": g,
        }, {
            "ipopt.max_iter": 100, "ipopt.print_level": 0,
            "ipopt.sb": "yes", "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
        })

        n_states = 3 * (N + 1)
        n_vars = n_states + 2 * N
        self._lbx_base = np.full(n_vars, -np.inf)
        self._ubx_base = np.full(n_vars, np.inf)
        self._lbg = np.zeros(3 * (N + 1))
        self._ubg = np.zeros(3 * (N + 1))

    def _get_phase_params(self, phase: DockingPhase):
        cfg = self.config
        if phase == DockingPhase.APPROACH:
            return (cfg.approach_Q, cfg.approach_R, cfg.approach_Qf,
                    cfg.approach_v_max, cfg.approach_omega_max)
        elif phase == DockingPhase.ALIGN:
            return (cfg.align_Q, cfg.align_R, cfg.align_Qf,
                    cfg.align_v_max, cfg.align_omega_max)
        elif phase == DockingPhase.FINAL_APPROACH:
            return (cfg.final_Q, cfg.final_R, cfg.final_Qf,
                    cfg.final_v_max, cfg.final_omega_max)
        else:
            return (cfg.approach_Q, cfg.approach_R, cfg.approach_Qf, 0.0, 0.0)

    def solve(self, current: Pose2D, target: Pose2D, phase: DockingPhase):
        """
        求解 MPC, 返回 (v, omega).

        Args:
            current: Follower 当前位姿
            target:  目标位姿 (staging / dock point)
            phase:   当前对接阶段 (决定权重与速度约束)
        """
        N = self.N
        cfg = self.config
        Q, R, Qf, v_max, omega_max = self._get_phase_params(phase)

        p = np.array([
            current.x, current.y, current.theta,
            target.x, target.y, target.theta,
            Q[0], Q[1], Q[2],
            R[0], R[1],
            Qf[0], Qf[1], Qf[2],
            v_max,
            self._prev_u[0], self._prev_u[1],
            cfg.smooth_dv, cfg.smooth_domega,
        ])

        lbx = self._lbx_base.copy()
        ubx = self._ubx_base.copy()
        n_states = 3 * (N + 1)
        for k in range(N):
            lbx[n_states + 2*k]     = -v_max
            ubx[n_states + 2*k]     =  v_max
            lbx[n_states + 2*k + 1] = -omega_max
            ubx[n_states + 2*k + 1] =  omega_max

        x0_guess = self._prev_sol if self._prev_sol is not None else (
            np.array([current.x, current.y, current.theta] * (N + 1)
                     + [0.0] * (2 * N))
        )

        sol = self._solver(
            x0=x0_guess, p=p, lbx=lbx, ubx=ubx,
            lbg=self._lbg, ubg=self._ubg,
        )
        self._prev_sol = np.array(sol["x"]).flatten()

        opt = self._prev_sol
        v_out = float(opt[n_states])
        w_out = float(opt[n_states + 1])
        self._prev_u = np.array([v_out, w_out])
        return v_out, w_out

    def set_prev_control(self, v: float, omega: float):
        """从外部设置上一步控制 (用于 DMPC→MPC 切换时桥接)."""
        self._prev_u = np.array([v, omega])

    def reset(self):
        self._prev_sol = None
        self._prev_u = np.zeros(2)
