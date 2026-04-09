"""独立安全层: 控制限幅 + 急停锁存.

该模块与 MPC/状态机解耦, 可在仿真与实机统一复用.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SafetyStatus:
    """安全层状态快照."""

    estop_active: bool = False
    estop_reason: str = ""
    clipped_v: bool = False
    clipped_omega: bool = False


class SafetyLayer:
    """速度安全层.

    功能:
    1) 对控制输出做统一硬限幅 (v, omega)
    2) 急停锁存: 一旦触发, 持续输出 (0, 0), 直到显式清除
    """

    def __init__(self, v_limit: float, omega_limit: float):
        if v_limit <= 0.0:
            raise ValueError("v_limit must be > 0")
        if omega_limit <= 0.0:
            raise ValueError("omega_limit must be > 0")

        self._v_limit = float(v_limit)
        self._omega_limit = float(omega_limit)
        self._estop_active = False
        self._estop_reason = ""
        self._last_status = SafetyStatus()

    @property
    def estop_active(self) -> bool:
        return self._estop_active

    def trigger_estop(self, reason: str = "manual"):
        """触发急停 (锁存)."""
        self._estop_active = True
        self._estop_reason = reason

    def clear_estop(self):
        """清除急停."""
        self._estop_active = False
        self._estop_reason = ""

    def apply(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float]:
        """对输入指令施加安全约束, 返回安全后的速度指令."""
        if self._estop_active:
            self._last_status = SafetyStatus(
                estop_active=True,
                estop_reason=self._estop_reason,
                clipped_v=False,
                clipped_omega=False,
            )
            return 0.0, 0.0

        v_safe = max(-self._v_limit, min(self._v_limit, float(v_cmd)))
        w_safe = max(-self._omega_limit, min(self._omega_limit, float(omega_cmd)))

        self._last_status = SafetyStatus(
            estop_active=False,
            estop_reason="",
            clipped_v=(v_safe != float(v_cmd)),
            clipped_omega=(w_safe != float(omega_cmd)),
        )
        return v_safe, w_safe

    def get_status(self) -> SafetyStatus:
        """返回最近一次 apply 后的状态."""
        return self._last_status
