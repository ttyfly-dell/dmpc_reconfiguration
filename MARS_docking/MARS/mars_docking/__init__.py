"""MARS Docking - 双轮足机器人自主对接运动规划系统"""

from .types import Pose2D, DockingPhase, DockingConfig
from .mpc_controller import MPCController
from .dmpc_controller import DMPCApproachController
from .state_machine import DockingStateMachine
from .multi_robot import ChainDockingCoordinator
from .utils import compute_staging_point, compute_dock_point
from .safety_layer import SafetyLayer, SafetyStatus