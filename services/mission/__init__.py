"""
Mission planning and simulation package.

This package provides:
- Mission definition: phases, waypoints, profiles
- Autopilot: PID-based flight control
- Mission simulator: forward simulation of mission profiles
- Legacy sweep and motor utilities (apc_map, motor, sweep, etc.)
"""

# Legacy utilities
from .util import floats_in_line, isa_density, tip_mach
from .apc_map import APCMap
from .motor import MotorProp
from .sweep import compute_sweep

# Mission definition system
from .mission_definition import (
    MissionPhaseType,
    Waypoint,
    MissionPhase,
    MissionProfile,
)

# Autopilot
from .autopilot import (
    PIDController,
    AutopilotGains,
    SimpleAutopilot,
)

# Mission simulator
from .mission_simulator import (
    SimulationConfig,
    PhaseResult,
    MissionResult,
    MissionSimulator,
    simulate_simple_cruise,
    set_mission_verbosity,
    mission_logger,
)

# 3-DOF Dynamics
from .dynamics_3dof import (
    AircraftConfig3DOF,
    compute_3dof_derivatives,
    integrate_3dof_euler,
    integrate_3dof_rk4,
    apply_bank_limiting,
    apply_ground_effect,
    compute_wind_components,
    compute_headwind,
    compute_turn_rate,
    compute_turn_radius,
)

# Ground Roll
from .ground_roll import (
    GroundRollConfig,
    compute_ground_roll_derivatives,
    integrate_ground_roll,
    check_liftoff_condition,
    check_touchdown_condition,
    transition_to_flight,
    transition_to_ground,
    compute_V_stall,
    compute_V_rot,
    estimate_takeoff_distance,
    estimate_landing_distance,
)

__all__ = [
    # Legacy
    "floats_in_line",
    "isa_density",
    "tip_mach",
    "APCMap",
    "MotorProp",
    "compute_sweep",
    # Mission definition
    "MissionPhaseType",
    "Waypoint",
    "MissionPhase",
    "MissionProfile",
    # Autopilot
    "PIDController",
    "AutopilotGains",
    "SimpleAutopilot",
    # Simulator
    "SimulationConfig",
    "PhaseResult",
    "MissionResult",
    "MissionSimulator",
    "simulate_simple_cruise",
    "set_mission_verbosity",
    "mission_logger",
    # 3-DOF Dynamics
    "AircraftConfig3DOF",
    "compute_3dof_derivatives",
    "integrate_3dof_euler",
    "integrate_3dof_rk4",
    "apply_bank_limiting",
    "apply_ground_effect",
    "compute_wind_components",
    "compute_headwind",
    "compute_turn_rate",
    "compute_turn_radius",
    # Ground Roll
    "GroundRollConfig",
    "compute_ground_roll_derivatives",
    "integrate_ground_roll",
    "check_liftoff_condition",
    "check_touchdown_condition",
    "transition_to_flight",
    "transition_to_ground",
    "compute_V_stall",
    "compute_V_rot",
    "estimate_takeoff_distance",
    "estimate_landing_distance",
]

