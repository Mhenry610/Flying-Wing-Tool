"""
Comprehensive Takeoff Analysis Script

This script performs a complete, physics-based takeoff ground roll calculation
using:
1. AeroBuildup for aerodynamic coefficients during ground roll
2. Propeller meta-model for thrust vs velocity
3. Motor equilibrium solver for realistic power/current
4. Numerical integration of the takeoff equation:
   T - D - mu*(W - L) = m*a

Output is GUI-ready with validated values.

Author: Sisyphus AI
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aerosandbox as asb

from core.models.project import WingProject
from services.geometry import AeroSandboxService
from services.propulsion.battery_model import BatteryPackConfig, create_lipo_pack
from services.propulsion.motor_model import DifferentiableMotorModel, MotorParameters
from services.propulsion.propeller_model import PropellerMetaModel, get_pretrained_model

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MotorConfig:
    """Motor configuration for takeoff analysis."""

    kv: float  # RPM/V
    R_internal: float  # Ohms
    I_no_load: float  # Amps
    n_motors: int = 2

    @property
    def params(self) -> MotorParameters:
        return MotorParameters(
            kv=self.kv,
            R_internal=self.R_internal,
            I_no_load=self.I_no_load,
        )


@dataclass
class PropellerConfig:
    """Propeller configuration."""

    diameter_in: float
    pitch_in: float
    family: str = "Standard"

    @property
    def diameter_m(self) -> float:
        return self.diameter_in * 0.0254

    @property
    def pitch_m(self) -> float:
        return self.pitch_in * 0.0254


@dataclass
class BatteryConfig:
    """Battery configuration with voltage sag model.

    Supports both simple (V_full only) and physics-based (OCV-SOC + IR drop) modes.
    When pack is provided, uses full voltage sag model. Otherwise falls back to
    legacy V_full * throttle behavior.
    """

    n_series: int  # e.g., 4 for 4S
    capacity_mAh: int
    c_rating: float = 40.0  # Max continuous discharge rate

    # Optional: physics-based pack model for voltage sag
    # If None, uses legacy V_full behavior
    _pack: Optional[BatteryPackConfig] = None

    @property
    def V_nominal(self) -> float:
        if self._pack is not None:
            return self._pack.V_nominal
        return self.n_series * 3.7

    @property
    def V_full(self) -> float:
        if self._pack is not None:
            return self._pack.V_max
        return self.n_series * 4.2

    @property
    def V_min(self) -> float:
        """Minimum pack voltage (cutoff)."""
        if self._pack is not None:
            return self._pack.V_min
        return self.n_series * 3.0

    @property
    def I_max(self) -> float:
        """Max discharge current based on C-rating."""
        if self._pack is not None:
            return self._pack.I_max_continuous
        return (self.capacity_mAh / 1000.0) * self.c_rating

    @property
    def R_internal(self) -> float:
        """Pack internal resistance [Ohms]."""
        if self._pack is not None:
            return self._pack.R_internal
        # Estimate from capacity: ~5mOhm per cell at 2Ah, scales inversely
        cell_capacity_Ah = self.capacity_mAh / 1000.0
        cell_R_mOhm = 5.0 * (2.0 / cell_capacity_Ah) if cell_capacity_Ah > 0 else 5.0
        return cell_R_mOhm / 1000.0 * self.n_series

    def get_terminal_voltage(
        self, SOC: float, I_discharge: float, T_battery: float = 25.0
    ) -> float:
        """Get terminal voltage under load with SOC and IR drop.

        V_terminal = OCV(SOC) - I * R_internal(SOC, T)

        Args:
            SOC: State of charge [0, 1]
            I_discharge: Discharge current [A] (positive = discharging)
            T_battery: Battery temperature [C]

        Returns:
            Terminal voltage [V]
        """
        if self._pack is not None:
            from services.propulsion.battery_model import DifferentiableBatteryModel

            model = DifferentiableBatteryModel(self._pack)
            return float(model.get_terminal_voltage(SOC, I_discharge, T_battery))

        # Fallback: simple linear OCV model + IR drop
        # OCV varies linearly from V_min (SOC=0) to V_full (SOC=1)
        OCV = self.V_min + SOC * (self.V_full - self.V_min)
        return OCV - I_discharge * self.R_internal

    def get_ocv(self, SOC: float) -> float:
        """Get open circuit voltage for given SOC."""
        if self._pack is not None:
            from services.propulsion.battery_model import DifferentiableBatteryModel

            model = DifferentiableBatteryModel(self._pack)
            return float(model.get_ocv(SOC))

        # Fallback: simple linear OCV model
        return self.V_min + SOC * (self.V_full - self.V_min)

    @classmethod
    def from_pack(cls, pack: BatteryPackConfig) -> "BatteryConfig":
        """Create BatteryConfig from a BatteryPackConfig for full voltage sag model."""
        return cls(
            n_series=pack.n_series,
            capacity_mAh=int(pack.capacity_mAh),
            c_rating=pack.cell.C_rate_max_continuous,
            _pack=pack,
        )


@dataclass
class TakeoffResult:
    """Results from takeoff analysis (3-DOF)."""

    # Core results
    ground_roll_m: float
    ground_roll_ft: float
    takeoff_time_s: float
    liftoff_velocity_mps: float

    # Propulsion at takeoff
    static_thrust_N: float
    thrust_at_liftoff_N: float
    max_battery_current_A: float
    max_power_W: float

    # Margins
    passes_requirement: bool
    requirement_m: float
    margin_m: float
    margin_percent: float

    # 3-DOF specific
    rotation_velocity_mps: float
    max_pitch_deg: float
    max_q_degps: float

    # Time history for plotting
    time_history: np.ndarray
    velocity_history: np.ndarray
    distance_history: np.ndarray
    thrust_history: np.ndarray
    acceleration_history: np.ndarray
    pitch_history: np.ndarray
    pitch_rate_history: np.ndarray
    alpha_history: np.ndarray
    gamma_history: np.ndarray
    altitude_history: np.ndarray
    elevon_history: np.ndarray

    # SOC/voltage tracking (optional, may be None for legacy runs)
    soc_history: Optional[np.ndarray] = None
    voltage_history: Optional[np.ndarray] = None
    initial_soc: float = 1.0
    final_soc: float = 1.0
    soc_consumed: float = 0.0  # SOC used during takeoff [0-1]

    def summary(self) -> str:
        """Return formatted summary string."""
        status = "PASS" if self.passes_requirement else "FAIL"
        lines = [
            "=" * 70,
            "TAKEOFF ANALYSIS RESULTS (3-DOF)",
            "=" * 70,
            "",
            f"Ground Roll Distance: {self.ground_roll_m:.1f} m ({self.ground_roll_ft:.0f} ft)",
            f"Rotation Velocity:    {self.rotation_velocity_mps:.1f} m/s",
            f"Takeoff Time:         {self.takeoff_time_s:.2f} s",
            f"Liftoff Velocity:     {self.liftoff_velocity_mps:.1f} m/s",
            "",
            "PITCH DYNAMICS:",
            f"  Max Pitch Angle:      {self.max_pitch_deg:.1f} deg",
            f"  Max Pitch Rate:       {self.max_q_degps:.1f} deg/s",
            "",
            "PROPULSION:",
            f"  Static Thrust:        {self.static_thrust_N:.1f} N",
            f"  Thrust at Liftoff:    {self.thrust_at_liftoff_N:.1f} N",
            f"  Max Battery Current:  {self.max_battery_current_A:.0f} A",
            f"  Max Power:            {self.max_power_W:.0f} W",
        ]

        # Add SOC info if tracked
        if self.soc_history is not None and len(self.soc_history) > 0:
            lines.extend(
                [
                    "",
                    "BATTERY STATE:",
                    f"  Initial SOC:          {self.initial_soc * 100:.1f}%",
                    f"  Final SOC:            {self.final_soc * 100:.1f}%",
                    f"  SOC Consumed:         {self.soc_consumed * 100:.2f}%",
                ]
            )
            if self.voltage_history is not None and len(self.voltage_history) > 0:
                v_min = min(self.voltage_history)
                v_max = max(self.voltage_history)
                lines.append(f"  Voltage Range:        {v_min:.1f} - {v_max:.1f} V")

        lines.extend(
            [
                "",
                f"REQUIREMENT: {self.requirement_m:.1f} m ({self.requirement_m * 3.281:.0f} ft)",
                f"STATUS: {status}",
                f"Margin: {self.margin_m:.1f} m ({self.margin_percent:.0f}%)",
                "=" * 70,
            ]
        )
        return "\n".join(lines)


@dataclass
class ClimbResult:
    """Results from climb phase analysis."""

    altitude_gain_m: float  # Actual altitude achieved
    target_altitude_m: float  # Requested altitude
    horizontal_distance_m: float
    climb_time_s: float
    climb_angle_deg: float
    average_climb_rate_mps: float
    average_velocity_mps: float
    average_throttle: float
    energy_consumed_Wh: float

    # Constraint tracking
    distance_constrained: bool  # True if stopped due to distance limit
    distance_limit_m: Optional[float]  # The distance constraint if any

    # Time history for plotting
    time_history: np.ndarray
    altitude_history: np.ndarray
    distance_history: np.ndarray
    velocity_history: np.ndarray

    @property
    def meets_altitude_target(self) -> bool:
        """Check if target altitude was achieved."""
        return self.altitude_gain_m >= self.target_altitude_m * 0.99  # 1% tolerance

    def summary(self) -> str:
        status = "OK" if self.meets_altitude_target else "ALTITUDE NOT REACHED"
        lines = [
            "=" * 70,
            "CLIMB PHASE RESULTS",
            "=" * 70,
            "",
        ]

        if not self.meets_altitude_target:
            lines.extend(
                [
                    "*** CLIMB CONSTRAINT FAILURE ***",
                    f"Target Altitude: {self.target_altitude_m:.1f} m ({self.target_altitude_m * 3.281:.0f} ft)",
                    f"Achieved Altitude: {self.altitude_gain_m:.1f} m ({self.altitude_gain_m * 3.281:.0f} ft) - {self.altitude_gain_m / self.target_altitude_m * 100:.0f}%",
                    f"Distance Limit: {self.distance_limit_m:.1f} m ({self.distance_limit_m * 3.281:.0f} ft)"
                    if self.distance_limit_m
                    else "",
                    "",
                ]
            )

        lines.extend(
            [
                f"Altitude Gain: {self.altitude_gain_m:.1f} m ({self.altitude_gain_m * 3.281:.0f} ft)",
                f"Horizontal Distance: {self.horizontal_distance_m:.1f} m ({self.horizontal_distance_m * 3.281:.0f} ft)",
                f"Climb Time: {self.climb_time_s:.1f} s",
                f"Climb Angle: {self.climb_angle_deg:.1f} deg",
                f"Average Climb Rate: {self.average_climb_rate_mps:.2f} m/s ({self.average_climb_rate_mps * 196.85:.0f} fpm)",
                f"Average Velocity: {self.average_velocity_mps:.1f} m/s",
                f"Average Throttle: {self.average_throttle * 100:.0f}%",
                f"Energy Consumed: {self.energy_consumed_Wh:.1f} Wh",
                "",
                f"STATUS: {status}",
                "=" * 70,
            ]
        )
        return "\n".join([l for l in lines if l is not None])


@dataclass
class CruiseResult:
    """Results from cruise phase analysis."""

    distance_m: float
    cruise_time_s: float
    cruise_velocity_mps: float
    cruise_altitude_m: float
    throttle_setting: float
    power_W: float
    current_A: float
    energy_consumed_Wh: float
    L_over_D: float
    time_history: np.ndarray
    velocity_history: np.ndarray
    distance_history: np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "CRUISE PHASE RESULTS",
            "=" * 70,
            "",
            f"Cruise Distance: {self.distance_m:.1f} m",
            f"Cruise Time: {self.cruise_time_s:.1f} s",
            f"Cruise Velocity: {self.cruise_velocity_mps:.1f} m/s ({self.cruise_velocity_mps * 1.944:.1f} kts)",
            f"Cruise Altitude: {self.cruise_altitude_m:.0f} m ({self.cruise_altitude_m * 3.281:.0f} ft)",
            f"Throttle: {self.throttle_setting * 100:.0f}%",
            f"Power: {self.power_W:.0f} W",
            f"Current: {self.current_A:.1f} A",
            f"Energy Consumed: {self.energy_consumed_Wh:.1f} Wh",
            f"L/D Ratio: {self.L_over_D:.1f}",
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass
class DescentResult:
    """Results from descent phase analysis."""

    altitude_loss_m: float
    horizontal_distance_m: float
    descent_time_s: float
    descent_angle_deg: float
    average_sink_rate_mps: float
    average_velocity_mps: float
    throttle_setting: float
    energy_consumed_Wh: float

    # Time history for plotting
    time_history: np.ndarray
    altitude_history: np.ndarray
    distance_history: np.ndarray
    velocity_history: np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "DESCENT PHASE RESULTS",
            "=" * 70,
            "",
            f"Altitude Loss: {self.altitude_loss_m:.1f} m ({self.altitude_loss_m * 3.281:.0f} ft)",
            f"Horizontal Distance: {self.horizontal_distance_m:.1f} m",
            f"Descent Time: {self.descent_time_s:.1f} s",
            f"Descent Angle: {self.descent_angle_deg:.1f} deg",
            f"Average Sink Rate: {self.average_sink_rate_mps:.2f} m/s ({self.average_sink_rate_mps * 196.85:.0f} fpm)",
            f"Average Velocity: {self.average_velocity_mps:.1f} m/s",
            f"Throttle: {self.throttle_setting * 100:.0f}%",
            f"Energy Consumed: {self.energy_consumed_Wh:.1f} Wh",
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass
class LandingResult:
    """Results from landing phase analysis."""

    approach_distance_m: float
    flare_distance_m: float
    ground_roll_m: float
    total_landing_distance_m: float

    approach_time_s: float
    flare_time_s: float
    ground_roll_time_s: float
    total_time_s: float

    approach_velocity_mps: float
    touchdown_velocity_mps: float
    flare_altitude_m: float

    energy_consumed_Wh: float

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "LANDING PHASE RESULTS",
            "=" * 70,
            "",
            f"Approach Distance: {self.approach_distance_m:.1f} m",
            f"Flare Distance: {self.flare_distance_m:.1f} m",
            f"Ground Roll: {self.ground_roll_m:.1f} m ({self.ground_roll_m * 3.281:.0f} ft)",
            f"Total Landing Distance: {self.total_landing_distance_m:.1f} m ({self.total_landing_distance_m * 3.281:.0f} ft)",
            "",
            f"Approach Time: {self.approach_time_s:.1f} s",
            f"Flare Time: {self.flare_time_s:.1f} s",
            f"Ground Roll Time: {self.ground_roll_time_s:.1f} s",
            f"Total Time: {self.total_time_s:.1f} s",
            "",
            f"Approach Velocity: {self.approach_velocity_mps:.1f} m/s",
            f"Touchdown Velocity: {self.touchdown_velocity_mps:.1f} m/s",
            f"Flare Altitude: {self.flare_altitude_m:.1f} m",
            f"Energy Consumed: {self.energy_consumed_Wh:.1f} Wh",
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass
class MissionResult:
    """Complete mission analysis results."""

    takeoff: TakeoffResult
    climb: ClimbResult
    cruise: CruiseResult
    descent: DescentResult
    landing: LandingResult

    # Totals
    total_distance_m: float
    total_time_s: float
    total_energy_Wh: float
    max_altitude_m: float
    avg_thrust_N: float
    max_thrust_N: float

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "COMPLETE MISSION SUMMARY",
            "=" * 70,
            "",
            "PHASE BREAKDOWN:",
            f"  1. Takeoff (3-DOF):      {self.takeoff.ground_roll_m:>7.1f} m  |  {self.takeoff.takeoff_time_s:>5.1f} s",
            f"  2. Climb to {self.climb.altitude_gain_m:.0f}m:       {self.climb.horizontal_distance_m:>7.1f} m  |  {self.climb.climb_time_s:>5.1f} s",
            f"  3. Cruise:               {self.cruise.distance_m:>7.1f} m  |  {self.cruise.cruise_time_s:>5.1f} s",
            f"  4. Descent:              {self.descent.horizontal_distance_m:>7.1f} m  |  {self.descent.descent_time_s:>5.1f} s",
            f"  5. Landing:              {self.landing.total_landing_distance_m:>7.1f} m  |  {self.landing.total_time_s:>5.1f} s",
            "",
            "TOTALS:",
            f"  Total Distance: {self.total_distance_m:.1f} m ({self.total_distance_m / 1000:.2f} km)",
            f"  Total Time: {self.total_time_s:.1f} s ({self.total_time_s / 60:.1f} min)",
            f"  Total Energy: {self.total_energy_Wh:.1f} Wh",
            f"  Max Altitude: {self.max_altitude_m:.0f} m ({self.max_altitude_m * 3.281:.0f} ft)",
            f"  Avg Thrust: {self.avg_thrust_N:.1f} N",
            f"  Max Thrust: {self.max_thrust_N:.1f} N",
            "",
            "ENERGY BREAKDOWN:",
            f"  Takeoff: {self.takeoff.max_power_W * self.takeoff.takeoff_time_s / 3600:.1f} Wh",
            f"  Climb:   {self.climb.energy_consumed_Wh:.1f} Wh",
            f"  Cruise:  {self.cruise.energy_consumed_Wh:.1f} Wh",
            f"  Descent: {self.descent.energy_consumed_Wh:.1f} Wh",
            f"  Landing: {self.landing.energy_consumed_Wh:.1f} Wh",
            "=" * 70,
        ]
        return "\n".join(lines)


# =============================================================================
# Mission Plotting
# =============================================================================


def plot_mission(
    mission: MissionResult,
    battery_capacity_Wh: float,
    propulsion: "TakeoffPropulsionModel",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Generate mission profile plots.

    Creates a 2x2 subplot figure with:
    - Altitude vs Time
    - Airspeed vs Time
    - Thrust vs Time
    - State of Charge vs Time

    Each plot has vertical lines marking phase transitions.

    Args:
        mission: Complete mission result
        battery_capacity_Wh: Battery capacity for SOC calculation
        propulsion: Propulsion model for thrust calculations
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    # Phase colors
    COLORS = {
        "takeoff": "#FF6B6B",  # Red
        "climb": "#4ECDC4",  # Teal
        "cruise": "#45B7D1",  # Blue
        "descent": "#96CEB4",  # Green
        "landing": "#FFEAA7",  # Yellow
    }

    # Build complete time series for each phase using physics histories

    # --- Phase timing ---
    t_takeoff_end = mission.takeoff.takeoff_time_s
    t_climb_end = t_takeoff_end + mission.climb.climb_time_s
    t_cruise_end = t_climb_end + mission.cruise.cruise_time_s
    t_descent_end = t_cruise_end + mission.descent.descent_time_s
    t_landing_end = t_descent_end + mission.landing.total_time_s

    phase_boundaries = [
        0,
        t_takeoff_end,
        t_climb_end,
        t_cruise_end,
        t_descent_end,
        t_landing_end,
    ]
    phase_names = ["Takeoff", "Climb", "Cruise", "Descent", "Landing"]
    phase_colors = [
        COLORS["takeoff"],
        COLORS["climb"],
        COLORS["cruise"],
        COLORS["descent"],
        COLORS["landing"],
    ]

    # --- Build arrays for each variable ---
    time_all = []
    altitude_all = []
    velocity_all = []
    thrust_all = []
    energy_cumulative = []

    current_energy_Wh = 0.0
    rho = 1.225

    # Precompute per-phase energy rates (Wh over phase duration)
    takeoff_energy_Wh = (
        mission.takeoff.max_power_W * mission.takeoff.takeoff_time_s / 3600
    )
    climb_energy_rate_W = (
        (mission.climb.energy_consumed_Wh * 3600 / mission.climb.climb_time_s)
        if mission.climb.climb_time_s > 0
        else 0.0
    )
    cruise_energy_rate_W = (
        (mission.cruise.energy_consumed_Wh * 3600 / mission.cruise.cruise_time_s)
        if mission.cruise.cruise_time_s > 0
        else 0.0
    )
    descent_energy_rate_W = (
        (mission.descent.energy_consumed_Wh * 3600 / mission.descent.descent_time_s)
        if mission.descent.descent_time_s > 0
        else 0.0
    )
    landing_energy_rate_W = (
        (
            mission.landing.energy_consumed_Wh
            * 3600
            / max(mission.landing.approach_time_s + mission.landing.flare_time_s, 1e-6)
        )
        if (mission.landing.approach_time_s + mission.landing.flare_time_s) > 0
        else 0.0
    )

    # 1. TAKEOFF PHASE
    for i, t in enumerate(mission.takeoff.time_history):
        time_all.append(t)
        altitude_all.append(0.0)
        velocity_all.append(mission.takeoff.velocity_history[i])
        thrust_all.append(
            mission.takeoff.thrust_history[i]
            if i < len(mission.takeoff.thrust_history)
            else mission.takeoff.thrust_at_liftoff_N
        )
        if i > 0:
            dt = mission.takeoff.time_history[i] - mission.takeoff.time_history[i - 1]
            current_energy_Wh += (
                (takeoff_energy_Wh / mission.takeoff.takeoff_time_s) * dt
                if mission.takeoff.takeoff_time_s > 0
                else 0.0
            )
        energy_cumulative.append(current_energy_Wh)

    # 2. CLIMB PHASE
    for i, t_local in enumerate(mission.climb.time_history):
        t = t_takeoff_end + t_local
        time_all.append(t)
        altitude_all.append(mission.climb.altitude_history[i])
        V = mission.climb.velocity_history[i]
        velocity_all.append(V)
        thrust_all.append(
            propulsion.solve_operating_point(
                V, throttle=mission.climb.average_throttle, rho=rho
            )["thrust_total"]
        )
        if i > 0:
            dt = mission.climb.time_history[i] - mission.climb.time_history[i - 1]
            current_energy_Wh += climb_energy_rate_W * dt / 3600
        energy_cumulative.append(current_energy_Wh)

    # 3. CRUISE PHASE
    for i, t_local in enumerate(mission.cruise.time_history):
        t = t_climb_end + t_local
        time_all.append(t)
        altitude_all.append(mission.cruise.cruise_altitude_m)
        V = mission.cruise.velocity_history[i]
        velocity_all.append(V)
        thrust_all.append(
            propulsion.solve_operating_point(
                V, throttle=mission.cruise.throttle_setting, rho=rho
            )["thrust_total"]
        )
        if i > 0:
            dt = mission.cruise.time_history[i] - mission.cruise.time_history[i - 1]
            current_energy_Wh += cruise_energy_rate_W * dt / 3600
        energy_cumulative.append(current_energy_Wh)

    # 4. DESCENT PHASE (glide: thrust ~0, use phase energy)
    for i, t_local in enumerate(mission.descent.time_history):
        t = t_cruise_end + t_local
        time_all.append(t)
        altitude_all.append(mission.descent.altitude_history[i])
        V = mission.descent.velocity_history[i]
        velocity_all.append(V)
        thrust_all.append(0.0)
        if i > 0:
            dt = mission.descent.time_history[i] - mission.descent.time_history[i - 1]
            current_energy_Wh += descent_energy_rate_W * dt / 3600
        energy_cumulative.append(current_energy_Wh)

    # 5. LANDING PHASE (reconstruct; thrust ~0, use landing energy on approach+flare)
    V_start = mission.descent.velocity_history[-1]
    approach_alt_start = 10.0
    flare_alt = mission.landing.flare_altitude_m
    approach_time = mission.landing.approach_time_s
    flare_time = mission.landing.flare_time_s
    ground_roll_time = mission.landing.ground_roll_time_s

    # Representative idle thrust for plotting consistency (set to zero to avoid artificial spikes)
    idle_thrust = 0.0

    # Approach: linear descent from approach_alt_start to flare_alt
    approach_steps = max(5, int(approach_time / 0.05)) if approach_time > 0 else 5
    for i in range(approach_steps):
        frac = i / max(1, approach_steps - 1)
        t = t_descent_end + frac * approach_time
        alt = approach_alt_start + (flare_alt - approach_alt_start) * frac
        V = V_start + (mission.landing.approach_velocity_mps - V_start) * frac
        time_all.append(t)
        altitude_all.append(max(0.0, alt))
        velocity_all.append(V)
        thrust_all.append(idle_thrust)
        if i > 0:
            dt = approach_time / max(1, approach_steps - 1)
            current_energy_Wh += landing_energy_rate_W * dt / 3600
        energy_cumulative.append(current_energy_Wh)

    t_approach_end = t_descent_end + approach_time

    # Flare: decelerate to touchdown velocity
    flare_steps = max(3, int(flare_time / 0.05)) if flare_time > 0 else 3
    for i in range(flare_steps):
        frac = i / max(1, flare_steps - 1)
        t = t_approach_end + frac * flare_time
        alt = flare_alt * (1 - frac)
        V = (
            mission.landing.approach_velocity_mps
            + (
                mission.landing.touchdown_velocity_mps
                - mission.landing.approach_velocity_mps
            )
            * frac
        )
        time_all.append(t)
        altitude_all.append(max(0.0, alt))
        velocity_all.append(V)
        thrust_all.append(idle_thrust)
        if i > 0:
            dt = flare_time / max(1, flare_steps - 1)
            current_energy_Wh += landing_energy_rate_W * dt / 3600
        energy_cumulative.append(current_energy_Wh)

    t_flare_end = t_approach_end + flare_time

    # Ground roll: linear decel to 0 (no energy assumed)
    roll_steps = max(5, int(ground_roll_time / 0.05)) if ground_roll_time > 0 else 5
    for i in range(roll_steps):
        frac = i / max(1, roll_steps - 1)
        t = t_flare_end + frac * ground_roll_time
        V = mission.landing.touchdown_velocity_mps * (1 - frac)
        time_all.append(t)
        altitude_all.append(0.0)
        velocity_all.append(max(0.0, V))
        thrust_all.append(0.0)
        energy_cumulative.append(current_energy_Wh)

    # Convert to numpy
    time_all = np.array(time_all)
    altitude_all = np.array(altitude_all)
    velocity_all = np.array(velocity_all)
    thrust_all = np.array(thrust_all)
    energy_cumulative = np.array(energy_cumulative)

    # State of charge (100% - used%)
    soc_all = 100.0 * (1.0 - energy_cumulative / battery_capacity_Wh)

    # --- Create figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mission Profile Analysis", fontsize=14, fontweight="bold")

    # Helper function to add phase shading and labels
    def add_phase_regions(ax, y_min, y_max):
        for i, (t_start, t_end) in enumerate(
            zip(phase_boundaries[:-1], phase_boundaries[1:])
        ):
            ax.axvspan(t_start, t_end, alpha=0.15, color=phase_colors[i])
            ax.axvline(t_end, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # 1. ALTITUDE vs TIME
    ax1 = axes[0, 0]
    add_phase_regions(ax1, 0, max(altitude_all) * 1.1)
    ax1.plot(time_all, altitude_all, "b-", linewidth=2, label="Altitude")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude Profile")
    ax1.set_xlim(0, time_all[-1])
    ax1.set_ylim(0, max(altitude_all) * 1.15 if max(altitude_all) > 0 else 10)
    ax1.grid(True, alpha=0.3)

    # Add altitude annotations
    ax1.axhline(
        mission.climb.target_altitude_m,
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"Target: {mission.climb.target_altitude_m:.0f}m",
    )
    ax1.axhline(
        mission.climb.altitude_gain_m,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Achieved: {mission.climb.altitude_gain_m:.0f}m",
    )
    ax1.legend(loc="upper right", fontsize=8)

    # 2. AIRSPEED vs TIME
    ax2 = axes[0, 1]
    add_phase_regions(ax2, 0, max(velocity_all) * 1.1)
    ax2.plot(time_all, velocity_all, "g-", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Airspeed (m/s)")
    ax2.set_title("Airspeed Profile")
    ax2.set_xlim(0, time_all[-1])
    ax2.set_ylim(0, max(velocity_all) * 1.15)
    ax2.grid(True, alpha=0.3)

    # Add secondary y-axis for knots
    ax2_kts = ax2.twinx()
    ax2_kts.set_ylabel("Airspeed (kts)")
    ax2_kts.set_ylim(0, max(velocity_all) * 1.15 * 1.944)

    # 3. THRUST vs TIME
    ax3 = axes[1, 0]
    add_phase_regions(ax3, 0, max(thrust_all) * 1.1)
    ax3.plot(time_all, thrust_all, "r-", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Thrust (N)")
    ax3.set_title("Thrust Profile")
    ax3.set_xlim(0, time_all[-1])
    ax3.set_ylim(0, max(thrust_all) * 1.15 if max(thrust_all) > 0 else 10)
    ax3.grid(True, alpha=0.3)

    # Add weight reference line
    weight_N = mission.takeoff.static_thrust_N / 0.58  # Approximate from T/W
    ax3.axhline(
        weight_N,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Weight: {weight_N:.0f}N",
    )
    ax3.legend(loc="upper right", fontsize=8)

    # 4. STATE OF CHARGE vs TIME
    ax4 = axes[1, 1]
    add_phase_regions(ax4, 0, 100)
    ax4.plot(time_all, soc_all, "m-", linewidth=2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("State of Charge (%)")
    ax4.set_title("Battery State of Charge")
    ax4.set_xlim(0, time_all[-1])
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)

    # Add SOC warning levels
    ax4.axhline(20, color="red", linestyle="--", alpha=0.5, label="20% Warning")
    ax4.axhline(50, color="orange", linestyle="--", alpha=0.5, label="50% Caution")
    ax4.legend(loc="upper right", fontsize=8)

    # Add phase legend at bottom
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.4, label=n)
        for n, c in zip(phase_names, phase_colors)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Aerodynamic Model (3-DOF)
# =============================================================================


class RigidBodyAeroModel:
    """
    Aerodynamic model for 3-DOF analysis.

    Uses AeroBuildup to compute CL, CD, and CM at various angles of attack
    and control surface deflections.

    With use_precomputed_polars=True (default), builds a polar lookup table at
    initialization for smooth, fast interpolation during simulation.
    """

    def __init__(
        self,
        wing_project: WingProject,
        ground_effect: bool = True,
        use_precomputed_polars: bool = True,
        alpha_range: Tuple[float, float, int] = (-5.0, 20.0, 26),  # 1° steps
        delta_e_range: Tuple[float, float, int] = (-25.0, 10.0, 8),  # ~5° steps
        verbose: bool = True,
    ):
        """
        Initialize aerodynamic model.

        Args:
            wing_project: WingProject definition
            ground_effect: Enable ground effect modeling
                            NOTE: This flag is currently stored but NOT USED.
                            Ground effect modeling has been removed from the
                            coefficient calculations. The flag remains for backward
                            compatibility but does not affect physics results.
                            If ground effect modeling is needed in the future,
                            appropriate corrections must be added to the polar
                            computation (e.g., modified CL/CD values based on
                            height-above-ground).
            use_precomputed_polars: If True, build polar lookup table at init
                                    (~30s startup, ~μs per call during sim)
                                    If False, use AeroBuildup per call with caching
            alpha_range: (min, max, n_points) for alpha grid [deg]
            delta_e_range: (min, max, n_points) for elevator grid [deg]
            verbose: Print progress during polar computation
        """
        self.wing_project = wing_project
        self.ground_effect = ground_effect
        self.use_precomputed_polars = use_precomputed_polars
        self.verbose = verbose
        self.aero_service = AeroSandboxService(wing_project)

        # Cache key parameters
        self.S = wing_project.planform.wing_area_m2
        self.span = wing_project.planform.actual_span()

        # Build wing to get AC and MAC
        wing = self.aero_service.build_wing()
        self.x_ac = wing.aerodynamic_center()[0]
        self.mac = wing.mean_aerodynamic_chord()

        # Calculate CG based on static margin
        static_margin = self.wing_project.twist_trim.static_margin_percent
        self.x_cg = self.x_ac - (static_margin / 100.0) * self.mac

        # Identify pitch control surface name
        self.pitch_control_name = "Elevon"  # Default fallback
        if self.wing_project.planform.control_surfaces:
            for cs in self.wing_project.planform.control_surfaces:
                if cs.surface_type.lower() in ["elevon", "elevator"]:
                    self.pitch_control_name = cs.name
                    break

        # Initialize polar storage
        self._polar_cache: Dict[
            Tuple[float, float, float, float], Tuple[float, float, float]
        ] = {}
        self._interp_CL = None
        self._interp_CD = None
        self._interp_CM = None
        self._alpha_grid = None
        self._delta_e_grid = None

        # Build precomputed polars if requested
        if use_precomputed_polars:
            self._build_polar_table(alpha_range, delta_e_range)

    def _build_polar_table(
        self,
        alpha_range: Tuple[float, float, int],
        delta_e_range: Tuple[float, float, int],
    ):
        """Build precomputed polar lookup table using AeroBuildup."""
        from scipy.interpolate import RegularGridInterpolator

        alpha_min, alpha_max, n_alpha = alpha_range
        delta_e_min, delta_e_max, n_delta_e = delta_e_range

        self._alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)
        self._delta_e_grid = np.linspace(delta_e_min, delta_e_max, n_delta_e)

        n_total = n_alpha * n_delta_e

        if self.verbose:
            print(f"\nBuilding aerodynamic polar table (Vectorized)...")
            print(f"  Alpha: {alpha_min}deg to {alpha_max}deg ({n_alpha} points)")
            print(
                f"  Delta_e: {delta_e_min}deg to {delta_e_max}deg ({n_delta_e} points)"
            )
            print(f"  Total: {n_total} evaluations")

        # Initialize arrays
        CL_data = np.zeros((n_alpha, n_delta_e))
        CD_data = np.zeros((n_alpha, n_delta_e))
        CM_data = np.zeros((n_alpha, n_delta_e))
        CMq_data = np.zeros((n_alpha, n_delta_e))

        start_time = __import__("time").time()

        # REYNOLDS NUMBER ASSUMPTION:
        # The polar table is built at a single velocity (20.0 m/s), making the
        # aerodynamic coefficients Reynolds-number independent. This is acceptable for:
        #   - Mission-level analysis where Re variations are modest
        #   - Similar airspeeds across takeoff, climb, cruise phases
        #
        # For highly accurate analysis spanning large velocity ranges (e.g., static
        # takeoff at low Re vs high-speed cruise), multiple velocity-dependent
        # tables or Re-correction factors would be needed.

        for j, delta_e in enumerate(self._delta_e_grid):
            try:
                # Build airplane once for this delta_e
                controls = {self.pitch_control_name: delta_e}
                airplane = self.aero_service.build_airplane(
                    xyz_ref=[self.x_cg, 0.0, 0.0], control_deflections=controls
                )

                # Vectorized alpha evaluation
                atmo = asb.Atmosphere(altitude=0.0)
                op_point = asb.OperatingPoint(
                    atmosphere=atmo,
                    velocity=20.0,
                    alpha=self._alpha_grid,
                )

                aero = asb.AeroBuildup(airplane, op_point)
                result = aero.run_with_stability_derivatives(q=True)

                # Reshape to ensure 1D array even if single alpha
                CL_data[:, j] = np.reshape(result.get("CL", result.get("Cl", 0.0)), -1)
                CD_data[:, j] = np.reshape(result.get("CD", result.get("Cd", 0.0)), -1)
                CM_data[:, j] = np.reshape(result.get("CM", result.get("Cm", 0.0)), -1)
                CMq_data[:, j] = np.reshape(result.get("Cmq", 0.0), -1)

                if self.verbose:
                    elapsed = __import__("time").time() - start_time
                    print(
                        f"  [{j + 1}/{n_delta_e}] delta_e={delta_e:5.1f}deg | "
                        f"CL avg={np.mean(CL_data[:, j]):.3f} | {elapsed:.1f}s elapsed"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed at delta_e={delta_e:.1f}deg: {e}")
                CL_data[:, j] = np.nan
                CD_data[:, j] = np.nan
                CM_data[:, j] = np.nan
                CMq_data[:, j] = np.nan

        elapsed = __import__("time").time() - start_time

        # Build interpolators
        grid_points = (self._alpha_grid, self._delta_e_grid)
        self._interp_CL = RegularGridInterpolator(
            grid_points, CL_data, bounds_error=False, fill_value=None
        )
        self._interp_CD = RegularGridInterpolator(
            grid_points, CD_data, bounds_error=False, fill_value=None
        )
        self._interp_CM = RegularGridInterpolator(
            grid_points, CM_data, bounds_error=False, fill_value=None
        )
        self._interp_CMq = RegularGridInterpolator(
            grid_points, CMq_data, bounds_error=False, fill_value=None
        )

        if self.verbose:
            print(f"\nPolar table complete in {elapsed:.1f}s")
            print(f"  CL range: [{np.nanmin(CL_data):.3f}, {np.nanmax(CL_data):.3f}]")
            print(f"  CD range: [{np.nanmin(CD_data):.3f}, {np.nanmax(CD_data):.3f}]")
            print(f"  CM range: [{np.nanmin(CM_data):.3f}, {np.nanmax(CM_data):.3f}]")
            print(
                f"  CMq range: [{np.nanmin(CMq_data):.3f}, {np.nanmax(CMq_data):.3f}]"
            )
            print(f"  Ready for fast interpolation (~us per call)")

    def _compute_coefficients_raw(
        self, alpha_deg: float, delta_e: float, V: float = 20.0, altitude: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Compute coefficients using AeroBuildup (slow, ~200ms per call)."""
        controls = {self.pitch_control_name: delta_e}
        airplane = self.aero_service.build_airplane(
            xyz_ref=[self.x_cg, 0.0, 0.0], control_deflections=controls
        )

        atmo = asb.Atmosphere(altitude=max(0, altitude))
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=max(V, 1.0),
            alpha=alpha_deg,
        )

        aero = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point,
        )
        result = aero.run_with_stability_derivatives(q=True)

        CL = float(result.get("CL", result.get("Cl", 0.0)))
        CD = float(result.get("CD", result.get("Cd", 0.0)))
        CM = float(result.get("CM", result.get("Cm", 0.0)))
        CMq = float(result.get("Cmq", 0.0))

        return CL, CD, CM, CMq

    def get_coefficients(
        self, alpha_deg: float, V: float, delta_e: float = 0.0, altitude: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Get aerodynamic coefficients CL, CD, CM, CMq.

        Args:
            alpha_deg: Angle of attack [degrees]
            V: Airspeed [m/s]
            delta_e: Pitch control deflection [degrees] (positive = down)
            altitude: Altitude [m]
                    NOTE: Altitude is NOT USED for ground effect modeling.
                    The ground_effect flag is stored but not applied to
                    coefficient calculations. This is intentional - ground effect
                    corrections would require separate lift/drag modifications
                    based on height-above-ground.

        Returns:
            (CL, CD, CM, CMq) Tuple
        """
        if self.use_precomputed_polars and self._interp_CL is not None:
            point = np.array([[alpha_deg, delta_e]])
            CL = float(self._interp_CL(point)[0])
            CD = float(self._interp_CD(point)[0])
            CM = float(self._interp_CM(point)[0])
            CMq = float(self._interp_CMq(point)[0])
            return CL, CD, CM, CMq

        # Fallback to direct AeroBuildup with rounding for caching
        alpha_rounded = round(alpha_deg * 2) / 2  # 0.5 deg bins
        delta_e_rounded = round(delta_e * 2) / 2  # 0.5 deg bins

        cache_key = (alpha_rounded, delta_e_rounded, round(V), round(altitude))
        if cache_key not in self._polar_cache:
            CL, CD, CM, CMq = self._compute_coefficients_raw(
                alpha_deg, delta_e, V, altitude
            )
            self._polar_cache[cache_key] = (CL, CD, CM, CMq)

        return self._polar_cache[cache_key]

    def get_forces_moments(
        self,
        V: float,
        alpha_deg: float,
        delta_e: float,
        rho: float,
        altitude: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Compute aerodynamic forces and pitching moment.

        Args:
            V: Airspeed [m/s]
            alpha_deg: Angle of attack [degrees]
            delta_e: Elevator deflection [degrees]
            rho: Air density [kg/m^3]
            altitude: Altitude [m]

        Returns:
            (L, D, M, CMq) Tuple of lift [N], drag [N], moment [N-m], and CMq derivative
        """
        CL, CD, CM, CMq = self.get_coefficients(alpha_deg, V, delta_e, altitude)

        q = 0.5 * rho * V**2 * self.S
        L = CL * q
        D = CD * q
        M = CM * q * self.mac

        return L, D, M, CMq

    def get_forces(
        self, V: float, alpha_deg: float, rho: float, altitude: float = 0.0
    ) -> Tuple[float, float]:
        """Backward compatibility for other calculators."""
        L, D, M, CMq = self.get_forces_moments(V, alpha_deg, 0.0, rho, altitude)
        return L, D


# =============================================================================
# Propulsion Model
# =============================================================================


class TakeoffPropulsionModel:
    """
    Propulsion model for takeoff analysis.

    Combines propeller meta-model with motor model to compute
    thrust and power at any airspeed.
    """

    def __init__(
        self,
        prop_config: PropellerConfig,
        motor_config: MotorConfig,
        battery_config: BatteryConfig,
    ):
        self.prop = prop_config
        self.motor_config = motor_config
        self.battery = battery_config

        # Load propeller model
        self.prop_model = get_pretrained_model(prop_config.family)

        # Create motor model
        self.motor_model = DifferentiableMotorModel(motor_config.params)

        # Cache
        self._thrust_cache: Dict[float, Dict] = {}

    def solve_operating_point(
        self,
        V_air: float,
        throttle: float = 1.0,
        rho: float = 1.225,
        SOC: float = 1.0,
        T_battery: float = 25.0,
        n_iterations: int = 10,
    ) -> Dict:
        """
        Solve motor-propeller equilibrium at given airspeed with voltage sag.

        Uses iterative coupling between:
        - Battery: V_terminal = OCV(SOC) - I * R_internal(SOC, T)
        - Motor: equilibrium at V_terminal
        - Propeller: torque/thrust at motor RPM

        Args:
            V_air: Airspeed [m/s]
            throttle: Throttle setting [0-1]
            rho: Air density [kg/m^3]
            SOC: State of charge [0, 1] (default 1.0 = full charge)
            T_battery: Battery temperature [C] (default 25.0)
            n_iterations: Iteration count

        Returns:
            Dict with thrust, power, current, rpm, voltage_terminal, etc.
        """
        # Cache key: round SOC to 1% to avoid cache misses during short phases
        # (takeoff typically uses <1% SOC, so this gives good cache hits)
        cache_key = (round(V_air, 2), round(throttle, 2), round(rho, 3), round(SOC, 2))
        if cache_key in self._thrust_cache:
            return self._thrust_cache[cache_key]

        D_m = self.prop.diameter_m
        P_m = self.prop.pitch_m
        n_motors = self.motor_config.n_motors

        # Get open circuit voltage at current SOC
        V_ocv = self.battery.get_ocv(SOC)

        # Initial guess: start with OCV (no load) scaled by throttle
        V_batt = V_ocv * throttle
        I_battery = 0.0  # Initial current guess

        # Initial guess for RPM (approximate based on Kv and voltage)
        rpm = self.motor_config.kv * V_batt * 0.8

        # Coupled iteration: solve battery voltage, motor, and prop together
        for iteration in range(n_iterations):
            omega = rpm * 2 * np.pi / 60

            # Get propeller torque and thrust at current RPM
            T, P_shaft = self.prop_model.get_performance(
                V=V_air, omega=omega, D=D_m, P=P_m, rho=rho
            )
            T = float(T)
            P_shaft = float(P_shaft)

            # Torque = Power / omega
            torque_req = P_shaft / max(omega, 1.0)

            # Solve motor for this torque at given voltage
            motor_res = self.motor_model.solve_from_voltage_and_torque(
                V_motor=V_batt, tau_load=torque_req
            )

            # Get new RPM from motor solver
            rpm_new = motor_res["rpm"]

            # Calculate new battery current (total from all motors)
            I_battery_new = n_motors * motor_res["current"]

            # Update terminal voltage with voltage sag
            # V_terminal = OCV(SOC) - I * R_internal
            V_terminal = self.battery.get_terminal_voltage(
                SOC, I_battery_new, T_battery
            )
            V_batt_new = V_terminal * throttle

            # Check current limit
            I_limit = self.battery.I_max
            if I_battery_new > I_limit:
                # Current limiting: cap the current and recalculate
                # This would require motor/prop to find a lower operating point
                # For simplicity, we report the limited state
                pass

            # Relaxed update for stability
            rpm = 0.5 * rpm + 0.5 * rpm_new
            V_batt = 0.5 * V_batt + 0.5 * V_batt_new
            I_battery = 0.5 * I_battery + 0.5 * I_battery_new

            if abs(rpm - rpm_new) < 1.0 and abs(V_batt - V_batt_new) < 0.01:
                break

        # Final operating point
        omega = rpm * 2 * np.pi / 60
        T, P_shaft = self.prop_model.get_performance(
            V=V_air, omega=omega, D=D_m, P=P_m, rho=rho
        )
        motor_res = self.motor_model.solve_from_voltage_and_torque(
            V_motor=V_batt, tau_load=P_shaft / max(omega, 1.0)
        )
        I_battery_final = n_motors * motor_res["current"]
        V_terminal_final = self.battery.get_terminal_voltage(
            SOC, I_battery_final, T_battery
        )

        # Calculate voltage sag magnitude
        voltage_sag = V_ocv - V_terminal_final

        result = {
            "thrust_per_motor": float(T),
            "thrust_total": n_motors * float(T),
            "power_shaft_per_motor": float(P_shaft),
            "power_shaft_total": n_motors * float(P_shaft),
            "current_per_motor": motor_res["current"],
            "current_battery": I_battery_final,
            "rpm": float(rpm),
            "omega": float(omega),
            "voltage_effective": motor_res["voltage"],
            "voltage_battery": V_batt,
            "voltage_terminal": V_terminal_final,
            "voltage_ocv": V_ocv,
            "voltage_sag": voltage_sag,
            "SOC": SOC,
        }

        self._thrust_cache[cache_key] = result
        return result

    def get_thrust(
        self,
        V_air: float,
        throttle: float = 1.0,
        rho: float = 1.225,
        SOC: float = 1.0,
    ) -> float:
        """Get total thrust at given airspeed.

        Args:
            V_air: Airspeed [m/s]
            throttle: Throttle setting [0-1]
            rho: Air density [kg/m^3]
            SOC: State of charge [0, 1] (default 1.0 = full charge)

        Returns:
            Total thrust [N]
        """
        result = self.solve_operating_point(V_air, throttle, rho, SOC=SOC)
        return result["thrust_total"]

    def get_static_thrust(self, rho: float = 1.225, SOC: float = 1.0) -> float:
        """Get static thrust (V=0).

        Args:
            rho: Air density [kg/m^3]
            SOC: State of charge [0, 1] (default 1.0 = full charge)

        Returns:
            Static thrust [N]
        """
        return self.get_thrust(0.0, throttle=1.0, rho=rho, SOC=SOC)


# =============================================================================
# Takeoff Calculator (3-DOF)
# =============================================================================


class TakeoffCalculator:
    """
    3-DOF Takeoff Simulator.

    Stages:
    1. Ground Roll (Fixed Alpha)
    2. Rotation (Pivot around Gear)
    3. Transition and Climb-out
    """

    def __init__(
        self,
        wing_project: WingProject,
        prop_config: PropellerConfig,
        motor_config: MotorConfig,
        battery_config: BatteryConfig,
        mass_kg: float,
        mu_roll: float = 0.03,
        ground_effect: bool = True,
        x_gear_mac_offset: float = 0.02,  # Fraction of MAC behind CG (2% for easier rotation)
        z_cg: float = 0.10,  # Height of CG above ground [m]
        z_thrust_line: float = 0.08,  # Height of thrust line above ground [m]
        thrust_angle_deg: float = 3.0,  # Reasonable thrust angle (3 deg up)
        iyy_heuristic_factor: float = 0.10,  # Factor for radius of gyration (k_y = factor * span)
        requirement_m: float = 30.48,  # Default 100 ft
        Cmq: Optional[float] = None,  # Pitch damping derivative (None = no damping)
        aero_model: Optional[RigidBodyAeroModel] = None,  # Optional existing aero model
    ):
        """
        Initialize 3-DOF takeoff calculator.

        Args:
            wing_project: WingProject definition
            prop_config: Propeller configuration
            motor_config: Motor configuration
            battery_config: Battery configuration
            mass_kg: Aircraft mass [kg]
            mu_roll: Ground rolling friction coefficient
            ground_effect: Enable ground effect modeling
            x_gear_mac_offset: Main gear position aft of CG as fraction of MAC
            z_cg: Height of CG above ground [m] (typical: 0.05-0.15 for small UAV)
            z_thrust_line: Height of thrust line above ground [m]
            thrust_angle_deg: Angle of thrust line relative to chord [deg]
                             Positive = tilted up (assists rotation)
            iyy_heuristic_factor: Factor for Iyy estimation (k_y = factor * span)
            requirement_m: Takeoff distance requirement [m]
            Cmq: Pitch damping derivative (dimensionless).
            aero_model: Optional existing RigidBodyAeroModel to avoid re-computing polars
        """
        self.wing_project = wing_project
        self.mass_kg = mass_kg
        self.mu_roll = mu_roll
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.requirement_m = requirement_m
        self.Cmq = Cmq  # Pitch damping derivative (or None)
        self.thrust_angle_deg = thrust_angle_deg

        # Create models
        if aero_model is not None:
            self.aero_model = aero_model
        else:
            self.aero_model = RigidBodyAeroModel(wing_project, ground_effect)
            
        self.propulsion = TakeoffPropulsionModel(
            prop_config, motor_config, battery_config
        )

        # Gear configuration
        self.x_cg = self.aero_model.x_cg
        self.mac = self.aero_model.mac
        self.S = self.aero_model.S  # Cache wing area for damping calc
        self.x_gear = self.x_cg + x_gear_mac_offset * self.mac
        self.z_cg = z_cg
        self.z_thrust_line = z_thrust_line

        # Pitch inertia estimation (Iyy = m * k_y^2)
        self.Iyy = mass_kg * (iyy_heuristic_factor * self.aero_model.span) ** 2

    def calculate_liftoff_velocity(
        self, CL_max: float = 1.2, margin: float = 1.2
    ) -> float:
        rho = 1.225
        V_stall = np.sqrt(2 * self.weight_N / (rho * self.aero_model.S * CL_max))
        return margin * V_stall

    def autotune_elevator(
        self,
        V_rotation: float,
        target_pitch_deg: float = 12.0,
        rho: float = 1.225,
    ) -> Tuple[float, float]:
        """
        Autotune elevator deflections using precomputed polar table.

        Finds:
        1. rotation_delta_e: Elevator deflection that achieves positive M_gear at rotation
        2. climb_delta_e: Elevator deflection that gives target pitch angle at reasonable climb speed

        Args:
            V_rotation: Velocity at which to initiate rotation [m/s]
            target_pitch_deg: Target pitch angle during climb [degrees]
            rho: Air density [kg/m^3]

        Returns:
            (rotation_delta_e, climb_delta_e) Tuple of elevator deflections [degrees]
        """
        if not self.aero_model.use_precomputed_polars:
            print("Warning: Autotuning requires precomputed polars. Using defaults.")
            return -20.0, -6.0

        # 1. Find rotation_delta_e: deflection that gives M_gear > 0 at V_rotation
        print("Autotuning elevator deflections...")
        print(f"  Rotation velocity: {V_rotation:.1f} m/s")
        print(f"  Target pitch: {target_pitch_deg:.1f}deg")

        # Get thrust at rotation velocity
        thrust_at_rotation = self.propulsion.get_thrust(
            V_rotation, throttle=1.0, rho=rho
        )

        # Search polar table for elevator that gives positive M_gear
        # across expected rotation alpha range (2-6 degrees)
        best_rotation_delta_e = -20.0
        max_moment_ratio = 0.0

        # Thrust angle in body frame
        eps = np.radians(self.thrust_angle_deg)
        
        # We search over expected rotation alpha range (ground roll to liftoff)
        # to find elevator that gives consistent positive moment across this range.
        # This is more robust than using a single alpha_test.
        alphas_rotation = np.linspace(2.0, 6.0, 5)  # Test 2°, 3.5°, 5°, 6.5°, 8°

        for delta_e in self.aero_model._delta_e_grid:
            # Test elevator across rotation alpha range
            moment_ratios = []
            for alpha_test in alphas_rotation:
                L, D, M_aero, _ = self.aero_model.get_forces_moments(
                    V_rotation, alpha_test, delta_e, rho, altitude=0.0
                )

                # Moment about gear pivot (x_gear, z=0)
                # 
                # PHYSICS NOTE: Since M_aero is computed about CG (xyz_ref=[x_cg,0,0]),
                # we need to transfer it to gear pivot, and compute all force moments
                # using position vector from gear to CG.
                #
                # Coordinate Convention (Ground Frame):
                #   - X is forward, Z is up, M is nose-up positive.
                #   - r_cg = (dx, dz) where dx is distance forward of gear.
                #   - M = dx * Fz - dz * Fx
                
                theta_test = np.radians(alpha_test)  # During ground roll, theta ≈ alpha
                sin_t = np.sin(theta_test)
                cos_t = np.cos(theta_test)
                
                # CG position relative to gear (ground frame)
                # If x is positive-aft, then (x_gear - x_cg) is the forward arm at theta=0.
                x_arm_body = self.x_gear - self.x_cg
                dx_cg = x_arm_body * cos_t - self.z_cg * sin_t
                dz_cg = x_arm_body * sin_t + self.z_cg * cos_t
                
                # Thrust line position relative to gear (ground frame)
                dx_thrust = x_arm_body * cos_t - self.z_thrust_line * sin_t
                dz_thrust = x_arm_body * sin_t + self.z_thrust_line * cos_t
                
                # Ground-frame components of forces
                # For autotune (ground roll), we use gamma=0 approximation.
                # Lift is up (+Z), Drag is back (-X), Weight is down (-Z).
                L_gx = 0.0
                L_gz = L
                D_gx = -D
                D_gz = 0.0
                
                T_x_ground = thrust_at_rotation * np.cos(theta_test + eps)
                T_z_ground = thrust_at_rotation * np.sin(theta_test + eps)
                
                # Total moment about gear pivot (ground frame)
                # M = dx*Fz - dz*Fx
                M_gear = (
                    M_aero
                    + dx_cg * (L_gz + D_gz - self.weight_N) 
                    - dz_cg * (L_gx + D_gx)
                    + dx_thrust * T_z_ground - dz_thrust * T_x_ground
                )

                moment_ratio = M_gear / (thrust_at_rotation * self.z_thrust_line) if thrust_at_rotation > 0 else 0
                moment_ratios.append(moment_ratio)

            # Check if this elevator gives positive moments across alpha range
            # and has good average moment capability
            avg_moment = np.mean(moment_ratios)
            min_moment = np.min(moment_ratios)
            # moment_ratios already contains normalized values (M / (T*z_thrust_line))
            # so min_moment_ratio is directly the minimum of moment_ratios
            min_moment_ratio = min_moment if thrust_at_rotation > 0 else 0
            
            # Select elevator that gives best average positive moment
            if min_moment_ratio > 0 and avg_moment > max_moment_ratio:
                max_moment_ratio = avg_moment
                best_rotation_delta_e = delta_e

        print(f"  Rotation delta_e: {best_rotation_delta_e:.1f}deg")

        # 2. Find climb_delta_e using iterative approach
        # Goal: achieve target pitch at reasonable climb speed (~1.2-1.5 * V_rotation)
        V_climb = 1.3 * V_rotation  # Reasonable climb speed

        # Get thrust at climb speed
        thrust_at_climb = self.propulsion.get_thrust(V_climb, throttle=1.0, rho=rho)

        # Binary search or grid search for delta_e that achieves target pitch
        # We want pitch angle where M_aero = 0 (trim condition)
        best_climb_delta_e = -6.0
        min_pitch_error = float("inf")
        debug_counter = 0

        for delta_e in self.aero_model._delta_e_grid:
            # Find trim alpha where sum of moments = 0 for this delta_e
            trim_alpha = None
            min_m_total = float("inf")

            # Search alpha range for trim
            for alpha_deg in self.aero_model._alpha_grid:
                L, D, M_aero, _ = self.aero_model.get_forces_moments(
                    V_climb, alpha_deg, delta_e, rho, altitude=0.0
                )
                
                # Total moment about CG (free flight)
                dz_cg_thrust = self.z_thrust_line - self.z_cg
                eps = np.radians(self.thrust_angle_deg)
                T_x = thrust_at_climb * np.cos(eps)
                
                # M = M_aero - T_x*dz
                M_thrust_cg = -T_x * dz_cg_thrust
                M_total = M_aero + M_thrust_cg

                if abs(M_total) < min_m_total:
                    min_m_total = abs(M_total)
                    trim_alpha = alpha_deg

            if trim_alpha is not None:
                # Approximate gamma from force balance: sin(gamma) = (T_wind_x - D) / W
                alpha_rad = np.radians(trim_alpha)
                T_wind_x = thrust_at_climb * np.cos(alpha_rad + eps)
                sin_gamma = (T_wind_x - D) / self.weight_N
                gamma_approx = np.degrees(np.arcsin(min(0.8, max(-0.1, sin_gamma))))
                
                theta_approx = trim_alpha + gamma_approx
                pitch_error = abs(theta_approx - target_pitch_deg)

                if pitch_error < min_pitch_error:
                    min_pitch_error = pitch_error
                    best_climb_delta_e = delta_e

                # Debug: print iterations
                debug_counter += 1
                if debug_counter <= 10 and self.aero_model.verbose:
                    print(
                        f"    delta_e={delta_e:.1f}deg, trim_alpha={trim_alpha:.1f}deg, gamma={gamma_approx:.1f}deg, theta={theta_approx:.1f}deg"
                    )


        print(
            f"  Climb delta_e: {best_climb_delta_e:.1f}deg (pitch error: {min_pitch_error:.1f}deg)"
        )

        return best_rotation_delta_e, best_climb_delta_e

    def simulate(
        self,
        V_rotation: float,
        alpha_ground_deg: float = 1.0,
        rotation_delta_e: float = -15.0,  # Up-elevon for moderate rotation (reduced from -20.0)
        climb_delta_e: float = -3.0,  # Up-elevon for climbout (very conservative)
        rho: float = 1.225,
        dt: float = 0.01,
        max_time: float = 40.0,
        initial_SOC: float = 1.0,  # Starting state of charge
        track_SOC: bool = True,  # Whether to integrate SOC during simulation
    ) -> TakeoffResult:
        """
        Run 3-DOF simulation with optional SOC tracking.

        When track_SOC=True, integrates battery state of charge and uses
        voltage sag model for realistic thrust degradation during takeoff.

        Args:
            V_rotation: Velocity to initiate rotation [m/s]
            alpha_ground_deg: Ground attitude [degrees]
            rotation_delta_e: Elevon deflection during rotation [degrees]
            climb_delta_e: Elevon deflection during climbout [degrees]
            rho: Air density [kg/m^3]
            dt: Time step [s]
            max_time: Maximum simulation time [s]
            initial_SOC: Starting state of charge [0-1] (default 1.0 = full)
            track_SOC: If True, integrate SOC and apply voltage sag (default True)

        Returns:
            TakeoffResult with full time histories
        """
        # Initial State
        x = 0.0
        h = 0.0
        V = 0.1
        gamma = 0.0
        theta = np.radians(alpha_ground_deg)
        q = 0.0
        t = 0.0
        SOC = initial_SOC

        # History
        times, velocities, distances = [t], [V], [x]
        thrusts, accelerations = [], []
        pitches, pitch_rates = [np.degrees(theta)], [q]
        alphas, gammas, altitudes = [alpha_ground_deg], [0.0], [0.0]
        elevons = []
        soc_history = [SOC]
        voltage_history = []

        # Track liftoff time for smooth elevator transition
        liftoff_time = None
        elevator_transition_time = 2.0  # seconds to transition from rotation to climb elevator (increased from 1.0s)

        # Tracking maxes
        max_current = 0.0
        max_power = 0.0

        # Simulation flags
        rotating = False
        airborne = False

        # Battery capacity for SOC integration
        capacity_Ah = self.propulsion.battery.capacity_mAh / 1000.0

        static_thrust = self.propulsion.get_static_thrust(rho, SOC=initial_SOC)

        while t < max_time:
            # 1. Determine Control Inputs
            if not rotating and V >= V_rotation:
                rotating = True

            if airborne:
                # Record liftoff time if not set
                if liftoff_time is None:
                    liftoff_time = t

                # Smooth elevator transition: blend from rotation to climb over 1 second
                if (
                    liftoff_time is not None
                    and (t - liftoff_time) < elevator_transition_time
                ):
                    transition_factor = (t - liftoff_time) / elevator_transition_time
                    delta_e = (
                        rotation_delta_e
                        + (climb_delta_e - rotation_delta_e) * transition_factor
                    )
                else:
                    delta_e = climb_delta_e
            elif rotating:
                delta_e = rotation_delta_e
            else:
                delta_e = 0.0  # Keep elevons neutral during high-speed roll

            elevons.append(delta_e)

            # 2. Get Aerodynamic and Propulsion Forces
            alpha_deg = np.degrees(theta - gamma)
            L, D, M_aero, CMq = self.aero_model.get_forces_moments(
                V, alpha_deg, delta_e, rho, altitude=h
            )
 
            # Calculate dynamic pitch damping from AeroBuildup
            #
            # CMq SIGN CONVENTION:
            # AeroBuildup returns CMq with the standard aerodynamic convention:
            #   - Negative CMq = pitch damping (stabilizing)
            #   - Positive CMq = anti-damping (destabilizing)
            #
            # The formula below assumes this sign convention:
            #   M_q = 0.5 * rho * V^2 * S * c * (c / 2V) * CMq * q_rate
            #   Simplifies to: M_q = 0.25 * rho * V * S * c^2 * CMq * q_rate
            #
            # If AeroBuildup were to use an opposite sign convention, CMq would be
            # destabilizing. For typical configurations, CMq should be negative.
            #
            M_damping = 0.25 * rho * V * self.aero_model.S * self.mac**2 * CMq * q


            prop_res = self.propulsion.solve_operating_point(
                V, throttle=1.0, rho=rho, SOC=SOC
            )
            T_mag = prop_res["thrust_total"]
            
            # Thrust direction in body frame (relative to body x-axis)
            # thrust_angle_deg is the angle of thrust line above body x-axis
            eps = np.radians(self.thrust_angle_deg)
            
            # Thrust components in body axes (for moment calculations)
            T_x_body = T_mag * np.cos(eps)
            T_z_body = T_mag * np.sin(eps)
            
            # Thrust components in inertial/ground axes (for position integration)
            # theta = pitch angle of body relative to ground
            # Total thrust angle in ground frame = theta + eps
            thrust_angle_ground = theta + eps
            T_x_inertial = T_mag * np.cos(thrust_angle_ground)
            T_z_inertial = T_mag * np.sin(thrust_angle_ground)
            
            # Thrust components in wind axes (for flight path dynamics)
            # alpha = theta - gamma (angle of attack)
            # The thrust vector makes angle (alpha + eps) with respect to velocity vector
            # T_wind_x: along velocity (positive forward)
            # T_wind_z: perpendicular to velocity (positive up/normal)
            alpha_rad = theta - gamma  # angle of attack in radians
            T_wind_x = T_mag * np.cos(alpha_rad + eps)
            T_wind_z = T_mag * np.sin(alpha_rad + eps)
            
            thrusts.append(T_mag)
            voltage_history.append(
                prop_res.get("voltage_terminal", prop_res.get("voltage_battery", 0))
            )

            # Track max power/current
            max_current = max(max_current, prop_res["current_battery"])
            max_power = max(max_power, prop_res["power_shaft_total"])

            # Integrate SOC if tracking enabled
            if track_SOC:
                I_battery = prop_res["current_battery"]
                # dSOC/dt = -I / (3600 * capacity_Ah)
                dSOC = -I_battery * dt / (3600.0 * capacity_Ah)
                SOC = max(0.0, SOC + dSOC)
                soc_history.append(SOC)

            # 3. Sum Forces and Moments
            if not airborne:
                # Normal force and ground friction
                # Vertical balance: N + L + T_z_inertial = W * cos(gamma)
                N = max(0, self.weight_N * np.cos(gamma) - L - T_z_inertial)
                F_roll = self.mu_roll * N

                # Liftoff criteria: Normal force goes to zero and we have vertical acceleration
                if N <= 0 and (
                    T_z_inertial + L
                    > self.weight_N * np.cos(gamma)
                ):
                    airborne = True

                # Moment about gear pivot (main gear contact point)
                #
                # Coordinate Convention (Ground Frame):
                #   - X is forward, Z is up, M is nose-up positive.
                #   - r_cg = (dx, dz) where dx is distance forward of gear.
                #   - M = dx * Fz - dz * Fx
                
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                
                # CG position relative to gear pivot (in ground frame)
                # If x is positive-aft, then (x_gear - x_cg) is the forward arm at theta=0.
                x_arm_body = self.x_gear - self.x_cg
                dx_gear_cg = x_arm_body * cos_theta - self.z_cg * sin_theta
                dz_gear_cg = x_arm_body * sin_theta + self.z_cg * cos_theta
                
                # Thrust line position relative to gear pivot (in ground frame)
                dx_gear_thrust = x_arm_body * cos_theta - self.z_thrust_line * sin_theta
                dz_gear_thrust = x_arm_body * sin_theta + self.z_thrust_line * cos_theta

                # Ground-frame components of aero forces
                # Wind to ground rotation is gamma (flight path angle)
                sin_gamma = np.sin(gamma)
                cos_gamma = np.cos(gamma)
                
                L_gx = -L * sin_gamma
                L_gz =  L * cos_gamma
                D_gx = -D * cos_gamma
                D_gz = -D * sin_gamma
                
                # Total moment about gear pivot (ground frame)
                # M = dx*Fz - dz*Fx
                M_gear = (
                    M_aero
                    + M_damping
                    + dx_gear_cg * (L_gz + D_gz - self.weight_N)
                    - dz_gear_cg * (L_gx + D_gx)
                    + dx_gear_thrust * T_z_inertial - dz_gear_thrust * T_x_inertial
                )
                
                if not airborne:
                    # During rotation, q is constrained by gear interaction
                    if M_gear > 0 or q > 0:
                        q_dot = M_gear / self.Iyy
                    else:
                        q_dot = 0.0
                        q = 0.0

                    # axial acceleration (using horizontal component of thrust)
                    a = (T_x_inertial - D - F_roll) / self.mass_kg
                    gamma_dot = 0.0
                else:
                    # Transition to free flight equations (wind-axis formulation)
                    # Tangential: m * dV/dt = T*cos(alpha+eps) - D - W*sin(gamma)
                    # Normal: m * V * dgamma/dt = T*sin(alpha+eps) + L - W*cos(gamma)
                    a = (T_wind_x - D - self.weight_N * np.sin(gamma)) / self.mass_kg
                    gamma_dot = (
                        T_wind_z + L - self.weight_N * np.cos(gamma)
                    ) / (self.mass_kg * max(V, 1.0))
                    q_dot = (M_aero + M_damping) / self.Iyy
            else:
                # Free flight (3-DOF) - Wind-axis equations of motion
                # 
                # Standard 3-DOF point-mass flight dynamics:
                # Tangential (along velocity): m * dV/dt = T*cos(alpha+eps) - D - W*sin(gamma)
                # Normal (perpendicular to velocity): m * V * dgamma/dt = T*sin(alpha+eps) + L - W*cos(gamma)
                #
                # Using wind-axis thrust components ensures correct physics when theta != gamma
                a = (T_wind_x - D - self.weight_N * np.sin(gamma)) / self.mass_kg
                gamma_dot = (
                    T_wind_z + L - self.weight_N * np.cos(gamma)
                ) / (self.mass_kg * max(V, 1.0))

                # Use AeroBuildup damping + additional artificial damping if provided
                if self.Cmq is not None and V > 1.0:
                    M_q_extra = 0.25 * rho * V * self.S * self.mac**2 * self.Cmq * q
                else:
                    M_q_extra = 0.0

                q_dot = (M_aero + M_damping + M_q_extra) / self.Iyy

            accelerations.append(a)

            # 4. Integrate State
            V += a * dt
            x += V * np.cos(gamma) * dt
            h += V * np.sin(gamma) * dt
            gamma += gamma_dot * dt
            q += q_dot * dt
            theta += q * dt

            # Ground constraints
            if h <= 0:
                h = 0
                gamma = max(0, gamma)

            if theta < np.radians(alpha_ground_deg) and not airborne:
                theta = np.radians(alpha_ground_deg)
                q = max(0, q)

            t += dt

            # Store histories
            times.append(t)
            velocities.append(V)
            distances.append(x)
            pitches.append(np.degrees(theta))
            pitch_rates.append(np.degrees(q))
            alphas.append(alpha_deg)
            gammas.append(np.degrees(gamma))
            altitudes.append(h)

            # Exit condition: stable climb reached
            if airborne and h > 15.0:  # 50 ft
                break

        # Final results
        elevons.append(elevons[-1])  # Align lengths

        requirement_m = self.requirement_m
        ground_roll = x
        for i, alt in enumerate(altitudes):
            if alt > 0.1:
                ground_roll = distances[i]
                break

        return TakeoffResult(
            ground_roll_m=ground_roll,
            ground_roll_ft=ground_roll * 3.281,
            takeoff_time_s=t,
            liftoff_velocity_mps=V,
            static_thrust_N=static_thrust,
            thrust_at_liftoff_N=T_mag,
            max_battery_current_A=max_current,
            max_power_W=max_power,
            passes_requirement=(ground_roll <= requirement_m),
            requirement_m=requirement_m,
            margin_m=requirement_m - ground_roll,
            margin_percent=(requirement_m - ground_roll) / requirement_m * 100,
            rotation_velocity_mps=V_rotation,
            max_pitch_deg=max(pitches),
            max_q_degps=max(pitch_rates),
            time_history=np.array(times),
            velocity_history=np.array(velocities),
            distance_history=np.array(distances),
            thrust_history=np.array(thrusts),
            acceleration_history=np.array(accelerations),
            pitch_history=np.array(pitches),
            pitch_rate_history=np.array(pitch_rates),
            alpha_history=np.array(alphas),
            gamma_history=np.array(gammas),
            altitude_history=np.array(altitudes),
            elevon_history=np.array(elevons),
            # SOC tracking data
            soc_history=np.array(soc_history) if track_SOC else None,
            voltage_history=np.array(voltage_history)
            if track_SOC and voltage_history
            else None,
            initial_soc=initial_SOC,
            final_soc=SOC,
            soc_consumed=initial_SOC - SOC,
        )

    def run_full_analysis(
        self,
        CL_max: float = 1.2,
        alpha_ground_deg: float = 1.0,
        autotune: bool = True,
        target_pitch_deg: float = 12.0,
    ) -> TakeoffResult:
        """
        Run complete 3-DOF analysis with computed rotation velocity.

        Args:
            CL_max: Maximum lift coefficient
            alpha_ground_deg: Ground attitude [degrees]
            autotune: If True, use autotuned elevator deflections
            target_pitch_deg: Target pitch angle during climb [degrees]

        Returns:
            TakeoffResult
        """
        V_stall = self.calculate_liftoff_velocity(CL_max, margin=1.0)
        V_rotation = 1.1 * V_stall

        if autotune:
            rotation_delta_e, climb_delta_e = self.autotune_elevator(
                V_rotation, target_pitch_deg=target_pitch_deg
            )
        else:
            rotation_delta_e = -15.0
            climb_delta_e = -3.0

        return self.simulate(
            V_rotation, alpha_ground_deg, rotation_delta_e, climb_delta_e
        )

    def sweep_motor_kv(
        self,
        kv_range: List[float],
        CL_max: float = 1.2,
    ) -> List[Dict]:
        """
        Sweep motor Kv values to find optimal configuration.

        Args:
            kv_range: List of Kv values to test
            CL_max: Maximum lift coefficient

        Returns:
            List of results for each Kv
        """
        results = []
        V_liftoff = self.calculate_liftoff_velocity(CL_max)

        for kv in kv_range:
            # Update motor config
            self.propulsion.motor_config = MotorConfig(
                kv=kv,
                R_internal=self.propulsion.motor_config.R_internal,
                I_no_load=self.propulsion.motor_config.I_no_load,
                n_motors=self.propulsion.motor_config.n_motors,
            )
            self.propulsion.motor_model = DifferentiableMotorModel(
                self.propulsion.motor_config.params
            )
            self.propulsion._thrust_cache.clear()

            # For 3-DOF, we need to provide rotation velocity
            V_rotation = 1.1 * V_liftoff / 1.2  # Approximate Vr
            result = self.simulate(V_rotation)

            results.append(
                {
                    "kv": kv,
                    "ground_roll_m": result.ground_roll_m,
                    "ground_roll_ft": result.ground_roll_ft,
                    "static_thrust_N": result.static_thrust_N,
                    "max_current_A": result.max_battery_current_A,
                    "passes": result.passes_requirement,
                }
            )

        return results


# =============================================================================
# Climb Calculator
# =============================================================================


class ClimbCalculator:
    """
    Climb phase calculator using quasi-steady approximation.

    PHYSICS MODEL:
    This calculator uses a quasi-steady climb model, NOT full dynamic simulation.
    
    The climb angle gamma is computed from force balance assuming:
        sin(gamma) = (T*cos(alpha+eps) - D) / W
    
    Then velocity is integrated using the same forces:
        m * dV/dt = T*cos(alpha+eps) - D - W*sin(gamma)
    
    IMPORTANT: Because gamma is derived from the force balance and dV/dt uses
    the same balance, the tangential acceleration tends toward zero in steady
    climb. This is intentional for mission-level analysis but is NOT a true
    dynamic simulation where velocity would evolve independently.
    
    For true dynamic climb simulation, gamma would need its own equation of
    motion based on normal force balance:
        m * V * dgamma/dt = T*sin(alpha+eps) + L - W*cos(gamma)
    
    The quasi-steady approximation is appropriate for:
    - Mission planning and energy estimation
    - Steady-state climb performance
    - Cases where climb transients are not critical
    """

    def __init__(
        self,
        aero_model: RigidBodyAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
        thrust_angle_deg: float = 0.0,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S
        self.thrust_angle_deg = thrust_angle_deg

    def calculate_climb(
        self,
        V_initial: float,
        target_altitude_m: float,
        throttle: float = 0.9,
        CL_climb: float = 0.6,
        dt: float = 0.05,
        max_time: float = 300.0,
        max_horizontal_distance_m: float = None,
    ) -> ClimbResult:
        """
        Calculate climb phase from liftoff to target altitude (quasi-steady model).

        QUASI-STEADY APPROXIMATION:
        The climb angle gamma is computed from tangential force balance:
            sin(gamma) = (T*cos(alpha+eps) - D) / W
        
        Then velocity is integrated:
            m * dV/dt = T*cos(alpha+eps) - D - W*sin(gamma)
        
        Since gamma is derived from the same force balance used in dV/dt, the
        acceleration tends toward zero in steady climb. This gives correct
        steady-state climb performance but NOT dynamic transient behavior.
        
        For mission planning and energy estimation, this approximation is
        typically sufficient. For dynamic analysis, use the full 3-DOF
        TakeoffCalculator simulation.
        """
        h = 0.0
        x = 0.0
        V = max(V_initial, 1.0)
        rho = 1.225
        t = 0.0

        # History arrays
        times = [t]
        altitudes = [h]
        distances = [x]
        velocities = [V]

        # Energy tracking
        total_energy_J = 0.0

        # Termination flags
        reached_altitude = False
        reached_distance_limit = False

        # Minimum velocity limit (stall protection)
        V_stall = np.sqrt(2 * self.weight_N / (rho * self.S * 1.4))  # CL_max ~ 1.4
        V_min = 1.1 * V_stall  # 10% above stall

        while t < max_time:
            # Check termination conditions
            if h >= target_altitude_m:
                reached_altitude = True
                break
            if max_horizontal_distance_m is not None and x >= max_horizontal_distance_m:
                reached_distance_limit = True
                break

            # --- Aerodynamic forces ---
            # Alpha from CL (approximate: CL_alpha ~ 5/rad for typical wing)
            alpha_deg = CL_climb / (5 * np.pi / 180)
            alpha_deg = min(alpha_deg, 12.0)  # Limit alpha to avoid stall

            # Get lift and drag
            L, D = self.aero_model.get_forces(V, alpha_deg, rho, altitude=h)

            # --- Propulsion ---
            prop_result = self.propulsion.solve_operating_point(
                V, throttle=throttle, rho=rho
            )
            T = prop_result["thrust_total"]
            P_shaft = prop_result["power_shaft_total"]

            # Thrust angle in wind axes: alpha + thrust_angle
            eps = np.radians(self.thrust_angle_deg)
            alpha_rad = np.radians(alpha_deg)
            angle_thrust_wind = alpha_rad + eps
            
            T_wind_x = T * np.cos(angle_thrust_wind)
            T_wind_z = T * np.sin(angle_thrust_wind)

            # --- Climb angle from force balance ---
            # Standard climb equation: sin(gamma) = (T*cos(alpha+eps) - D) / W
            sin_gamma = (T_wind_x - D) / self.weight_N
            sin_gamma = min(0.85, max(-0.1, sin_gamma))  # Clamp to physical limits
            gamma = np.arcsin(sin_gamma)
            cos_gamma = np.cos(gamma)

            # Check if lift is sufficient for this climb angle
            # L + T_wind_z must be >= W*cos(gamma). If not, we sink.
            L_needed = self.weight_N * cos_gamma - T_wind_z
            if L < L_needed and V > 1.0:
                # Need to increase alpha to get more lift (up to limit)
                # But for this simple simulation, we just reduce gamma to what lift can support
                cos_gamma_max = min(1.0, (L + T_wind_z) / self.weight_N)
                gamma = np.arccos(cos_gamma_max)
                sin_gamma = np.sin(gamma)

            # --- Tangential equation of motion ---
            # m * dV/dt = T_wind_x - D - W*sin(gamma)
            F_tangential = T_wind_x - D - self.weight_N * sin_gamma
            dV_dt = F_tangential / self.mass_kg

            # --- Integrate velocity ---
            V_new = V + dV_dt * dt
            V_new = max(V_min, V_new)  # Stall protection

            # --- Integrate position ---
            V_avg = 0.5 * (V + V_new)
            V_vertical = V_avg * np.sin(gamma)
            V_horizontal = V_avg * np.cos(gamma)

            h_new = h + V_vertical * dt
            x_new = x + V_horizontal * dt

            # --- Update state ---
            V = V_new
            h = max(0, h_new)  # Don't go below ground
            x = x_new
            t += dt

            # Energy consumed this step
            total_energy_J += P_shaft * dt

            # Store history
            times.append(t)
            altitudes.append(h)
            distances.append(x)
            velocities.append(V)


        # Actual altitude achieved
        actual_altitude = min(h, target_altitude_m)

        # Calculate summary results
        climb_time = t
        avg_climb_rate = actual_altitude / t if t > 0 else 0
        climb_angle = np.degrees(np.arctan2(actual_altitude, x)) if x > 0 else 0
        energy_Wh = total_energy_J / 3600

        return ClimbResult(
            altitude_gain_m=actual_altitude,
            target_altitude_m=target_altitude_m,
            horizontal_distance_m=x,
            climb_time_s=climb_time,
            climb_angle_deg=climb_angle,
            average_climb_rate_mps=avg_climb_rate,
            average_velocity_mps=np.mean(velocities),
            average_throttle=throttle,
            energy_consumed_Wh=energy_Wh,
            distance_constrained=reached_distance_limit,
            distance_limit_m=max_horizontal_distance_m,
            time_history=np.array(times),
            altitude_history=np.array(altitudes),
            distance_history=np.array(distances),
            velocity_history=np.array(velocities),
        )


# =============================================================================
# Cruise Calculator
# =============================================================================


class CruiseCalculator:
    """
    Cruise phase calculator.
    """

    def __init__(
        self,
        aero_model: RigidBodyAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
        thrust_angle_deg: float = 0.0,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S
        self.thrust_angle_deg = thrust_angle_deg

    def calculate_cruise(
        self,
        distance_m: float,
        altitude_m: float,
        V_initial: float,
        CL_cruise: float = 0.5,
        dt: float = 0.05,
        max_time: float = 600.0,
    ) -> CruiseResult:
        """
        Calculate cruise phase for given distance with physics-based velocity evolution.
        """
        # Air density at altitude
        rho = 1.225 * np.exp(-altitude_m / 8500)

        # Thrust angle in radians
        eps = np.radians(self.thrust_angle_deg)
        alpha_target_deg = CL_cruise / (5 * np.pi / 180)
        alpha_rad = np.radians(alpha_target_deg)

        # Target cruise velocity from L + T*sin(alpha+eps) = W
        # Assuming T ~= D at cruise, we can solve iteratively or just use L=W as first guess
        V_target = np.sqrt(2 * self.weight_N / (rho * self.S * CL_cruise))

        # Determine throttle such that T*cos(alpha+eps) ~= D at V_target
        _, D_target = self.aero_model.get_forces(
            V_target, alpha_target_deg, rho, altitude=altitude_m
        )
        throttle_low, throttle_high = 0.05, 1.0
        for _ in range(25):
            throttle = 0.5 * (throttle_low + throttle_high)
            T_test = self.propulsion.solve_operating_point(
                V_target, throttle=throttle, rho=rho
            )["thrust_total"]
            if T_test * np.cos(alpha_rad + eps) < D_target:
                throttle_low = throttle
            else:
                throttle_high = throttle
        throttle = 0.5 * (throttle_low + throttle_high)

        # Integrate velocity and distance with fixed throttle
        V = max(0.1, V_initial)
        x = 0.0
        t = 0.0
        times = [t]
        velocities = [V]
        distances = [x]
        total_energy_J = 0.0

        while x < distance_m and t < max_time:
            # Lift/drag at current speed using target CL (approx quasi-level flight)
            q = 0.5 * rho * V**2
            CL_required = self.weight_N / (q * self.S) if q > 1e-6 else CL_cruise
            CL_use = max(0.1, min(CL_required, CL_cruise * 1.2))
            alpha_deg = CL_use / (5 * np.pi / 180)
            alpha_deg = min(alpha_deg, 12.0)
            _, D = self.aero_model.get_forces(V, alpha_deg, rho, altitude=altitude_m)

            # Thrust at fixed throttle
            prop_result = self.propulsion.solve_operating_point(
                V, throttle=throttle, rho=rho
            )
            T = prop_result["thrust_total"]
            P_shaft = prop_result["power_shaft_total"]

            # Tangential EOM for level flight: m dV/dt = T*cos(alpha+eps) - D
            alpha_rad_loop = np.radians(alpha_deg)
            a = (T * np.cos(alpha_rad_loop + eps) - D) / self.mass_kg
            V_new = max(0.1, V + a * dt)
            V_avg = 0.5 * (V + V_new)

            # Position update
            x += V_avg * dt
            t += dt
            V = V_new

            total_energy_J += P_shaft * dt

            times.append(t)
            velocities.append(V)
            distances.append(x)

        cruise_time = t
        power_W = total_energy_J / cruise_time if cruise_time > 0 else 0.0
        energy_Wh = total_energy_J / 3600
        L_over_D = (self.weight_N) / D if D > 0 else 10.0


        return CruiseResult(
            distance_m=distance_m,
            cruise_time_s=cruise_time,
            cruise_velocity_mps=np.mean(velocities[-10:])
            if len(velocities) >= 10
            else velocities[-1],
            cruise_altitude_m=altitude_m,
            throttle_setting=throttle,
            power_W=power_W,
            current_A=prop_result["current_battery"],
            energy_consumed_Wh=energy_Wh,
            L_over_D=L_over_D,
            time_history=np.array(times),
            velocity_history=np.array(velocities),
            distance_history=np.array(distances),
        )


# =============================================================================
# Descent Calculator
# =============================================================================


class DescentCalculator:
    """
    Descent phase calculator.
    """

    def __init__(
        self,
        aero_model: RigidBodyAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
        thrust_angle_deg: float = 0.0,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S
        self.thrust_angle_deg = thrust_angle_deg

    def calculate_descent(
        self,
        start_altitude_m: float,
        end_altitude_m: float,
        V_initial: float,
        throttle: float = 0.0,
        CL_descent: float = 0.7,
        dt: float = 0.05,
        max_time: float = 300.0,
    ) -> DescentResult:
        """
        Calculate descent phase with dynamic velocity evolution.

        PHYSICS MODEL:
        Unlike the original kinematic version, this integrates velocity
        using the full tangential equation of motion:
            m * dV/dt = T*cos(alpha+eps) - D - W*sin(gamma)
        
        This allows velocity to evolve naturally during descent when thrust
        is applied or when transitioning between different flight regimes.
        For pure gliding (throttle=0), velocity will tend toward
        the equilibrium glide speed determined by L/D ratio.
        
        The climb/descent angle gamma is computed from quasi-steady
        normal force balance:
            sin(gamma) = (D - T*cos(alpha+eps)) / W
        
        This decoupling allows for more realistic descent dynamics.
        """
        rho = 1.225
        eps = np.radians(self.thrust_angle_deg)

        # Target descent speed from CL_descent, start from incoming if higher
        V_descent = np.sqrt(2 * self.weight_N / (rho * self.S * CL_descent))
        V = max(V_descent, V_initial)
        h = start_altitude_m
        x = 0.0
        t = 0.0
        gamma = 0.0  # Initial flight path angle (will be computed)
 
        times = [t]
        altitudes = [h]
        distances = [x]
        velocities = [V]
        total_energy_J = 0.0

        # Minimum velocity for descent (stall protection)
        V_min = 1.1 * np.sqrt(2 * self.weight_N / (rho * self.S * 1.4))

        while h > end_altitude_m and t < max_time:
            # Get aerodynamic forces at current state
            alpha_deg = CL_descent / (5 * np.pi / 180)
            alpha_deg = min(alpha_deg, 12.0)
            L, D = self.aero_model.get_forces(V, alpha_deg, rho, altitude=h)
            
            # Get propulsion (throttle may be non-zero for powered descent)
            if throttle > 0:
                prop_res = self.propulsion.solve_operating_point(V, throttle=throttle, rho=rho)
                T = prop_res["thrust_total"]
                total_energy_J += prop_res["power_shaft_total"] * dt
            else:
                T = 0.0
 
            # Wind-axis thrust components
            alpha_rad = np.radians(alpha_deg)
            T_wind_x = T * np.cos(alpha_rad + eps)
            T_wind_z = T * np.sin(alpha_rad + eps)
            
            # Flight path angle from quasi-steady normal force balance
            # Assuming quasi-steady (dgamma/dt ≈ 0):
            # T_wind_z + L = W*cos(gamma)
            L_eff = L + T_wind_z
            cos_gamma = min(1.0, max(-1.0, L_eff / self.weight_N))
            gamma = np.arccos(cos_gamma)
            
            # Tangential equation of motion:
            # m * dV/dt = T_wind_x - D - W*sin(gamma)
            a = (T_wind_x - D - self.weight_N * np.sin(gamma)) / self.mass_kg
            
            # Update velocity with stall protection
            V_new = V + a * dt
            V_new = max(V_min, V_new)
            V_avg = 0.5 * (V + V_new)
            
            # Position update
            # Negative gamma means we're going down (descent)
            V_vertical = -V_avg * np.sin(gamma)
            V_horizontal = V_avg * np.cos(gamma)
            h = h + V_vertical * dt
            x = x + V_horizontal * dt
            t += dt
            V = V_new

            times.append(t)
            altitudes.append(h)
            distances.append(x)
            velocities.append(V)

        altitude_loss = start_altitude_m - max(h, end_altitude_m)
        descent_angle = np.degrees(np.arctan2(altitude_loss, x)) if x > 0 else 5.0
        avg_sink_rate = altitude_loss / t if t > 0 else 0

        return DescentResult(
            altitude_loss_m=altitude_loss,
            horizontal_distance_m=x,
            descent_time_s=t,
            descent_angle_deg=descent_angle,
            average_sink_rate_mps=avg_sink_rate,
            average_velocity_mps=np.mean(velocities),
            throttle_setting=throttle,
            energy_consumed_Wh=total_energy_J / 3600,
            time_history=np.array(times),
            altitude_history=np.array(altitudes),
            distance_history=np.array(distances),
            velocity_history=np.array(velocities),
        )


# =============================================================================
# Landing Calculator
# =============================================================================


class LandingCalculator:
    """
    Landing phase calculator.
    """

    def __init__(
        self,
        aero_model: RigidBodyAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
        mu_brake: float = 0.1,
        thrust_angle_deg: float = 0.0,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S
        self.mu_brake = mu_brake
        self.thrust_angle_deg = thrust_angle_deg

    def calculate_landing(
        self,
        approach_altitude_m: float,
        V_initial: float,
        flare_altitude_m: float = 2.0,
        glide_slope_deg: float = 5.0,
        throttle: float = 0.0,
        CL_max: float = 1.2,
        dt: float = 0.05,
    ) -> LandingResult:
        """
        Calculate complete landing sequence with physics-based velocity evolution.

        Args:
            approach_altitude_m: Starting altitude for approach [m]
            V_initial: Incoming velocity from descent [m/s]
            flare_altitude_m: Altitude to begin flare [m]
            glide_slope_deg: Approach glide slope [deg]
            throttle: Throttle during approach (idle)
            CL_max: Max CL available in landing config
            dt: Integration time step [s]

        Returns:
            LandingResult
        """
        rho = 1.225

        # Reference approach speed ~1.3 * Vstall
        V_stall = np.sqrt(2 * self.weight_N / (rho * self.S * CL_max))
        V_ref = 1.3 * V_stall
        V_approach = max(
            V_ref, 0.8 * V_initial
        )  # decel from incoming but cap not below V_ref
        altitude_drop = max(0.0, approach_altitude_m - flare_altitude_m)
        approach_distance = (
            altitude_drop / np.tan(np.radians(glide_slope_deg))
            if glide_slope_deg > 0
            else 0.0
        )
        approach_time = approach_distance / V_approach if V_approach > 0 else 0.0

        # Idle approach/flare energy (set to zero like descent idle)
        approach_energy_J = 0.0

        # Flare: decelerate to touchdown at 85% approach speed over ~2.5 s
        flare_time = 2.5
        V_touchdown = V_approach * 0.85
        flare_distance = (V_approach + V_touchdown) / 2 * flare_time
        flare_energy_J = 0.0

        # Ground roll with braking + aero drag integration
        ground_distance = 0.0
        ground_time = 0.0
        V = V_touchdown
        while V > 0.1:
            # Aero drag at small alpha
            _, D = self.aero_model.get_forces(V, 0.0, rho, altitude=0.0)
            F_brake = self.mu_brake * self.weight_N
            a = -(D + F_brake) / self.mass_kg
            V_new = max(0.0, V + a * dt)
            V_avg = 0.5 * (V + V_new)
            ground_distance += V_avg * dt
            ground_time += dt
            V = V_new

        total_distance = approach_distance + flare_distance + ground_distance
        total_time = approach_time + flare_time + ground_time
        total_energy_Wh = (approach_energy_J + flare_energy_J) / 3600

        return LandingResult(
            approach_distance_m=approach_distance,
            flare_distance_m=flare_distance,
            ground_roll_m=ground_distance,
            total_landing_distance_m=total_distance,
            approach_time_s=approach_time,
            flare_time_s=flare_time,
            ground_roll_time_s=ground_time,
            total_time_s=total_time,
            approach_velocity_mps=V_approach,
            touchdown_velocity_mps=V_touchdown,
            flare_altitude_m=flare_altitude_m,
            energy_consumed_Wh=total_energy_Wh,
        )


# =============================================================================
# Mission Analyzer
# =============================================================================


class MissionAnalyzer:
    """
    Complete mission analyzer combining all phases.

    Phases:
    1. Takeoff ground roll (3-DOF)
    2. Climb to cruise altitude
    3. Cruise for specified distance
    4. Descent to approach altitude
    5. Landing (approach, flare, ground roll)
    """

    def __init__(
        self,
        wing_project: WingProject,
        prop_config: PropellerConfig,
        motor_config: MotorConfig,
        battery_config: BatteryConfig,
        mass_kg: float,
        mu_roll: float = 0.03,
        mu_brake: float = 0.1,
        takeoff_requirement_m: float = 30.48,
        thrust_angle_deg: float = 2.0,  # Default 2 degrees up
    ):
        self.wing_project = wing_project
        self.mass_kg = mass_kg
        self.thrust_angle_deg = thrust_angle_deg

        # Create shared models
        self.aero_model = RigidBodyAeroModel(wing_project, ground_effect=True)
        self.propulsion = TakeoffPropulsionModel(
            prop_config, motor_config, battery_config
        )

        # Create phase calculators
        self.takeoff_calc = TakeoffCalculator(
            wing_project,
            prop_config,
            motor_config,
            battery_config,
            mass_kg,
            mu_roll,
            ground_effect=True,
            requirement_m=takeoff_requirement_m,
            Cmq=None,  # Rely on AeroBuildup's CMq for damping
            aero_model=self.aero_model,  # Reuse existing aero model
            thrust_angle_deg=thrust_angle_deg,
        )
        # Use simpler aero interfaces for other phases for now
        self.climb_calc = ClimbCalculator(self.aero_model, self.propulsion, mass_kg, thrust_angle_deg=thrust_angle_deg)
        self.cruise_calc = CruiseCalculator(self.aero_model, self.propulsion, mass_kg, thrust_angle_deg=thrust_angle_deg)
        self.descent_calc = DescentCalculator(self.aero_model, self.propulsion, mass_kg, thrust_angle_deg=thrust_angle_deg)
        self.landing_calc = LandingCalculator(
            self.aero_model, self.propulsion, mass_kg, mu_brake, thrust_angle_deg=thrust_angle_deg
        )

    def run_mission(
        self,
        cruise_altitude_m: float = 100.0,
        cruise_distance_m: float = 300.0,
        climb_distance_limit_m: float = None,
        CL_max: float = 1.2,
        CL_cruise: float = 0.5,
    ) -> MissionResult:
        """
        Run complete mission analysis.

        Args:
            cruise_altitude_m: Target cruise altitude [m]
            cruise_distance_m: Cruise distance [m]
            climb_distance_limit_m: Max horizontal distance for climb [m] (optional constraint)
            CL_max: Maximum CL for takeoff/landing
            CL_cruise: Cruise CL

        Returns:
            MissionResult with all phases
        """
        print("Phase 1: Takeoff...")
        takeoff = self.takeoff_calc.run_full_analysis(
            CL_max=CL_max, autotune=True, target_pitch_deg=12.0, alpha_ground_deg=5.0
        )

        print("Phase 2: Climb...")
        # Use full throttle and higher CL for steeper climb if constraint is tight
        climb_throttle = 1.0  # Full throttle for max climb performance
        climb_CL = 0.8  # Higher CL = slower speed = steeper climb angle

        climb = self.climb_calc.calculate_climb(
            V_initial=takeoff.liftoff_velocity_mps,
            target_altitude_m=cruise_altitude_m,
            throttle=climb_throttle,
            CL_climb=climb_CL,
            max_horizontal_distance_m=climb_distance_limit_m,
        )

        # Use actual achieved altitude for subsequent phases
        actual_cruise_altitude = climb.altitude_gain_m

        # Report climb status
        if climb.distance_constrained:
            required_angle = np.degrees(
                np.arctan2(cruise_altitude_m, climb_distance_limit_m)
            )
            sin_required = np.sin(np.radians(required_angle))
            required_excess_thrust = self.mass_kg * 9.81 * sin_required

            print(f"  *** CLIMB DISTANCE CONSTRAINT APPLIED ***")
            print(f"  Target altitude: {cruise_altitude_m:.1f} m")
            print(
                f"  Achieved altitude: {actual_cruise_altitude:.1f} m ({actual_cruise_altitude / cruise_altitude_m * 100:.0f}%)"
            )
            print(
                f"  Climb distance: {climb.horizontal_distance_m:.1f} m (limit: {climb_distance_limit_m:.1f} m)"
            )
            print(f"  Required climb angle for target: {required_angle:.1f} deg")
            print(f"  Achieved climb angle: {climb.climb_angle_deg:.1f} deg")
            print(
                f"  Thrust-to-weight ratio: {takeoff.static_thrust_N / (self.mass_kg * 9.81):.2f}"
            )
            print(f"  ")
            print(
                f"  CONTINUING MISSION AT {actual_cruise_altitude:.1f}m CRUISE ALTITUDE"
            )
        else:
            print(
                f"  OK: Reached {actual_cruise_altitude:.1f}m in {climb.horizontal_distance_m:.1f}m horizontal"
            )

        print("Phase 3: Cruise...")
        cruise = self.cruise_calc.calculate_cruise(
            distance_m=cruise_distance_m,
            altitude_m=actual_cruise_altitude,  # Use achieved altitude
            V_initial=climb.velocity_history[-1],
            CL_cruise=CL_cruise,
        )

        print("Phase 4: Descent...")
        approach_altitude = 10.0  # Descend to 10m before landing approach
        descent = self.descent_calc.calculate_descent(
            start_altitude_m=actual_cruise_altitude,  # Use achieved altitude
            end_altitude_m=approach_altitude,
            V_initial=cruise.velocity_history[-1],
            throttle=0.0,
            CL_descent=0.7,
        )

        print("Phase 5: Landing...")
        landing = self.landing_calc.calculate_landing(
            approach_altitude_m=approach_altitude,
            V_initial=descent.velocity_history[-1],
            flare_altitude_m=2.0,
            glide_slope_deg=5.0,
            throttle=0.0,
            CL_max=CL_max,
        )

        # Calculate totals
        total_distance = (
            takeoff.ground_roll_m
            + climb.horizontal_distance_m
            + cruise.distance_m
            + descent.horizontal_distance_m
            + landing.total_landing_distance_m
        )

        total_time = (
            takeoff.takeoff_time_s
            + climb.climb_time_s
            + cruise.cruise_time_s
            + descent.descent_time_s
            + landing.total_time_s
        )

        # Energy: estimate takeoff energy from power * time
        takeoff_energy = takeoff.max_power_W * takeoff.takeoff_time_s / 3600
        total_energy = (
            takeoff_energy
            + climb.energy_consumed_Wh
            + cruise.energy_consumed_Wh
            + descent.energy_consumed_Wh
            + landing.energy_consumed_Wh
        )

        # Representative thrust per phase for reporting
        def get_avg_thrust(result, phase_name):
            if hasattr(result, "thrust_history") and len(result.thrust_history) > 0:
                return float(np.mean(result.thrust_history))

            # Fallback for phases that don't track thrust history (climb/cruise)
            # Use actual propulsion model at average velocity
            V_avg = getattr(result, "average_velocity_mps", 0)
            throttle = getattr(
                result, "average_throttle", getattr(result, "throttle_setting", 0)
            )
            if V_avg > 0 and throttle > 0:
                res = self.propulsion.solve_operating_point(V_avg, throttle=throttle)
                return res["thrust_total"]
            return 0.0

        T_takeoff = get_avg_thrust(takeoff, "takeoff")
        T_climb = get_avg_thrust(climb, "climb")
        T_cruise = get_avg_thrust(cruise, "cruise")
        T_descent = 0.0
        T_landing = 0.0

        thrust_times = [
            takeoff.takeoff_time_s,
            climb.climb_time_s,
            cruise.cruise_time_s,
            descent.descent_time_s,
            landing.total_time_s,
        ]
        thrust_levels = [T_takeoff, T_climb, T_cruise, T_descent, T_landing]
        avg_thrust = (
            sum(t * l for t, l in zip(thrust_times, thrust_levels)) / total_time
            if total_time > 0
            else 0.0
        )
        # For Max Thrust, check takeoff history and static
        max_thrust = takeoff.static_thrust_N
        if len(takeoff.thrust_history) > 0:
            max_thrust = max(max_thrust, np.max(takeoff.thrust_history))

        return MissionResult(
            takeoff=takeoff,
            climb=climb,
            cruise=cruise,
            descent=descent,
            landing=landing,
            total_distance_m=total_distance,
            total_time_s=total_time,
            total_energy_Wh=total_energy,
            max_altitude_m=actual_cruise_altitude,  # Use achieved altitude
            avg_thrust_N=avg_thrust,
            max_thrust_N=max_thrust,
        )


# =============================================================================
# Main Script
# =============================================================================


def main():
    """Run complete mission analysis on IntendedValidation.json project."""

    # Load project
    project_path = PROJECT_ROOT / "IntendedValidation2.json"
    with open(project_path, "r") as f:
        data = json.load(f)

    from core.state import Project

    project = Project.from_dict(data)
    wing_project = project.wing

    # Extract parameters
    mass_kg = wing_project.twist_trim.gross_takeoff_weight_kg
    CL_max = wing_project.twist_trim.estimated_cl_max

    # Get aero config if available
    if "mission" in data and "aero_config" in data["mission"]:
        aero_config = data["mission"]["aero_config"]
        mu_roll = aero_config.get("mu_roll", 0.03)
        CL_max = aero_config.get("cl_max", CL_max)
        CL_cruise = aero_config.get("cl_cruise", 0.5)
        thrust_angle_deg = aero_config.get("thrust_angle_deg", 3.0)
    else:
        mu_roll = 0.03
        CL_cruise = 0.5
        thrust_angle_deg = 3.0

    # Mission parameters
    CRUISE_ALTITUDE_M = 100.0
    CRUISE_DISTANCE_M = 200.0  # Reduced from 300m
    CLIMB_DISTANCE_LIMIT_M = 121.9  # 400 ft constraint

    print("=" * 70)
    print("COMPLETE MISSION ANALYSIS - IntendedValidation.json")
    print("=" * 70)
    print()
    print("MISSION PROFILE:")
    print(f"  1. Takeoff ground roll")
    print(
        f"  2. Climb to {CRUISE_ALTITUDE_M:.0f} m ({CRUISE_ALTITUDE_M * 3.281:.0f} ft)"
    )
    print(f"  3. Cruise for {CRUISE_DISTANCE_M:.0f} m")
    print(f"  4. Descend to approach altitude")
    print(f"  5. Land with flare")
    print()
    print("AIRCRAFT CONFIGURATION:")
    print(f"  Wing Area: {wing_project.planform.wing_area_m2:.3f} m^2")
    print(f"  Aspect Ratio: {wing_project.planform.aspect_ratio:.2f}")
    print(f"  Gross Weight: {mass_kg:.1f} kg ({mass_kg * 9.81:.1f} N)")
    print(f"  CL_max: {CL_max:.2f}")
    print(f"  CL_cruise: {CL_cruise:.2f}")
    print(f"  Rolling Friction: {mu_roll:.3f}")
    print()

    # Define propulsion configuration
    prop_config = PropellerConfig(
        diameter_in=12.0,
        pitch_in=6.0,
        family="Standard",
    )

    motor_config = MotorConfig(
        kv=1400,
        R_internal=0.026,
        I_no_load=2.61,
        n_motors=2,
    )

    battery_config = BatteryConfig(
        n_series=4,
        capacity_mAh=2200,
        c_rating=80.0,
    )

    print("PROPULSION CONFIGURATION:")
    print(
        f"  Propeller: {prop_config.diameter_in}x{prop_config.pitch_in} ({prop_config.family})"
    )
    print(
        f"  Motor: {motor_config.kv} Kv, {motor_config.R_internal * 1000:.0f} mOhm, {motor_config.n_motors} motors"
    )
    print(
        f"  Battery: {battery_config.n_series}S {battery_config.capacity_mAh}mAh ({battery_config.capacity_mAh * battery_config.V_nominal / 1000:.1f} Wh), {battery_config.c_rating}C ({battery_config.I_max:.1f}A limit)"
    )
    print()

    # Create mission analyzer
    analyzer = MissionAnalyzer(
        wing_project=wing_project,
        prop_config=prop_config,
        motor_config=MotorConfig(
            kv=data["mission"]["motor"]["KV_rpm_per_V"],
            R_internal=data["mission"]["motor"]["Ri_mOhm"] / 1000.0,
            I_no_load=data["mission"]["motor"]["Io_A"],
            n_motors=2,  # Assuming 2 as per script setup
        )
        if "mission" in data and "motor" in data["mission"]
        else motor_config,
        battery_config=battery_config,
        mass_kg=mass_kg,
        mu_roll=mu_roll,
        mu_brake=0.1,
        takeoff_requirement_m=30.48,  # Pass desired requirement
        thrust_angle_deg=thrust_angle_deg,
    )

    # Run complete mission
    print("Running mission analysis...")
    print()

    mission = analyzer.run_mission(
        cruise_altitude_m=CRUISE_ALTITUDE_M,
        cruise_distance_m=CRUISE_DISTANCE_M,
        climb_distance_limit_m=CLIMB_DISTANCE_LIMIT_M,
        CL_max=CL_max,
        CL_cruise=CL_cruise,
    )

    print()

    # Print individual phase results
    print(mission.takeoff.summary())
    print()
    print(mission.climb.summary())
    print()
    print(mission.cruise.summary())
    print()
    print(mission.descent.summary())
    print()
    print(mission.landing.summary())
    print()

    # Print mission summary
    print(mission.summary())
    print()

    # Battery check
    battery_capacity_Wh = battery_config.capacity_mAh * battery_config.V_nominal / 1000
    battery_usage_percent = (mission.total_energy_Wh / battery_capacity_Wh) * 100

    print("BATTERY ANALYSIS:")
    print(f"  Battery Capacity: {battery_capacity_Wh:.1f} Wh")
    print(
        f"  Energy Used: {mission.total_energy_Wh:.1f} Wh ({battery_usage_percent:.1f}%)"
    )
    print(
        f"  Remaining: {battery_capacity_Wh - mission.total_energy_Wh:.1f} Wh ({100 - battery_usage_percent:.1f}%)"
    )

    if battery_usage_percent > 80:
        print("  STATUS: WARNING - Battery usage exceeds 80%!")
    elif battery_usage_percent > 50:
        print("  STATUS: CAUTION - Battery usage over 50%")
    else:
        print("  STATUS: OK - Adequate battery margin")

    print()
    print("=" * 70)
    print("Mission analysis complete. All values are GUI-ready.")
    print("=" * 70)

    # Generate plots
    print()
    print("Generating mission plots...")

    plot_mission(
        mission=mission,
        battery_capacity_Wh=battery_capacity_Wh,
        propulsion=analyzer.propulsion,
        save_path=PROJECT_ROOT / "outputs" / "mission_profile.png",
        show=True,
    )

    return mission


if __name__ == "__main__":
    main()
