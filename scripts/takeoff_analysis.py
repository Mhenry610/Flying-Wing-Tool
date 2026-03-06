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
    """Battery configuration."""

    n_series: int  # e.g., 4 for 4S
    capacity_mAh: int

    @property
    def V_nominal(self) -> float:
        return self.n_series * 3.7

    @property
    def V_full(self) -> float:
        return self.n_series * 4.2

    @property
    def I_max(self) -> float:
        """Max discharge current (assume 40C for high-C pack)."""
        return self.capacity_mAh / 1000.0 * 80.0


@dataclass
class TakeoffResult:
    """Results from takeoff analysis."""

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

    # Time history for plotting
    time_history: np.ndarray
    velocity_history: np.ndarray
    distance_history: np.ndarray
    thrust_history: np.ndarray
    acceleration_history: np.ndarray

    def summary(self) -> str:
        """Return formatted summary string."""
        status = "PASS" if self.passes_requirement else "FAIL"
        lines = [
            "=" * 70,
            "TAKEOFF ANALYSIS RESULTS",
            "=" * 70,
            "",
            f"Ground Roll Distance: {self.ground_roll_m:.1f} m ({self.ground_roll_ft:.0f} ft)",
            f"Takeoff Time: {self.takeoff_time_s:.2f} s",
            f"Liftoff Velocity: {self.liftoff_velocity_mps:.1f} m/s",
            "",
            "PROPULSION:",
            f"  Static Thrust (2 motors): {self.static_thrust_N:.1f} N",
            f"  Thrust at Liftoff: {self.thrust_at_liftoff_N:.1f} N",
            f"  Max Battery Current: {self.max_battery_current_A:.0f} A",
            f"  Max Power: {self.max_power_W:.0f} W",
            "",
            f"REQUIREMENT: {self.requirement_m:.1f} m ({self.requirement_m * 3.281:.0f} ft)",
            f"STATUS: {status}",
            f"Margin: {self.margin_m:.1f} m ({self.margin_percent:.0f}%)",
            "=" * 70,
        ]
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
            f"  1. Takeoff Ground Roll:  {self.takeoff.ground_roll_m:>7.1f} m  |  {self.takeoff.takeoff_time_s:>5.1f} s",
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
# Aerodynamic Model
# =============================================================================


class GroundRollAeroModel:
    """
    Aerodynamic model for ground roll phase.

    Uses AeroBuildup to compute CL and CD at various angles of attack
    during the ground roll. Accounts for ground effect.
    """

    def __init__(
        self,
        wing_project: WingProject,
        ground_effect: bool = True,
    ):
        self.wing_project = wing_project
        self.ground_effect = ground_effect
        self.aero_service = AeroSandboxService(wing_project)

        # Cache key parameters
        self.S = wing_project.planform.wing_area_m2
        self.span = wing_project.planform.actual_span()

        # Build wing once
        self._wing = self.aero_service.build_wing()
        self._airplane = self._build_airplane()

        # Cache polars for speed
        self._polar_cache: Dict[float, Tuple[float, float]] = {}

    def _build_airplane(self) -> asb.Airplane:
        """Build AeroSandbox airplane."""
        x_np = self._wing.aerodynamic_center()[0]
        mac = self._wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac

        return asb.Airplane(
            name=self.wing_project.name,
            wings=[self._wing],
            xyz_ref=[x_cg, 0.0, 0.0],
        )

    def get_cl_cd(
        self, alpha_deg: float, V: float, altitude: float = 0.0
    ) -> Tuple[float, float]:
        """
        Get CL and CD at given alpha and velocity.

        Args:
            alpha_deg: Angle of attack [degrees]
            V: Airspeed [m/s]
            altitude: Altitude AGL [m] (for ground effect)

        Returns:
            (CL, CD)
        """
        # Check cache
        cache_key = (round(alpha_deg, 1), round(V, 1))
        if cache_key in self._polar_cache:
            return self._polar_cache[cache_key]

        # Run AeroBuildup
        atmo = asb.Atmosphere(altitude=0)  # Sea level
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=max(V, 1.0),  # Avoid div by zero
            alpha=alpha_deg,
        )

        aero = asb.AeroBuildup(
            airplane=self._airplane,
            op_point=op_point,
        )
        result = aero.run()

        CL = float(result.get("CL", result.get("Cl", 0.0)))
        CD = float(result.get("CD", result.get("Cd", 0.0)))

        # Apply ground effect if enabled and close to ground
        if self.ground_effect and altitude < self.span:
            h_b = max(altitude, 0.1) / self.span
            phi_ge = (16 * h_b) ** 2 / (1 + (16 * h_b) ** 2)

            # Lift augmentation (up to 10% at h=0)
            CL = CL * (1 + 0.1 * (1 - phi_ge))

            # Induced drag reduction
            CD = CD * (phi_ge + (1 - phi_ge) * 0.5)

        self._polar_cache[cache_key] = (CL, CD)
        return CL, CD

    def get_forces(
        self,
        V: float,
        alpha_deg: float,
        rho: float,
        altitude: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Get lift and drag forces.

        Args:
            V: Airspeed [m/s]
            alpha_deg: Angle of attack [degrees]
            rho: Air density [kg/m^3]
            altitude: Altitude AGL [m]

        Returns:
            (Lift [N], Drag [N])
        """
        CL, CD = self.get_cl_cd(alpha_deg, V, altitude)
        q = 0.5 * rho * V**2
        L = q * self.S * CL
        D = q * self.S * CD
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
        n_iterations: int = 30,
    ) -> Dict:
        """
        Solve motor-propeller equilibrium at given airspeed.

        Uses fixed-point iteration to find where motor torque = prop torque.

        Args:
            V_air: Airspeed [m/s]
            throttle: Throttle setting [0-1]
            rho: Air density [kg/m^3]
            n_iterations: Iteration count

        Returns:
            Dict with thrust, power, current, rpm, etc.
        """
        cache_key = (round(V_air, 2), round(throttle, 2))
        if cache_key in self._thrust_cache:
            return self._thrust_cache[cache_key]

        # Battery voltage (use full charge for takeoff)
        V_batt = self.battery.V_full * throttle

        Kv = self.motor_config.kv
        Rm = self.motor_config.R_internal
        I0 = self.motor_config.I_no_load
        D_m = self.prop.diameter_m
        P_m = self.prop.pitch_m
        n_motors = self.motor_config.n_motors

        # Initial guess for motor current
        I_motor = 20.0

        # Fixed-point iteration
        for _ in range(n_iterations):
            # Motor voltage after resistance drop
            V_eff = V_batt - I_motor * Rm
            if V_eff <= 0:
                break

            # Motor RPM and angular velocity
            rpm = Kv * V_eff
            omega = rpm * 2 * np.pi / 60

            # Get propeller performance
            T, P_shaft = self.prop_model.get_performance(
                V=V_air, omega=omega, D=D_m, P=P_m, rho=rho
            )
            T = float(T)
            P_shaft = float(P_shaft)

            # Update current estimate (motor power balance)
            if V_eff > 0:
                I_new = P_shaft / V_eff + I0
            else:
                I_new = I0

            # Relaxed update for stability
            I_motor = 0.6 * I_motor + 0.4 * I_new

        # Final values
        V_eff = max(0.1, V_batt - I_motor * Rm)
        rpm = Kv * V_eff
        omega = rpm * 2 * np.pi / 60
        T, P_shaft = self.prop_model.get_performance(
            V=V_air, omega=omega, D=D_m, P=P_m, rho=rho
        )
        T = float(T)
        P_shaft = float(P_shaft)

        result = {
            "thrust_per_motor": T,
            "thrust_total": n_motors * T,
            "power_shaft_per_motor": P_shaft,
            "power_shaft_total": n_motors * P_shaft,
            "current_per_motor": I_motor,
            "current_battery": n_motors * I_motor,
            "rpm": rpm,
            "omega": omega,
            "voltage_effective": V_eff,
            "voltage_battery": V_batt,
        }

        self._thrust_cache[cache_key] = result
        return result

    def get_thrust(
        self, V_air: float, throttle: float = 1.0, rho: float = 1.225
    ) -> float:
        """Get total thrust at given airspeed."""
        result = self.solve_operating_point(V_air, throttle, rho)
        return result["thrust_total"]

    def get_static_thrust(self, rho: float = 1.225) -> float:
        """Get static thrust (V=0)."""
        return self.get_thrust(0.0, throttle=1.0, rho=rho)


# =============================================================================
# Takeoff Calculator
# =============================================================================


class TakeoffCalculator:
    """
    Complete takeoff ground roll calculator.

    Integrates:
    - Aerodynamic model (AeroBuildup + ground effect)
    - Propulsion model (propeller meta-model + motor)
    - Numerical integration of equations of motion
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
    ):
        self.wing_project = wing_project
        self.mass_kg = mass_kg
        self.mu_roll = mu_roll
        self.g = 9.81
        self.weight_N = mass_kg * self.g

        # Create models
        self.aero_model = GroundRollAeroModel(wing_project, ground_effect)
        self.propulsion = TakeoffPropulsionModel(
            prop_config, motor_config, battery_config
        )

        # Get wing area
        self.S = wing_project.planform.wing_area_m2

    def calculate_liftoff_velocity(
        self, CL_max: float = 1.2, margin: float = 1.2
    ) -> float:
        """
        Calculate liftoff velocity.

        V_liftoff = margin * V_stall
        V_stall = sqrt(2*W / (rho * S * CL_max))

        Args:
            CL_max: Maximum lift coefficient
            margin: Safety margin (typically 1.2)

        Returns:
            Liftoff velocity [m/s]
        """
        rho = 1.225  # Sea level
        V_stall = np.sqrt(2 * self.weight_N / (rho * self.S * CL_max))
        return margin * V_stall

    def calculate_ground_roll(
        self,
        V_liftoff: float,
        alpha_ground_deg: float = 2.0,
        rho: float = 1.225,
        dt: float = 0.01,
        max_time: float = 60.0,
    ) -> TakeoffResult:
        """
        Calculate ground roll distance using numerical integration.

        Integrates: T - D - mu*(W - L) = m*a

        Args:
            V_liftoff: Liftoff velocity [m/s]
            alpha_ground_deg: Ground attitude angle [degrees]
            rho: Air density [kg/m^3]
            dt: Time step [s]
            max_time: Maximum simulation time [s]

        Returns:
            TakeoffResult with all analysis outputs
        """
        # Initialize state
        V = 0.1  # Small initial velocity to avoid singularities
        x = 0.0
        t = 0.0

        # History arrays
        times = [t]
        velocities = [V]
        distances = [x]
        thrusts = []
        accelerations = []

        # Track max values
        max_current = 0.0
        max_power = 0.0

        # Get static thrust
        static_result = self.propulsion.solve_operating_point(
            0.0, throttle=1.0, rho=rho
        )
        static_thrust = static_result["thrust_total"]

        # Integration loop
        while V < V_liftoff and t < max_time:
            # Get thrust at current velocity
            prop_result = self.propulsion.solve_operating_point(
                V, throttle=1.0, rho=rho
            )
            T = prop_result["thrust_total"]

            # Track max current/power
            max_current = max(max_current, prop_result["current_battery"])
            max_power = max(max_power, prop_result["power_shaft_total"])

            # Get aerodynamic forces
            L, D = self.aero_model.get_forces(V, alpha_ground_deg, rho, altitude=0.0)

            # Ground reaction and rolling friction
            N = max(0, self.weight_N - L)  # Normal force
            F_roll = self.mu_roll * N

            # Net force and acceleration
            F_net = T - D - F_roll
            a = F_net / self.mass_kg

            # Store history
            thrusts.append(T)
            accelerations.append(a)

            # Euler integration
            V_new = V + a * dt
            x_new = x + V * dt + 0.5 * a * dt**2
            t += dt

            V = max(0.1, V_new)  # Prevent negative velocity
            x = x_new

            times.append(t)
            velocities.append(V)
            distances.append(x)

        # Final thrust at liftoff
        liftoff_result = self.propulsion.solve_operating_point(
            V_liftoff, throttle=1.0, rho=rho
        )
        thrust_at_liftoff = liftoff_result["thrust_total"]

        # Convert to numpy arrays
        times = np.array(times)
        velocities = np.array(velocities)
        distances = np.array(distances)
        thrusts = np.array(thrusts)
        accelerations = np.array(accelerations)

        # Calculate margins (100 ft = 30.48 m requirement)
        requirement_m = 30.48
        margin_m = requirement_m - x
        margin_percent = (margin_m / requirement_m) * 100

        return TakeoffResult(
            ground_roll_m=x,
            ground_roll_ft=x * 3.281,
            takeoff_time_s=t,
            liftoff_velocity_mps=V_liftoff,
            static_thrust_N=static_thrust,
            thrust_at_liftoff_N=thrust_at_liftoff,
            max_battery_current_A=max_current,
            max_power_W=max_power,
            passes_requirement=(x <= requirement_m),
            requirement_m=requirement_m,
            margin_m=margin_m,
            margin_percent=margin_percent,
            time_history=times[:-1],  # Align with force arrays
            velocity_history=velocities[:-1],
            distance_history=distances[:-1],
            thrust_history=thrusts,
            acceleration_history=accelerations,
        )

    def run_full_analysis(
        self,
        CL_max: float = 1.2,
        alpha_ground_deg: float = 2.0,
    ) -> TakeoffResult:
        """
        Run complete takeoff analysis.

        Args:
            CL_max: Maximum lift coefficient
            alpha_ground_deg: Ground attitude [degrees]

        Returns:
            TakeoffResult
        """
        V_liftoff = self.calculate_liftoff_velocity(CL_max)
        return self.calculate_ground_roll(V_liftoff, alpha_ground_deg)

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

            result = self.calculate_ground_roll(V_liftoff)

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
    Climb phase calculator.

    Integrates the equations of motion for climbing flight:
    - Tangential: m * dV/dt = T - D - W*sin(gamma)
    - Climb angle from quasi-steady normal force balance

    Velocity evolves naturally from initial condition via physics.
    """

    def __init__(
        self,
        aero_model: GroundRollAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S

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
        Calculate climb phase from liftoff to target altitude.

        Integrates the tangential equation of motion:
            m * dV/dt = T - D - W*sin(gamma)

        The climb angle gamma is determined by excess thrust:
            sin(gamma) = (T - D) / W  (for quasi-steady climb)

        Velocity evolves naturally from V_initial - no discontinuities.

        Args:
            V_initial: Initial velocity (from liftoff) [m/s]
            target_altitude_m: Target altitude [m]
            throttle: Throttle setting [0-1]
            CL_climb: Target CL for climb (determines alpha)
            dt: Time step [s]
            max_time: Maximum simulation time [s]
            max_horizontal_distance_m: Distance constraint [m] (optional)

        Returns:
            ClimbResult
        """
        rho = 1.225  # Sea level (approximate)

        # Initialize state variables
        V = V_initial  # Start exactly where takeoff ended
        h = 0.0  # Altitude above liftoff point
        x = 0.0  # Horizontal distance
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
            # CL required for L = W*cos(gamma) at current V
            # For small gamma, L ≈ W, so CL ≈ W / (0.5*rho*V^2*S)
            q = 0.5 * rho * V**2
            CL_required = self.weight_N / (q * self.S) if q > 1 else CL_climb
            CL_actual = min(CL_required, CL_climb * 1.2)  # Don't exceed CL limit

            # Alpha from CL (approximate: CL_alpha ~ 5/rad for typical wing)
            alpha_deg = CL_actual / (5 * np.pi / 180)
            alpha_deg = min(alpha_deg, 12.0)  # Limit alpha to avoid stall

            # Get lift and drag
            L, D = self.aero_model.get_forces(V, alpha_deg, rho, altitude=h)

            # --- Propulsion ---
            prop_result = self.propulsion.solve_operating_point(
                V, throttle=throttle, rho=rho
            )
            T = prop_result["thrust_total"]
            P_shaft = prop_result["power_shaft_total"]

            # --- Climb angle from force balance ---
            # For quasi-steady climb: sin(gamma) = (T - D) / W
            excess_thrust = T - D
            sin_gamma = excess_thrust / self.weight_N
            sin_gamma = min(0.85, max(-0.1, sin_gamma))  # Clamp to physical limits
            gamma = np.arcsin(sin_gamma)
            cos_gamma = np.cos(gamma)

            # --- Tangential equation of motion ---
            # m * dV/dt = T - D - W*sin(gamma)
            # Since sin(gamma) = (T-D)/W, this simplifies but we keep general form
            # for numerical accuracy when not in perfect equilibrium
            F_tangential = T - D - self.weight_N * sin_gamma
            dV_dt = F_tangential / self.mass_kg

            # --- Integrate velocity using modified Euler ---
            V_new = V + dV_dt * dt
            V_new = max(V_min, V_new)  # Stall protection

            # --- Integrate position ---
            # Use average velocity for more accurate integration
            V_avg = 0.5 * (V + V_new)
            V_vertical = V_avg * np.sin(gamma)
            V_horizontal = V_avg * cos_gamma

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
    Level cruise phase calculator.

    Integrates velocity to reach cruise equilibrium with constant throttle
    (chosen to satisfy T ≈ D at the target cruise condition). Velocity evolves
    from the incoming condition, eliminating discontinuities.
    """

    def __init__(
        self,
        aero_model: GroundRollAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S

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

        Args:
            distance_m: Cruise distance [m]
            altitude_m: Cruise altitude [m]
            V_initial: Incoming velocity from previous phase [m/s]
            CL_cruise: Cruise CL target
            dt: Integration time step [s]
            max_time: Safety cap on simulation time [s]

        Returns:
            CruiseResult
        """
        # Air density at altitude (simplified: exponential model for low altitude)
        rho = 1.225 * np.exp(-altitude_m / 8500)

        # Target cruise velocity from L = W
        V_target = np.sqrt(2 * self.weight_N / (rho * self.S * CL_cruise))

        # Determine throttle such that T ~= D at V_target
        alpha_target_deg = CL_cruise / (5 * np.pi / 180)
        _, D_target = self.aero_model.get_forces(
            V_target, alpha_target_deg, rho, altitude=altitude_m
        )
        throttle_low, throttle_high = 0.05, 1.0
        for _ in range(25):
            throttle = 0.5 * (throttle_low + throttle_high)
            T_test = self.propulsion.solve_operating_point(
                V_target, throttle=throttle, rho=rho
            )["thrust_total"]
            if T_test < D_target:
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

            # Tangential EOM for level flight (gamma ≈ 0): m dV/dt = T - D
            a = (T - D) / self.mass_kg
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

    Integrates velocity and flight path angle during descent. Velocity evolves
    from the incoming condition; no artificial smoothing.
    """

    def __init__(
        self,
        aero_model: GroundRollAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S

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
        Calculate descent phase.

        Uses a glide-based model: set descent CL, compute L/D, pick flight-path
        angle from L/D (gamma ≈ atan(1/(L/D))), and descend at (roughly) constant
        airspeed set by CL_descent. Throttle defaults near idle to avoid overly
        shallow powered descents.
        """
        rho = 1.225

        # Target descent speed from CL_descent, start from incoming if higher
        V_descent = np.sqrt(2 * self.weight_N / (rho * self.S * CL_descent))
        V = max(V_descent, V_initial)
        h = start_altitude_m
        x = 0.0
        t = 0.0

        times = [t]
        altitudes = [h]
        distances = [x]
        velocities = [V]
        total_energy_J = 0.0

        while h > end_altitude_m and t < max_time:
            q = 0.5 * rho * V**2
            alpha_deg = CL_descent / (5 * np.pi / 180)
            alpha_deg = min(alpha_deg, 12.0)
            L, D = self.aero_model.get_forces(V, alpha_deg, rho, altitude=h)
            L_over_D = L / D if D > 1e-6 else 10.0
            gamma = np.arctan(1.0 / L_over_D)

            # Glide: neglect thrust/power (idle)
            P_shaft = 0.0

            V_vertical = -V * np.sin(gamma)
            V_horizontal = V * np.cos(gamma)
            h = h + V_vertical * dt
            x = x + V_horizontal * dt
            t += dt

            # No propulsion energy in idle glide
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

    Three sub-phases:
    1. Final approach (integrated along glide slope)
    2. Flare (deceleration with increasing lift/drag)
    3. Ground roll (braking + aero drag integration)
    """

    def __init__(
        self,
        aero_model: GroundRollAeroModel,
        propulsion: TakeoffPropulsionModel,
        mass_kg: float,
        mu_brake: float = 0.1,
    ):
        self.aero_model = aero_model
        self.propulsion = propulsion
        self.mass_kg = mass_kg
        self.g = 9.81
        self.weight_N = mass_kg * self.g
        self.S = aero_model.S
        self.mu_brake = mu_brake  # Braking friction coefficient

    def calculate_landing(
        self,
        approach_altitude_m: float,
        V_initial: float,
        flare_altitude_m: float = 2.0,
        glide_slope_deg: float = 5.0,
        throttle: float = 0.05,
        CL_max: float = 1.6,
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
    1. Takeoff ground roll
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
    ):
        self.wing_project = wing_project
        self.mass_kg = mass_kg

        # Create shared models
        self.aero_model = GroundRollAeroModel(wing_project, ground_effect=True)
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
        )
        self.climb_calc = ClimbCalculator(self.aero_model, self.propulsion, mass_kg)
        self.cruise_calc = CruiseCalculator(self.aero_model, self.propulsion, mass_kg)
        self.descent_calc = DescentCalculator(self.aero_model, self.propulsion, mass_kg)
        self.landing_calc = LandingCalculator(
            self.aero_model, self.propulsion, mass_kg, mu_brake
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
        takeoff = self.takeoff_calc.run_full_analysis(CL_max=CL_max)

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

        # Representative thrust per phase for reporting (power/velocity based to avoid prop model overflow)
        def thrust_from_power(power_W: float, V: float) -> float:
            return power_W / max(V, 0.1)

        T_takeoff = (
            float(np.max(takeoff.thrust_history))
            if len(takeoff.thrust_history) > 0
            else takeoff.thrust_at_liftoff_N
        )
        T_climb = (
            thrust_from_power(
                climb.energy_consumed_Wh * 3600 / max(climb.climb_time_s, 1e-6),
                climb.average_velocity_mps,
            )
            if climb.climb_time_s > 0
            else 0.0
        )
        T_cruise = (
            thrust_from_power(
                cruise.energy_consumed_Wh * 3600 / max(cruise.cruise_time_s, 1e-6),
                cruise.cruise_velocity_mps,
            )
            if cruise.cruise_time_s > 0
            else 0.0
        )
        T_descent = (
            thrust_from_power(
                descent.energy_consumed_Wh * 3600 / max(descent.descent_time_s, 1e-6),
                descent.average_velocity_mps,
            )
            if descent.descent_time_s > 0
            else 0.0
        )
        T_landing = (
            thrust_from_power(
                landing.energy_consumed_Wh * 3600 / max(landing.total_time_s, 1e-6),
                max(landing.touchdown_velocity_mps, 0.1),
            )
            if landing.total_time_s > 0
            else 0.0
        )
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
        max_thrust = max(thrust_levels + [takeoff.static_thrust_N])

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
    project_path = PROJECT_ROOT / "IntendedValidation.json"
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
    else:
        mu_roll = 0.03
        CL_cruise = 0.5

    # Mission parameters
    CRUISE_ALTITUDE_M = 100.0
    CRUISE_DISTANCE_M = 2000.0  # Reduced from 300m
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
        pitch_in=4.0,
        family="Standard",
    )

    motor_config = MotorConfig(
        kv=1100,
        R_internal=0.090,
        I_no_load=1.36,
        n_motors=2,
    )

    battery_config = BatteryConfig(
        n_series=4,
        capacity_mAh=2200,
    )

    print("PROPULSION CONFIGURATION:")
    print(
        f"  Propeller: {prop_config.diameter_in}x{prop_config.pitch_in} ({prop_config.family})"
    )
    print(
        f"  Motor: {motor_config.kv} Kv, {motor_config.R_internal * 1000:.0f} mOhm, {motor_config.n_motors} motors"
    )
    print(
        f"  Battery: {battery_config.n_series}S {battery_config.capacity_mAh}mAh ({battery_config.capacity_mAh * battery_config.V_nominal / 1000:.1f} Wh)"
    )
    print()

    # Create mission analyzer
    analyzer = MissionAnalyzer(
        wing_project=wing_project,
        prop_config=prop_config,
        motor_config=motor_config,
        battery_config=battery_config,
        mass_kg=mass_kg,
        mu_roll=mu_roll,
        mu_brake=0.1,
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
