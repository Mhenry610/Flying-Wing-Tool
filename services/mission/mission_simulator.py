"""
Mission Simulator

Forward simulation of prescribed mission profiles using autopilot control
and integrated propulsion/dynamics models.

This module provides:
- SimulationConfig: Simulation parameters
- PhaseResult: Results for a single phase
- MissionResult: Complete mission results with summary
- MissionSimulator: Main simulator class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
import numpy as np
import logging

# Configure mission simulation logger
mission_logger = logging.getLogger("mission_sim")


def set_mission_verbosity(level: str = "INFO") -> None:
    """
    Set mission simulation logging verbosity.
    
    Args:
        level: Logging level - "DEBUG" for calculation details,
               "INFO" for phase summaries, "WARNING" to disable
    """
    mission_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not mission_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        mission_logger.addHandler(handler)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .mission_definition import MissionPhaseType, MissionPhase, MissionProfile, Waypoint
from .autopilot import SimpleAutopilot, AutopilotGains

if TYPE_CHECKING:
    from services.dynamics.aircraft_dynamics import FlyingWingDynamics6DOF
    from services.propulsion.propulsion_system import IntegratedPropulsionSystem
    from models.wing_project import WingProject


@dataclass
class SimulationConfig:
    """
    Simulation configuration.
    
    Attributes:
        dt: Fixed timestep for Euler integration [s]
        dt_output: Output recording interval [s]
        max_simulation_time: Maximum total simulation time [s]
        stop_on_ground_contact: Stop phase if altitude goes to zero
        verbose: Print progress messages
        verbose_calc_interval: How often to log calculation details [s]
    """
    dt: float = 0.05                      # Integration timestep [s]
    dt_output: float = 0.5                # Output recording rate [s]
    max_simulation_time: float = 7200.0   # 2 hours max
    stop_on_ground_contact: bool = True   # Stop on ground contact
    verbose: bool = False                 # Print progress
    verbose_calc_interval: float = 5.0    # Log calc details every N seconds
    dynamics_mode: Optional[str] = "3dof" # "simple", "3dof", or "full"
    integration_method: str = "rk4"       # "euler" or "rk4"
    mu_roll: float = 0.04                 # Rolling friction coefficient
    mu_brake: float = 0.40                # Braking friction coefficient
    alpha_ground_deg: float = 5.0         # Ground attitude [deg]
    V_rot_factor: float = 1.1             # V_rot = V_rot_factor * V_stall
    flare_height_m: float = 7.0           # Flare trigger height AGL [m]
    max_bank_deg: float = 45.0            # Bank limit [deg]
    max_load_factor: Optional[float] = None  # Optional n_max
    max_turn_rate_deg_s: Optional[float] = None  # Optional turn rate cap


@dataclass
class PhaseResult:
    """
    Results for a single mission phase.
    
    Contains time histories of states, controls, and auxiliary variables,
    plus summary metrics for the phase.
    """
    name: str
    phase_type: MissionPhaseType
    start_time: float
    end_time: float
    
    # Time history arrays
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    states: Dict[str, np.ndarray] = field(default_factory=dict)
    controls: Dict[str, np.ndarray] = field(default_factory=dict)
    auxiliary: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Summary metrics
    energy_consumed_Wh: float = 0.0
    distance_traveled_m: float = 0.0
    altitude_change_m: float = 0.0
    SOC_start: float = 1.0
    SOC_end: float = 1.0
    
    @property
    def duration(self) -> float:
        """Phase duration in seconds."""
        return self.end_time - self.start_time
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert to pandas DataFrame.
        
        Returns:
            DataFrame with time as index and all states/controls/aux as columns
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        
        data = {'time': self.time}
        
        # Add states
        for key, values in self.states.items():
            data[f'state_{key}'] = values
        
        # Add controls
        for key, values in self.controls.items():
            data[f'control_{key}'] = values
        
        # Add auxiliary
        for key, values in self.auxiliary.items():
            data[f'aux_{key}'] = values
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        return df


@dataclass
class MissionResult:
    """
    Complete mission simulation results.
    
    Contains results for all phases plus overall mission metrics.
    """
    mission: MissionProfile
    phases: List[PhaseResult] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None
    
    # Overall metrics
    total_time_s: float = 0.0
    total_distance_m: float = 0.0
    total_energy_Wh: float = 0.0
    final_SOC: float = 0.0
    max_altitude_m: float = 0.0
    max_speed_m_s: float = 0.0
    max_power_W: float = 0.0
    max_motor_temp_C: float = 25.0
    max_battery_temp_C: float = 25.0
    
    def summary(self) -> Dict[str, Any]:
        """
        Return mission summary as dictionary.
        
        Returns:
            Dict with key mission metrics
        """
        return {
            'mission_name': self.mission.name,
            'success': self.success,
            'failure_reason': self.failure_reason,
            'total_time_s': self.total_time_s,
            'total_time_min': self.total_time_s / 60.0,
            'total_distance_m': self.total_distance_m,
            'total_distance_km': self.total_distance_m / 1000.0,
            'total_energy_Wh': self.total_energy_Wh,
            'final_SOC': self.final_SOC,
            'final_SOC_percent': self.final_SOC * 100.0,
            'max_altitude_m': self.max_altitude_m,
            'max_speed_m_s': self.max_speed_m_s,
            'max_power_W': self.max_power_W,
            'max_motor_temp_C': self.max_motor_temp_C,
            'max_battery_temp_C': self.max_battery_temp_C,
            'num_phases': len(self.phases),
            'phases_completed': sum(1 for p in self.phases if p.end_time > p.start_time),
        }
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Combine all phase DataFrames into single DataFrame.
        
        Returns:
            DataFrame with continuous time index across all phases
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        
        if not self.phases:
            return pd.DataFrame()
        
        dfs = []
        for phase in self.phases:
            df = phase.to_dataframe()
            df['phase'] = phase.name
            df['phase_type'] = phase.phase_type.name
            dfs.append(df)
        
        return pd.concat(dfs, axis=0)
    
    def print_summary(self):
        """Print formatted mission summary."""
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"Mission: {s['mission_name']}")
        print(f"{'='*50}")
        print(f"Status: {'SUCCESS' if s['success'] else 'FAILED'}")
        if s['failure_reason']:
            print(f"Failure: {s['failure_reason']}")
        print(f"\nTime:     {s['total_time_min']:.1f} min")
        print(f"Distance: {s['total_distance_km']:.2f} km")
        print(f"Energy:   {s['total_energy_Wh']:.1f} Wh")
        print(f"Final SOC: {s['final_SOC_percent']:.1f}%")
        print(f"\nMax Altitude: {s['max_altitude_m']:.1f} m")
        print(f"Max Speed:    {s['max_speed_m_s']:.1f} m/s")
        print(f"Max Power:    {s['max_power_W']:.0f} W")
        print(f"\nPhases: {s['phases_completed']}/{s['num_phases']} completed")
        print(f"{'='*50}\n")

    def print_detailed_summary(self):
        """Print a verbose, phase-by-phase summary."""
        s = self.summary()
        lines = [
            "=" * 70,
            f"MISSION RESULTS: {s['mission_name']}",
            "=" * 70,
            f"Status: {'SUCCESS' if s['success'] else 'FAILED'}",
        ]
        if s['failure_reason']:
            lines.append(f"Failure: {s['failure_reason']}")

        lines.extend(
            [
                "",
                f"Total Time: {s['total_time_min']:.2f} min",
                f"Total Distance: {s['total_distance_km']:.3f} km",
                f"Total Energy: {s['total_energy_Wh']:.2f} Wh",
                f"Final SOC: {s['final_SOC_percent']:.1f}%",
                f"Max Altitude: {s['max_altitude_m']:.1f} m",
                f"Max Speed: {s['max_speed_m_s']:.1f} m/s",
                f"Max Power: {s['max_power_W']:.0f} W",
                "",
                "PHASE BREAKDOWN:",
                "-" * 70,
            ]
        )

        for phase in self.phases:
            duration_s = phase.duration
            duration_min = duration_s / 60.0
            energy_Wh = phase.energy_consumed_Wh
            distance_m = phase.distance_traveled_m
            altitude_delta = phase.altitude_change_m
            soc_drop = (phase.SOC_start - phase.SOC_end) * 100.0

            airspeed = phase.states.get('airspeed') if phase.states else None
            altitude = phase.states.get('altitude') if phase.states else None
            power = phase.auxiliary.get('power') if phase.auxiliary else None
            thrust = phase.auxiliary.get('thrust') if phase.auxiliary else None
            headwind = phase.auxiliary.get('headwind') if phase.auxiliary else None
            load_factor = phase.auxiliary.get('load_factor') if phase.auxiliary else None
            bank = phase.auxiliary.get('bank') if phase.auxiliary else None

            def stats(arr):
                if arr is None or len(arr) == 0:
                    return None
                return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

            airspeed_stats = stats(airspeed)
            altitude_stats = stats(altitude)
            power_stats = stats(power)
            thrust_stats = stats(thrust)
            headwind_stats = stats(headwind)
            load_factor_stats = stats(load_factor)
            bank_stats = stats(bank)

            lines.append(f"{phase.name} ({phase.phase_type.name})")
            lines.append(f"  Duration: {duration_min:.2f} min | Distance: {distance_m:.1f} m | Energy: {energy_Wh:.2f} Wh")
            lines.append(f"  Altitude Delta: {altitude_delta:.1f} m | SOC Delta: -{soc_drop:.2f}%")

            if airspeed_stats:
                lines.append(
                    f"  Airspeed (min/avg/max): {airspeed_stats[0]:.1f} / {airspeed_stats[1]:.1f} / {airspeed_stats[2]:.1f} m/s"
                )
            if altitude_stats:
                lines.append(
                    f"  Altitude (min/avg/max): {altitude_stats[0]:.1f} / {altitude_stats[1]:.1f} / {altitude_stats[2]:.1f} m"
                )
            if power_stats:
                lines.append(
                    f"  Power (min/avg/max): {power_stats[0]:.0f} / {power_stats[1]:.0f} / {power_stats[2]:.0f} W"
                )
            if thrust_stats:
                lines.append(
                    f"  Thrust (min/avg/max): {thrust_stats[0]:.1f} / {thrust_stats[1]:.1f} / {thrust_stats[2]:.1f} N"
                )
            if headwind_stats:
                lines.append(
                    f"  Headwind (min/avg/max): {headwind_stats[0]:.1f} / {headwind_stats[1]:.1f} / {headwind_stats[2]:.1f} m/s"
                )
            if bank_stats:
                lines.append(
                    f"  Bank (min/avg/max): {bank_stats[0]:.1f} / {bank_stats[1]:.1f} / {bank_stats[2]:.1f} deg"
                )
            if load_factor_stats:
                lines.append(
                    f"  Load Factor (min/avg/max): {load_factor_stats[0]:.2f} / {load_factor_stats[1]:.2f} / {load_factor_stats[2]:.2f}"
                )

            lines.append("-")

        lines.append("=" * 70)
        print("\n".join(lines))


class MissionSimulator:
    """
    Forward simulation of prescribed missions.
    
    Uses Euler integration with autopilot control. Supports either:
    - Full dynamics (FlyingWingDynamics6DOF + IntegratedPropulsionSystem)
    - Simplified point-mass model (for faster simulation)
    
    Attributes:
        config: SimulationConfig instance
        autopilot: SimpleAutopilot instance
        use_simple_dynamics: Use simplified point-mass model
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        autopilot: Optional[SimpleAutopilot] = None,
        dynamics: Optional['FlyingWingDynamics6DOF'] = None,
        propulsion: Optional['IntegratedPropulsionSystem'] = None,
        aero_model: Optional[Callable] = None,
        wing_project: Optional['WingProject'] = None,
        mass_kg: float = 3.0,
        wing_area_m2: float = 0.5,
        wingspan_m: float = 2.0,
        CL0: float = 0.3,
        CLa: float = 5.5,
        CD0: float = 0.025,
        k_induced: float = 0.04,
        CL_max: float = 1.4,
        max_thrust_N: float = 15.0,
        battery_capacity_Wh: float = 100.0,
        propulsive_efficiency: float = 0.55,
        pitch_rate_gain: float = 6.0,
        pitch_rate_damping: float = 3.0,
        bank_rate_gain: float = 8.0,
        bank_rate_damping: float = 4.0,
        build_polar_verbose: bool = True,
    ):
        """
        Initialize mission simulator.
        
        Args:
            config: Simulation configuration
            autopilot: Autopilot instance (created if None)
            dynamics: 6-DOF dynamics model (optional)
            propulsion: Propulsion system model (optional)
            aero_model: Aerodynamic model callable (optional)
            wing_project: WingProject for auto-building RigidBodyAeroModel (optional)
                         If provided and aero_model is None, builds polar table
                         with verbose output matching takeoff_analysis3DOF.py format
            mass_kg: Aircraft mass (for simple dynamics)
            wing_area_m2: Wing area (for simple dynamics)
            build_polar_verbose: Print polar building progress (default True)
        """
        self.config = config or SimulationConfig()
        self.autopilot = autopilot or SimpleAutopilot()
        self.dynamics = dynamics
        self.propulsion = propulsion
        self.wing_project = wing_project
        
        # Build RigidBodyAeroModel from wing_project if no aero_model provided
        if aero_model is None and wing_project is not None:
            from services.aero_model import create_rigid_body_aero_model
            self.aero_model = create_rigid_body_aero_model(
                wing_project=wing_project,
                use_precomputed_polars=True,
                verbose=build_polar_verbose,
            )
        else:
            self.aero_model = aero_model
        
        self.mass_kg = mass_kg
        self.wing_area_m2 = wing_area_m2
        self.wingspan_m = wingspan_m
        self.CL0 = CL0
        self.CLa = CLa
        self.CD0 = CD0
        self.k_induced = k_induced
        self.CL_max = CL_max
        self.max_thrust_N = max_thrust_N
        self.battery_capacity_Wh = battery_capacity_Wh
        self.propulsive_efficiency = propulsive_efficiency
        self.pitch_rate_gain = pitch_rate_gain
        self.pitch_rate_damping = pitch_rate_damping
        self.bank_rate_gain = bank_rate_gain
        self.bank_rate_damping = bank_rate_damping
        
        if self.config.dynamics_mode is None:
            self.dynamics_mode = "full" if dynamics is not None else "simple"
        else:
            self.dynamics_mode = self.config.dynamics_mode
        
        self.max_bank_rad = np.radians(self.config.max_bank_deg)
        self.max_turn_rate_rad_s = None
        if self.config.max_turn_rate_deg_s is not None:
            self.max_turn_rate_rad_s = np.radians(self.config.max_turn_rate_deg_s)
    
    def simulate_mission(self, mission: MissionProfile) -> MissionResult:
        """
        Simulate complete mission.
        
        Args:
            mission: MissionProfile to simulate
        
        Returns:
            MissionResult with all phase results and metrics
        """
        # Validate mission
        issues = mission.validate()
        if issues:
            return MissionResult(
                mission=mission,
                success=False,
                failure_reason=f"Invalid mission: {'; '.join(issues)}",
            )
        
        # Initialize state
        state = self._create_initial_state(mission)
        
        # Initialize result
        result = MissionResult(mission=mission)
        current_time = 0.0
        
        # Simulate each phase
        for phase_idx, phase in enumerate(mission.phases):
            if self.config.verbose:
                print(f"Simulating phase {phase_idx + 1}/{len(mission.phases)}: {phase.name}")
            
            # Reset autopilot for new phase
            self.autopilot.reset()
            
            # Simulate phase
            phase_result, state, end_time, failure = self._simulate_phase(
                phase=phase,
                initial_state=state,
                start_time=current_time,
                mission=mission,
            )
            
            result.phases.append(phase_result)
            current_time = end_time
            
            # Check for mission failure
            if failure:
                result.failure_reason = failure
                break
            
            # Check SOC limit
            if state.get('SOC', 1.0) < mission.min_SOC:
                result.failure_reason = f"Battery depleted (SOC={state['SOC']*100:.1f}%)"
                break
            
            # Check max simulation time
            if current_time > self.config.max_simulation_time:
                result.failure_reason = "Maximum simulation time exceeded"
                break
        
        # Compute overall metrics
        result.success = (result.failure_reason is None)
        result.total_time_s = current_time
        result.total_distance_m = sum(p.distance_traveled_m for p in result.phases)
        result.total_energy_Wh = sum(p.energy_consumed_Wh for p in result.phases)
        result.final_SOC = state.get('SOC', 0.0)
        
        # Find max values from phase histories
        for phase in result.phases:
            if 'altitude' in phase.states and len(phase.states['altitude']) > 0:
                result.max_altitude_m = max(result.max_altitude_m, np.max(phase.states['altitude']))
            if 'airspeed' in phase.states and len(phase.states['airspeed']) > 0:
                result.max_speed_m_s = max(result.max_speed_m_s, np.max(phase.states['airspeed']))
            if 'power' in phase.auxiliary and len(phase.auxiliary['power']) > 0:
                result.max_power_W = max(result.max_power_W, np.max(phase.auxiliary['power']))
        
        return result
    
    def _create_initial_state(self, mission: MissionProfile) -> Dict[str, float]:
        """Create initial state from mission definition."""
        on_ground = mission.initial_altitude <= 0.0
        return {
            # Position (ENU)
            'x': 0.0,
            'y': 0.0,
            'altitude': max(0.0, mission.initial_altitude),
            
            # Velocity
            'airspeed': max(0.0, mission.initial_speed),
            'climb_rate': 0.0,
            'heading': mission.initial_heading,
            
            # Attitude
            'pitch': 0.0,
            'bank': 0.0,
            'yaw': mission.initial_heading,
            
            # Rates
            'p': 0.0,  # Roll rate
            'q': 0.0,  # Pitch rate
            'r': 0.0,  # Yaw rate
            
            # 3-DOF state
            'gamma': 0.0,
            'track': mission.initial_heading,
            'on_ground': on_ground,
            'ground_distance': 0.0,
            'V_ground': max(0.0, mission.initial_speed),
            
            # Energy
            'SOC': mission.initial_SOC,
            
            # Thermal
            'T_motor': mission.T_ambient,
            'T_esc': mission.T_ambient,
            'T_battery': mission.T_ambient,
            
            # Environment
            'T_ambient': mission.T_ambient,
        }
    
    def _simulate_phase(
        self,
        phase: MissionPhase,
        initial_state: Dict[str, float],
        start_time: float,
        mission: MissionProfile,
    ) -> tuple:
        """
        Simulate a single phase.
        
        Returns:
            (PhaseResult, final_state, end_time, failure_reason)
        """
        dt = self.config.dt
        dt_output = self.config.dt_output
        
        state = initial_state.copy()
        time = start_time
        
        # Recording buffers
        times = []
        states_history = {k: [] for k in state.keys()}
        controls_history = {'throttle': [], 'elevator': [], 'aileron': [], 'rudder': [], 'brake': []}
        aux_history: Dict[str, List[float]] = {}
        
        # Phase metrics
        energy_consumed = 0.0
        distance_traveled = 0.0
        initial_altitude = state['altitude']
        initial_SOC = state['SOC']
        
        last_output_time = start_time
        last_calc_log_time = start_time
        failure_reason = None
        
        # Phase simulation loop
        max_phase_time = phase.duration or 3600.0  # Default 1 hour max per phase
        phase_start_time = time
        
        # Log phase start
        if self.config.verbose:
            mission_logger.info(f"\n{'='*60}")
            mission_logger.info(f"PHASE START: {phase.name} ({phase.phase_type.name})")
            mission_logger.info(f"{'='*60}")
            mission_logger.info(f"  Time: {time:.1f}s | Alt: {state['altitude']:.1f}m | V: {state['airspeed']:.1f}m/s | SOC: {state['SOC']*100:.1f}%")
            if phase.target_altitude is not None:
                mission_logger.info(f"  Target Alt: {phase.target_altitude:.1f}m")
            if phase.target_speed is not None:
                mission_logger.info(f"  Target Speed: {phase.target_speed:.1f}m/s")
            if phase.duration is not None:
                mission_logger.info(f"  Duration: {phase.duration:.1f}s")
        
        while True:
            # Check phase end conditions
            phase_complete, reason = self._check_phase_end(phase, state, time - phase_start_time)
            if phase_complete:
                break
            
            # Check for failures
            if state['altitude'] < -1.0:  # Below ground (with tolerance)
                failure_reason = "Aircraft crashed (altitude < 0)"
                break
            
            if state['airspeed'] < 0.5 and not phase.phase_type.is_ground_phase():
                if state['altitude'] > 5.0:  # Stall in flight
                    failure_reason = "Aircraft stalled (airspeed too low)"
                    break
            
            # Get autopilot commands
            try:
                controls = self.autopilot.compute_controls(state, phase, dt)
                # Update state using dynamics
                state, aux = self._integrate_step(state, controls, dt, mission, phase)
                time += dt
            except Exception as exc:
                failure_reason = f"Simulation error in phase '{phase.name}': {exc}"
                break

            for key, value in state.items():
                if not np.all(np.isfinite(value)):
                    failure_reason = f"Non-finite state '{key}' in phase '{phase.name}'"
                    mission_logger.error(f"  CRITICAL: {failure_reason} at t={time:.2f}s")
                    break
            if failure_reason:
                break
            for key, value in aux.items():
                if not np.all(np.isfinite(value)):
                    failure_reason = f"Non-finite aux '{key}' in phase '{phase.name}'"
                    mission_logger.error(f"  CRITICAL: {failure_reason} at t={time:.2f}s")
                    break
            if failure_reason:
                break
            
            # Accumulate metrics
            power = aux.get('power', 0.0)
            energy_consumed += power * dt / 3600.0  # Wh
            
            # Distance traveled (horizontal)
            if 'V_ground' in state:
                distance_traveled += state['V_ground'] * dt
            else:
                v_horizontal = state['airspeed'] * np.cos(np.radians(state.get('pitch', 0.0)))
                distance_traveled += v_horizontal * dt
            
            # Record at output interval
            if time - last_output_time >= dt_output:
                times.append(time)
                for k, v in state.items():
                    states_history[k].append(v)
                for k, v in controls.items():
                    controls_history[k].append(v)
                for k, v in aux.items():
                    aux_history.setdefault(k, []).append(v)
                last_output_time = time
            
            # Log calculation details at interval
            if self.config.verbose and (time - last_calc_log_time >= self.config.verbose_calc_interval):
                self._log_calc_details(time, state, controls, aux, phase)
                last_calc_log_time = time
                # Flush stdout to ensure output is visible before potential crash
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
            
            # Safety: max phase time
            if time - phase_start_time > max_phase_time:
                break
        
        # Create phase result
        phase_result = PhaseResult(
            name=phase.name,
            phase_type=phase.phase_type,
            start_time=start_time,
            end_time=time,
            time=np.array(times),
            states={k: np.array(v) for k, v in states_history.items()},
            controls={k: np.array(v) for k, v in controls_history.items()},
            auxiliary={k: np.array(v) for k, v in aux_history.items()},
            energy_consumed_Wh=energy_consumed,
            distance_traveled_m=distance_traveled,
            altitude_change_m=state['altitude'] - initial_altitude,
            SOC_start=initial_SOC,
            SOC_end=state['SOC'],
        )
        
        # Log phase end summary
        if self.config.verbose:
            phase_duration = time - start_time
            soc_used = (initial_SOC - state['SOC']) * 100
            throttle_stats = None
            thrust_stats = None
            if controls_history.get('throttle'):
                throttle_vals = np.array(controls_history['throttle'], dtype=float)
                throttle_stats = (
                    float(np.min(throttle_vals)),
                    float(np.mean(throttle_vals)),
                    float(np.max(throttle_vals)),
                )
            if aux_history.get('thrust'):
                thrust_vals = np.array(aux_history['thrust'], dtype=float)
                thrust_stats = (
                    float(np.min(thrust_vals)),
                    float(np.mean(thrust_vals)),
                    float(np.max(thrust_vals)),
                )
            mission_logger.info(f"\n{'-'*60}")
            mission_logger.info(f"PHASE END: {phase.name}")
            mission_logger.info(f"{'-'*60}")
            mission_logger.info(f"  Duration: {phase_duration:.1f}s | Distance: {distance_traveled:.1f}m | Energy: {energy_consumed:.2f}Wh")
            mission_logger.info(f"  Alt: {initial_altitude:.1f}m -> {state['altitude']:.1f}m (delta: {state['altitude']-initial_altitude:+.1f}m)")
            mission_logger.info(f"  Speed: {initial_state['airspeed']:.1f}m/s -> {state['airspeed']:.1f}m/s")
            mission_logger.info(f"  SOC: {initial_SOC*100:.1f}% -> {state['SOC']*100:.1f}% (used: {soc_used:.2f}%)")
            if throttle_stats is not None:
                mission_logger.info(
                    f"  Throttle (min/avg/max): {throttle_stats[0]:.2f} / {throttle_stats[1]:.2f} / {throttle_stats[2]:.2f}"
                )
            if thrust_stats is not None:
                mission_logger.info(
                    f"  Thrust (min/avg/max): {thrust_stats[0]:.1f} / {thrust_stats[1]:.1f} / {thrust_stats[2]:.1f} N"
                )
            if failure_reason:
                mission_logger.info(f"  FAILURE: {failure_reason}")
        
        return phase_result, state, time, failure_reason
    
    def _check_phase_end(
        self,
        phase: MissionPhase,
        state: Dict[str, float],
        phase_time: float,
    ) -> tuple:
        """
        Check if phase end conditions are met.
        
        Returns:
            (is_complete, reason)
        """
        # Duration
        if phase.duration is not None and phase_time >= phase.duration:
            return True, "duration"
        
        # Ground contact detection for landing phases
        if phase.phase_type in (MissionPhaseType.LANDING_FLARE, MissionPhaseType.LANDING_ROLL):
            if state['altitude'] <= 0.1:  # Within 10cm of ground
                if phase.phase_type == MissionPhaseType.LANDING_FLARE:
                    return True, "ground_contact"
                # For landing roll, also check speed
                if phase.end_speed is not None and state['airspeed'] <= phase.end_speed:
                    return True, "speed_reached"
                elif phase.end_speed is None and state['airspeed'] <= 2.0:
                    return True, "stopped"
        
        # Altitude (with hysteresis to avoid oscillation)
        if phase.end_altitude is not None:
            alt = state['altitude']
            target = phase.end_altitude
            # Check if we've reached target altitude (with small tolerance)
            if phase.phase_type in (MissionPhaseType.CLIMB, MissionPhaseType.ROTATION):
                if alt >= target - 1.0:
                    return True, "altitude_reached"
            elif phase.phase_type in (MissionPhaseType.DESCENT, MissionPhaseType.APPROACH, 
                                      MissionPhaseType.LANDING_FLARE):
                if alt <= target + 0.5:  # Tighter tolerance for landing
                    return True, "altitude_reached"
        
        # Speed
        if phase.end_speed is not None:
            speed = state['airspeed']
            target = phase.end_speed
            if phase.phase_type == MissionPhaseType.TAKEOFF_ROLL:
                if speed >= target:
                    return True, "speed_reached"
            elif phase.phase_type == MissionPhaseType.LANDING_ROLL:
                if speed <= target:
                    return True, "speed_reached"
        
        # Distance
        if phase.end_distance is not None:
            # Would need to track phase distance separately
            pass
        
        # Waypoint
        if phase.waypoint is not None:
            wp = phase.waypoint
            dx = wp.x - state['x']
            dy = wp.y - state['y']
            distance = np.sqrt(dx*dx + dy*dy)
            alt_error = abs(state['altitude'] - wp.altitude)
            
            if distance < wp.position_tolerance and alt_error < wp.altitude_tolerance:
                return True, "waypoint_reached"
        
        return False, None
    
    def _log_calc_details(
        self,
        time: float,
        state: Dict[str, float],
        controls: Dict[str, float],
        aux: Dict[str, Any],
        phase: MissionPhase,
    ) -> None:
        """Log detailed calculation values for debugging."""
        mission_logger.info(f"\n  t={time:.1f}s [{phase.name}]")
        
        # State
        alt = state.get('altitude', 0.0)
        V = state.get('airspeed', 0.0)
        gamma = state.get('gamma', 0.0)
        track = state.get('track', 0.0)
        V_ground = state.get('V_ground', V)
        on_ground = state.get('on_ground', False)
        
        mission_logger.info(f"    State: Alt={alt:.1f}m V={V:.2f}m/s gamma={gamma:.2f}deg track={track:.1f}deg V_gnd={V_ground:.2f}m/s on_ground={on_ground}")
        
        # Controls
        throttle = controls.get('throttle', 0.0)
        elevator = controls.get('elevator', 0.0)
        brake = controls.get('brake', 0.0)
        mission_logger.info(f"    Controls: throttle={throttle:.2f} elevator={elevator:.3f} brake={brake:.2f}")
        
        # Forces and coefficients
        lift = aux.get('lift', 0.0)
        drag = aux.get('drag', 0.0)
        thrust = aux.get('thrust', 0.0)
        CL = aux.get('CL', 0.0)
        CD = aux.get('CD', 0.0)
        mission_logger.info(f"    Aero: L={lift:.2f}N D={drag:.2f}N CL={CL:.3f} CD={CD:.4f}")
        mission_logger.info(f"    Prop: T={thrust:.2f}N")
        
        # Power and energy
        power = aux.get('power', 0.0)
        soc = state.get('SOC', 1.0) * 100
        mission_logger.info(f"    Power: P={power:.1f}W SOC={soc:.1f}%")
        
        # Ground roll specific
        if on_ground:
            normal = aux.get('normal_force', 0.0)
            friction = aux.get('friction', 0.0)
            headwind = aux.get('headwind', 0.0)
            accel = aux.get('accel', 0.0)
            mission_logger.info(f"    Ground: N={normal:.2f}N Friction={friction:.2f}N Headwind={headwind:.1f}m/s Accel={accel:.2f}m/s²")
        
        # Flight specific
        else:
            bank = aux.get('bank', 0.0)
            load_factor = aux.get('load_factor', 1.0)
            headwind = aux.get('headwind', 0.0)
            mission_logger.info(f"    Flight: Bank={bank:.1f}deg n={load_factor:.2f} Headwind={headwind:.1f}m/s")
    
    def _integrate_step(
        self,
        state: Dict[str, float],
        controls: Dict[str, float],
        dt: float,
        mission: MissionProfile,
        phase: MissionPhase,
    ) -> tuple:
        """
        Integrate one timestep.
        
        Uses selected dynamics mode.
        
        Returns:
            (new_state, auxiliary_outputs)
        """
        if self.dynamics_mode == "simple":
            return self._integrate_simple(state, controls, dt, mission, phase)
        if self.dynamics_mode == "3dof":
            if phase.phase_type == MissionPhaseType.LANDING_FLARE:
                return self._integrate_3dof(state, controls, dt, mission, phase)
            if phase.phase_type.is_ground_phase() or state.get('on_ground', False):
                return self._integrate_ground(state, controls, dt, mission, phase)
            return self._integrate_3dof(state, controls, dt, mission, phase)
        return self._integrate_full(state, controls, dt, mission, phase)

    def _get_density(self, altitude_m: float, mission: MissionProfile) -> float:
        rho0 = 1.225
        scale_height = 8500.0
        effective_alt = max(0.0, altitude_m + mission.pressure_altitude)
        return rho0 * np.exp(-effective_alt / scale_height)

    def _get_wind_enu(self, mission: MissionProfile) -> tuple:
        wind_from_rad = np.radians(mission.wind_direction)
        wind_to_rad = wind_from_rad + np.pi
        wind_east = mission.wind_speed * np.sin(wind_to_rad)
        wind_north = mission.wind_speed * np.cos(wind_to_rad)
        return wind_east, wind_north

    def _get_headwind(self, mission: MissionProfile) -> float:
        wind_from_rad = np.radians(mission.wind_direction)
        runway_heading_rad = np.radians(mission.runway_heading)
        return mission.wind_speed * np.cos(wind_from_rad - runway_heading_rad)

    def _get_aero_coeffs(
        self,
        alpha_deg: float,
        airspeed: float,
        altitude_m: float,
        controls: Dict[str, float],
    ) -> tuple:
        if self.aero_model is not None:
            result = self.aero_model(
                alpha=alpha_deg,
                beta=0.0,
                airspeed=airspeed,
                p=0.0,
                q=0.0,
                r=0.0,
                elevator=controls.get('elevator', 0.0),
                aileron=controls.get('aileron', 0.0),
                rudder=controls.get('rudder', 0.0),
            )
            CL = result.get('CL', 0.0)
            CD = result.get('CD', self.CD0)
        else:
            alpha_rad = np.radians(alpha_deg)
            CL = self.CL0 + self.CLa * alpha_rad
            CD = self.CD0 + self.k_induced * CL**2
        CL = float(np.clip(CL, -self.CL_max, self.CL_max))
        CD = float(max(CD, 0.0))
        return CL, CD

    def _compute_propulsion(
        self,
        throttle: float,
        airspeed: float,
        rho: float,
        SOC: float,
        temperatures: Dict[str, float],
        T_ambient: float,
    ) -> tuple:
        throttle = float(np.clip(throttle, 0.0, 1.0))
        airspeed = float(max(0.0, airspeed))
        rho = float(max(rho, 0.0))
        SOC = float(np.clip(SOC, 0.0, 1.0))

        if SOC <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}

        def _fallback_linear() -> tuple:
            thrust = throttle * self.max_thrust_N
            power = max(0.0, thrust * airspeed / max(self.propulsive_efficiency, 1e-3))
            dSOC_dt = -power / (self.battery_capacity_Wh * 3600.0)
            return thrust, power, dSOC_dt, 0.0, 0.0, 0.0, {}

        if self.propulsion is None:
            return _fallback_linear()

        try:
            result = self.propulsion.solve_equilibrium(
                throttle=throttle,
                V_freestream=airspeed,
                rho=rho,
                SOC=SOC,
                temperatures=temperatures,
                T_ambient=T_ambient,
            )
        except Exception as exc:
            mission_logger.warning(
                "Propulsion solver failed: %s. Falling back to linear model.", exc
            )
            return _fallback_linear()
        thrust = result.get('thrust_total', 0.0)
        power = result.get('power_battery', 0.0)
        dSOC_dt = result.get('dSOC_dt', 0.0)
        dT_motor_dt = result.get('dT_motor_dt')
        dT_esc_dt = result.get('dT_esc_dt')
        dT_battery_dt = result.get('dT_battery_dt')

        if not np.isfinite(thrust) or not np.isfinite(power) or not np.isfinite(dSOC_dt):
            mission_logger.warning(
                "Propulsion returned non-finite values (thrust=%s, power=%s, dSOC_dt=%s). "
                "Falling back to linear model.",
                thrust,
                power,
                dSOC_dt,
            )
            return _fallback_linear()
        if np.isfinite(self.max_thrust_N) and self.max_thrust_N > 0:
            if abs(thrust) > self.max_thrust_N * 10.0:
                mission_logger.warning(
                    "Propulsion thrust %.2f N exceeds sanity limit (%.2f N). "
                    "Falling back to linear model.",
                    thrust,
                    self.max_thrust_N * 10.0,
                )
                return _fallback_linear()

        if dT_motor_dt is None or dT_esc_dt is None or dT_battery_dt is None:
            thermal_model = getattr(self.propulsion, 'thermal_model', None)
            if thermal_model is not None:
                thermal_result = thermal_model.compute_derivatives(
                    heat_motor=result.get('heat_motor', 0.0),
                    heat_esc=result.get('heat_esc', 0.0),
                    heat_battery=result.get('heat_battery', 0.0),
                    T_motor=temperatures['motor'],
                    T_esc=temperatures['esc'],
                    T_battery=temperatures['battery'],
                    V_freestream=airspeed,
                    T_ambient=T_ambient,
                )
                dT_motor_dt = thermal_result.get('dT_motor_dt', 0.0)
                dT_esc_dt = thermal_result.get('dT_esc_dt', 0.0)
                dT_battery_dt = thermal_result.get('dT_battery_dt', 0.0)
            else:
                dT_motor_dt = 0.0
                dT_esc_dt = 0.0
                dT_battery_dt = 0.0

        return thrust, power, dSOC_dt, dT_motor_dt, dT_esc_dt, dT_battery_dt, result

    def _integrate_3dof(
        self,
        state: Dict[str, float],
        controls: Dict[str, float],
        dt: float,
        mission: MissionProfile,
        phase: MissionPhase,
    ) -> tuple:
        new_state = state.copy()
        g = 9.81
        V_min = 2.0

        pitch = new_state.get('pitch', 0.0)
        bank = new_state.get('bank', 0.0)
        pitch_rate = self.pitch_rate_gain * controls['elevator'] - self.pitch_rate_damping * pitch
        bank_rate = self.bank_rate_gain * controls['aileron'] - self.bank_rate_damping * bank
        pitch = np.clip(pitch + pitch_rate * dt, phase.pitch_min, phase.pitch_max)
        bank = np.clip(bank + bank_rate * dt, -phase.bank_max, phase.bank_max)

        gamma_deg = new_state.get('gamma', 0.0)
        alpha_cmd_deg = np.clip(pitch - gamma_deg, phase.pitch_min, phase.pitch_max)
        bank_cmd_deg = bank

        wind_east, wind_north = self._get_wind_enu(mission)

        def derivatives(local_state: Dict[str, float]) -> Dict[str, float]:
            gamma_local = np.radians(local_state['gamma'])
            chi_local = np.radians(local_state['track'])
            bank_local = np.radians(bank_cmd_deg)
            if self.config.max_load_factor is not None:
                max_bank = np.arccos(1.0 / self.config.max_load_factor)
                bank_local = np.clip(bank_local, -max_bank, max_bank)
            rho = self._get_density(local_state['altitude'], mission)
            V = max(local_state['airspeed'], V_min)
            CL, CD = self._get_aero_coeffs(alpha_cmd_deg, V, local_state['altitude'], controls)
            q = 0.5 * rho * V**2
            L = q * self.wing_area_m2 * CL
            D = q * self.wing_area_m2 * CD

            temps = {
                'motor': local_state['T_motor'],
                'esc': local_state['T_esc'],
                'battery': local_state['T_battery'],
            }
            thrust, power, dSOC_dt, dT_motor_dt, dT_esc_dt, dT_battery_dt, _ = self._compute_propulsion(
                controls['throttle'],
                V,
                rho,
                local_state['SOC'],
                temps,
                local_state['T_ambient'],
            )

            alpha_rad = np.radians(alpha_cmd_deg)
            dV_dt = (thrust * np.cos(alpha_rad) - D) / self.mass_kg - g * np.sin(gamma_local)
            
            # Clamp dV_dt to prevent numerical blowup (max ±5g longitudinal)
            dV_dt = float(np.clip(dV_dt, -50.0, 50.0))
            
            dgamma_dt = (thrust * np.sin(alpha_rad) + L * np.cos(bank_local) - self.mass_kg * g * np.cos(gamma_local)) / (self.mass_kg * V)
            
            # Clamp dgamma_dt to prevent numerical blowup (max ±30 deg/s)
            dgamma_dt = float(np.clip(dgamma_dt, -np.radians(30), np.radians(30)))
            
            dchi_dt = g * np.tan(bank_local) / V
            if self.max_turn_rate_rad_s is not None:
                dchi_dt = float(np.clip(dchi_dt, -self.max_turn_rate_rad_s, self.max_turn_rate_rad_s))

            V_ground_east = V * np.cos(gamma_local) * np.sin(chi_local) + wind_east
            V_ground_north = V * np.cos(gamma_local) * np.cos(chi_local) + wind_north
            ground_speed = np.sqrt(V_ground_east**2 + V_ground_north**2)

            return {
                'x': V_ground_east,
                'y': V_ground_north,
                'altitude': V * np.sin(gamma_local),
                'airspeed': dV_dt,
                'gamma': np.degrees(dgamma_dt),
                'track': np.degrees(dchi_dt),
                'SOC': dSOC_dt,
                'T_motor': dT_motor_dt,
                'T_esc': dT_esc_dt,
                'T_battery': dT_battery_dt,
                'ground_distance': ground_speed,
                'power': power,
                'thrust': thrust,
                'drag': D,
                'lift': L,
                'CL': CL,
                'CD': CD,
                'bank_used': np.degrees(bank_local),
                'load_factor': 1.0 / max(np.cos(bank_local), 1e-3),
                'V_ground': ground_speed,
                'V_air': V,
            }

        if self.config.integration_method == "rk4":
            k1 = derivatives(new_state)
            k2_state = new_state.copy()
            for key, value in k1.items():
                if key in k2_state:
                    k2_state[key] = new_state[key] + 0.5 * dt * value
            k2 = derivatives(k2_state)
            k3_state = new_state.copy()
            for key, value in k2.items():
                if key in k3_state:
                    k3_state[key] = new_state[key] + 0.5 * dt * value
            k3 = derivatives(k3_state)
            k4_state = new_state.copy()
            for key, value in k3.items():
                if key in k4_state:
                    k4_state[key] = new_state[key] + dt * value
            k4 = derivatives(k4_state)
            for key in ('x', 'y', 'altitude', 'airspeed', 'gamma', 'track', 'SOC', 'T_motor', 'T_esc', 'T_battery', 'ground_distance'):
                new_state[key] += (dt / 6.0) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
            aux = k1
        else:
            k1 = derivatives(new_state)
            for key in ('x', 'y', 'altitude', 'airspeed', 'gamma', 'track', 'SOC', 'T_motor', 'T_esc', 'T_battery', 'ground_distance'):
                new_state[key] += dt * k1[key]
            aux = k1

        new_state['pitch'] = pitch
        new_state['bank'] = bank
        new_state['q'] = pitch_rate
        new_state['p'] = bank_rate
        new_state['climb_rate'] = new_state['airspeed'] * np.sin(np.radians(new_state['gamma']))
        new_state['heading'] = new_state['track'] % 360.0
        new_state['on_ground'] = False
        new_state['V_ground'] = aux['V_ground']
        new_state['airspeed'] = max(0.0, new_state['airspeed'])
        new_state['altitude'] = max(0.0, new_state['altitude'])

        if new_state['altitude'] <= 0.0 and phase.phase_type in (
            MissionPhaseType.LANDING_FLARE,
            MissionPhaseType.APPROACH,
            MissionPhaseType.DESCENT,
        ):
            new_state['altitude'] = 0.0
            new_state['gamma'] = 0.0
            new_state['on_ground'] = True
            new_state['V_ground'] = max(0.0, new_state['airspeed'] * np.cos(np.radians(state.get('gamma', 0.0))))
            new_state['track'] = new_state['heading']

        aux_out = {
            'power': aux['power'],
            'thrust': aux['thrust'],
            'drag': aux['drag'],
            'lift': aux['lift'],
            'CL': aux['CL'],
            'CD': aux['CD'],
            'bank': aux['bank_used'],
            'load_factor': aux['load_factor'],
            'headwind': self._get_headwind(mission),
        }

        return new_state, aux_out

    def _integrate_ground(
        self,
        state: Dict[str, float],
        controls: Dict[str, float],
        dt: float,
        mission: MissionProfile,
        phase: MissionPhase,
    ) -> tuple:
        new_state = state.copy()
        g = 9.81
        rho = self._get_density(0.0, mission)
        headwind = self._get_headwind(mission)
        alpha_ground = self.config.alpha_ground_deg

        V_ground = max(0.0, new_state.get('V_ground', new_state['airspeed']))
        V_air = max(0.0, V_ground + headwind)

        CL, CD = self._get_aero_coeffs(alpha_ground, V_air, 0.0, controls)
        q = 0.5 * rho * V_air**2
        L = q * self.wing_area_m2 * CL
        D = q * self.wing_area_m2 * CD
        N = max(0.0, self.mass_kg * g - L)

        mu = self.config.mu_roll * (1.0 - controls['brake']) + self.config.mu_brake * controls['brake']
        temps = {
            'motor': new_state['T_motor'],
            'esc': new_state['T_esc'],
            'battery': new_state['T_battery'],
        }
        thrust, power, dSOC_dt, dT_motor_dt, dT_esc_dt, dT_battery_dt, _ = self._compute_propulsion(
            controls['throttle'],
            V_air,
            rho,
            new_state['SOC'],
            temps,
            new_state['T_ambient'],
        )

        accel = (thrust - D - mu * N) / self.mass_kg
        V_ground = max(0.0, V_ground + accel * dt)

        heading_rad = np.radians(new_state.get('heading', mission.runway_heading))
        new_state['x'] += V_ground * np.sin(heading_rad) * dt
        new_state['y'] += V_ground * np.cos(heading_rad) * dt
        new_state['ground_distance'] += V_ground * dt
        new_state['airspeed'] = V_air
        new_state['V_ground'] = V_ground
        new_state['gamma'] = 0.0
        new_state['track'] = new_state.get('heading', mission.runway_heading)
        new_state['climb_rate'] = 0.0
        new_state['on_ground'] = True
        new_state['altitude'] = 0.0
        new_state['pitch'] = np.clip(new_state['pitch'], phase.pitch_min, phase.pitch_max)
        new_state['bank'] = np.clip(new_state['bank'], -phase.bank_max, phase.bank_max)
        new_state['SOC'] = max(0.0, new_state['SOC'] + dSOC_dt * dt)
        new_state['T_motor'] += dT_motor_dt * dt
        new_state['T_esc'] += dT_esc_dt * dt
        new_state['T_battery'] += dT_battery_dt * dt

        if phase.phase_type in (MissionPhaseType.TAKEOFF_ROLL, MissionPhaseType.ROTATION):
            V_rot = phase.end_speed or phase.target_speed
            if V_rot is None:
                V_stall = np.sqrt(2.0 * self.mass_kg * g / (rho * self.wing_area_m2 * max(self.CL_max, 1e-3)))
                V_rot = self.config.V_rot_factor * V_stall
            if V_air >= V_rot and L >= self.mass_kg * g:
                new_state['on_ground'] = False
                new_state['altitude'] = 0.1
                new_state['gamma'] = self.config.alpha_ground_deg
                new_state['track'] = new_state.get('heading', mission.runway_heading)

        aux = {
            'power': power,
            'thrust': thrust,
            'drag': D,
            'lift': L,
            'CL': CL,
            'CD': CD,
            'headwind': headwind,
            'normal_force': N,
            'friction': mu * N,
            'accel': accel,
        }

        return new_state, aux

    def _integrate_simple(
        self,
        state: Dict[str, float],
        controls: Dict[str, float],
        dt: float,
        mission: MissionProfile,
        phase: MissionPhase,
    ) -> tuple:
        """
        Simple point-mass integration.
        
        Simplified physics for fast, stable simulation.
        Key simplification: climb rate is directly proportional to pitch angle
        when thrust exceeds drag.
        """
        g = 9.81
        rho = 1.225
        
        # Current state
        V_actual = state['airspeed']  # Actual airspeed (can be 0 on ground)
        V = max(V_actual, 5.0)  # Clamped for aero calculations (avoid div/0)
        alt = state['altitude']
        heading = state['heading']
        pitch = state['pitch']
        bank = state['bank']
        SOC = state['SOC']
        
        m = self.mass_kg
        S = self.wing_area_m2
        weight = m * g
        
        # Dynamic pressure
        q = 0.5 * rho * V**2
        
        # Simple drag model
        CD = self.CD0 + 0.05 * (pitch / 10.0)**2  # Induced drag increases with pitch
        drag = q * S * CD
        
        # Propulsion
        throttle = controls['throttle']
        thrust = throttle * self.max_thrust_N
        
        # Power
        power = max(0, thrust * V / max(self.propulsive_efficiency, 1e-3))
        
        # Climb rate calculation - simplified energy method
        # Excess thrust determines climb capability
        excess_thrust = thrust - drag
        
        # At positive pitch with excess thrust, we climb
        # Climb rate = V * sin(gamma), where gamma is flight path angle
        # Simplified: gamma ≈ pitch when in steady climb
        if excess_thrust > 0 and pitch > 0:
            # Can climb - rate proportional to pitch and excess thrust
            max_climb = excess_thrust / weight * V  # Max possible from energy
            pitch_climb = V * np.sin(np.radians(pitch))
            target_climb_rate = min(pitch_climb, max_climb)
        elif pitch > 0:
            # Not enough thrust to climb, but pitched up - slow descent
            target_climb_rate = -0.5
        else:
            # Pitched down or level with insufficient thrust - descend
            target_climb_rate = V * np.sin(np.radians(pitch))
        
        # Smooth climb rate transition
        climb_rate = 0.9 * state['climb_rate'] + 0.1 * target_climb_rate
        
        # Speed change - simplified
        # Increase speed if thrust > drag and pitched down
        # Decrease if climbing (trading speed for altitude)
        if pitch > 5.0 and climb_rate > 0.5:
            # Climbing - speed decreases slightly
            accel = -0.3
        elif pitch < -5.0:
            # Descending - speed increases
            accel = 0.5
        else:
            # Level-ish - thrust vs drag
            accel = (thrust - drag) / m * 0.3  # Damped response
        
        # Turn rate from bank
        if V > 5.0 and abs(bank) > 2.0:
            turn_rate = g * np.tan(np.radians(bank)) / V
        else:
            turn_rate = 0.0
        
        # Control surface effects
        elevator = controls['elevator']
        aileron = controls['aileron']
        
        # Pitch dynamics
        pitch_rate = self.pitch_rate_gain * elevator - self.pitch_rate_damping * pitch
        
        # Bank dynamics
        bank_rate = self.bank_rate_gain * aileron - self.bank_rate_damping * bank
        
        # Integrate
        new_state = state.copy()
        
        # Velocity integration - use clamped V for aero-based acceleration
        # This prevents instability at very low speeds in flight
        new_state['airspeed'] = np.clip(V + accel * dt, 0.0, 35.0)
        new_state['climb_rate'] = np.clip(climb_rate, -8.0, 5.0)
        
        # Position - use clamped V for flight, actual for ground
        v_horizontal = V * np.cos(np.radians(min(pitch, 20.0)))
        new_state['x'] = state['x'] + v_horizontal * np.sin(np.radians(heading)) * dt
        new_state['y'] = state['y'] + v_horizontal * np.cos(np.radians(heading)) * dt
        new_state['altitude'] = max(0.0, alt + climb_rate * dt)
        new_state['ground_distance'] = state.get('ground_distance', 0.0) + v_horizontal * dt
        
        # Attitude
        new_state['heading'] = (heading + np.degrees(turn_rate) * dt) % 360.0
        new_state['pitch'] = np.clip(pitch + pitch_rate * dt, phase.pitch_min, phase.pitch_max)
        new_state['bank'] = np.clip(bank + bank_rate * dt, -phase.bank_max, phase.bank_max)
        
        if new_state['airspeed'] > 1e-3:
            gamma_rad = np.arcsin(np.clip(new_state['climb_rate'] / new_state['airspeed'], -1.0, 1.0))
        else:
            gamma_rad = 0.0
        new_state['gamma'] = np.degrees(gamma_rad)
        new_state['track'] = new_state['heading']
        new_state['V_ground'] = new_state['airspeed'] * np.cos(gamma_rad)
        new_state['on_ground'] = new_state['altitude'] <= 0.0
        
        # Energy
        energy_used = power * dt / 3600.0
        new_state['SOC'] = max(0.0, SOC - energy_used / self.battery_capacity_Wh)
        
        # Ground contact and takeoff logic
        if new_state['altitude'] <= 0.0:
            new_state['altitude'] = 0.0
            
            # Check if aircraft can take off
            # Need sufficient speed and positive pitch
            takeoff_speed = phase.end_speed or phase.target_speed or 10.0
            if new_state['airspeed'] >= takeoff_speed and pitch > 3.0 and thrust > drag:
                # Allow takeoff - generate positive climb rate
                liftoff_rate = min(2.0, (new_state['airspeed'] - takeoff_speed) * 0.3 * np.sin(np.radians(pitch)))
                new_state['climb_rate'] = max(0.1, liftoff_rate)
                new_state['altitude'] = 0.1  # Just above ground
            else:
                # On ground, can't take off
                new_state['climb_rate'] = 0.0
                new_state['pitch'] = max(0.0, min(new_state['pitch'], 15.0))  # Limit ground pitch
                
                # Ground speed dynamics - OVERRIDE flight-based calculation
                # Use V_actual (true previous speed) for ground operations
                if controls['brake'] > 0.5:
                    # Full brakes: ~6 m/s² base + speed-dependent aero drag
                    brake_decel = 6.0 + 2.0 * (V_actual / 20.0)
                    new_state['airspeed'] = max(0.0, V_actual - brake_decel * dt)
                elif thrust < 0.1 * self.max_thrust_N:
                    # Idle: rolling friction + aero drag
                    idle_decel = 0.5 + 1.5 * (V_actual / 20.0)
                    new_state['airspeed'] = max(0.0, V_actual - idle_decel * dt)
                # Ground acceleration
                elif V_actual < 25.0:
                    ground_accel = (thrust - 0.05 * weight) / m  # Rolling friction ~5% weight
                    new_state['airspeed'] = max(0.0, V_actual + ground_accel * dt)
        
        # Auxiliary
        aux = {
            'power': power,
            'thrust': thrust,
            'drag': drag,
            'lift': q * S * 0.5,  # Approximate
            'CL': 0.5,
            'excess_power': excess_thrust * V,
        }
        
        return new_state, aux
    
    def _integrate_full(
        self,
        state: Dict[str, float],
        controls: Dict[str, float],
        dt: float,
        mission: MissionProfile,
        phase: MissionPhase,
    ) -> tuple:
        """
        Full 6-DOF integration with propulsion.
        
        Uses FlyingWingDynamics6DOF and IntegratedPropulsionSystem.
        """
        # TODO: Implement full dynamics integration when dynamics/propulsion are provided
        # For now, fall back to simple
        return self._integrate_simple(state, controls, dt, mission, phase)


# Convenience function
def simulate_simple_cruise(
    cruise_altitude: float = 100.0,
    cruise_speed: float = 18.0,
    cruise_duration: float = 300.0,
    mass_kg: float = 3.0,
    wing_area_m2: float = 0.5,
    verbose: bool = True,
    verbose_mode: str = "summary",
) -> MissionResult:
    """
    Quick simulation of a simple cruise mission.
    
    Args:
        cruise_altitude: Cruise altitude [m]
        cruise_speed: Cruise speed [m/s]
        cruise_duration: Time at cruise [s]
        mass_kg: Aircraft mass [kg]
        wing_area_m2: Wing area [m^2]
        verbose: Print progress
    
    Returns:
        MissionResult
    """
    # Create mission
    mission = MissionProfile.simple_cruise(
        cruise_altitude=cruise_altitude,
        cruise_speed=cruise_speed,
        cruise_duration=cruise_duration,
    )
    
    # Create simulator
    config = SimulationConfig(verbose=verbose)
    simulator = MissionSimulator(
        config=config,
        mass_kg=mass_kg,
        wing_area_m2=wing_area_m2,
    )
    
    # Run simulation
    result = simulator.simulate_mission(mission)
    
    if verbose:
        if verbose_mode == "detailed":
            result.print_detailed_summary()
        else:
            result.print_summary()
    
    return result
