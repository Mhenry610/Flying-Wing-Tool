"""
Full Mission Phase Comparison Script

Runs a complete 9-phase mission through the MissionSimulator and validates
each phase produces physically reasonable results.

Phases tested:
1. Ground Idle (5s stationary)
2. Takeoff Roll (accelerate to V_rot)
3. Rotation (pitch up, liftoff)
4. Climb (climb to cruise altitude)
5. Cruise (level flight for duration)
6. Descent (descend to approach altitude)
7. Approach (final approach)
8. Flare (touchdown)
9. Landing Roll (decelerate to stop)

Author: Sisyphus AI
Date: January 2026
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.state import Project
from services.aero_model import create_rigid_body_aero_model
from services.mission.mission_simulator import (
    MissionSimulator,
    SimulationConfig,
    MissionResult,
    PhaseResult,
)
from services.mission.mission_definition import (
    MissionProfile,
    MissionPhase,
    MissionPhaseType,
)
from services.propulsion.battery_model import create_lipo_pack
from services.propulsion.motor_model import DifferentiableMotorModel, MotorParameters
from services.propulsion.propeller_model import get_pretrained_model
from services.propulsion.propulsion_system import IntegratedPropulsionSystem


@dataclass
class PhaseValidation:
    """Validation results for a single phase."""
    phase_name: str
    phase_type: str
    duration_s: float
    distance_m: float
    energy_Wh: float
    start_altitude_m: float
    end_altitude_m: float
    start_speed_mps: float
    end_speed_mps: float
    soc_start: float
    soc_end: float
    
    # Validation checks
    checks: Dict[str, Tuple[bool, str]] = None  # (passed, message)
    
    @property
    def all_passed(self) -> bool:
        if self.checks is None:
            return True
        return all(passed for passed, _ in self.checks.values())
    
    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Phase: {self.phase_name} ({self.phase_type})",
            f"{'='*60}",
            f"  Duration: {self.duration_s:.2f}s",
            f"  Distance: {self.distance_m:.1f}m",
            f"  Energy:   {self.energy_Wh:.3f}Wh",
            f"  Altitude: {self.start_altitude_m:.1f}m -> {self.end_altitude_m:.1f}m",
            f"  Speed:    {self.start_speed_mps:.1f}m/s -> {self.end_speed_mps:.1f}m/s",
            f"  SOC:      {self.soc_start*100:.1f}% -> {self.soc_end*100:.1f}%",
        ]
        
        if self.checks:
            lines.append("\n  Validation Checks:")
            for check_name, (passed, msg) in self.checks.items():
                status = "[PASS]" if passed else "[FAIL]"
                lines.append(f"    {status} {check_name}: {msg}")
        
        return "\n".join(lines)


def validate_phase(phase_result: PhaseResult, expected_behavior: Dict) -> PhaseValidation:
    """Validate a single phase result against expected behavior."""
    
    # Extract state values from time history
    states = phase_result.states
    times = phase_result.time
    
    if len(times) > 0:
        start_alt = states.get('altitude', [0])[0]
        end_alt = states.get('altitude', [0])[-1]
        start_speed = states.get('airspeed', [0])[0]
        end_speed = states.get('airspeed', [0])[-1]
    else:
        start_alt = end_alt = 0
        start_speed = end_speed = 0
    
    validation = PhaseValidation(
        phase_name=phase_result.name,
        phase_type=phase_result.phase_type.name,
        duration_s=phase_result.end_time - phase_result.start_time,
        distance_m=phase_result.distance_traveled_m,
        energy_Wh=phase_result.energy_consumed_Wh,
        start_altitude_m=start_alt,
        end_altitude_m=end_alt,
        start_speed_mps=start_speed,
        end_speed_mps=end_speed,
        soc_start=phase_result.SOC_start,
        soc_end=phase_result.SOC_end,
        checks={},
    )
    
    # Apply validation checks based on expected behavior
    checks = {}
    
    # Check duration is reasonable
    if 'min_duration' in expected_behavior:
        passed = validation.duration_s >= expected_behavior['min_duration']
        checks['duration_min'] = (passed, f"{validation.duration_s:.1f}s >= {expected_behavior['min_duration']}s")
    
    if 'max_duration' in expected_behavior:
        passed = validation.duration_s <= expected_behavior['max_duration']
        checks['duration_max'] = (passed, f"{validation.duration_s:.1f}s <= {expected_behavior['max_duration']}s")
    
    # Check speed is reasonable (not runaway)
    if 'max_speed' in expected_behavior:
        max_speed_observed = np.max(states.get('airspeed', [0])) if len(states.get('airspeed', [])) > 0 else end_speed
        passed = max_speed_observed <= expected_behavior['max_speed']
        checks['max_speed'] = (passed, f"{max_speed_observed:.1f}m/s <= {expected_behavior['max_speed']}m/s")
    
    # Check altitude change direction
    if 'altitude_change' in expected_behavior:
        alt_change = end_alt - start_alt
        expected = expected_behavior['altitude_change']
        if expected == 'increase':
            passed = alt_change >= 0
            checks['altitude'] = (passed, f"Change {alt_change:+.1f}m (expected increase)")
        elif expected == 'decrease':
            passed = alt_change <= 0
            checks['altitude'] = (passed, f"Change {alt_change:+.1f}m (expected decrease)")
        elif expected == 'stable':
            passed = abs(alt_change) < 10  # Within 10m
            checks['altitude'] = (passed, f"Change {alt_change:+.1f}m (expected stable)")
    
    # Check speed change direction
    if 'speed_change' in expected_behavior:
        speed_change = end_speed - start_speed
        expected = expected_behavior['speed_change']
        if expected == 'increase':
            passed = speed_change >= 0
            checks['speed'] = (passed, f"Change {speed_change:+.1f}m/s (expected increase)")
        elif expected == 'decrease':
            passed = speed_change <= 0
            checks['speed'] = (passed, f"Change {speed_change:+.1f}m/s (expected decrease)")
        elif expected == 'stable':
            passed = abs(speed_change) < 5  # Within 5m/s
            checks['speed'] = (passed, f"Change {speed_change:+.1f}m/s (expected stable)")
    
    # Check energy consumption is positive (except for idle)
    if expected_behavior.get('uses_energy', True):
        passed = validation.energy_Wh >= 0
        checks['energy'] = (passed, f"{validation.energy_Wh:.3f}Wh consumed")
    
    # Check no NaN/Inf values
    for key, arr in states.items():
        if len(arr) > 0:
            has_nan = np.any(np.isnan(arr))
            has_inf = np.any(np.isinf(arr))
            if has_nan or has_inf:
                checks[f'no_nan_{key}'] = (False, f"NaN/Inf detected in {key}")
    
    validation.checks = checks
    return validation


def run_full_mission_comparison(
    wing_project,
    mass_kg: float,
    wing_area_m2: float,
    CL_max: float,
    mu_roll: float,
    cruise_speed: float,
    cruise_alt: float,
    cruise_duration_s: float,
) -> Tuple[MissionResult, List[PhaseValidation]]:
    """Run a complete 9-phase mission and validate each phase."""
    
    print("\n" + "=" * 70)
    print("RUNNING FULL MISSION SIMULATION")
    print("=" * 70)
    
    # Build aero model
    print("\nBuilding RigidBodyAeroModel...")
    start = time.time()
    aero_model = create_rigid_body_aero_model(
        wing_project=wing_project,
        use_precomputed_polars=True,
        verbose=True,
    )
    aero_time = time.time() - start
    print(f"Aero model built in {aero_time:.1f}s")
    
    # Calculate key speeds
    rho = 1.225
    g = 9.81
    W = mass_kg * g
    V_stall = np.sqrt(2 * W / (rho * wing_area_m2 * CL_max))
    V_rot = 1.1 * V_stall
    V_ref = 1.3 * V_stall
    
    print(f"\nFlight speeds:")
    print(f"  V_stall: {V_stall:.1f} m/s")
    print(f"  V_rot:   {V_rot:.1f} m/s")
    print(f"  V_ref:   {V_ref:.1f} m/s")
    print(f"  V_cruise: {cruise_speed:.1f} m/s")
    
    # Build mission profile
    phases = [
        MissionPhase(
            name="Ground Idle",
            phase_type=MissionPhaseType.GROUND_IDLE,
            duration=5.0,
        ),
        MissionPhase(
            name="Takeoff Roll",
            phase_type=MissionPhaseType.TAKEOFF_ROLL,
            end_speed=V_rot,
        ),
        MissionPhase(
            name="Rotation",
            phase_type=MissionPhaseType.ROTATION,
            duration=4.0,
            end_altitude=5.0,
        ),
        MissionPhase(
            name="Climb",
            phase_type=MissionPhaseType.CLIMB,
            end_altitude=cruise_alt,
            target_speed=cruise_speed,
        ),
        MissionPhase(
            name="Cruise",
            phase_type=MissionPhaseType.CRUISE,
            duration=cruise_duration_s,
            target_altitude=cruise_alt,
            end_altitude=cruise_alt,
            target_speed=cruise_speed,
            target_heading=0.0,
        ),
        MissionPhase(
            name="Descent",
            phase_type=MissionPhaseType.DESCENT,
            end_altitude=50.0,
            target_speed=V_ref,
        ),
        MissionPhase(
            name="Approach",
            phase_type=MissionPhaseType.APPROACH,
            end_altitude=10.0,
            target_descent_rate=2.0,
            target_speed=V_ref,
            target_heading=0.0,
        ),
        MissionPhase(
            name="Flare",
            phase_type=MissionPhaseType.LANDING_FLARE,
            end_altitude=0.0,
            target_descent_rate=0.5,
            target_heading=0.0,
        ),
        MissionPhase(
            name="Landing Roll",
            phase_type=MissionPhaseType.LANDING_ROLL,
            end_speed=2.0,
            target_heading=0.0,
        ),
    ]
    
    mission = MissionProfile(
        name="Full Comparison Mission",
        phases=phases,
        initial_altitude=0.0,
        initial_speed=0.0,
        initial_heading=0.0,
        initial_SOC=1.0,
        T_ambient=15.0,
        pressure_altitude=0.0,
        wind_speed=0.0,
        wind_direction=0.0,
        runway_heading=0.0,
        runway_length=500.0,
        surface_friction=mu_roll,
    )
    
    # Configure simulator
    config = SimulationConfig(
        dt=0.02,
        dt_output=1.0,  # Output every second
        dynamics_mode="3dof",
        mu_roll=mu_roll,
        alpha_ground_deg=5.0,
        V_rot_factor=1.1,
        verbose=True,
    )
    
    # Create simulator
    simulator = MissionSimulator(
        config=config,
        aero_model=aero_model,
        mass_kg=mass_kg,
        wing_area_m2=wing_area_m2,
        wingspan_m=wing_project.planform.actual_span(),
        CL_max=CL_max,
        max_thrust_N=100.0,
        battery_capacity_Wh=500.0,
    )
    
    # Run simulation
    print("\n" + "-" * 70)
    print("SIMULATING MISSION...")
    print("-" * 70)
    
    result = simulator.simulate_mission(mission)
    
    # Define expected behavior for each phase
    expected_behaviors = {
        "Ground Idle": {
            'min_duration': 4.9,
            'max_duration': 5.1,
            'altitude_change': 'stable',
            'speed_change': 'stable',
            'max_speed': 1.0,
            'uses_energy': False,
        },
        "Takeoff Roll": {
            'min_duration': 0.1,
            'max_duration': 30.0,
            'altitude_change': 'stable',
            'speed_change': 'increase',
            'max_speed': 30.0,  # Should not exceed reasonable takeoff speed
        },
        "Rotation": {
            'min_duration': 1.0,
            'max_duration': 10.0,
            'altitude_change': 'increase',
            'max_speed': 50.0,
        },
        "Climb": {
            'max_duration': 300.0,
            'altitude_change': 'increase',
            'max_speed': 60.0,  # Reasonable climb speed
        },
        "Cruise": {
            'min_duration': cruise_duration_s * 0.9,
            'max_duration': cruise_duration_s * 1.5,
            'altitude_change': 'stable',
            'speed_change': 'stable',
            'max_speed': 40.0,  # Cruise speed should be stable
        },
        "Descent": {
            'max_duration': 300.0,
            'altitude_change': 'decrease',
            'max_speed': 50.0,
        },
        "Approach": {
            'max_duration': 120.0,
            'altitude_change': 'decrease',
            'max_speed': 35.0,
        },
        "Flare": {
            'max_duration': 30.0,
            'altitude_change': 'decrease',
            'max_speed': 30.0,
        },
        "Landing Roll": {
            'max_duration': 60.0,
            'altitude_change': 'stable',
            'speed_change': 'decrease',
            'max_speed': 30.0,
        },
    }
    
    # Validate each phase
    validations = []
    for phase_result in result.phases:
        expected = expected_behaviors.get(phase_result.name, {})
        validation = validate_phase(phase_result, expected)
        validations.append(validation)
    
    return result, validations


def print_mission_summary(result: MissionResult, validations: List[PhaseValidation]):
    """Print mission summary with phase-by-phase validation."""
    
    print("\n" + "=" * 70)
    print("MISSION PHASE VALIDATION RESULTS")
    print("=" * 70)
    
    for v in validations:
        print(v)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL MISSION SUMMARY")
    print("=" * 70)
    
    total_passed = sum(1 for v in validations if v.all_passed)
    total_phases = len(validations)
    
    print(f"\n  Total Time:     {result.total_time_s:.1f}s")
    print(f"  Total Distance: {result.total_distance_m:.1f}m")
    print(f"  Total Energy:   {result.total_energy_Wh:.2f}Wh")
    print(f"  Final SOC:      {result.final_SOC*100:.1f}%")
    print(f"  Max Altitude:   {result.max_altitude_m:.1f}m")
    print(f"  Max Speed:      {result.max_speed_m_s:.1f}m/s")
    print(f"  Success:        {result.success}")
    if result.failure_reason:
        print(f"  Failure:        {result.failure_reason}")
    
    print(f"\n  Phase Validation: {total_passed}/{total_phases} phases passed all checks")
    
    # List failed checks
    failed_phases = [v for v in validations if not v.all_passed]
    if failed_phases:
        print("\n  FAILED PHASES:")
        for v in failed_phases:
            print(f"    - {v.phase_name}:")
            for check_name, (passed, msg) in v.checks.items():
                if not passed:
                    print(f"        {check_name}: {msg}")
    else:
        print("\n  [OK] All phases passed validation!")
    
    return total_passed == total_phases


def main():
    """Main comparison function."""
    print("=" * 70)
    print("Full Mission Phase Comparison Test")
    print("=" * 70)
    
    # Load IntendedValidation.json
    validation_path = PROJECT_ROOT / "IntendedValidation.json"
    if not validation_path.exists():
        print(f"ERROR: {validation_path} not found")
        return False
    
    print(f"\nLoading: {validation_path}")
    
    with open(validation_path) as f:
        data = json.load(f)
    
    project = Project.from_dict(data)
    wing_project = project.wing
    
    # Parameters
    mass_kg = data["wing"]["twist_trim"]["gross_takeoff_weight_kg"]  # 12.0
    wing_area_m2 = data["wing"]["planform"]["wing_area_m2"]  # 1.25
    CL_max = data["mission"]["aero_config"]["cl_max"]  # 1.6
    mu_roll = data["mission"]["aero_config"]["mu_roll"]  # 0.03
    
    # Mission parameters
    cruise_speed = 20.0  # m/s
    cruise_alt = 30.0    # m
    cruise_duration_s = 30.0  # 30 second cruise
    
    print(f"\nConfiguration:")
    print(f"  Mass: {mass_kg} kg")
    print(f"  Wing Area: {wing_area_m2} m²")
    print(f"  CL_max: {CL_max}")
    print(f"  mu_roll: {mu_roll}")
    print(f"  Cruise Speed: {cruise_speed} m/s")
    print(f"  Cruise Altitude: {cruise_alt} m")
    print(f"  Cruise Duration: {cruise_duration_s} s")
    
    # Run full mission
    result, validations = run_full_mission_comparison(
        wing_project=wing_project,
        mass_kg=mass_kg,
        wing_area_m2=wing_area_m2,
        CL_max=CL_max,
        mu_roll=mu_roll,
        cruise_speed=cruise_speed,
        cruise_alt=cruise_alt,
        cruise_duration_s=cruise_duration_s,
    )
    
    # Print results
    all_passed = print_mission_summary(result, validations)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
