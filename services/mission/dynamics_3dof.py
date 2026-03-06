"""
3-DOF Flight Dynamics Module

Implements the core 3-DOF (speed-gamma-track) equations of motion for
mission simulation. This module provides accurate mission-level physics
within +/-2% of 6-DOF results at significantly lower computational cost.

Key Features:
    - Speed-gamma-track flight dynamics (Section 3 of 3DOF_MISSION_DYNAMICS_SPEC.md)
    - Coordinated turn limits via bank angle cap and optional load factor limit
    - Wind model integration for ground speed and position
    - Ground effect modeling for takeoff/landing accuracy
    - Full propulsion integration support

References:
    - SPECS/3DOF_MISSION_DYNAMICS_SPEC.md Sections 2-3, 5-6, 8

Example:
    >>> from services.mission.dynamics_3dof import compute_3dof_derivatives, AircraftConfig3DOF
    >>> config = AircraftConfig3DOF(mass_kg=3.0, wing_area_m2=0.5, wingspan_m=2.0)
    >>> state = {'x': 0, 'y': 0, 'altitude': 100, 'airspeed': 18, 'gamma': 5, 'track': 0}
    >>> derivs = compute_3dof_derivatives(state, controls, config, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable, Any
import numpy as np


@dataclass
class AircraftConfig3DOF:
    """
    Aircraft parameters for 3-DOF simulation.
    
    Defines the physical and aerodynamic properties required for
    3-DOF flight dynamics computation.
    
    Attributes:
        mass_kg: Aircraft mass [kg]
        wing_area_m2: Reference wing area [m^2]
        wingspan_m: Wing span [m]
        CL_max: Maximum lift coefficient (for stall calculations)
        CD0: Zero-lift drag coefficient
        k: Induced drag factor (CD = CD0 + k*CL^2)
        CLa: Lift curve slope [1/rad]
        CL0: Zero-alpha lift coefficient
        max_bank_deg: Maximum allowed bank angle [deg]
        max_load_factor: Maximum allowed load factor (optional)
        thrust_incidence_deg: Thrust line incidence angle relative to body x-axis [deg]
    
    Example:
        >>> config = AircraftConfig3DOF(
        ...     mass_kg=3.0,
        ...     wing_area_m2=0.5,
        ...     wingspan_m=2.0,
        ...     CL_max=1.4,
        ...     CD0=0.025,
        ...     k=0.04,
        ... )
    """
    mass_kg: float
    wing_area_m2: float
    wingspan_m: float
    CL_max: float = 1.4
    CD0: float = 0.025
    k: float = 0.04
    CLa: float = 5.5  # [1/rad]
    CL0: float = 0.3
    max_bank_deg: float = 45.0
    max_load_factor: Optional[float] = None
    thrust_incidence_deg: float = 0.0
    
    @property
    def max_bank_rad(self) -> float:
        """Maximum bank angle in radians."""
        return np.radians(self.max_bank_deg)
    
    @property
    def thrust_incidence_rad(self) -> float:
        """Thrust incidence in radians."""
        return np.radians(self.thrust_incidence_deg)
    
    def get_stall_speed(self, rho: float = 1.225) -> float:
        """
        Compute stall speed for current configuration.
        
        Args:
            rho: Air density [kg/m^3]
            
        Returns:
            Stall speed [m/s]
        """
        g = 9.81
        W = self.mass_kg * g
        V_stall = np.sqrt(2 * W / (rho * self.wing_area_m2 * self.CL_max))
        return float(V_stall)


def apply_bank_limiting(
    bank_cmd_rad: float,
    max_bank_rad: float,
    max_load_factor: Optional[float] = None,
) -> float:
    """
    Apply bank angle and load factor limiting.
    
    Per Section 5 of the spec, bank angle is limited by:
    1. Phase-defined bank_max
    2. Optional load factor limit (n_max)
    
    Args:
        bank_cmd_rad: Commanded bank angle [rad]
        max_bank_rad: Maximum allowed bank angle [rad]
        max_load_factor: Maximum load factor (optional)
        
    Returns:
        Limited bank angle [rad]
    """
    # First, apply bank angle limit
    bank = np.clip(bank_cmd_rad, -max_bank_rad, max_bank_rad)
    
    # Then apply load factor limit if specified
    if max_load_factor is not None and max_load_factor > 1.0:
        max_bank_from_n = np.arccos(1.0 / max_load_factor)
        bank = np.clip(bank, -max_bank_from_n, max_bank_from_n)
    
    return float(bank)


def apply_ground_effect(
    CL: float,
    CD: float,
    CD_induced: Optional[float],
    altitude_m: float,
    wingspan_m: float,
) -> Tuple[float, float]:
    """
    Apply ground effect correction to aerodynamic coefficients.
    
    Ground effect reduces induced drag when h < wingspan. Uses the
    Prandtl ground effect model per Section 8.2 of the spec.
    
    Args:
        CL: Lift coefficient
        CD: Total drag coefficient
        CD_induced: Induced drag component (if None, estimated from k*CL^2)
        altitude_m: Altitude above ground [m]
        wingspan_m: Aircraft wingspan [m]
        
    Returns:
        Tuple of (CL_corrected, CD_corrected)
    """
    h_b = altitude_m / max(wingspan_m, 0.1)
    
    if h_b >= 1.0:
        return CL, CD
    
    # Prandtl ground effect factor
    # phi_GE -> 0 at ground, -> 1 at h/b = 1
    phi_GE = (16 * h_b)**2 / (1 + (16 * h_b)**2)
    
    # Estimate induced drag if not provided
    if CD_induced is None:
        # Assume parasitic drag is the smaller component
        # CD_i = k * CL^2 typically dominates at high CL (low speed)
        # This is a rough estimate; prefer passing CD_i from aero model
        k_estimate = 0.04
        CD_induced = k_estimate * CL**2
    
    CD_parasitic = max(0.0, CD - CD_induced)
    
    # Ground effect corrections
    CD_i_GE = CD_induced * phi_GE
    CL_GE = CL * (1 + 0.1 * (1 - phi_GE))  # Slight lift increase
    CD_GE = CD_parasitic + CD_i_GE
    
    return float(CL_GE), float(CD_GE)


def compute_wind_components(
    wind_speed: float,
    wind_direction_from_deg: float,
) -> Tuple[float, float]:
    """
    Compute wind vector components in ENU frame.
    
    Per Section 6.2 of the spec, wind_direction is the direction
    wind is FROM (meteorological convention).
    
    Args:
        wind_speed: Wind speed magnitude [m/s]
        wind_direction_from_deg: Direction wind is FROM [deg]
        
    Returns:
        Tuple of (wind_east, wind_north) components [m/s]
    """
    # Wind TO direction (direction wind is blowing toward)
    wind_to_rad = np.radians(wind_direction_from_deg + 180) % (2 * np.pi)
    
    # Wind components in ENU frame
    wind_east = wind_speed * np.sin(wind_to_rad)
    wind_north = wind_speed * np.cos(wind_to_rad)
    
    return float(wind_east), float(wind_north)


def compute_headwind(
    wind_speed: float,
    wind_direction_from_deg: float,
    runway_heading_deg: float,
) -> float:
    """
    Compute headwind component along runway.
    
    Per Section 6.3 of the spec, headwind is positive when
    wind is FROM the runway heading direction.
    
    Args:
        wind_speed: Wind speed magnitude [m/s]
        wind_direction_from_deg: Direction wind is FROM [deg]
        runway_heading_deg: Runway heading [deg]
        
    Returns:
        Headwind component [m/s] (positive = headwind, negative = tailwind)
    """
    wind_from_rad = np.radians(wind_direction_from_deg)
    runway_heading_rad = np.radians(runway_heading_deg)
    
    headwind = wind_speed * np.cos(wind_from_rad - runway_heading_rad)
    return float(headwind)


def get_density(altitude_m: float, pressure_altitude_offset: float = 0.0) -> float:
    """
    Compute air density from exponential atmosphere model.
    
    Args:
        altitude_m: Altitude AGL [m]
        pressure_altitude_offset: Pressure altitude offset [m]
        
    Returns:
        Air density [kg/m^3]
    """
    rho0 = 1.225
    scale_height = 8500.0
    effective_alt = max(0.0, altitude_m + pressure_altitude_offset)
    return rho0 * np.exp(-effective_alt / scale_height)


def compute_3dof_derivatives(
    state: Dict[str, float],
    controls: Dict[str, float],
    config: AircraftConfig3DOF,
    aero_model: Optional[Callable] = None,
    propulsion_model: Optional[Callable] = None,
    wind_east: float = 0.0,
    wind_north: float = 0.0,
    T_ambient: float = 15.0,
    pressure_altitude_offset: float = 0.0,
) -> Dict[str, float]:
    """
    Compute 3-DOF state derivatives.
    
    Implements the full 3-DOF equations of motion per Sections 3.1-3.4
    of the 3DOF_MISSION_DYNAMICS_SPEC.md.
    
    State Vector (Section 2.1):
        x, y: Position in ENU frame [m]
        altitude: Altitude AGL [m]
        airspeed: True airspeed [m/s]
        gamma: Flight path angle [deg]
        track: Track angle [deg] (0 = North, 90 = East)
        SOC: Battery state of charge [0-1]
        T_motor, T_esc, T_battery: Component temperatures [C]
        ground_distance: Cumulative ground track [m]
    
    Args:
        state: Current state dictionary
        controls: Control inputs (throttle, alpha_cmd, bank_cmd, brake)
        config: Aircraft configuration
        aero_model: Callable returning (CL, CD) or dict with 'CL', 'CD'
        propulsion_model: Callable returning thrust and derivatives
        wind_east: Wind component blowing East [m/s]
        wind_north: Wind component blowing North [m/s]
        T_ambient: Ambient temperature [C]
        pressure_altitude_offset: Pressure altitude offset [m]
        
    Returns:
        Dictionary of state derivatives
    """
    g = 9.81
    V_min = 2.0  # Minimum speed to avoid singularities
    
    # Unpack state
    altitude = state.get('altitude', 0.0)
    V = max(state.get('airspeed', V_min), V_min)
    gamma_deg = state.get('gamma', 0.0)
    track_deg = state.get('track', 0.0)
    SOC = state.get('SOC', 1.0)
    T_motor = state.get('T_motor', T_ambient)
    T_esc = state.get('T_esc', T_ambient)
    T_battery = state.get('T_battery', T_ambient)
    
    # Unpack controls
    throttle = np.clip(controls.get('throttle', 0.0), 0.0, 1.0)
    alpha_cmd_deg = controls.get('alpha_cmd', 5.0)
    bank_cmd_deg = controls.get('bank_cmd', 0.0)
    
    # Convert angles to radians
    gamma_rad = np.radians(gamma_deg)
    track_rad = np.radians(track_deg)
    alpha_rad = np.radians(alpha_cmd_deg)
    bank_cmd_rad = np.radians(bank_cmd_deg)
    
    # Apply bank limiting (Section 5)
    bank_rad = apply_bank_limiting(
        bank_cmd_rad,
        config.max_bank_rad,
        config.max_load_factor,
    )
    
    # Environment
    rho = get_density(altitude, pressure_altitude_offset)
    
    # Aerodynamics (Section 8)
    if aero_model is not None:
        aero_result = aero_model(
            alpha=alpha_cmd_deg,
            beta=0.0,
            airspeed=V,
            p=0.0, q=0.0, r=0.0,
            elevator=controls.get('elevator', 0.0),
            aileron=controls.get('aileron', 0.0),
            rudder=controls.get('rudder', 0.0),
        )
        if isinstance(aero_result, dict):
            CL = aero_result.get('CL', 0.0)
            CD = aero_result.get('CD', config.CD0)
            CD_i = aero_result.get('CD_i', None)
        else:
            CL, CD = aero_result
            CD_i = None
    else:
        # Fallback parabolic polar
        CL = config.CL0 + config.CLa * alpha_rad
        CD = config.CD0 + config.k * CL**2
        CD_i = config.k * CL**2
    
    # Clamp CL to physical limits
    CL = float(np.clip(CL, -config.CL_max, config.CL_max))
    CD = float(max(CD, 0.0))
    
    # Apply ground effect (Section 8.2)
    if altitude < config.wingspan_m:
        CL, CD = apply_ground_effect(CL, CD, CD_i, altitude, config.wingspan_m)
    
    # Dynamic pressure and forces
    q = 0.5 * rho * V**2
    S = config.wing_area_m2
    L = q * S * CL
    D = q * S * CD
    
    # Propulsion (Section 7)
    T = 0.0
    power = 0.0
    dSOC_dt = 0.0
    dT_motor_dt = 0.0
    dT_esc_dt = 0.0
    dT_battery_dt = 0.0
    
    if propulsion_model is not None:
        prop_result = propulsion_model(
            throttle=throttle,
            V_freestream=V,
            rho=rho,
            SOC=SOC,
            temperatures={'motor': T_motor, 'esc': T_esc, 'battery': T_battery},
            T_ambient=T_ambient,
        )
        if isinstance(prop_result, dict):
            T = prop_result.get('thrust_total', 0.0)
            power = prop_result.get('power_battery', 0.0)
            dSOC_dt = prop_result.get('dSOC_dt', 0.0)
            dT_motor_dt = prop_result.get('dT_motor_dt', 0.0)
            dT_esc_dt = prop_result.get('dT_esc_dt', 0.0)
            dT_battery_dt = prop_result.get('dT_battery_dt', 0.0)
        else:
            T = prop_result
    else:
        # Fallback simple propulsion
        T_max = 15.0
        T = throttle * T_max
        power = T * V / 0.55 if V > 0 else 0.0
        battery_capacity_Wh = 100.0
        dSOC_dt = -power / (battery_capacity_Wh * 3600.0)
    
    # Mass and weight
    m = config.mass_kg
    W = m * g
    
    # Account for thrust incidence (Section 3.2 note)
    alpha_thrust = alpha_rad + config.thrust_incidence_rad
    
    # Dynamic equations (Section 3.2)
    # dV/dt = (T*cos(alpha) - D) / m - g*sin(gamma)
    dV_dt = (T * np.cos(alpha_thrust) - D) / m - g * np.sin(gamma_rad)
    
    # dgamma/dt = (T*sin(alpha) + L*cos(bank) - W*cos(gamma)) / (m*V)
    dgamma_dt_rad = (T * np.sin(alpha_thrust) + L * np.cos(bank_rad) - W * np.cos(gamma_rad)) / (m * V)
    dgamma_dt = np.degrees(dgamma_dt_rad)
    
    # dchi/dt = (L*sin(bank)) / (m*V*cos(gamma)) - force-based form for all flight conditions
    cos_gamma = np.cos(gamma_rad)
    if abs(cos_gamma) > 0.01:  # Avoid singularity at 90 deg pitch
        dchi_dt_rad = (L * np.sin(bank_rad)) / (m * V * cos_gamma)
    else:
        dchi_dt_rad = 0.0
    dchi_dt = np.degrees(dchi_dt_rad)
    
    # Kinematic equations with wind (Section 3.1)
    # Ground velocity = Air velocity + Wind
    V_air_east = V * cos_gamma * np.sin(track_rad)
    V_air_north = V * cos_gamma * np.cos(track_rad)
    
    V_ground_east = V_air_east + wind_east
    V_ground_north = V_air_north + wind_north
    
    dx_dt = V_ground_east
    dy_dt = V_ground_north
    dh_dt = V * np.sin(gamma_rad)
    
    # Ground distance rate (always wind-affected, Section 6.4)
    V_ground = np.sqrt(V_ground_east**2 + V_ground_north**2)
    ds_ground_dt = V_ground
    
    # Load factor (for monitoring)
    n = L / (m * g)
    
    return {
        # Position derivatives
        'x': dx_dt,
        'y': dy_dt,
        'altitude': dh_dt,
        
        # Velocity derivatives
        'airspeed': dV_dt,
        'gamma': dgamma_dt,
        'track': dchi_dt,
        
        # Energy/thermal derivatives
        'SOC': dSOC_dt,
        'T_motor': dT_motor_dt,
        'T_esc': dT_esc_dt,
        'T_battery': dT_battery_dt,
        
        # Distance derivative
        'ground_distance': ds_ground_dt,
        
        # Auxiliary outputs (for monitoring/logging)
        'power': power,
        'thrust': T,
        'drag': D,
        'lift': L,
        'CL': CL,
        'CD': CD,
        'bank_used': np.degrees(bank_rad),
        'load_factor': n,
        'V_ground': V_ground,
        'V_air': V,
    }


def integrate_3dof_euler(
    state: Dict[str, float],
    derivatives: Dict[str, float],
    dt: float,
) -> Dict[str, float]:
    """
    Euler integration step for 3-DOF state.
    
    Args:
        state: Current state dictionary
        derivatives: State derivatives from compute_3dof_derivatives
        dt: Time step [s]
        
    Returns:
        Updated state dictionary
    """
    new_state = state.copy()
    
    # Integrate primary states
    state_keys = [
        'x', 'y', 'altitude', 'airspeed', 'gamma', 'track',
        'SOC', 'T_motor', 'T_esc', 'T_battery', 'ground_distance'
    ]
    
    for key in state_keys:
        if key in derivatives and key in new_state:
            new_state[key] = new_state[key] + derivatives[key] * dt
    
    # Enforce physical constraints
    new_state['airspeed'] = max(0.0, new_state['airspeed'])
    new_state['altitude'] = max(0.0, new_state['altitude'])
    new_state['SOC'] = np.clip(new_state['SOC'], 0.0, 1.0)
    new_state['track'] = new_state['track'] % 360.0
    
    # Update derived states
    gamma_rad = np.radians(new_state['gamma'])
    new_state['climb_rate'] = new_state['airspeed'] * np.sin(gamma_rad)
    new_state['heading'] = new_state['track']  # In no-wind, heading = track
    new_state['V_ground'] = derivatives.get('V_ground', new_state['airspeed'])
    
    return new_state


def integrate_3dof_rk4(
    state: Dict[str, float],
    controls: Dict[str, float],
    config: AircraftConfig3DOF,
    dt: float,
    aero_model: Optional[Callable] = None,
    propulsion_model: Optional[Callable] = None,
    wind_east: float = 0.0,
    wind_north: float = 0.0,
    T_ambient: float = 15.0,
    pressure_altitude_offset: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    RK4 integration step for 3-DOF state.
    
    Provides higher accuracy than Euler at 4x computational cost.
    
    Args:
        state: Current state dictionary
        controls: Control inputs
        config: Aircraft configuration
        dt: Time step [s]
        aero_model: Aerodynamic model callable
        propulsion_model: Propulsion model callable
        wind_east: Wind component blowing East [m/s]
        wind_north: Wind component blowing North [m/s]
        T_ambient: Ambient temperature [C]
        pressure_altitude_offset: Pressure altitude offset [m]
        
    Returns:
        Tuple of (new_state, auxiliary_outputs)
    """
    def derivs(s):
        return compute_3dof_derivatives(
            s, controls, config, aero_model, propulsion_model,
            wind_east, wind_north, T_ambient, pressure_altitude_offset,
        )
    
    state_keys = [
        'x', 'y', 'altitude', 'airspeed', 'gamma', 'track',
        'SOC', 'T_motor', 'T_esc', 'T_battery', 'ground_distance'
    ]
    
    # k1
    k1 = derivs(state)
    
    # k2
    s2 = state.copy()
    for key in state_keys:
        if key in k1 and key in s2:
            s2[key] = state[key] + 0.5 * dt * k1[key]
    k2 = derivs(s2)
    
    # k3
    s3 = state.copy()
    for key in state_keys:
        if key in k2 and key in s3:
            s3[key] = state[key] + 0.5 * dt * k2[key]
    k3 = derivs(s3)
    
    # k4
    s4 = state.copy()
    for key in state_keys:
        if key in k3 and key in s4:
            s4[key] = state[key] + dt * k3[key]
    k4 = derivs(s4)
    
    # Combine
    new_state = state.copy()
    for key in state_keys:
        if key in k1 and key in new_state:
            new_state[key] = state[key] + (dt / 6.0) * (
                k1[key] + 2*k2[key] + 2*k3[key] + k4[key]
            )
    
    # Enforce physical constraints
    new_state['airspeed'] = max(0.0, new_state['airspeed'])
    new_state['altitude'] = max(0.0, new_state['altitude'])
    new_state['SOC'] = np.clip(new_state['SOC'], 0.0, 1.0)
    new_state['track'] = new_state['track'] % 360.0
    
    # Update derived states
    gamma_rad = np.radians(new_state['gamma'])
    new_state['climb_rate'] = new_state['airspeed'] * np.sin(gamma_rad)
    new_state['heading'] = new_state['track']
    new_state['V_ground'] = k1.get('V_ground', new_state['airspeed'])
    new_state['on_ground'] = False
    
    return new_state, k1


def compute_turn_rate(
    bank_deg: float,
    airspeed: float,
    gamma_deg: float = 0.0,
    use_force_based: bool = True,
    lift: Optional[float] = None,
    mass_kg: Optional[float] = None,
) -> float:
    """
    Compute turn rate from bank angle.
    
    Args:
        bank_deg: Bank angle [deg]
        airspeed: Airspeed [m/s]
        gamma_deg: Flight path angle [deg]
        use_force_based: Use force-based form (recommended for non-level flight)
        lift: Lift force [N] (required if use_force_based=True)
        mass_kg: Aircraft mass [kg] (required if use_force_based=True)
        
    Returns:
        Turn rate [deg/s]
    """
    g = 9.81
    bank_rad = np.radians(bank_deg)
    gamma_rad = np.radians(gamma_deg)
    
    V = max(airspeed, 2.0)  # Avoid division by zero
    cos_gamma = np.cos(gamma_rad)
    
    if abs(cos_gamma) < 0.01:
        return 0.0
    
    if use_force_based and lift is not None and mass_kg is not None:
        # Force-based form (Section 3.2)
        chi_dot_rad = (lift * np.sin(bank_rad)) / (mass_kg * V * cos_gamma)
    else:
        # Simplified level-flight approximation
        chi_dot_rad = g * np.tan(bank_rad) / V
    
    return np.degrees(chi_dot_rad)


def compute_turn_radius(
    bank_deg: float,
    airspeed: float,
    gamma_deg: float = 0.0,
) -> float:
    """
    Compute turn radius from bank angle and speed.
    
    Args:
        bank_deg: Bank angle [deg]
        airspeed: Airspeed [m/s]
        gamma_deg: Flight path angle [deg]
        
    Returns:
        Turn radius [m] (inf for straight flight)
    """
    g = 9.81
    bank_rad = np.radians(bank_deg)
    gamma_rad = np.radians(gamma_deg)
    
    V = max(airspeed, 2.0)
    
    tan_bank = np.tan(bank_rad)
    if abs(tan_bank) < 0.001:
        return float('inf')
    
    # Level flight approximation
    R = V**2 / (g * abs(tan_bank))
    
    return R
