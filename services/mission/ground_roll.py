"""
Ground Roll Physics Module

Implements ground roll equations for takeoff and landing, including:
- Takeoff roll with headwind effects
- Rotation and liftoff detection
- Landing roll with braking
- Ground effect on aerodynamics

This module provides accurate ground phase physics per Section 4
of the 3DOF_MISSION_DYNAMICS_SPEC.md.

Key Features:
    - 1-DOF along-runway dynamics with lift reducing normal force
    - Headwind/tailwind effects on effective airspeed
    - Rolling and braking friction models
    - Canonical liftoff condition (N <= N_threshold OR L >= 0.95*W)
    - Landing flare and touchdown detection

References:
    - SPECS/3DOF_MISSION_DYNAMICS_SPEC.md Section 4

Example:
    >>> from services.mission.ground_roll import compute_ground_roll_derivatives
    >>> state = {'V_ground': 10, 'ground_distance': 0, 'SOC': 1.0}
    >>> derivs = compute_ground_roll_derivatives(state, controls, config, headwind=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Any
import numpy as np

from .dynamics_3dof import AircraftConfig3DOF, get_density


@dataclass
class GroundRollConfig:
    """
    Ground roll physics configuration.
    
    Per Section 4 and Section 11.1 of the spec.
    
    Attributes:
        mu_roll: Rolling friction coefficient (typical 0.02-0.05 for grass)
        mu_brake: Braking friction coefficient (typical 0.3-0.5)
        alpha_ground_deg: Ground attitude (nose-up angle on gear) [deg]
        V_rot_factor: V_rot = V_rot_factor * V_stall
        flare_height_m: Flare trigger height AGL [m]
        liftoff_N_threshold_fraction: N <= threshold*W triggers liftoff
        
    Example:
        >>> config = GroundRollConfig(
        ...     mu_roll=0.04,
        ...     mu_brake=0.40,
        ...     alpha_ground_deg=5.0,
        ... )
    """
    mu_roll: float = 0.04
    mu_brake: float = 0.40
    alpha_ground_deg: float = 5.0
    V_rot_factor: float = 1.1
    flare_height_m: float = 7.0
    liftoff_N_threshold_fraction: float = 0.1
    V_stop_threshold: float = 2.0  # Landing roll complete when V < this


def compute_V_stall(
    config: AircraftConfig3DOF,
    rho: float = 1.225,
) -> float:
    """
    Compute stall speed.
    
    Args:
        config: Aircraft configuration
        rho: Air density [kg/m^3]
        
    Returns:
        Stall speed [m/s]
    """
    g = 9.81
    W = config.mass_kg * g
    V_stall = np.sqrt(2 * W / (rho * config.wing_area_m2 * config.CL_max))
    return float(V_stall)


def compute_V_rot(
    config: AircraftConfig3DOF,
    ground_config: GroundRollConfig,
    rho: float = 1.225,
    V_rot_override: Optional[float] = None,
) -> float:
    """
    Compute rotation speed.
    
    Per Section 4.2 priority order:
    1. Use V_rot_override (from phase.end_speed) if provided
    2. Else compute from V_rot_factor * V_stall
    
    Args:
        config: Aircraft configuration
        ground_config: Ground roll configuration
        rho: Air density [kg/m^3]
        V_rot_override: Override rotation speed [m/s]
        
    Returns:
        Rotation speed [m/s]
    """
    if V_rot_override is not None:
        return V_rot_override
    
    V_stall = compute_V_stall(config, rho)
    return ground_config.V_rot_factor * V_stall


def check_liftoff_condition(
    V_air: float,
    V_rot: float,
    lift: float,
    weight: float,
    normal_force: float,
    liftoff_N_threshold_fraction: float = 0.1,
) -> Tuple[bool, str]:
    """
    Check if liftoff conditions are met.
    
    This is the CANONICAL liftoff condition per Section 4.2 of the spec.
    
    Primary Condition - Transition to flight when BOTH are satisfied:
        V_air >= V_rot AND N <= N_liftoff_threshold
    
    Alternative Condition (simpler implementations):
        V_air >= V_rot AND L >= 0.95 * W
    
    Args:
        V_air: Effective airspeed (ground speed + headwind) [m/s]
        V_rot: Rotation speed [m/s]
        lift: Current lift force [N]
        weight: Aircraft weight [N]
        normal_force: Normal force on gear [N]
        liftoff_N_threshold_fraction: N_threshold = fraction * W
        
    Returns:
        Tuple of (liftoff_occurred, reason)
    """
    N_threshold = liftoff_N_threshold_fraction * weight
    
    if V_air < V_rot:
        return False, "below_rotation_speed"
    
    # Primary condition: gear essentially unloaded
    if normal_force <= N_threshold:
        return True, "gear_unloaded"
    
    # Alternative condition: sufficient lift
    if lift >= 0.95 * weight:
        return True, "sufficient_lift"
    
    return False, "waiting_for_liftoff"


def check_touchdown_condition(
    altitude: float,
    gamma_deg: float,
    V_vertical: float,
) -> bool:
    """
    Check if touchdown conditions are met.
    
    Per Section 9.1, touchdown occurs when:
        h <= 0 AND V_vertical <= 0 (descending or level)
    
    Args:
        altitude: Altitude AGL [m]
        gamma_deg: Flight path angle [deg]
        V_vertical: Vertical speed [m/s] (positive = climbing)
        
    Returns:
        True if touchdown should occur
    """
    return altitude <= 0.1 and V_vertical <= 0.1


def compute_ground_roll_derivatives(
    state: Dict[str, float],
    controls: Dict[str, float],
    config: AircraftConfig3DOF,
    ground_config: GroundRollConfig,
    headwind: float = 0.0,
    rho: float = 1.225,
    T_ambient: float = 15.0,
    aero_model: Optional[Callable] = None,
    propulsion_model: Optional[Callable] = None,
    is_landing: bool = False,
) -> Dict[str, float]:
    """
    Compute ground roll state derivatives.
    
    Implements Section 4.1 (takeoff) and Section 4.3 (landing) physics.
    
    Ground roll uses 1-DOF along-runway dynamics:
        V_air = V_ground + headwind
        L = q * S * CL_ground(alpha_ground)
        N = max(0, W - L)
        F_friction = mu * N
        dV_ground/dt = (T - D - F_friction) / m  [takeoff]
        dV_ground/dt = -(D + F_friction) / m     [landing, T=0]
    
    Args:
        state: Current state (V_ground, ground_distance, SOC, temps)
        controls: Control inputs (throttle, brake)
        config: Aircraft configuration
        ground_config: Ground roll configuration
        headwind: Headwind component [m/s] (positive = headwind)
        rho: Air density [kg/m^3]
        T_ambient: Ambient temperature [C]
        aero_model: Aerodynamic model callable
        propulsion_model: Propulsion model callable
        is_landing: True for landing roll (thrust = 0)
        
    Returns:
        Dictionary of state derivatives and auxiliary outputs
    """
    g = 9.81
    
    # Unpack state
    V_ground = max(0.0, state.get('V_ground', state.get('airspeed', 0.0)))
    SOC = state.get('SOC', 1.0)
    T_motor = state.get('T_motor', T_ambient)
    T_esc = state.get('T_esc', T_ambient)
    T_battery = state.get('T_battery', T_ambient)
    heading = state.get('heading', 0.0)
    
    # Unpack controls
    throttle = np.clip(controls.get('throttle', 0.0), 0.0, 1.0)
    brake = np.clip(controls.get('brake', 0.0), 0.0, 1.0)
    
    # Effective airspeed includes headwind
    V_air = max(0.1, V_ground + headwind)
    
    # Ground attitude (fixed on gear)
    alpha_ground_deg = ground_config.alpha_ground_deg
    alpha_ground_rad = np.radians(alpha_ground_deg)
    
    # Aerodynamics at ground attitude
    if aero_model is not None:
        aero_result = aero_model(
            alpha=alpha_ground_deg,
            beta=0.0,
            airspeed=V_air,
            p=0.0, q=0.0, r=0.0,
            elevator=controls.get('elevator', 0.0),
            aileron=controls.get('aileron', 0.0),
            rudder=controls.get('rudder', 0.0),
        )
        if isinstance(aero_result, dict):
            CL = aero_result.get('CL', 0.0)
            CD = aero_result.get('CD', config.CD0)
        else:
            CL, CD = aero_result
    else:
        # Fallback parabolic polar
        CL = config.CL0 + config.CLa * alpha_ground_rad
        CD = config.CD0 + config.k * CL**2
    
    # Ground effect is very strong at h~0
    # Apply maximum ground effect reduction
    phi_GE = 0.5  # At h/b ~ 0.05, induced drag reduced ~50%
    CD_induced = config.k * CL**2
    CD_parasitic = max(0.0, CD - CD_induced)
    CD_GE = CD_parasitic + CD_induced * phi_GE
    
    # Forces
    q = 0.5 * rho * V_air**2
    S = config.wing_area_m2
    L = q * S * CL
    D = q * S * CD_GE
    
    W = config.mass_kg * g
    m = config.mass_kg
    
    # Normal force (lift reduces ground contact)
    N = max(0.0, W - L)
    
    # Friction (combined rolling and braking)
    mu = ground_config.mu_roll * (1.0 - brake) + ground_config.mu_brake * brake
    F_friction = mu * N
    
    # Propulsion
    T = 0.0
    power = 0.0
    dSOC_dt = 0.0
    dT_motor_dt = 0.0
    dT_esc_dt = 0.0
    dT_battery_dt = 0.0
    
    if not is_landing:  # Takeoff: use propulsion
        if propulsion_model is not None:
            prop_result = propulsion_model(
                throttle=throttle,
                V_freestream=V_air,
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
            power = T * V_air / 0.55 if V_air > 0 else 0.0
            battery_capacity_Wh = 100.0
            dSOC_dt = -power / (battery_capacity_Wh * 3600.0)
    
    # Ground roll acceleration (Section 4.1 & 4.3)
    if is_landing:
        # Landing: T = 0, decelerate
        accel = -(D + F_friction) / m
    else:
        # Takeoff: accelerate
        accel = (T - D - F_friction) / m
    
    # Clamp to prevent reverse motion on ground
    if V_ground <= 0 and accel < 0:
        accel = 0.0
    
    # Position derivatives
    heading_rad = np.radians(heading)
    dx_dt = V_ground * np.sin(heading_rad)
    dy_dt = V_ground * np.cos(heading_rad)
    ds_dt = V_ground
    
    return {
        # Velocity derivatives
        'V_ground': accel,
        'airspeed': accel,  # On ground, airspeed follows ground speed + headwind change
        
        # Position derivatives
        'x': dx_dt,
        'y': dy_dt,
        'ground_distance': ds_dt,
        
        # Energy/thermal derivatives
        'SOC': dSOC_dt,
        'T_motor': dT_motor_dt,
        'T_esc': dT_esc_dt,
        'T_battery': dT_battery_dt,
        
        # Ground state (no change)
        'altitude': 0.0,
        'gamma': 0.0,
        'track': 0.0,
        
        # Auxiliary outputs
        'power': power,
        'thrust': T,
        'drag': D,
        'lift': L,
        'normal_force': N,
        'friction': F_friction,
        'CL': CL,
        'CD': CD_GE,
        'V_air': V_air,
        'headwind': headwind,
        'accel': accel,
    }


def integrate_ground_roll(
    state: Dict[str, float],
    derivatives: Dict[str, float],
    dt: float,
    runway_heading_deg: float = 0.0,
) -> Dict[str, float]:
    """
    Integrate ground roll state for one timestep.
    
    Args:
        state: Current state dictionary
        derivatives: State derivatives from compute_ground_roll_derivatives
        dt: Time step [s]
        runway_heading_deg: Runway heading [deg]
        
    Returns:
        Updated state dictionary
    """
    new_state = state.copy()
    
    # Integrate velocity
    dV = derivatives.get('V_ground', 0.0) * dt
    new_state['V_ground'] = max(0.0, state.get('V_ground', 0.0) + dV)
    new_state['airspeed'] = derivatives.get('V_air', new_state['V_ground'])
    
    # Integrate position
    new_state['x'] = state.get('x', 0.0) + derivatives['x'] * dt
    new_state['y'] = state.get('y', 0.0) + derivatives['y'] * dt
    new_state['ground_distance'] = state.get('ground_distance', 0.0) + derivatives['ground_distance'] * dt
    
    # Integrate energy/thermal
    new_state['SOC'] = np.clip(state.get('SOC', 1.0) + derivatives['SOC'] * dt, 0.0, 1.0)
    new_state['T_motor'] = state.get('T_motor', 25.0) + derivatives['T_motor'] * dt
    new_state['T_esc'] = state.get('T_esc', 25.0) + derivatives['T_esc'] * dt
    new_state['T_battery'] = state.get('T_battery', 25.0) + derivatives['T_battery'] * dt
    
    # Ground state
    new_state['altitude'] = 0.0
    new_state['gamma'] = 0.0
    new_state['track'] = runway_heading_deg
    new_state['heading'] = runway_heading_deg
    new_state['on_ground'] = True
    new_state['climb_rate'] = 0.0
    
    return new_state


def transition_to_flight(
    state: Dict[str, float],
    alpha_ground_deg: float,
    runway_heading_deg: float,
) -> Dict[str, float]:
    """
    Transition state from ground to flight mode at liftoff.
    
    Per Section 4.2 and Section 9.2, initializes flight state:
        gamma = alpha_ground_deg (initial flight path = ground pitch)
        track = runway_heading_deg
        h = 0.1 (just above ground)
        V = V_air (airspeed)
        on_ground = False
    
    Args:
        state: Ground roll state at liftoff
        alpha_ground_deg: Ground attitude [deg]
        runway_heading_deg: Runway heading [deg]
        
    Returns:
        Flight state dictionary
    """
    flight_state = state.copy()
    
    flight_state['gamma'] = alpha_ground_deg
    flight_state['track'] = runway_heading_deg
    flight_state['heading'] = runway_heading_deg
    flight_state['altitude'] = 0.1
    flight_state['on_ground'] = False
    
    # Airspeed is already set (V_air from ground roll)
    # Climb rate from initial gamma
    gamma_rad = np.radians(alpha_ground_deg)
    flight_state['climb_rate'] = flight_state['airspeed'] * np.sin(gamma_rad)
    
    return flight_state


def transition_to_ground(
    state: Dict[str, float],
    runway_heading_deg: float,
) -> Dict[str, float]:
    """
    Transition state from flight to ground mode at touchdown.
    
    Per Section 9.2 Flight → Ground transition:
        h = 0.0
        gamma = 0.0
        V_ground = V * cos(gamma_pre_touchdown)
        on_ground = True
    
    Args:
        state: Flight state at touchdown
        runway_heading_deg: Runway heading [deg]
        
    Returns:
        Ground roll state dictionary
    """
    ground_state = state.copy()
    
    # Store pre-touchdown gamma for ground speed calculation
    gamma_pre = state.get('gamma', 0.0)
    gamma_rad = np.radians(gamma_pre)
    
    ground_state['altitude'] = 0.0
    ground_state['gamma'] = 0.0
    ground_state['track'] = runway_heading_deg
    ground_state['heading'] = runway_heading_deg
    ground_state['on_ground'] = True
    ground_state['climb_rate'] = 0.0
    
    # Ground speed from airspeed and pre-touchdown gamma
    ground_state['V_ground'] = max(0.0, state['airspeed'] * np.cos(gamma_rad))
    
    return ground_state


def estimate_takeoff_distance(
    config: AircraftConfig3DOF,
    ground_config: GroundRollConfig,
    thrust_N: float,
    rho: float = 1.225,
    headwind: float = 0.0,
) -> float:
    """
    Estimate takeoff ground roll distance analytically.
    
    Uses the approximation from Appendix A.3:
        S_g ≈ V_LOF^2 / (2 * a_avg)
        
    With headwind correction:
        S_g_wind ≈ S_g * ((V_LOF - V_w) / V_LOF)^2
    
    Args:
        config: Aircraft configuration
        ground_config: Ground roll configuration
        thrust_N: Average thrust [N]
        rho: Air density [kg/m^3]
        headwind: Headwind component [m/s]
        
    Returns:
        Estimated ground roll distance [m]
    """
    g = 9.81
    W = config.mass_kg * g
    m = config.mass_kg
    S = config.wing_area_m2
    
    # Liftoff speed
    V_stall = compute_V_stall(config, rho)
    V_LOF = ground_config.V_rot_factor * V_stall
    
    # Average acceleration (simplified)
    # At mid-roll speed, estimate forces
    V_mid = V_LOF / 2
    V_air_mid = V_mid + headwind
    
    # Simplified aero at ground attitude
    alpha_rad = np.radians(ground_config.alpha_ground_deg)
    CL = config.CL0 + config.CLa * alpha_rad
    CD = config.CD0 + config.k * CL**2
    
    q_mid = 0.5 * rho * V_air_mid**2
    L_mid = q_mid * S * CL
    D_mid = q_mid * S * CD
    N_mid = max(0, W - L_mid)
    F_friction_mid = ground_config.mu_roll * N_mid
    
    a_avg = (thrust_N - D_mid - F_friction_mid) / m
    a_avg = max(a_avg, 0.1)  # Prevent negative/zero
    
    # Ground roll distance (no wind)
    S_g = V_LOF**2 / (2 * a_avg)
    
    # Headwind correction
    if headwind > 0:
        V_ground_LOF = max(V_LOF - headwind, 0.1)
        S_g_wind = S_g * (V_ground_LOF / V_LOF)**2
        return float(S_g_wind)
    elif headwind < 0:  # Tailwind
        V_ground_LOF = V_LOF + abs(headwind)
        S_g_wind = S_g * (V_ground_LOF / V_LOF)**2
        return float(S_g_wind)
    
    return float(S_g)


def estimate_landing_distance(
    config: AircraftConfig3DOF,
    ground_config: GroundRollConfig,
    touchdown_speed: float,
    rho: float = 1.225,
    headwind: float = 0.0,
) -> float:
    """
    Estimate landing roll distance analytically.
    
    Args:
        config: Aircraft configuration
        ground_config: Ground roll configuration
        touchdown_speed: Touchdown ground speed [m/s]
        rho: Air density [kg/m^3]
        headwind: Headwind component [m/s]
        
    Returns:
        Estimated landing roll distance [m]
    """
    g = 9.81
    W = config.mass_kg * g
    m = config.mass_kg
    S = config.wing_area_m2
    
    # Average braking
    V_mid = touchdown_speed / 2
    V_air_mid = V_mid + headwind
    
    alpha_rad = np.radians(ground_config.alpha_ground_deg)
    CL = config.CL0 + config.CLa * alpha_rad
    CD = config.CD0 + config.k * CL**2
    
    q_mid = 0.5 * rho * V_air_mid**2
    L_mid = q_mid * S * CL
    D_mid = q_mid * S * CD
    N_mid = max(0, W - L_mid)
    
    # Assume full braking
    F_friction_mid = ground_config.mu_brake * N_mid
    
    a_decel = (D_mid + F_friction_mid) / m
    a_decel = max(a_decel, 0.1)
    
    # Landing roll distance
    S_land = touchdown_speed**2 / (2 * a_decel)
    
    return float(S_land)
