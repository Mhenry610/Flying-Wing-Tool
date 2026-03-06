"""
Integrated Propulsion System

Couples propeller, motor, ESC, battery, and thermal models into a single
differentiable interface for trajectory optimization and dynamics simulation.

Key Features:
    - Motor-propeller equilibrium solver (fixed-point iteration, AD-compatible)
    - Throttle-to-thrust mapping with efficiency tracking
    - Battery power and SOC dynamics
    - Thermal heat generation from component losses
    - Derating based on thermal limits

System Chain:
    Battery -> ESC -> Motor -> Propeller -> Thrust
    
    throttle [0,1] 
      -> V_motor = throttle * V_battery * eta_esc
      -> omega = solve_equilibrium(V_motor, propeller_torque)
      -> thrust = Ct * rho * n^2 * D^4

References:
    - Section 6 of Propulsion-Mission_SPEC.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np

from .motor_model import MotorParameters, DifferentiableMotorModel
from .battery_model import BatteryPackConfig, DifferentiableBatteryModel
from .thermal_model import (
    ThermalNetworkModel, 
    ThermalState, 
    HeatInputs,
    create_propulsion_thermal_network,
    compute_propulsion_heat_inputs,
)

# Type alias for AD-compatible arrays
ArrayLike = Union[float, np.ndarray]


@dataclass
class MotorMount:
    """
    Motor mount position and orientation in body frame.
    
    Defines where a motor is mounted on the aircraft and which direction
    it produces thrust. This enables proper moment arm calculations for
    multi-motor configurations.
    
    Attributes:
        x_b: X position in body frame [m] (positive = forward of CG)
        y_b: Y position in body frame [m] (positive = right of centerline)
        z_b: Z position in body frame [m] (positive = below CG)
        thrust_vector: Unit vector of thrust direction in body frame
                      Default [1, 0, 0] = forward thrust
        rotation_direction: +1 for CW (looking forward), -1 for CCW
                           Used for propeller torque reaction moments
        name: Optional identifier (e.g., 'left', 'right', 'pusher')
        
    Example:
        >>> # Twin motor on flying wing, 0.3m either side of centerline
        >>> left_motor = MotorMount(x_b=0.1, y_b=-0.3, z_b=0.05, rotation_direction=-1, name='left')
        >>> right_motor = MotorMount(x_b=0.1, y_b=0.3, z_b=0.05, rotation_direction=1, name='right')
    """
    
    x_b: float = 0.0
    y_b: float = 0.0
    z_b: float = 0.0
    thrust_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    rotation_direction: int = 1  # +1 CW, -1 CCW (looking forward)
    name: str = ""
    
    def __post_init__(self):
        """Validate and normalize thrust vector."""
        self.thrust_vector = np.asarray(self.thrust_vector, dtype=float)
        norm = np.linalg.norm(self.thrust_vector)
        if norm < 1e-6:
            raise ValueError("thrust_vector cannot be zero")
        self.thrust_vector = self.thrust_vector / norm
        
        if self.rotation_direction not in (1, -1):
            raise ValueError(f"rotation_direction must be +1 or -1, got {self.rotation_direction}")
    
    @property
    def position(self) -> np.ndarray:
        """Position as numpy array [x_b, y_b, z_b]."""
        return np.array([self.x_b, self.y_b, self.z_b])
    
    def get_thrust_force(self, thrust_magnitude: float) -> np.ndarray:
        """Get thrust force vector [Fx, Fy, Fz] in body frame."""
        return thrust_magnitude * self.thrust_vector
    
    def get_thrust_moment(self, thrust_magnitude: float) -> np.ndarray:
        """
        Get moment from thrust about CG: M = r × F
        
        Args:
            thrust_magnitude: Scalar thrust [N]
            
        Returns:
            Moment vector [Mx, My, Mz] in body frame [N·m]
        """
        r = self.position
        F = self.get_thrust_force(thrust_magnitude)
        return np.cross(r, F)
    
    def get_torque_reaction_moment(self, torque: float) -> np.ndarray:
        """
        Get moment from propeller torque reaction.
        
        The propeller exerts a reaction torque on the airframe opposite
        to its rotation direction. This creates a rolling moment.
        
        Args:
            torque: Propeller shaft torque [N·m]
            
        Returns:
            Reaction moment vector [Mx, My, Mz] in body frame [N·m]
        """
        # Torque reaction is opposite to rotation, along thrust axis
        # If prop spins CW (looking forward), reaction torque is CCW
        # This creates a negative rolling moment (right wing down) if motor is on centerline
        return -self.rotation_direction * torque * self.thrust_vector


def create_twin_motor_mounts(
    y_spacing: float,
    x_offset: float = 0.0,
    z_offset: float = 0.0,
    cant_angle_deg: float = 0.0,
    counter_rotating: bool = True,
) -> List[MotorMount]:
    """
    Create symmetric twin motor mount configuration.
    
    Standard flying wing configuration with motors on trailing edge,
    either side of centerline.
    
    Args:
        y_spacing: Distance from centerline to each motor [m]
        x_offset: X position relative to CG [m] (positive = forward)
        z_offset: Z position relative to CG [m] (positive = below)
        cant_angle_deg: Motor cant angle [deg] (positive = thrust up)
        counter_rotating: If True, motors rotate opposite directions
                         to cancel torque reaction
                         
    Returns:
        List of [left_motor, right_motor] MotorMount objects
        
    Example:
        >>> # 30cm spacing, counter-rotating, 5° up-cant
        >>> mounts = create_twin_motor_mounts(
        ...     y_spacing=0.30,
        ...     x_offset=-0.10,  # Behind CG
        ...     cant_angle_deg=5.0,
        ... )
    """
    cant_rad = np.radians(cant_angle_deg)
    
    # Thrust vector with cant angle (positive cant = thrust tilted up)
    thrust_vec = np.array([np.cos(cant_rad), 0.0, -np.sin(cant_rad)])
    
    # Left motor (negative y)
    left = MotorMount(
        x_b=x_offset,
        y_b=-y_spacing,
        z_b=z_offset,
        thrust_vector=thrust_vec.copy(),
        rotation_direction=-1 if counter_rotating else 1,
        name='left',
    )
    
    # Right motor (positive y)
    right = MotorMount(
        x_b=x_offset,
        y_b=y_spacing,
        z_b=z_offset,
        thrust_vector=thrust_vec.copy(),
        rotation_direction=1,
        name='right',
    )
    
    return [left, right]


def create_single_motor_mount(
    x_offset: float = 0.0,
    z_offset: float = 0.0,
    cant_angle_deg: float = 0.0,
    pusher: bool = False,
) -> List[MotorMount]:
    """
    Create single motor mount (nose tractor or tail pusher).
    
    Args:
        x_offset: X position relative to CG [m]
        z_offset: Z position relative to CG [m]
        cant_angle_deg: Motor cant angle [deg]
        pusher: If True, thrust is rearward (negative x)
        
    Returns:
        List containing single MotorMount
    """
    cant_rad = np.radians(cant_angle_deg)
    direction = -1.0 if pusher else 1.0
    
    thrust_vec = np.array([
        direction * np.cos(cant_rad),
        0.0,
        -np.sin(cant_rad),
    ])
    
    return [MotorMount(
        x_b=x_offset,
        y_b=0.0,
        z_b=z_offset,
        thrust_vector=thrust_vec,
        rotation_direction=1,
        name='pusher' if pusher else 'tractor',
    )]


@dataclass
class ESCParameters:
    """
    Electronic Speed Controller parameters.
    
    The ESC converts DC battery voltage to 3-phase AC for the motor.
    Models efficiency losses and current limits.
    
    Attributes:
        efficiency: Power conversion efficiency [0-1]
        I_max_continuous: Maximum continuous current [A]
        I_max_burst: Maximum burst current (10s) [A]
        mass: ESC mass [kg]
        thermal_resistance: Thermal resistance to ambient [C/W]
        thermal_capacitance: Heat capacity [J/C]
        T_max: Maximum operating temperature [C]
        
    Example:
        >>> esc = ESCParameters(
        ...     efficiency=0.95,
        ...     I_max_continuous=60,
        ...     I_max_burst=80,
        ...     mass=0.050,
        ... )
    """
    
    efficiency: float = 0.95
    I_max_continuous: float = 60.0
    I_max_burst: float = 80.0
    mass: float = 0.050
    
    # Thermal properties
    thermal_resistance: float = 5.0    # [C/W]
    thermal_capacitance: float = 20.0  # [J/C]
    T_max: float = 100.0               # [C]
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency}")
        if self.I_max_continuous <= 0:
            raise ValueError(f"I_max_continuous must be positive, got {self.I_max_continuous}")
    
    def get_output_power(self, P_input: ArrayLike) -> ArrayLike:
        """Get output power given input power."""
        return P_input * self.efficiency
    
    def get_heat_generation(self, P_input: ArrayLike) -> ArrayLike:
        """Get heat dissipation from ESC losses."""
        return P_input * (1 - self.efficiency)


# ESC presets for common sizes
ESC_PRESETS = {
    '20A': ESCParameters(
        efficiency=0.94,
        I_max_continuous=20,
        I_max_burst=25,
        mass=0.020,
        T_max=85,
    ),
    '30A': ESCParameters(
        efficiency=0.95,
        I_max_continuous=30,
        I_max_burst=40,
        mass=0.030,
        T_max=90,
    ),
    '40A': ESCParameters(
        efficiency=0.95,
        I_max_continuous=40,
        I_max_burst=55,
        mass=0.040,
        T_max=95,
    ),
    '60A': ESCParameters(
        efficiency=0.96,
        I_max_continuous=60,
        I_max_burst=80,
        mass=0.055,
        T_max=100,
    ),
    '80A': ESCParameters(
        efficiency=0.96,
        I_max_continuous=80,
        I_max_burst=100,
        mass=0.075,
        T_max=100,
    ),
}


def get_esc_preset(name: str) -> ESCParameters:
    """Get ESC preset by name."""
    if name not in ESC_PRESETS:
        available = list(ESC_PRESETS.keys())
        raise ValueError(f"Unknown ESC preset '{name}'. Available: {available}")
    return ESC_PRESETS[name]


def list_esc_presets() -> List[str]:
    """List available ESC presets."""
    return list(ESC_PRESETS.keys())


@dataclass
class PropellerSpec:
    """
    Propeller specification for integrated system.
    
    Primary inputs are diameter and pitch in inches (user-friendly).
    Automatically loads the appropriate pretrained meta-model from APC data
    to provide accurate Ct/Cp predictions across the full J range.
    
    Attributes:
        diameter_in: Propeller diameter [inches] (e.g., 10 for a 10" prop)
        pitch_in: Propeller pitch [inches] (e.g., 4.7 for a 10x4.7 prop)
        family: Propeller family ('Electric', 'SlowFly', 'Standard')
        
    Example:
        >>> prop = PropellerSpec(diameter_in=10, pitch_in=4.7)  # 10x4.7 prop
        >>> prop = PropellerSpec(diameter_in=12, pitch_in=6, family='SlowFly')
        
    Note:
        Uses pretrained meta-models from APC propeller test data.
        Falls back to empirical estimation if model not available.
    """
    
    diameter_in: float
    pitch_in: float
    family: str = 'Electric'  # 'Electric', 'SlowFly', 'Standard'
    
    # Internal: loaded meta-model (set in __post_init__)
    _meta_model: Optional[object] = field(default=None, init=False, repr=False)
    _use_fallback: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Validate and load meta-model."""
        if self.diameter_in <= 0:
            raise ValueError(f"diameter_in must be positive, got {self.diameter_in}")
        if self.pitch_in <= 0:
            raise ValueError(f"pitch_in must be positive, got {self.pitch_in}")
        
        # Normalize family name
        family_map = {
            'electric': 'Electric',
            'slowfly': 'SlowFly', 
            'slow_fly': 'SlowFly',
            'standard': 'Standard',
            'sport': 'Standard',
        }
        self.family = family_map.get(self.family.lower(), self.family)
        
        # Try to load pretrained meta-model
        self._load_meta_model()
    
    def _load_meta_model(self):
        """Load pretrained meta-model for this family."""
        try:
            from .propeller_model import get_pretrained_model, list_pretrained_families
            
            available = list_pretrained_families()
            if not available:
                raise ValueError("No pre-trained propeller models found in data/propeller_models")
            if self.family not in available:
                raise ValueError(
                    f"No pre-trained model for family '{self.family}'. Available: {available}"
                )
            self._meta_model = get_pretrained_model(self.family)
            self._use_fallback = False
        except Exception as exc:
            raise ValueError("Failed to load propeller meta-model") from exc
    
    @property
    def diameter_m(self) -> float:
        """Diameter in meters."""
        return self.diameter_in * 0.0254
    
    @property
    def pitch_m(self) -> float:
        """Pitch in meters."""
        return self.pitch_in * 0.0254
    
    @property
    def P_D_ratio(self) -> float:
        """Pitch to diameter ratio."""
        return self.pitch_in / self.diameter_in
    
    @property
    def Ct_static(self) -> float:
        """Static thrust coefficient (J=0)."""
        Ct, _ = self.get_coefficients(0.0)
        return float(Ct)
    
    @property
    def Cp_static(self) -> float:
        """Static power coefficient (J=0)."""
        _, Cp = self.get_coefficients(0.0)
        return float(Cp)
    
    def get_coefficients(self, J: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get thrust and power coefficients for given advance ratio.
        
        Uses pretrained APC meta-model for accurate predictions.
        
        Args:
            J: Advance ratio V/(n*D)
            
        Returns:
            Tuple of (Ct, Cp)
        """
        if self._meta_model is not None and not self._use_fallback:
            # Use pretrained meta-model
            Ct, Cp = self._meta_model.get_coefficients(J, self.diameter_m, self.pitch_m)
            return Ct, Cp
        if self._meta_model is None or self._use_fallback:
            raise ValueError("Propeller meta-model not loaded; cannot compute coefficients")
        
        # Fallback: empirical estimation from P/D ratio
        return self._get_coefficients_fallback(J)
    
    def _get_coefficients_fallback(self, J: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Fallback coefficient estimation when meta-model not available.
        
        Based on empirical correlations from propeller theory.
        """
        P_D = self.P_D_ratio
        J = np.asarray(J)
        
        # Empirical fits
        Ct_static = 0.09 + 0.02 * P_D
        Cp_static = 0.025 + 0.035 * P_D
        J_zero_thrust = 0.7 + 0.3 * P_D
        
        # Clamp to physical bounds
        Ct_static = np.clip(Ct_static, 0.05, 0.15)
        Cp_static = np.clip(Cp_static, 0.02, 0.08)
        J_zero_thrust = np.clip(J_zero_thrust, 0.5, 1.2)
        
        # Linear decrease model
        Ct = Ct_static * np.maximum(0, 1 - J / J_zero_thrust)
        J_zero_Cp = J_zero_thrust * 1.3
        Cp = Cp_static * np.maximum(0, 1 - 0.7 * (J / J_zero_Cp)**1.5)
        
        return Ct, Cp
    
    def get_thrust_and_power(
        self, 
        V: ArrayLike, 
        omega: ArrayLike, 
        rho: ArrayLike = 1.225
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get dimensional thrust and power.
        
        Args:
            V: Freestream velocity [m/s]
            omega: Angular velocity [rad/s]
            rho: Air density [kg/m^3]
            
        Returns:
            Tuple of (Thrust [N], Power [W])
        """
        n = omega / (2 * np.pi)  # rev/s
        D = self.diameter_m
        
        # Avoid division by zero at zero RPM
        n_safe = np.maximum(np.abs(n), 1e-6)
        J = V / (n_safe * D)
        
        Ct, Cp = self.get_coefficients(J)
        
        # Thrust and power (zero if not rotating)
        is_rotating = np.abs(n) > 0.1  # Threshold for "rotating"
        
        thrust = np.where(is_rotating, Ct * rho * n**2 * D**4, 0.0)
        power = np.where(is_rotating, Cp * rho * n**3 * D**5, 0.0)
        
        return thrust, power
    
    def get_torque(
        self, 
        V: ArrayLike, 
        omega: ArrayLike, 
        rho: ArrayLike = 1.225
    ) -> ArrayLike:
        """
        Get propeller torque (power / omega).
        
        Args:
            V: Freestream velocity [m/s]
            omega: Angular velocity [rad/s]
            rho: Air density [kg/m^3]
            
        Returns:
            Torque [N*m]
        """
        _, power = self.get_thrust_and_power(V, omega, rho)
        omega_safe = np.maximum(np.abs(omega), 1e-6)
        return power / omega_safe


# Maximum supported motors (covers single to octocopter+)
MAX_MOTORS = 10


@dataclass
class PropulsionSystemConfig:
    """
    Complete propulsion system configuration.
    
    Defines all components and their parameters for the integrated system.
    Supports both simple n_motors configuration and explicit motor mounts
    for proper moment arm calculations.
    
    Attributes:
        propeller: Propeller specification
        motor: Motor parameters  
        esc: ESC parameters
        battery: Battery pack configuration
        motor_mounts: List of MotorMount objects defining motor positions.
                     If not provided, uses n_motors with centerline mounting.
        n_motors: Number of motors (used only if motor_mounts not provided)
        thrust_cant_angle_deg: Motor cant angle [deg] (used only if motor_mounts not provided)
        
    Example:
        >>> # Simple single motor configuration
        >>> config = PropulsionSystemConfig(
        ...     propeller=PropellerSpec(diameter_in=10, pitch_in=4.7),
        ...     motor=get_motor_preset('2212_920'),
        ...     esc=get_esc_preset('30A'),
        ...     battery=create_lipo_pack(4, 1, 2200),
        ... )
        
        >>> # Twin motor with explicit mounts
        >>> mounts = create_twin_motor_mounts(y_spacing=0.30, x_offset=-0.10)
        >>> config = PropulsionSystemConfig(
        ...     motor_mounts=mounts,
        ...     propeller=PropellerSpec(diameter_in=10, pitch_in=4.7),
        ...     motor=get_motor_preset('2212_920'),
        ...     esc=get_esc_preset('30A'),
        ...     battery=create_lipo_pack(4, 1, 2200),
        ... )
    """
    
    propeller: PropellerSpec
    motor: MotorParameters
    esc: ESCParameters
    battery: BatteryPackConfig
    motor_mounts: Optional[List[MotorMount]] = None
    n_motors: int = 1
    thrust_cant_angle_deg: float = 0.0
    
    def __post_init__(self):
        """Validate configuration and create default motor mounts if needed."""
        if self.motor_mounts is None:
            # Create default centerline mounts
            if self.n_motors < 1:
                raise ValueError(f"n_motors must be >= 1, got {self.n_motors}")
            if self.n_motors > MAX_MOTORS:
                raise ValueError(f"n_motors must be <= {MAX_MOTORS}, got {self.n_motors}")
            
            cant_rad = np.radians(self.thrust_cant_angle_deg)
            thrust_vec = np.array([np.cos(cant_rad), 0.0, -np.sin(cant_rad)])
            
            self.motor_mounts = [
                MotorMount(
                    x_b=0.0, y_b=0.0, z_b=0.0,
                    thrust_vector=thrust_vec.copy(),
                    rotation_direction=1,
                    name=f'motor_{i}',
                )
                for i in range(self.n_motors)
            ]
        else:
            # motor_mounts provided, update n_motors to match
            if len(self.motor_mounts) > MAX_MOTORS:
                raise ValueError(f"Maximum {MAX_MOTORS} motors supported, got {len(self.motor_mounts)}")
            self.n_motors = len(self.motor_mounts)
    
    @property
    def total_mass(self) -> float:
        """Total propulsion system mass [kg]."""
        return (
            self.n_motors * self.motor.mass +
            self.n_motors * self.esc.mass +
            self.battery.mass
            # Note: propeller mass typically small, often ignored
        )
    
    @property
    def thrust_vector(self) -> np.ndarray:
        """
        Net unit thrust vector in body frame.
        
        For multi-motor configs, this is the average of individual thrust vectors.
        """
        if not self.motor_mounts:
            cant = np.deg2rad(self.thrust_cant_angle_deg)
            return np.array([np.cos(cant), 0, -np.sin(cant)])
        
        total = np.zeros(3)
        for mount in self.motor_mounts:
            total += mount.thrust_vector
        norm = np.linalg.norm(total)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return total / norm


class IntegratedPropulsionSystem:
    """
    Complete differentiable propulsion system model.
    
    Couples battery, ESC, motor, and propeller into a single interface
    for trajectory optimization and dynamics simulation.
    
    The system solves for motor-propeller equilibrium at each operating
    point using fixed-point iteration (AD-compatible).
    
    Example:
        >>> system = IntegratedPropulsionSystem(config)
        >>> result = system.solve_equilibrium(
        ...     throttle=0.7,
        ...     V_freestream=15.0,
        ...     rho=1.225,
        ...     SOC=0.8,
        ... )
        >>> print(f"Thrust: {result['thrust_total']:.1f} N")
        >>> print(f"Power: {result['power_battery']:.1f} W")
    """
    
    def __init__(
        self, 
        config: PropulsionSystemConfig,
        include_thermal: bool = True,
    ):
        """
        Initialize integrated propulsion system.
        
        Args:
            config: System configuration
            include_thermal: Whether to track thermal states
        """
        self.config = config
        self.include_thermal = include_thermal
        
        # Create component models
        self.motor_model = DifferentiableMotorModel(config.motor)
        self.battery_model = DifferentiableBatteryModel(config.battery)
        
        # Create thermal network if enabled
        if include_thermal:
            self.thermal_network = create_propulsion_thermal_network(
                motor_capacitance=config.motor.thermal_capacitance,
                motor_T_max=config.motor.T_max,
                esc_capacitance=config.esc.thermal_capacitance,
                esc_T_max=config.esc.T_max,
                battery_capacitance=config.battery.cell.thermal_capacitance * config.battery.n_parallel,
                battery_T_max=config.battery.cell.T_max,
            )
        else:
            self.thermal_network = None
    
    def solve_equilibrium(
        self,
        throttle: Union[float, np.ndarray],
        V_freestream: ArrayLike,
        rho: ArrayLike = 1.225,
        SOC: ArrayLike = 0.8,
        temperatures: Optional[Dict[str, float]] = None,
        T_ambient: float = 25.0,
        n_iterations: int = 20,
    ) -> Dict[str, ArrayLike]:
        """
        Solve propulsion system equilibrium for given operating point.
        
        Finds the steady-state operating point where motor torque
        equals propeller torque, using fixed-point iteration.
        
        Supports differential thrust via per-motor throttle array.
        
        Args:
            throttle: Throttle command [0, 1]. Can be:
                     - Single float: same throttle for all motors
                     - Array of length n_motors: individual throttle per motor
            V_freestream: Freestream airspeed [m/s]
            rho: Air density [kg/m^3]
            SOC: Battery state of charge [0, 1]
            temperatures: Component temperatures [C] (optional)
            T_ambient: Ambient temperature [C]
            n_iterations: Fixed-point iterations for equilibrium
            
        Returns:
            Dict with all propulsion outputs:
                - thrust_total: Total thrust magnitude [N]
                - thrust_per_motor: Array of thrust per motor [N] (length n_motors)
                - force_body: Total force vector [Fx, Fy, Fz] in body frame [N]
                - moment_body: Total moment vector [Mx, My, Mz] in body frame [N·m]
                - forces_per_motor: List of force vectors per motor [N]
                - moments_per_motor: List of moment vectors per motor [N·m]
                - power_battery: Battery power draw [W]
                - power_shaft: Shaft power per motor [W] (average if differential)
                - current_battery: Battery current [A]
                - current_motor: Motor current per motor [A] (average if differential)
                - voltage_battery: Battery terminal voltage [V]
                - voltage_motor: Motor voltage [V] (average if differential)
                - omega: Motor/propeller angular velocity [rad/s] (array if differential)
                - rpm: Motor/propeller RPM (array if differential)
                - J: Advance ratio (average)
                - Ct, Cp: Thrust and power coefficients (average)
                - eta_propeller: Propeller efficiency
                - eta_motor: Motor efficiency
                - eta_overall: System efficiency (thrust power / battery power)
                - dSOC_dt: Battery SOC rate of change [1/s]
                - heat_motor, heat_esc, heat_battery: Heat generation [W]
                - derate_motor, derate_esc, derate_battery: Thermal derating [0-1]
        """
        n_motors = self.config.n_motors
        
        # Normalize throttle to per-motor array
        throttle_arr = np.atleast_1d(throttle)
        if throttle_arr.size == 1:
            throttle_arr = np.full(n_motors, float(throttle_arr[0]))
        elif throttle_arr.size != n_motors:
            raise ValueError(
                f"throttle must be scalar or array of length {n_motors}, "
                f"got length {throttle_arr.size}"
            )
        
        # Initialize temperatures if not provided
        if temperatures is None:
            temperatures = {'motor': T_ambient, 'esc': T_ambient, 'battery': T_ambient}
        
        # Get thermal derating factors
        if self.thermal_network is not None:
            derate_factors = self.thermal_network.get_derating_factors(temperatures)
        else:
            derate_factors = {'motor': 1.0, 'esc': 1.0, 'battery': 1.0}
        
        # Apply derating to throttle (reduces power when hot)
        thermal_derate = min(derate_factors['motor'], derate_factors['esc'])
        throttle_effective = throttle_arr * thermal_derate
        
        # Get battery voltage
        T_battery = temperatures.get('battery', T_ambient)
        
        # Start with nominal voltage as initial guess for iteration
        V_battery_eff = self.battery_model.get_terminal_voltage(SOC, 0, T_battery)
        
        # System-level iteration to converge battery voltage with motor equilibrium
        # Use damping to prevent oscillation between high/low power states
        for sys_iter in range(5):
            thrust_per_motor = np.zeros(n_motors)
            omega_per_motor = np.zeros(n_motors)
            torque_per_motor = np.zeros(n_motors)
            I_motor_per = np.zeros(n_motors)
            P_shaft_per = np.zeros(n_motors)
            P_esc_in_per = np.zeros(n_motors)
            Q_motor_per = np.zeros(n_motors)
            Q_esc_per = np.zeros(n_motors)
            
            for i in range(n_motors):
                thr_eff = throttle_effective[i]
                V_motor_cmd = thr_eff * V_battery_eff
                
                omega_limit = self.config.motor.no_load_speed(V_motor_cmd)
                if not np.isfinite(omega_limit) or omega_limit < 0:
                    omega_limit = 0.0

                # Bisection for motor-prop equilibrium (fixed iteration count for AD)
                # Find omega where motor_torque == prop_torque
                # Motor torque: tau_m = Kt * ((V - Ke*omega)/R - I0)
                # Prop torque: tau_p = Cp * rho * (omega/2pi)^2 * D^5 / omega (= power/omega)
                omega_low = 0.0
                omega_high = omega_limit

                for _ in range(max(1, n_iterations)):
                    omega = (omega_low + omega_high) / 2.0
                    tau_prop = self.config.propeller.get_torque(V_freestream, omega, rho)

                    # Motor current and torque at this omega
                    I_motor = (V_motor_cmd - self.motor_model.Ke * omega) / self.motor_model.R
                    tau_motor = (I_motor - self.motor_model.I0) * self.motor_model.Kt

                    # If motor torque > prop torque, equilibrium is at higher omega
                    if tau_motor > tau_prop:
                        omega_low = omega
                    else:
                        omega_high = omega

                omega = (omega_low + omega_high) / 2.0
                tau_prop = self.config.propeller.get_torque(V_freestream, omega, rho)
                motor_result = self.motor_model.solve_from_omega_and_torque(omega, tau_prop)
                
                # Get final propeller performance
                thrust_i, _ = self.config.propeller.get_thrust_and_power(
                    V_freestream, omega, rho
                )
                
                thrust_per_motor[i] = thrust_i
                omega_per_motor[i] = omega
                torque_per_motor[i] = motor_result['torque']
                I_motor_per[i] = motor_result['current']
                P_shaft_per[i] = motor_result['power_shaft']
                
                P_motor_elec = motor_result['power_electrical']
                P_esc_in_per[i] = P_motor_elec / self.config.esc.efficiency
                Q_motor_per[i] = motor_result['power_loss']
                Q_esc_per[i] = self.config.esc.get_heat_generation(P_esc_in_per[i])
            
            # Update battery voltage for next pass with damping to prevent oscillation
            P_battery_total = np.sum(P_esc_in_per)
            I_battery_total = self.battery_model.get_current_for_power(SOC, P_battery_total, T_battery)
            V_battery_new = self.battery_model.get_terminal_voltage(SOC, I_battery_total, T_battery)
            
            # Damped update: blend new voltage with previous (0.5 = equal weight)
            damping = 0.5
            V_battery_eff = damping * V_battery_new + (1 - damping) * V_battery_eff
            
        # Final values after system iteration pass
        P_battery = np.sum(P_esc_in_per)
        I_battery = I_battery_total
        V_battery_loaded = V_battery_eff
        
        # Compute forces and moments from motor mounts
        force_body = np.zeros(3)
        moment_body = np.zeros(3)
        forces_per_motor = []
        moments_per_motor = []
        
        for i, mount in enumerate(self.config.motor_mounts):
            thrust_i = thrust_per_motor[i]
            torque_i = torque_per_motor[i]
            
            # Thrust force vector
            F_thrust = mount.get_thrust_force(thrust_i)
            
            # Thrust moment about CG
            M_thrust = mount.get_thrust_moment(thrust_i)
            
            # Propeller torque reaction moment
            M_torque_reaction = mount.get_torque_reaction_moment(torque_i)
            
            # Total moment from this motor
            M_total = M_thrust + M_torque_reaction
            
            force_body += F_thrust
            moment_body += M_total
            
            forces_per_motor.append(F_thrust)
            moments_per_motor.append(M_total)
        
        # Average values for reporting
        omega_avg = np.mean(omega_per_motor)
        n_rps = omega_avg / (2 * np.pi)
        D = self.config.propeller.diameter_m
        J = V_freestream / (max(n_rps, 0.01) * D)
        
        Ct, Cp = self.config.propeller.get_coefficients(J)
        
        # Efficiencies
        P_shaft_total = np.sum(P_shaft_per)
        eta_motor = P_shaft_total / max(P_battery * self.config.esc.efficiency, 1e-6)
        eta_propeller = float(np.divide(J * Ct, Cp, out=np.zeros_like(Cp, dtype=float), where=Cp > 1e-12))
        
        thrust_total = np.sum(thrust_per_motor)
        thrust_power = thrust_total * V_freestream
        eta_overall = thrust_power / max(P_battery, 1e-6)
        
        # SOC dynamics
        dSOC_dt = self.battery_model.get_soc_derivative(I_battery)
        
        # Total heat generation
        Q_motor_total = np.sum(Q_motor_per)
        Q_esc_total = np.sum(Q_esc_per)
        Q_battery = self.battery_model.get_heat_generation(I_battery, SOC, T_battery)
        
        return {
            # Forces and moments (for dynamics)
            'force_body': force_body,
            'moment_body': moment_body,
            'forces_per_motor': forces_per_motor,
            'moments_per_motor': moments_per_motor,
            
            # Thrust and power
            'thrust_total': thrust_total,
            'thrust_per_motor': thrust_per_motor,
            'power_battery': P_battery,
            'power_shaft': np.mean(P_shaft_per),
            'power_esc_in': np.mean(P_esc_in_per),
            
            # Electrical
            'current_battery': I_battery,
            'current_motor': np.mean(I_motor_per),
            'voltage_battery': V_battery_loaded,
            'voltage_motor': np.mean(throttle_effective) * V_battery_loaded,
            
            # Mechanical (per-motor arrays for differential thrust)
            'omega': omega_per_motor if n_motors > 1 else omega_per_motor[0],
            'rpm': omega_per_motor * 60 / (2 * np.pi) if n_motors > 1 else omega_per_motor[0] * 60 / (2 * np.pi),
            'torque': torque_per_motor if n_motors > 1 else torque_per_motor[0],
            
            # Propeller (average)
            'J': J,
            'Ct': Ct,
            'Cp': Cp,
            
            # Efficiencies
            'eta_propeller': eta_propeller,
            'eta_motor': eta_motor,
            'eta_esc': self.config.esc.efficiency,
            'eta_overall': eta_overall,
            
            # Battery dynamics
            'SOC': SOC,
            'dSOC_dt': dSOC_dt,
            
            # Thermal
            'heat_motor': Q_motor_total,
            'heat_esc': Q_esc_total,
            'heat_battery': Q_battery,
            
            # Derating
            'derate_motor': derate_factors['motor'],
            'derate_esc': derate_factors['esc'],
            'derate_battery': derate_factors['battery'],
            'thermal_derate': thermal_derate,
            
            # Throttle
            'throttle_effective': throttle_effective,
            'throttle_per_motor': throttle_arr,
        }
    
    def get_max_thrust(
        self,
        V_freestream: ArrayLike,
        rho: ArrayLike = 1.225,
        SOC: ArrayLike = 0.8,
        temperatures: Optional[Dict[str, float]] = None,
        T_ambient: float = 25.0,
    ) -> ArrayLike:
        """
        Get maximum available thrust at current conditions.
        
        Args:
            V_freestream: Freestream airspeed [m/s]
            rho: Air density [kg/m^3]
            SOC: Battery state of charge [0, 1]
            temperatures: Component temperatures [C]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Maximum thrust [N]
        """
        result = self.solve_equilibrium(
            throttle=1.0,
            V_freestream=V_freestream,
            rho=rho,
            SOC=SOC,
            temperatures=temperatures,
            T_ambient=T_ambient,
        )
        return result['thrust_total']
    
    def get_static_thrust(
        self,
        throttle: ArrayLike = 1.0,
        SOC: ArrayLike = 0.8,
        T_ambient: float = 25.0,
    ) -> ArrayLike:
        """
        Get static thrust (V=0).
        
        Args:
            throttle: Throttle command [0, 1]
            SOC: Battery state of charge [0, 1]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Static thrust [N]
        """
        result = self.solve_equilibrium(
            throttle=throttle,
            V_freestream=0.0,
            rho=1.225,
            SOC=SOC,
            temperatures={'motor': T_ambient, 'esc': T_ambient, 'battery': T_ambient},
            T_ambient=T_ambient,
        )
        return result['thrust_total']
    
    def get_throttle_for_thrust(
        self,
        thrust_target: ArrayLike,
        V_freestream: ArrayLike,
        rho: ArrayLike = 1.225,
        SOC: ArrayLike = 0.8,
        temperatures: Optional[Dict[str, float]] = None,
        T_ambient: float = 25.0,
        n_iterations: int = 10,
    ) -> ArrayLike:
        """
        Find throttle setting for desired thrust (inverse problem).
        
        Uses bisection search for robustness.
        
        Args:
            thrust_target: Desired thrust [N]
            V_freestream: Freestream airspeed [m/s]
            rho: Air density [kg/m^3]
            SOC: Battery state of charge [0, 1]
            temperatures: Component temperatures [C]
            T_ambient: Ambient temperature [C]
            n_iterations: Bisection iterations
            
        Returns:
            Throttle setting [0, 1]
        """
        # Bisection search
        throttle_lo = 0.0
        throttle_hi = 1.0
        
        for _ in range(n_iterations):
            throttle_mid = (throttle_lo + throttle_hi) / 2
            result = self.solve_equilibrium(
                throttle=throttle_mid,
                V_freestream=V_freestream,
                rho=rho,
                SOC=SOC,
                temperatures=temperatures,
                T_ambient=T_ambient,
            )
            thrust_mid = result['thrust_total']
            
            if thrust_mid < thrust_target:
                throttle_lo = throttle_mid
            else:
                throttle_hi = throttle_mid
        
        return (throttle_lo + throttle_hi) / 2
    
    def get_temperature_derivatives(
        self,
        temperatures: Dict[str, float],
        heat_inputs: Dict[str, float],
        V_freestream: float,
        T_ambient: float = 25.0,
    ) -> Dict[str, float]:
        """
        Get temperature derivatives for dynamics integration.
        
        Args:
            temperatures: Current component temperatures [C]
            heat_inputs: Current heat generation [W]
            V_freestream: Freestream airspeed [m/s]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Dict of dT/dt [C/s] for each component
        """
        if self.thermal_network is None:
            return {'motor': 0.0, 'esc': 0.0, 'battery': 0.0}
        
        return self.thermal_network.get_temperature_derivatives(
            temperatures=temperatures,
            heat_inputs=heat_inputs,
            airspeed=V_freestream,
            T_ambient=T_ambient,
        )
    
    def summary(self) -> str:
        """Return formatted system summary."""
        cfg = self.config
        prop = cfg.propeller
        
        # Handle both old-style (diameter_m only) and new-style (diameter_in, pitch_in)
        if hasattr(prop, 'diameter_in') and hasattr(prop, 'pitch_in'):
            prop_str = f"{prop.diameter_in:.0f}x{prop.pitch_in:.1f} ({prop.family})"
        else:
            prop_str = f"{prop.diameter_m*1000:.0f} mm"
        
        lines = [
            "Integrated Propulsion System",
            "=" * 40,
            f"Motors: {cfg.n_motors}x",
            f"",
            "Propeller:",
            f"  Size: {prop_str}",
            f"  P/D ratio: {prop.P_D_ratio:.2f}" if hasattr(prop, 'P_D_ratio') else "",
            f"  Static Ct: {prop.Ct_static:.3f}",
            f"  Static Cp: {prop.Cp_static:.3f}",
            f"",
            "Motor:",
            f"  Kv: {cfg.motor.kv:.0f} RPM/V",
            f"  Resistance: {cfg.motor.R_internal*1000:.1f} mOhm",
            f"  I_no_load: {cfg.motor.I_no_load:.2f} A",
            f"  Mass: {cfg.motor.mass*1000:.0f} g",
            f"",
            "ESC:",
            f"  Efficiency: {cfg.esc.efficiency*100:.0f}%",
            f"  I_max: {cfg.esc.I_max_continuous:.0f} A continuous",
            f"",
            "Battery:",
            f"  Config: {cfg.battery.n_series}S{cfg.battery.n_parallel}P",
            f"  Capacity: {cfg.battery.capacity_Ah*1000:.0f} mAh",
            f"  V_nominal: {cfg.battery.V_nominal:.1f} V",
            f"  Mass: {cfg.battery.mass*1000:.0f} g",
            f"",
            f"Total mass: {cfg.total_mass*1000:.0f} g",
        ]
        # Filter out empty strings
        lines = [l for l in lines if l]
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_propulsion_system(
    prop_diameter_in: float = 10.0,
    prop_pitch_in: float = 4.7,
    motor_kv: float = 920,
    battery_s: int = 4,
    battery_mah: int = 2200,
    esc_amps: int = 30,
    n_motors: int = 1,
    prop_family: str = 'electric',
) -> IntegratedPropulsionSystem:
    """
    Create a simple propulsion system with sensible defaults.
    
    Args:
        prop_diameter_in: Propeller diameter [inches] (e.g., 10 for 10x4.7)
        prop_pitch_in: Propeller pitch [inches] (e.g., 4.7 for 10x4.7)
        motor_kv: Motor Kv rating [RPM/V]
        battery_s: Battery series cell count (e.g., 4 for 4S)
        battery_mah: Battery capacity [mAh]
        esc_amps: ESC current rating [A]
        n_motors: Number of motors
        prop_family: Propeller family ('electric', 'slowfly', 'sport')
        
    Returns:
        Configured IntegratedPropulsionSystem
        
    Example:
        >>> # 10x4.7 prop, 920Kv motor, 4S 2200mAh battery
        >>> system = create_simple_propulsion_system(
        ...     prop_diameter_in=10,
        ...     prop_pitch_in=4.7,
        ...     motor_kv=920,
        ...     battery_s=4,
        ...     battery_mah=2200,
        ... )
    """
    from .motor_model import estimate_motor_params_from_spec
    from .battery_model import create_lipo_pack
    
    # Create propeller spec from diameter and pitch
    prop = PropellerSpec(
        diameter_in=prop_diameter_in,
        pitch_in=prop_pitch_in,
        family=prop_family,
    )
    
    # Estimate motor parameters from Kv
    # Nominal voltage for the battery
    V_nominal = battery_s * 3.7
    
    # Estimate power based on Kv and voltage (rough empirical)
    # Higher Kv motors at same voltage = more power (typically)
    P_max_estimate = V_nominal * (V_nominal / (60.0 / (2 * np.pi * motor_kv))) * 0.5
    P_max_estimate = max(100, min(2000, P_max_estimate))  # Clamp to reasonable range
    
    motor = estimate_motor_params_from_spec(
        kv=motor_kv,
        P_max=P_max_estimate,
        V_nominal=V_nominal,
    )
    
    # Get ESC preset
    esc_name = f'{esc_amps}A'
    if esc_name not in ESC_PRESETS:
        # Fall back to closest
        esc = ESCParameters(I_max_continuous=esc_amps)
    else:
        esc = get_esc_preset(esc_name)
    
    # Create battery pack
    battery = create_lipo_pack(
        n_series=battery_s,
        n_parallel=1,
        capacity_mAh=battery_mah,
    )
    
    # Create config
    config = PropulsionSystemConfig(
        n_motors=n_motors,
        propeller=prop,
        motor=motor,
        esc=esc,
        battery=battery,
    )
    
    return IntegratedPropulsionSystem(config)
