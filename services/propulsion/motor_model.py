"""
Differentiable BLDC Motor Model

Implements the Drela motor model with analytical solutions for AD-compatibility.
Replaces bisection-based motor solvers with closed-form equations.

Key equations:
    V_motor = Ke·ω + I·R              (electrical - back-EMF + resistive drop)
    τ_motor = Kt·(I - I₀)             (mechanical - torque production)
    P_shaft = τ·ω                     (shaft power output)
    Q_loss = I²·R + I₀·Ke·ω           (heat generation)

Where:
    Ke = 1 / (Kv · π/30) = back-EMF constant [V/(rad/s)]
    Kt = 60 / (2π · Kv) = torque constant [N·m/A]
    Note: For ideal BLDC, Ke = Kt (in SI units)

References:
    - Drela, M. "First-Order DC Electric Motor Model"
    - AeroSandbox motor modeling conventions
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Type alias for AD-compatible arrays
ArrayLike = Union[float, np.ndarray]


@dataclass
class MotorParameters:
    """
    BLDC motor parameters following the Drela model.
    
    The motor is characterized by:
    - Kv: Speed constant relating no-load speed to voltage
    - R: Internal resistance causing I²R losses  
    - I₀: No-load current representing friction/windage losses
    
    Attributes:
        kv: Velocity constant [rpm/V] - motor speed per volt at no load
        R_internal: Internal winding resistance [Ω]
        I_no_load: No-load current draw [A] - represents mechanical losses
        I_max: Maximum continuous current [A] - thermal limit
        P_max: Maximum continuous power [W] - thermal limit
        mass: Motor mass [kg] - for mass budgets
        thermal_resistance: Stator to ambient [°C/W]
        thermal_capacitance: Heat capacity [J/°C]
        T_max: Maximum winding temperature [°C]
    
    Example:
        >>> motor = MotorParameters(
        ...     kv=1000,
        ...     R_internal=0.050,
        ...     I_no_load=1.5,
        ...     I_max=40,
        ...     P_max=500
        ... )
        >>> print(f"Kt = {motor.Kt:.4f} N·m/A")
        Kt = 0.0095 N·m/A
    """
    
    # Primary electrical parameters
    kv: float                           # Velocity constant [rpm/V]
    R_internal: float                   # Internal resistance [Ω]
    I_no_load: float                    # No-load current [A]
    
    # Limits
    I_max: float = np.inf               # Maximum continuous current [A]
    P_max: float = np.inf               # Maximum continuous power [W]
    
    # Physical properties
    mass: float = 0.0                   # Motor mass [kg]
    
    # Thermal properties (lumped model)
    thermal_resistance: float = 5.0     # Stator to ambient [°C/W]
    thermal_capacitance: float = 50.0   # Heat capacity [J/°C]
    T_max: float = 150.0                # Maximum winding temperature [°C]
    
    # Optional metadata
    name: str = ""
    manufacturer: str = ""
    
    def __post_init__(self):
        """Validate parameters."""
        if self.kv <= 0:
            raise ValueError(f"kv must be positive, got {self.kv}")
        if self.R_internal < 0:
            raise ValueError(f"R_internal must be non-negative, got {self.R_internal}")
        if self.I_no_load < 0:
            raise ValueError(f"I_no_load must be non-negative, got {self.I_no_load}")
    
    @property
    def kv_rad_per_V(self) -> float:
        """Velocity constant in rad/s per Volt."""
        return self.kv * (2 * np.pi / 60)
    
    @property
    def Kt(self) -> float:
        """
        Torque constant [N·m/A].
        
        For an ideal BLDC motor: Kt = 60 / (2π · Kv)
        This is the torque produced per amp of current above I₀.
        """
        return 60.0 / (2 * np.pi * self.kv)
    
    @property
    def Ke(self) -> float:
        """
        Back-EMF constant [V/(rad/s)].
        
        For an ideal BLDC motor: Ke = 1 / (Kv · π/30) = Kt
        This is the voltage generated per rad/s of rotation.
        """
        return 1.0 / self.kv_rad_per_V
    
    @property
    def Ke_rpm(self) -> float:
        """Back-EMF constant [V/rpm]."""
        return 1.0 / self.kv
    
    def no_load_speed(self, V_motor: ArrayLike) -> ArrayLike:
        """
        Calculate no-load speed for given motor voltage.
        
        At no load: V = Ke·ω + I₀·R
        So: ω = (V - I₀·R) / Ke
        
        Args:
            V_motor: Applied motor voltage [V]
            
        Returns:
            No-load angular velocity [rad/s]
        """
        return (V_motor - self.I_no_load * self.R_internal) / self.Ke
    
    def no_load_rpm(self, V_motor: ArrayLike) -> ArrayLike:
        """Calculate no-load speed in RPM."""
        return self.no_load_speed(V_motor) * 30 / np.pi
    
    def stall_current(self, V_motor: ArrayLike) -> ArrayLike:
        """
        Calculate stall (locked rotor) current.
        
        At stall: ω = 0, so V = I·R
        Therefore: I_stall = V / R
        
        Args:
            V_motor: Applied motor voltage [V]
            
        Returns:
            Stall current [A]
        """
        return V_motor / self.R_internal
    
    def stall_torque(self, V_motor: ArrayLike) -> ArrayLike:
        """
        Calculate stall (maximum) torque.
        
        τ_stall = Kt · (I_stall - I₀)
        
        Args:
            V_motor: Applied motor voltage [V]
            
        Returns:
            Stall torque [N·m]
        """
        I_stall = self.stall_current(V_motor)
        return self.Kt * (I_stall - self.I_no_load)
    
    def max_efficiency_point(self, V_motor: float) -> Dict[str, float]:
        """
        Calculate the operating point of maximum efficiency.
        
        For a Drela motor model, max efficiency occurs at:
            I_opt = sqrt(I₀ · V / R)
        
        This is derived from dη/dI = 0.
        
        Args:
            V_motor: Applied motor voltage [V]
            
        Returns:
            Dict with omega, torque, current, power_out, power_in, efficiency
        """
        # Optimal current for max efficiency
        I_opt = np.sqrt(self.I_no_load * V_motor / self.R_internal)
        
        # Corresponding speed: V = Ke·ω + I·R
        omega = (V_motor - I_opt * self.R_internal) / self.Ke
        
        # Torque
        tau = self.Kt * (I_opt - self.I_no_load)
        
        # Power
        P_in = V_motor * I_opt
        P_out = tau * omega
        
        # Efficiency
        eta = P_out / P_in if P_in > 0 else 0.0
        
        return {
            'omega': omega,
            'rpm': omega * 30 / np.pi,
            'torque': tau,
            'current': I_opt,
            'power_in': P_in,
            'power_out': P_out,
            'efficiency': eta,
        }
    
    def summary(self) -> str:
        """Return a formatted summary of motor parameters."""
        lines = [
            f"Motor: {self.name}" if self.name else "Motor Parameters",
            f"  Kv: {self.kv:.0f} rpm/V",
            f"  R_internal: {self.R_internal*1000:.1f} mOhm",
            f"  I_no_load: {self.I_no_load:.2f} A",
            f"  Kt: {self.Kt*1000:.2f} mN-m/A",
            f"  Ke: {self.Ke*1000:.2f} mV/(rad/s)",
        ]
        if self.I_max < np.inf:
            lines.append(f"  I_max: {self.I_max:.0f} A")
        if self.P_max < np.inf:
            lines.append(f"  P_max: {self.P_max:.0f} W")
        if self.mass > 0:
            lines.append(f"  Mass: {self.mass*1000:.0f} g")
        return "\n".join(lines)


class DifferentiableMotorModel:
    """
    Fully differentiable BLDC motor model.
    
    Provides analytical solutions for motor operating points given various
    input combinations. All methods return Dict results suitable for use
    in optimization problems.
    
    The motor equations are:
        V_motor = Ke·ω + I·R              (electrical)
        τ_motor = Kt·(I - I₀)             (mechanical)
        P_shaft = τ·ω                     (shaft power)
        P_elec = V·I                      (electrical power)
        Q_loss = I²·R + P_friction        (heat generation)
    
    Three solution modes:
        1. Given V and ω → solve for I, τ
        2. Given V and τ → solve for I, ω  
        3. Given ω and τ → solve for I, V
    
    Example:
        >>> params = MotorParameters(kv=1000, R_internal=0.05, I_no_load=1.5)
        >>> motor = DifferentiableMotorModel(params)
        >>> result = motor.solve_from_voltage_and_omega(V_motor=22.2, omega=800)
        >>> print(f"Current: {result['current']:.1f} A, Torque: {result['torque']*1000:.1f} mN·m")
    """
    
    def __init__(self, params: MotorParameters):
        """
        Initialize motor model.
        
        Args:
            params: Motor parameters dataclass
        """
        self.params = params
    
    @property
    def Kt(self) -> float:
        """Torque constant [N·m/A]."""
        return self.params.Kt
    
    @property
    def Ke(self) -> float:
        """Back-EMF constant [V/(rad/s)]."""
        return self.params.Ke
    
    @property
    def R(self) -> float:
        """Internal resistance [Ω]."""
        return self.params.R_internal
    
    @property
    def I0(self) -> float:
        """No-load current [A]."""
        return self.params.I_no_load
    
    def solve_from_voltage_and_omega(
        self, 
        V_motor: ArrayLike, 
        omega: ArrayLike
    ) -> Dict[str, ArrayLike]:
        """
        Solve motor state given applied voltage and shaft speed.
        
        This is the most common operating mode: ESC applies voltage,
        propeller load determines speed, solve for current and torque.
        
        From V = Ke·ω + I·R:
            I = (V - Ke·ω) / R
        
        Then:
            τ = Kt·(I - I₀)
        
        Args:
            V_motor: Applied motor voltage [V]
            omega: Angular velocity [rad/s]
            
        Returns:
            Dict with keys:
                current: Motor current [A]
                torque: Shaft torque [N·m]
                power_electrical: Input power [W]
                power_shaft: Output power [W]
                power_loss: Heat dissipation [W]
                efficiency: η = P_out/P_in
                back_emf: Ke·ω [V]
        """
        # Back-EMF
        back_emf = self.Ke * omega
        
        # Current from electrical equation
        current = (V_motor - back_emf) / self.R
        
        # Torque from mechanical equation
        torque = self.Kt * (current - self.I0)
        
        # Power calculations
        power_electrical = V_motor * current
        power_shaft = torque * omega
        power_loss = power_electrical - power_shaft
        
        # Efficiency (avoid division by zero)
        efficiency = np.zeros_like(power_electrical)
        np.divide(
            power_shaft,
            power_electrical,
            out=efficiency,
            where=power_electrical > 1e-6,
        )
        
        return {
            'current': current,
            'torque': torque,
            'power_electrical': power_electrical,
            'power_shaft': power_shaft,
            'power_loss': power_loss,
            'efficiency': efficiency,
            'back_emf': back_emf,
            'omega': omega,
            'rpm': omega * 30 / np.pi,
            'voltage': V_motor,
        }
    
    def solve_from_voltage_and_torque(
        self, 
        V_motor: ArrayLike, 
        tau_load: ArrayLike
    ) -> Dict[str, ArrayLike]:
        """
        Solve motor state given applied voltage and load torque.
        
        At equilibrium, motor torque equals load torque:
            τ_motor = τ_load
            Kt·(I - I₀) = τ_load
            I = τ_load/Kt + I₀
        
        Then from V = Ke·ω + I·R:
            ω = (V - I·R) / Ke
        
        Args:
            V_motor: Applied motor voltage [V]
            tau_load: Load torque [N·m]
            
        Returns:
            Dict with motor operating state
        """
        # Current from torque equation
        current = tau_load / self.Kt + self.I0
        
        # Speed from electrical equation
        omega = (V_motor - current * self.R) / self.Ke
        
        # Torque equals load (at equilibrium)
        torque = tau_load
        
        # Power calculations
        power_electrical = V_motor * current
        power_shaft = torque * omega
        power_loss = power_electrical - power_shaft
        
        # Efficiency
        efficiency = np.zeros_like(power_electrical)
        np.divide(
            power_shaft,
            power_electrical,
            out=efficiency,
            where=power_electrical > 1e-6,
        )
        
        return {
            'current': current,
            'torque': torque,
            'power_electrical': power_electrical,
            'power_shaft': power_shaft,
            'power_loss': power_loss,
            'efficiency': efficiency,
            'back_emf': self.Ke * omega,
            'omega': omega,
            'rpm': omega * 30 / np.pi,
            'voltage': V_motor,
        }
    
    def solve_from_omega_and_torque(
        self, 
        omega: ArrayLike, 
        tau_load: ArrayLike
    ) -> Dict[str, ArrayLike]:
        """
        Solve motor state given shaft speed and load torque.
        
        This mode determines the voltage required to maintain
        a given speed under a given load.
        
        From τ = Kt·(I - I₀):
            I = τ/Kt + I₀
        
        From V = Ke·ω + I·R:
            V = Ke·ω + (τ/Kt + I₀)·R
        
        Args:
            omega: Angular velocity [rad/s]
            tau_load: Load torque [N·m]
            
        Returns:
            Dict with motor operating state including required voltage
        """
        # Current from torque equation
        current = tau_load / self.Kt + self.I0
        
        # Voltage from electrical equation
        V_motor = self.Ke * omega + current * self.R
        
        # Torque equals load
        torque = tau_load
        
        # Power calculations
        power_electrical = V_motor * current
        power_shaft = torque * omega
        power_loss = power_electrical - power_shaft
        
        # Efficiency
        efficiency = np.zeros_like(power_electrical)
        np.divide(
            power_shaft,
            power_electrical,
            out=efficiency,
            where=power_electrical > 1e-6,
        )
        
        return {
            'current': current,
            'torque': torque,
            'power_electrical': power_electrical,
            'power_shaft': power_shaft,
            'power_loss': power_loss,
            'efficiency': efficiency,
            'back_emf': self.Ke * omega,
            'omega': omega,
            'rpm': omega * 30 / np.pi,
            'voltage': V_motor,
        }
    
    def solve_for_power(
        self,
        V_motor: ArrayLike,
        P_shaft_target: ArrayLike
    ) -> Dict[str, ArrayLike]:
        """
        Solve for operating point that delivers target shaft power.
        
        P_shaft = τ·ω = Kt·(I - I₀)·ω
        
        Also: P_shaft = (V - I·R)·I·(Kt/Ke) - Kt·I₀·(V - I·R)/Ke
        
        This is a quadratic in I. We solve and take the lower current
        solution (more efficient operating point).
        
        Args:
            V_motor: Applied motor voltage [V]
            P_shaft_target: Desired shaft power [W]
            
        Returns:
            Dict with motor operating state
        """
        # Quadratic coefficients for I
        # P = τ·ω = Kt·(I - I₀)·(V - I·R)/Ke
        # P·Ke = Kt·(I - I₀)·(V - I·R)
        # P·Ke = Kt·(I·V - I²·R - I₀·V + I₀·I·R)
        # 0 = -Kt·R·I² + Kt·(V + I₀·R)·I - (Kt·I₀·V + P·Ke)
        
        a = -self.Kt * self.R
        b = self.Kt * (V_motor + self.I0 * self.R)
        c = -(self.Kt * self.I0 * V_motor + P_shaft_target * self.Ke)
        
        # Quadratic formula - take lower root for efficiency
        discriminant = b**2 - 4*a*c
        discriminant = np.maximum(discriminant, 0)  # Clamp for AD
        
        # Lower current = higher efficiency
        I_low = (-b + np.sqrt(discriminant)) / (2*a)
        I_high = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Choose the solution with lower current (more efficient)
        current = np.where(I_low > 0, I_low, I_high)
        
        # Now solve remaining states
        omega = (V_motor - current * self.R) / self.Ke
        torque = self.Kt * (current - self.I0)
        
        power_electrical = V_motor * current
        power_shaft = torque * omega
        power_loss = power_electrical - power_shaft
        
        efficiency = np.zeros_like(power_electrical)
        np.divide(
            power_shaft,
            power_electrical,
            out=efficiency,
            where=power_electrical > 1e-6,
        )
        
        return {
            'current': current,
            'torque': torque,
            'power_electrical': power_electrical,
            'power_shaft': power_shaft,
            'power_loss': power_loss,
            'efficiency': efficiency,
            'back_emf': self.Ke * omega,
            'omega': omega,
            'rpm': omega * 30 / np.pi,
            'voltage': V_motor,
        }
    
    def get_heat_generation(
        self,
        current: ArrayLike,
        omega: ArrayLike
    ) -> Dict[str, ArrayLike]:
        """
        Calculate heat generation breakdown.
        
        Heat sources:
            Q_resistive = I²·R           (copper losses)
            Q_friction = I₀·Ke·ω         (mechanical losses, approximation)
            Q_total = Q_resistive + Q_friction
        
        Args:
            current: Motor current [A]
            omega: Angular velocity [rad/s]
            
        Returns:
            Dict with heat generation components [W]
        """
        Q_resistive = current**2 * self.R
        Q_friction = self.I0 * self.Ke * omega  # Approximation
        Q_total = Q_resistive + Q_friction
        
        return {
            'Q_resistive': Q_resistive,
            'Q_friction': Q_friction,
            'Q_total': Q_total,
        }
    
    def get_limits(
        self,
        V_motor: float,
        T_ambient: float = 25.0,
        T_motor: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate operational limits for given conditions.
        
        Limits include:
            - Current limit (from I_max or thermal)
            - Power limit (from P_max or thermal)
            - Torque limit (from current limit)
            - Speed limit (no-load speed)
        
        Args:
            V_motor: Applied motor voltage [V]
            T_ambient: Ambient temperature [°C]
            T_motor: Current motor temperature [°C] (optional)
            
        Returns:
            Dict with operational limits
        """
        params = self.params
        
        # Base limits from specs
        I_limit_spec = params.I_max
        P_limit_spec = params.P_max
        
        # Thermal derating if temperature provided
        if T_motor is not None:
            T_headroom = params.T_max - T_motor
            T_derate_range = 0.2 * params.T_max  # Start derating at 80% T_max
            
            if T_headroom < T_derate_range:
                derate_factor = max(0, T_headroom / T_derate_range)
            else:
                derate_factor = 1.0
        else:
            derate_factor = 1.0
            
            # Calculate thermal steady-state limit
            # At steady state: Q = I²R = (T_max - T_ambient) / R_thermal
            Q_max = (params.T_max - T_ambient) / params.thermal_resistance
            I_thermal = np.sqrt(Q_max / self.R) if Q_max > 0 else np.inf
            I_limit_spec = min(I_limit_spec, I_thermal)
        
        # Apply derating
        I_limit = I_limit_spec * derate_factor
        P_limit = P_limit_spec * derate_factor
        
        # Derived limits
        tau_limit = self.Kt * (I_limit - self.I0)
        omega_no_load = params.no_load_speed(V_motor)
        omega_limit = omega_no_load  # Speed limited by voltage
        
        # Current limit from power at typical operating point
        # P = V·I, so I_power = P_limit / V
        I_from_power = P_limit / V_motor if V_motor > 0 else np.inf
        I_limit = min(I_limit, I_from_power)
        
        return {
            'I_max': I_limit,
            'P_max': P_limit,
            'tau_max': tau_limit,
            'omega_max': omega_limit,
            'rpm_max': omega_limit * 30 / np.pi,
            'derate_factor': derate_factor,
            'I_stall': params.stall_current(V_motor),
            'tau_stall': params.stall_torque(V_motor),
        }
    
    def get_efficiency_map(
        self,
        V_motor: float,
        omega_range: Optional[np.ndarray] = None,
        torque_range: Optional[np.ndarray] = None,
        n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Generate efficiency map for motor at given voltage.
        
        Args:
            V_motor: Applied motor voltage [V]
            omega_range: Speed range [rad/s], defaults to 0 to no-load
            torque_range: Torque range [N·m], defaults to 0 to stall
            n_points: Grid resolution
            
        Returns:
            Dict with omega_grid, torque_grid, efficiency_grid, power_grid
        """
        params = self.params
        
        # Default ranges
        if omega_range is None:
            omega_max = params.no_load_speed(V_motor)
            omega_range = np.linspace(0.05 * omega_max, 0.95 * omega_max, n_points)
        
        if torque_range is None:
            tau_max = params.stall_torque(V_motor)
            torque_range = np.linspace(0.05 * tau_max, 0.95 * tau_max, n_points)
        
        # Create grid
        omega_grid, torque_grid = np.meshgrid(omega_range, torque_range)
        
        # Calculate required current for each (omega, tau) point
        current_grid = torque_grid / self.Kt + self.I0
        
        # Calculate required voltage (may exceed V_motor)
        V_required = self.Ke * omega_grid + current_grid * self.R
        
        # Mask infeasible points
        feasible = V_required <= V_motor * 1.01  # 1% tolerance
        
        # Calculate efficiency
        power_in = V_motor * current_grid
        power_out = torque_grid * omega_grid
        efficiency_grid = np.where(
            feasible & (power_in > 1e-6),
            power_out / power_in,
            np.nan
        )
        
        return {
            'omega': omega_grid,
            'torque': torque_grid,
            'efficiency': efficiency_grid,
            'power_out': power_out,
            'power_in': power_in,
            'current': current_grid,
            'feasible': feasible,
        }


# =============================================================================
# Common Motor Presets
# =============================================================================

MOTOR_PRESETS: Dict[str, MotorParameters] = {
    # Small multirotors / micro wings
    '1806_2300KV': MotorParameters(
        kv=2300,
        R_internal=0.150,
        I_no_load=0.6,
        I_max=12,
        P_max=120,
        mass=0.024,
        thermal_resistance=8.0,
        thermal_capacitance=20.0,
        T_max=150,
        name="1806 2300KV",
    ),
    
    # Common 5" quad / small wing motor  
    '2212_1000KV': MotorParameters(
        kv=1000,
        R_internal=0.090,
        I_no_load=0.8,
        I_max=20,
        P_max=250,
        mass=0.053,
        thermal_resistance=6.0,
        thermal_capacitance=35.0,
        T_max=150,
        name="2212 1000KV",
    ),
    
    '2212_920KV': MotorParameters(
        kv=920,
        R_internal=0.085,
        I_no_load=0.7,
        I_max=18,
        P_max=220,
        mass=0.056,
        thermal_resistance=6.0,
        thermal_capacitance=35.0,
        T_max=150,
        name="2212 920KV",
    ),
    
    # Medium wing / large quad
    '2814_900KV': MotorParameters(
        kv=900,
        R_internal=0.055,
        I_no_load=1.0,
        I_max=35,
        P_max=500,
        mass=0.098,
        thermal_resistance=4.5,
        thermal_capacitance=60.0,
        T_max=150,
        name="2814 900KV",
    ),
    
    '2814_700KV': MotorParameters(
        kv=700,
        R_internal=0.065,
        I_no_load=0.9,
        I_max=30,
        P_max=450,
        mass=0.098,
        thermal_resistance=4.5,
        thermal_capacitance=60.0,
        T_max=150,
        name="2814 700KV",
    ),
    
    # Larger flying wing
    '3508_700KV': MotorParameters(
        kv=700,
        R_internal=0.040,
        I_no_load=1.2,
        I_max=40,
        P_max=700,
        mass=0.135,
        thermal_resistance=3.5,
        thermal_capacitance=80.0,
        T_max=150,
        name="3508 700KV",
    ),
    
    '3508_580KV': MotorParameters(
        kv=580,
        R_internal=0.050,
        I_no_load=1.0,
        I_max=35,
        P_max=600,
        mass=0.135,
        thermal_resistance=3.5,
        thermal_capacitance=80.0,
        T_max=150,
        name="3508 580KV",
    ),
    
    # Large fixed-wing
    '4010_370KV': MotorParameters(
        kv=370,
        R_internal=0.045,
        I_no_load=1.3,
        I_max=45,
        P_max=800,
        mass=0.195,
        thermal_resistance=3.0,
        thermal_capacitance=100.0,
        T_max=150,
        name="4010 370KV",
    ),
    
    # High-power (large aircraft)
    '5010_360KV': MotorParameters(
        kv=360,
        R_internal=0.030,
        I_no_load=1.5,
        I_max=60,
        P_max=1200,
        mass=0.280,
        thermal_resistance=2.5,
        thermal_capacitance=140.0,
        T_max=150,
        name="5010 360KV",
    ),
}


def get_motor_preset(name: str) -> MotorParameters:
    """
    Get a pre-defined motor by name.
    
    Args:
        name: Motor preset name (e.g., '2212_1000KV')
        
    Returns:
        MotorParameters instance
        
    Raises:
        ValueError: If preset not found
    """
    if name not in MOTOR_PRESETS:
        available = ", ".join(sorted(MOTOR_PRESETS.keys()))
        raise ValueError(f"Unknown motor preset '{name}'. Available: {available}")
    
    # Return a copy to prevent mutation
    import copy
    return copy.deepcopy(MOTOR_PRESETS[name])


def list_motor_presets() -> list:
    """List available motor preset names."""
    return sorted(MOTOR_PRESETS.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_motor_params_from_spec(
    kv: float,
    P_max: float,
    V_nominal: float,
    efficiency_peak: float = 0.85
) -> MotorParameters:
    """
    Estimate motor parameters from common datasheet specs.
    
    This is an approximation when full motor data isn't available.
    The Drela model parameters are estimated from:
    - Peak power and voltage give typical current
    - Peak efficiency constrains I₀ and R relationship
    
    Args:
        kv: Motor Kv [rpm/V]
        P_max: Maximum power rating [W]
        V_nominal: Nominal operating voltage [V]
        efficiency_peak: Typical peak efficiency (default 0.85)
        
    Returns:
        Estimated MotorParameters
    """
    # Typical max current
    I_max = P_max / V_nominal
    
    # Kt from kv
    Kt = 60.0 / (2 * np.pi * kv)
    
    # At peak efficiency, losses are minimized
    # Empirical: R ~ 0.5 * V / I_max for hobby motors (rough)
    R_internal = 0.4 * V_nominal / I_max
    
    # No-load current typically 3-8% of max current
    I_no_load = 0.05 * I_max
    
    # Estimate mass from power (empirical correlation)
    # Roughly 0.1-0.2 g per watt for quality motors
    mass = P_max * 0.00015  # kg
    
    return MotorParameters(
        kv=kv,
        R_internal=R_internal,
        I_no_load=I_no_load,
        I_max=I_max,
        P_max=P_max,
        mass=mass,
    )
