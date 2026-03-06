"""
6-DOF Aircraft Dynamics Model

Full rigid-body dynamics using Euler angle parameterization, extended
with propulsion (SOC) and thermal states for integrated simulation.

State Vector (16 states):
    0-2:   x_e, y_e, z_e     - Earth position [m] (NED: North-East-Down)
    3-5:   u_b, v_b, w_b     - Body velocity [m/s] (forward, right, down)
    6-8:   phi, theta, psi   - Euler angles [rad] (roll, pitch, yaw)
    9-11:  p, q, r           - Angular rates [rad/s] (body axes)
    12:    SOC               - Battery state of charge [0-1]
    13-15: T_motor, T_esc, T_battery - Temperatures [C]

Control Vector:
    throttle    - Throttle command [0-1]
    elevator    - Elevator deflection [deg or rad]
    aileron     - Aileron deflection [deg or rad]
    rudder      - Rudder deflection [deg or rad]

Coordinate Frames:
    Earth (NED): x=North, y=East, z=Down (altitude = -z_e)
    Body: x=Forward, y=Right, z=Down

References:
    - Drela, "Flight Vehicle Aerodynamics", Section 9.8
    - Stevens & Lewis, "Aircraft Control and Simulation"
    - AeroSandbox DynamicsRigidBody3DBodyEuler
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np

# Try to use AeroSandbox numpy for AD compatibility
try:
    import aerosandbox.numpy as asnp
except ImportError:
    asnp = np

# Type alias
ArrayLike = Union[float, np.ndarray]


@dataclass
class MassProperties:
    """
    Aircraft mass properties.
    
    Attributes:
        mass: Total mass [kg]
        Ixx, Iyy, Izz: Principal moments of inertia [kg*m^2]
        Ixz: Product of inertia (symmetric aircraft) [kg*m^2]
        Ixy, Iyz: Cross products (usually zero for symmetric) [kg*m^2]
        cg_x, cg_y, cg_z: CG position in body frame [m]
    """
    mass: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float = 0.0
    Ixy: float = 0.0
    Iyz: float = 0.0
    cg_x: float = 0.0
    cg_y: float = 0.0
    cg_z: float = 0.0
    
    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")


@dataclass
class AircraftState:
    """
    Complete aircraft state vector.
    
    Uses NED (North-East-Down) Earth frame and standard body axes.
    """
    # Position (Earth frame, NED)
    x_e: float = 0.0        # North position [m]
    y_e: float = 0.0        # East position [m]
    z_e: float = 0.0        # Down position [m] (altitude = -z_e)
    
    # Velocity (Body frame)
    u_b: float = 0.0        # Forward velocity [m/s]
    v_b: float = 0.0        # Right velocity [m/s]
    w_b: float = 0.0        # Down velocity [m/s]
    
    # Euler angles (yaw-pitch-roll convention)
    phi: float = 0.0        # Roll angle [rad]
    theta: float = 0.0      # Pitch angle [rad]
    psi: float = 0.0        # Yaw angle [rad]
    
    # Angular rates (Body frame)
    p: float = 0.0          # Roll rate [rad/s]
    q: float = 0.0          # Pitch rate [rad/s]
    r: float = 0.0          # Yaw rate [rad/s]
    
    # Propulsion state
    SOC: float = 1.0        # Battery state of charge [0-1]
    
    # Thermal states
    T_motor: float = 25.0   # Motor temperature [C]
    T_esc: float = 25.0     # ESC temperature [C]
    T_battery: float = 25.0 # Battery temperature [C]
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array (16 elements)."""
        return np.array([
            self.x_e, self.y_e, self.z_e,
            self.u_b, self.v_b, self.w_b,
            self.phi, self.theta, self.psi,
            self.p, self.q, self.r,
            self.SOC,
            self.T_motor, self.T_esc, self.T_battery,
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AircraftState':
        """Create from numpy array."""
        return cls(
            x_e=arr[0], y_e=arr[1], z_e=arr[2],
            u_b=arr[3], v_b=arr[4], w_b=arr[5],
            phi=arr[6], theta=arr[7], psi=arr[8],
            p=arr[9], q=arr[10], r=arr[11],
            SOC=arr[12],
            T_motor=arr[13], T_esc=arr[14], T_battery=arr[15],
        )
    
    @property
    def altitude(self) -> float:
        """Altitude above ground [m] (assuming ground at z=0)."""
        return -self.z_e
    
    @property
    def airspeed(self) -> float:
        """Total airspeed [m/s]."""
        return np.sqrt(self.u_b**2 + self.v_b**2 + self.w_b**2)
    
    @property
    def alpha(self) -> float:
        """Angle of attack [rad]."""
        return np.arctan2(self.w_b, self.u_b)
    
    @property
    def beta(self) -> float:
        """Sideslip angle [rad]."""
        V = self.airspeed
        if V < 1e-6:
            return 0.0
        return np.arcsin(self.v_b / V)
    
    @property
    def temperatures(self) -> Dict[str, float]:
        """Get thermal state as dict."""
        return {
            'motor': self.T_motor,
            'esc': self.T_esc,
            'battery': self.T_battery,
        }


@dataclass
class ControlInputs:
    """
    Aircraft control inputs.
    
    Attributes:
        throttle: Throttle command [0-1]
        elevator: Elevator deflection [deg]
        aileron: Aileron deflection [deg] (positive = right roll)
        rudder: Rudder deflection [deg] (positive = right yaw)
    """
    throttle: float = 0.0
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.throttle, self.elevator, self.aileron, self.rudder])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlInputs':
        """Create from numpy array."""
        return cls(
            throttle=arr[0],
            elevator=arr[1] if len(arr) > 1 else 0,
            aileron=arr[2] if len(arr) > 2 else 0,
            rudder=arr[3] if len(arr) > 3 else 0,
        )


# State indices
class StateIndex:
    """Indices for state vector elements."""
    X_E = 0
    Y_E = 1
    Z_E = 2
    U_B = 3
    V_B = 4
    W_B = 5
    PHI = 6
    THETA = 7
    PSI = 8
    P = 9
    Q = 10
    R = 11
    SOC = 12
    T_MOTOR = 13
    T_ESC = 14
    T_BATTERY = 15
    N_STATES = 16


def rotation_matrix_body_to_earth(phi: ArrayLike, theta: ArrayLike, psi: ArrayLike) -> np.ndarray:
    """
    Compute rotation matrix from body to Earth frame.
    
    Uses ZYX (yaw-pitch-roll) Euler angle convention.
    
    Args:
        phi: Roll angle [rad]
        theta: Pitch angle [rad]
        psi: Yaw angle [rad]
        
    Returns:
        3x3 rotation matrix R_be such that v_earth = R_be @ v_body
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(theta)
    sthe = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    
    R = np.array([
        [cthe*cpsi, sphi*sthe*cpsi - cphi*spsi, cphi*sthe*cpsi + sphi*spsi],
        [cthe*spsi, sphi*sthe*spsi + cphi*cpsi, cphi*sthe*spsi - sphi*cpsi],
        [-sthe,     sphi*cthe,                  cphi*cthe                 ],
    ])
    
    return R


def rotation_matrix_earth_to_body(phi: ArrayLike, theta: ArrayLike, psi: ArrayLike) -> np.ndarray:
    """
    Compute rotation matrix from Earth to body frame.
    
    R_eb = R_be.T
    """
    return rotation_matrix_body_to_earth(phi, theta, psi).T


class FlyingWingDynamics6DOF:
    """
    Full 6-DOF dynamics for flying wing aircraft.
    
    Computes state derivatives for rigid-body dynamics extended with
    propulsion (SOC) and thermal states.
    
    Features:
        - Full 6-DOF rigid body equations of motion
        - Ground effect aerodynamic corrections
        - Integrated propulsion system (throttle -> thrust + SOC dynamics)
        - Thermal state dynamics (component temperatures)
        - Wind support (steady wind in Earth frame)
    
    Example:
        >>> dynamics = FlyingWingDynamics6DOF(
        ...     mass_props=mass_props,
        ...     aero_model=aero_func,
        ...     propulsion=propulsion_system,
        ... )
        >>> state = AircraftState(u_b=20, altitude=100)
        >>> controls = ControlInputs(throttle=0.5, elevator=-2)
        >>> derivs = dynamics.compute_derivatives(state, controls)
    """
    
    def __init__(
        self,
        mass_props: MassProperties,
        aero_model: Callable,
        propulsion: Optional[object] = None,
        reference_area: float = 1.0,
        reference_chord: float = 0.3,
        reference_span: float = 2.0,
        include_ground_effect: bool = True,
        include_thermal: bool = True,
        gravity: float = 9.81,
    ):
        """
        Initialize dynamics model.
        
        Args:
            mass_props: Aircraft mass properties
            aero_model: Callable that returns aero forces/moments
                        aero_model(alpha, beta, airspeed, p, q, r, controls) -> dict
                        Returns: CL, CD, Cm, Cl, Cn, CY (or forces directly)
            propulsion: IntegratedPropulsionSystem (optional)
            reference_area: Wing reference area [m^2]
            reference_chord: Mean aerodynamic chord [m]
            reference_span: Wing span [m]
            include_ground_effect: Apply ground effect corrections
            include_thermal: Track thermal states
            gravity: Gravitational acceleration [m/s^2]
        """
        self.mass_props = mass_props
        self.aero_model = aero_model
        self.propulsion = propulsion
        
        self.S_ref = reference_area
        self.c_ref = reference_chord
        self.b_ref = reference_span
        
        self.include_ground_effect = include_ground_effect
        self.include_thermal = include_thermal
        self.g = gravity
    
    def compute_ground_effect_factor(self, altitude: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute ground effect correction factors.
        
        Based on Prandtl's ground effect model.
        
        Args:
            altitude: Height above ground [m]
            
        Returns:
            Tuple of (CL_factor, CD_factor)
            CL_factor > 1 (lift augmentation)
            CD_factor < 1 (induced drag reduction)
        """
        if not self.include_ground_effect:
            return 1.0, 1.0
        
        # Normalized height (altitude / wingspan)
        h_b = altitude / self.b_ref
        
        # Prandtl ground effect factor
        # phi_ge = 1 when far from ground, approaches 0 near ground
        phi_ge = (16 * h_b)**2 / (1 + (16 * h_b)**2)
        
        # Ground effect is significant when h < b
        # CL increases slightly, CD_induced decreases significantly
        CL_factor = 1 + 0.1 * (1 - phi_ge)  # Up to 10% lift increase
        CD_factor = phi_ge + (1 - phi_ge) * 0.5  # Up to 50% induced drag reduction
        
        return CL_factor, CD_factor
    
    def compute_aero_forces_moments(
        self,
        state: AircraftState,
        controls: ControlInputs,
        rho: float = 1.225,
    ) -> Dict[str, float]:
        """
        Compute aerodynamic forces and moments.
        
        Args:
            state: Aircraft state
            controls: Control inputs
            rho: Air density [kg/m^3]
            
        Returns:
            Dict with Fx_aero, Fy_aero, Fz_aero, Mx_aero, My_aero, Mz_aero (body frame)
        """
        V = state.airspeed
        alpha = state.alpha
        beta = state.beta
        
        # Avoid issues at zero airspeed
        if V < 0.1:
            return {
                'Fx_aero': 0.0, 'Fy_aero': 0.0, 'Fz_aero': 0.0,
                'Mx_aero': 0.0, 'My_aero': 0.0, 'Mz_aero': 0.0,
            }
        
        # Dynamic pressure
        q_bar = 0.5 * rho * V**2
        
        # Get aero coefficients from model
        aero = self.aero_model(
            alpha=np.degrees(alpha),
            beta=np.degrees(beta),
            airspeed=V,
            p=state.p,
            q=state.q,
            r=state.r,
            elevator=controls.elevator,
            aileron=controls.aileron,
            rudder=controls.rudder,
        )
        
        # Extract coefficients
        CL = aero.get('CL', 0.0)
        CD = aero.get('CD', 0.0)
        CY = aero.get('CY', 0.0)
        Cl = aero.get('Cl', 0.0)  # Roll moment
        Cm = aero.get('Cm', 0.0)  # Pitch moment
        Cn = aero.get('Cn', 0.0)  # Yaw moment
        
        # Apply ground effect corrections
        CL_factor, CD_factor = self.compute_ground_effect_factor(state.altitude)
        CL = CL * CL_factor
        CD = CD * CD_factor
        
        # Convert from stability to body axes
        # Lift is perpendicular to velocity, drag is parallel
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        # Forces in body frame
        Fx_aero = q_bar * self.S_ref * (-CD * ca + CL * sa)
        Fy_aero = q_bar * self.S_ref * CY
        Fz_aero = q_bar * self.S_ref * (-CD * sa - CL * ca)
        
        # Moments in body frame
        Mx_aero = q_bar * self.S_ref * self.b_ref * Cl
        My_aero = q_bar * self.S_ref * self.c_ref * Cm
        Mz_aero = q_bar * self.S_ref * self.b_ref * Cn
        
        return {
            'Fx_aero': Fx_aero,
            'Fy_aero': Fy_aero,
            'Fz_aero': Fz_aero,
            'Mx_aero': Mx_aero,
            'My_aero': My_aero,
            'Mz_aero': Mz_aero,
        }
    
    def compute_propulsion_forces(
        self,
        state: AircraftState,
        controls: ControlInputs,
        rho: float = 1.225,
        T_ambient: float = 25.0,
    ) -> Dict[str, float]:
        """
        Compute propulsion forces, moments, and state derivatives.
        
        Uses the new multi-motor support with proper moment arm calculations.
        Supports differential thrust if controls.throttle is an array.
        
        Args:
            state: Aircraft state
            controls: Control inputs (throttle - scalar or per-motor array)
            rho: Air density [kg/m^3]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Dict with:
                - Fx_prop, Fy_prop, Fz_prop: Total propulsion force [N]
                - Mx_prop, My_prop, Mz_prop: Total propulsion moment [N·m]
                - dSOC_dt: Battery SOC derivative [1/s]
                - dT_motor_dt, dT_esc_dt, dT_battery_dt: Thermal derivatives [C/s]
                - thrust: Total thrust magnitude [N]
                - power: Battery power [W]
        """
        if self.propulsion is None:
            return {
                'Fx_prop': 0.0, 'Fy_prop': 0.0, 'Fz_prop': 0.0,
                'Mx_prop': 0.0, 'My_prop': 0.0, 'Mz_prop': 0.0,
                'dSOC_dt': 0.0,
                'dT_motor_dt': 0.0, 'dT_esc_dt': 0.0, 'dT_battery_dt': 0.0,
                'thrust': 0.0, 'power': 0.0,
            }
        
        # Get propulsion operating point (supports per-motor throttle array)
        prop_result = self.propulsion.solve_equilibrium(
            throttle=controls.throttle,
            V_freestream=state.airspeed,
            rho=rho,
            SOC=state.SOC,
            temperatures=state.temperatures,
            T_ambient=T_ambient,
        )
        
        # Extract force and moment vectors from solve_equilibrium
        # These are computed with proper motor mount positions and torque reactions
        force_body = prop_result.get('force_body', None)
        moment_body = prop_result.get('moment_body', None)
        
        if force_body is not None and moment_body is not None:
            # New multi-motor path with proper moment arms
            Fx_prop = force_body[0]
            Fy_prop = force_body[1]
            Fz_prop = force_body[2]
            Mx_prop = moment_body[0]
            My_prop = moment_body[1]
            Mz_prop = moment_body[2]
        else:
            # Fallback for backward compatibility
            thrust = prop_result['thrust_total']
            thrust_vector = getattr(self.propulsion.config, 'thrust_vector', np.array([1, 0, 0]))
            Fx_prop = thrust * thrust_vector[0]
            Fy_prop = thrust * thrust_vector[1]
            Fz_prop = thrust * thrust_vector[2]
            Mx_prop = 0.0
            My_prop = 0.0
            Mz_prop = 0.0
        
        # SOC derivative
        dSOC_dt = prop_result['dSOC_dt']
        
        # Thermal derivatives
        if self.include_thermal and hasattr(self.propulsion, 'get_temperature_derivatives'):
            heat_inputs = {
                'motor': prop_result['heat_motor'],
                'esc': prop_result['heat_esc'],
                'battery': prop_result['heat_battery'],
            }
            dT = self.propulsion.get_temperature_derivatives(
                temperatures=state.temperatures,
                heat_inputs=heat_inputs,
                V_freestream=state.airspeed,
                T_ambient=T_ambient,
            )
            dT_motor_dt = dT.get('motor', 0.0)
            dT_esc_dt = dT.get('esc', 0.0)
            dT_battery_dt = dT.get('battery', 0.0)
        else:
            dT_motor_dt = 0.0
            dT_esc_dt = 0.0
            dT_battery_dt = 0.0
        
        return {
            'Fx_prop': Fx_prop,
            'Fy_prop': Fy_prop,
            'Fz_prop': Fz_prop,
            'Mx_prop': Mx_prop,
            'My_prop': My_prop,
            'Mz_prop': Mz_prop,
            'dSOC_dt': dSOC_dt,
            'dT_motor_dt': dT_motor_dt,
            'dT_esc_dt': dT_esc_dt,
            'dT_battery_dt': dT_battery_dt,
            'thrust': prop_result['thrust_total'],
            'power': prop_result['power_battery'],
        }
    
    def compute_gravity_forces(self, state: AircraftState) -> Dict[str, float]:
        """
        Compute gravity force in body frame.
        
        Args:
            state: Aircraft state (for attitude)
            
        Returns:
            Dict with Fx_grav, Fy_grav, Fz_grav
        """
        m = self.mass_props.mass
        g = self.g
        
        # Gravity in Earth frame: [0, 0, m*g] (down is positive z)
        # Transform to body frame using R_eb = R_be.T
        phi = state.phi
        theta = state.theta
        
        # Gravity components in body frame
        Fx_grav = -m * g * np.sin(theta)
        Fy_grav = m * g * np.cos(theta) * np.sin(phi)
        Fz_grav = m * g * np.cos(theta) * np.cos(phi)
        
        return {
            'Fx_grav': Fx_grav,
            'Fy_grav': Fy_grav,
            'Fz_grav': Fz_grav,
        }
    
    def compute_derivatives(
        self,
        state: AircraftState,
        controls: ControlInputs,
        rho: float = 1.225,
        T_ambient: float = 25.0,
        wind_ned: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all state derivatives.
        
        This is the main dynamics function: dx/dt = f(x, u)
        
        Args:
            state: Current aircraft state
            controls: Control inputs
            rho: Air density [kg/m^3]
            T_ambient: Ambient temperature [C]
            wind_ned: Wind velocity in NED frame [m/s] (optional)
            
        Returns:
            Dict of state derivatives (same keys as state)
        """
        # Shorthand
        m = self.mass_props.mass
        Ixx = self.mass_props.Ixx
        Iyy = self.mass_props.Iyy
        Izz = self.mass_props.Izz
        Ixz = self.mass_props.Ixz
        
        phi = state.phi
        theta = state.theta
        psi = state.psi
        u = state.u_b
        v = state.v_b
        w = state.w_b
        p = state.p
        q = state.q
        r = state.r
        
        # Trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        # Compute all forces
        grav = self.compute_gravity_forces(state)
        aero = self.compute_aero_forces_moments(state, controls, rho)
        prop = self.compute_propulsion_forces(state, controls, rho, T_ambient)
        
        # Total forces and moments
        Fx = grav['Fx_grav'] + aero['Fx_aero'] + prop['Fx_prop']
        Fy = grav['Fy_grav'] + aero['Fy_aero'] + prop['Fy_prop']
        Fz = grav['Fz_grav'] + aero['Fz_aero'] + prop['Fz_prop']
        
        L = aero['Mx_aero'] + prop['Mx_prop']  # Roll moment
        M = aero['My_aero'] + prop['My_prop']  # Pitch moment
        N = aero['Mz_aero'] + prop['Mz_prop']  # Yaw moment
        
        # ============ Translational Dynamics ============
        # Body frame accelerations (with Coriolis terms)
        # m * (du/dt + q*w - r*v) = Fx
        du_dt = Fx / m - q * w + r * v
        dv_dt = Fy / m - r * u + p * w
        dw_dt = Fz / m - p * v + q * u
        
        # ============ Rotational Dynamics ============
        # Euler equations with Ixz coupling (symmetric aircraft)
        # Simplified form assuming Ixy = Iyz = 0
        
        # Determinant for coupled equations
        Gamma = Ixx * Izz - Ixz**2
        
        if abs(Gamma) < 1e-10:
            # Fallback for diagonal inertia
            dp_dt = L / Ixx
            dq_dt = M / Iyy
            dr_dt = N / Izz
        else:
            # Full coupled equations
            # From Stevens & Lewis
            c1 = ((Ixx - Iyy + Izz) * Ixz) / Gamma
            c2 = (Izz * (Izz - Iyy) + Ixz**2) / Gamma
            c3 = Izz / Gamma
            c4 = Ixz / Gamma
            c5 = (Izz - Ixx) / Iyy
            c6 = Ixz / Iyy
            c7 = 1.0 / Iyy
            c8 = (Ixx * (Ixx - Iyy) + Ixz**2) / Gamma
            c9 = Ixx / Gamma
            
            dp_dt = c1 * p * q - c2 * q * r + c3 * L + c4 * N
            dq_dt = c5 * p * r - c6 * (p**2 - r**2) + c7 * M
            dr_dt = c8 * p * q - c1 * q * r + c4 * L + c9 * N
        
        # ============ Kinematic Equations ============
        # Position derivatives (body velocity -> Earth position)
        dx_e = (cthe * cpsi) * u + (sphi * sthe * cpsi - cphi * spsi) * v + (cphi * sthe * cpsi + sphi * spsi) * w
        dy_e = (cthe * spsi) * u + (sphi * sthe * spsi + cphi * cpsi) * v + (cphi * sthe * spsi - sphi * cpsi) * w
        dz_e = -sthe * u + sphi * cthe * v + cphi * cthe * w
        
        # Euler angle derivatives
        # Handle singularity at theta = +/- 90 degrees
        if abs(cthe) < 1e-6:
            dphi_dt = p
            dthe_dt = q * cphi - r * sphi
            dpsi_dt = 0
        else:
            tthe = sthe / cthe
            dphi_dt = p + (q * sphi + r * cphi) * tthe
            dthe_dt = q * cphi - r * sphi
            dpsi_dt = (q * sphi + r * cphi) / cthe
        
        # ============ Propulsion/Thermal Derivatives ============
        dSOC_dt = prop['dSOC_dt']
        dT_motor_dt = prop['dT_motor_dt']
        dT_esc_dt = prop['dT_esc_dt']
        dT_battery_dt = prop['dT_battery_dt']
        
        return {
            # Position
            'd_x_e': dx_e,
            'd_y_e': dy_e,
            'd_z_e': dz_e,
            # Velocity
            'd_u_b': du_dt,
            'd_v_b': dv_dt,
            'd_w_b': dw_dt,
            # Angles
            'd_phi': dphi_dt,
            'd_theta': dthe_dt,
            'd_psi': dpsi_dt,
            # Angular rates
            'd_p': dp_dt,
            'd_q': dq_dt,
            'd_r': dr_dt,
            # Propulsion
            'd_SOC': dSOC_dt,
            # Thermal
            'd_T_motor': dT_motor_dt,
            'd_T_esc': dT_esc_dt,
            'd_T_battery': dT_battery_dt,
            # Auxiliary outputs
            'thrust': prop.get('thrust', 0),
            'power': prop.get('power', 0),
            'Fx_total': Fx,
            'Fy_total': Fy,
            'Fz_total': Fz,
            'L_total': L,
            'M_total': M,
            'N_total': N,
        }
    
    def derivatives_array(
        self,
        t: float,
        state_array: np.ndarray,
        controls: ControlInputs,
        rho: float = 1.225,
        T_ambient: float = 25.0,
    ) -> np.ndarray:
        """
        Compute derivatives in array form for ODE solvers.
        
        Args:
            t: Time (unused but required for ODE interface)
            state_array: State as numpy array (16 elements)
            controls: Control inputs
            rho: Air density [kg/m^3]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Derivatives as numpy array (16 elements)
        """
        state = AircraftState.from_array(state_array)
        derivs = self.compute_derivatives(state, controls, rho, T_ambient)
        
        return np.array([
            derivs['d_x_e'], derivs['d_y_e'], derivs['d_z_e'],
            derivs['d_u_b'], derivs['d_v_b'], derivs['d_w_b'],
            derivs['d_phi'], derivs['d_theta'], derivs['d_psi'],
            derivs['d_p'], derivs['d_q'], derivs['d_r'],
            derivs['d_SOC'],
            derivs['d_T_motor'], derivs['d_T_esc'], derivs['d_T_battery'],
        ])
    
    def simulate(
        self,
        initial_state: AircraftState,
        controls: Union[ControlInputs, Callable],
        duration: float,
        dt: float = 0.01,
        rho: float = 1.225,
        T_ambient: float = 25.0,
        method: str = 'rk4',
    ) -> Dict[str, np.ndarray]:
        """
        Simulate aircraft dynamics over time.
        
        Args:
            initial_state: Starting state
            controls: ControlInputs (constant) or callable(t) -> ControlInputs
            duration: Simulation duration [s]
            dt: Time step [s]
            rho: Air density [kg/m^3]
            T_ambient: Ambient temperature [C]
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            Dict with 'time' and state arrays
        """
        n_steps = int(duration / dt) + 1
        time = np.linspace(0, duration, n_steps)
        
        # Allocate output arrays
        states = np.zeros((n_steps, StateIndex.N_STATES))
        states[0, :] = initial_state.to_array()
        
        # Auxiliary outputs
        thrust = np.zeros(n_steps)
        power = np.zeros(n_steps)
        
        for i in range(1, n_steps):
            t = time[i-1]
            x = states[i-1, :]
            
            # Get controls at current time
            if callable(controls):
                ctrl = controls(t)
            else:
                ctrl = controls
            
            if method == 'euler':
                # Simple Euler integration
                dx = self.derivatives_array(t, x, ctrl, rho, T_ambient)
                states[i, :] = x + dx * dt
            else:
                # RK4 integration
                k1 = self.derivatives_array(t, x, ctrl, rho, T_ambient)
                k2 = self.derivatives_array(t + dt/2, x + k1*dt/2, ctrl, rho, T_ambient)
                k3 = self.derivatives_array(t + dt/2, x + k2*dt/2, ctrl, rho, T_ambient)
                k4 = self.derivatives_array(t + dt, x + k3*dt, ctrl, rho, T_ambient)
                states[i, :] = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            
            # Clamp SOC to [0, 1]
            states[i, StateIndex.SOC] = np.clip(states[i, StateIndex.SOC], 0, 1)
            
            # Store auxiliary outputs
            state_i = AircraftState.from_array(states[i, :])
            derivs = self.compute_derivatives(state_i, ctrl, rho, T_ambient)
            thrust[i] = derivs.get('thrust', 0)
            power[i] = derivs.get('power', 0)
        
        return {
            'time': time,
            'x_e': states[:, StateIndex.X_E],
            'y_e': states[:, StateIndex.Y_E],
            'z_e': states[:, StateIndex.Z_E],
            'u_b': states[:, StateIndex.U_B],
            'v_b': states[:, StateIndex.V_B],
            'w_b': states[:, StateIndex.W_B],
            'phi': states[:, StateIndex.PHI],
            'theta': states[:, StateIndex.THETA],
            'psi': states[:, StateIndex.PSI],
            'p': states[:, StateIndex.P],
            'q': states[:, StateIndex.Q],
            'r': states[:, StateIndex.R],
            'SOC': states[:, StateIndex.SOC],
            'T_motor': states[:, StateIndex.T_MOTOR],
            'T_esc': states[:, StateIndex.T_ESC],
            'T_battery': states[:, StateIndex.T_BATTERY],
            'altitude': -states[:, StateIndex.Z_E],
            'airspeed': np.sqrt(states[:, StateIndex.U_B]**2 + 
                               states[:, StateIndex.V_B]**2 + 
                               states[:, StateIndex.W_B]**2),
            'thrust': thrust,
            'power': power,
            'states': states,
        }


# =============================================================================
# Simple Aero Model for Testing
# =============================================================================

def create_simple_aero_model(
    CL0: float = 0.3,
    CLa: float = 5.5,  # per rad
    CD0: float = 0.03,
    K: float = 0.04,   # induced drag factor
    Cma: float = -1.0, # pitch stability (per rad)
    Cmq: float = -15.0,
    Cmde: float = -1.0, # per deg
    Clp: float = -0.5,
    Clda: float = 0.1,  # per deg
    Cnr: float = -0.2,
    Cndr: float = -0.05, # per deg
) -> Callable:
    """
    Create a simple linear aerodynamic model for testing.
    
    Returns a callable suitable for FlyingWingDynamics6DOF.
    """
    def aero_model(
        alpha: float,  # deg
        beta: float,   # deg
        airspeed: float,
        p: float, q: float, r: float,
        elevator: float = 0, aileron: float = 0, rudder: float = 0,
    ) -> Dict[str, float]:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        
        # Lift and drag
        CL = CL0 + CLa * alpha_rad + Cmde * 0.1 * elevator  # Simple elevator effect
        CD = CD0 + K * CL**2
        
        # Lateral
        CY = -0.3 * beta_rad
        
        # Moments (simplified)
        c = 0.3  # reference chord for rate normalization
        b = 2.0  # span
        V = max(airspeed, 1.0)
        
        Cm = Cma * alpha_rad + Cmq * (q * c / (2 * V)) + Cmde * np.radians(elevator)
        Cl = Clp * (p * b / (2 * V)) + Clda * np.radians(aileron)
        Cn = Cnr * (r * b / (2 * V)) + Cndr * np.radians(rudder)
        
        return {
            'CL': CL, 'CD': CD, 'CY': CY,
            'Cl': Cl, 'Cm': Cm, 'Cn': Cn,
        }
    
    return aero_model
