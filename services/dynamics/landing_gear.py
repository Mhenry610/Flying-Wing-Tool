"""
Landing Gear Model

Spring-damper strut model with tire friction for takeoff/landing phases.

Features:
    - Strut compression dynamics (spring-damper)
    - Tire vertical forces (spring-damper in series with strut)
    - Tire friction: rolling, braking, lateral
    - Ground contact detection
    - Support for tricycle, taildragger, and flying wing skid configs

Coordinate Frames:
    Body: x=Forward, y=Right, z=Down
    Earth (NED): x=North, y=East, z=Down

References:
    - Section 8 of Propulsion-Mission_SPEC.md
    - JSBSim ground reactions model
    - Roskam, "Airplane Flight Dynamics and Automatic Flight Controls"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import numpy as np

# Type alias
ArrayLike = Union[float, np.ndarray]


class GearType(Enum):
    """Landing gear configuration types."""
    TRICYCLE = auto()      # Nose wheel + 2 main
    TAILDRAGGER = auto()   # 2 main + tail wheel
    FLYING_WING = auto()   # Center skid + wingtip skids
    CUSTOM = auto()        # User-defined


class SurfaceType(Enum):
    """Runway/ground surface types."""
    CONCRETE = auto()
    ASPHALT = auto()
    SHORT_GRASS = auto()
    LONG_GRASS = auto()
    DIRT = auto()
    SNOW = auto()
    ICE = auto()
    CUSTOM = auto()


@dataclass
class SurfaceProperties:
    """
    Ground surface friction and mechanical properties.
    
    Attributes:
        mu_rolling: Rolling friction coefficient (resistance to forward motion)
        mu_braking: Braking friction coefficient (max braking force)
        mu_static: Static/lateral friction coefficient (side grip limit)
        softness: Surface softness [0-1] where 0=hard (concrete), 1=very soft
                 Affects sinkage and adds rolling resistance on soft surfaces
        name: Surface identifier
        
    Example:
        >>> grass = SurfaceProperties(
        ...     mu_rolling=0.05,
        ...     mu_braking=0.4,
        ...     mu_static=0.5,
        ...     softness=0.3,
        ...     name='grass_field',
        ... )
    """
    
    mu_rolling: float = 0.03
    mu_braking: float = 0.5
    mu_static: float = 0.8
    softness: float = 0.0      # 0 = hard, 1 = very soft
    name: str = "custom"
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0 <= self.mu_rolling <= 1:
            raise ValueError(f"mu_rolling must be in [0, 1], got {self.mu_rolling}")
        if not 0 <= self.mu_braking <= 1:
            raise ValueError(f"mu_braking must be in [0, 1], got {self.mu_braking}")
        if not 0 <= self.mu_static <= 1:
            raise ValueError(f"mu_static must be in [0, 1], got {self.mu_static}")
        if not 0 <= self.softness <= 1:
            raise ValueError(f"softness must be in [0, 1], got {self.softness}")


# Preset surface properties based on typical values
SURFACE_PRESETS: Dict[SurfaceType, SurfaceProperties] = {
    SurfaceType.CONCRETE: SurfaceProperties(
        mu_rolling=0.02,
        mu_braking=0.7,
        mu_static=0.9,
        softness=0.0,
        name="concrete",
    ),
    SurfaceType.ASPHALT: SurfaceProperties(
        mu_rolling=0.02,
        mu_braking=0.6,
        mu_static=0.8,
        softness=0.0,
        name="asphalt",
    ),
    SurfaceType.SHORT_GRASS: SurfaceProperties(
        mu_rolling=0.05,
        mu_braking=0.4,
        mu_static=0.5,
        softness=0.2,
        name="short_grass",
    ),
    SurfaceType.LONG_GRASS: SurfaceProperties(
        mu_rolling=0.10,
        mu_braking=0.3,
        mu_static=0.4,
        softness=0.4,
        name="long_grass",
    ),
    SurfaceType.DIRT: SurfaceProperties(
        mu_rolling=0.04,
        mu_braking=0.5,
        mu_static=0.6,
        softness=0.15,
        name="dirt",
    ),
    SurfaceType.SNOW: SurfaceProperties(
        mu_rolling=0.03,
        mu_braking=0.2,
        mu_static=0.3,
        softness=0.5,
        name="snow",
    ),
    SurfaceType.ICE: SurfaceProperties(
        mu_rolling=0.01,
        mu_braking=0.1,
        mu_static=0.1,
        softness=0.0,
        name="ice",
    ),
}


def get_surface_properties(surface: Union[SurfaceType, SurfaceProperties, None]) -> Optional[SurfaceProperties]:
    """
    Get surface properties from type enum or pass through custom properties.
    
    Args:
        surface: SurfaceType enum, SurfaceProperties instance, or None
        
    Returns:
        SurfaceProperties or None (use gear defaults)
    """
    if surface is None:
        return None
    elif isinstance(surface, SurfaceProperties):
        return surface
    elif isinstance(surface, SurfaceType):
        if surface == SurfaceType.CUSTOM:
            raise ValueError("SurfaceType.CUSTOM requires passing SurfaceProperties directly")
        return SURFACE_PRESETS.get(surface)
    else:
        raise TypeError(f"Expected SurfaceType or SurfaceProperties, got {type(surface)}")


@dataclass
class LandingGearParameters:
    """
    Single gear leg parameters.
    
    Models a spring-damper strut with a tire at the bottom.
    The strut and tire act in series for vertical load.
    
    Attributes:
        name: Identifier for this gear leg
        
        # Strut properties
        strut_length: Uncompressed strut length [m]
        strut_stiffness: Spring constant [N/m]
        strut_damping: Damping coefficient [N·s/m]
        strut_max_compression: Maximum compression before bottoming [m]
        
        # Tire properties
        tire_radius: Unloaded tire radius [m]
        tire_stiffness: Vertical stiffness [N/m]
        tire_damping: Vertical damping [N·s/m]
        
        # Friction coefficients
        mu_static: Static friction coefficient (lateral grip limit)
        mu_rolling: Rolling friction coefficient
        mu_braking: Braking friction coefficient (max with full brake)
        mu_side: Cornering stiffness coefficient [1/rad]
        
        # Position relative to CG (body frame)
        x_b: X position [m] (positive = forward of CG)
        y_b: Y position [m] (positive = right of centerline)
        z_b: Z position [m] (positive = below CG, typically positive)
        
        # Capabilities
        is_steerable: Whether this gear can be steered
        has_brake: Whether this gear has a brake
        max_steer_angle_deg: Maximum steering angle [deg]
        
    Example:
        >>> main_gear = LandingGearParameters(
        ...     name='main_left',
        ...     strut_length=0.15,
        ...     strut_stiffness=5000,
        ...     strut_damping=500,
        ...     strut_max_compression=0.10,
        ...     tire_radius=0.05,
        ...     tire_stiffness=20000,
        ...     tire_damping=200,
        ...     x_b=0.05,
        ...     y_b=-0.20,
        ...     z_b=0.10,
        ...     has_brake=True,
        ... )
    """
    
    name: str = "gear"
    
    # Strut properties
    strut_length: float = 0.15           # [m]
    strut_stiffness: float = 5000.0      # [N/m]
    strut_damping: float = 500.0         # [N·s/m]
    strut_max_compression: float = 0.10  # [m]
    
    # Tire properties
    tire_radius: float = 0.05            # [m]
    tire_stiffness: float = 20000.0      # [N/m]
    tire_damping: float = 200.0          # [N·s/m]
    
    # Friction coefficients
    mu_static: float = 0.8               # Static/lateral limit
    mu_rolling: float = 0.03             # Rolling resistance
    mu_braking: float = 0.5              # Braking friction
    mu_side: float = 0.7                 # Cornering stiffness per rad
    
    # Position relative to CG (body frame)
    x_b: float = 0.0
    y_b: float = 0.0
    z_b: float = 0.10                    # Below CG
    
    # Capabilities
    is_steerable: bool = False
    has_brake: bool = False
    max_steer_angle_deg: float = 30.0
    
    def __post_init__(self):
        """Validate parameters."""
        if self.strut_length <= 0:
            raise ValueError(f"strut_length must be positive, got {self.strut_length}")
        if self.tire_radius <= 0:
            raise ValueError(f"tire_radius must be positive, got {self.tire_radius}")
        if self.strut_stiffness <= 0:
            raise ValueError(f"strut_stiffness must be positive, got {self.strut_stiffness}")
        if self.tire_stiffness <= 0:
            raise ValueError(f"tire_stiffness must be positive, got {self.tire_stiffness}")
    
    @property
    def position_body(self) -> np.ndarray:
        """Position as numpy array [x_b, y_b, z_b]."""
        return np.array([self.x_b, self.y_b, self.z_b])
    
    @property
    def total_length(self) -> float:
        """Total length from attachment to ground contact [m]."""
        return self.strut_length + self.tire_radius
    
    @property
    def combined_stiffness(self) -> float:
        """Effective stiffness of strut + tire in series [N/m]."""
        return 1.0 / (1.0 / self.strut_stiffness + 1.0 / self.tire_stiffness)
    
    @property
    def combined_damping(self) -> float:
        """Effective damping of strut + tire in series [N·s/m]."""
        # For dampers in series, combine like springs
        return 1.0 / (1.0 / self.strut_damping + 1.0 / self.tire_damping)


@dataclass
class LandingGearSet:
    """
    Complete landing gear configuration.
    
    Supports various configurations:
    - Tricycle: nose + main_left + main_right
    - Taildragger: main_left + main_right + tail
    - Flying wing: center_skid + left_skid + right_skid
    
    Attributes:
        gears: List of all gear legs
        gear_type: Configuration type
        
    Example:
        >>> # Tricycle configuration
        >>> gear_set = LandingGearSet.create_tricycle(
        ...     nose_x=-0.30, main_x=0.05, main_y=0.20,
        ...     main_z=0.10, nose_z=0.08,
        ... )
    """
    
    gears: List[LandingGearParameters] = field(default_factory=list)
    gear_type: GearType = GearType.CUSTOM
    
    def __post_init__(self):
        """Validate gear set."""
        if not self.gears:
            # Create default single center gear
            self.gears = [LandingGearParameters(name='center')]
    
    @property
    def n_gears(self) -> int:
        """Number of gear legs."""
        return len(self.gears)
    
    def get_gear(self, name: str) -> Optional[LandingGearParameters]:
        """Get gear by name."""
        for gear in self.gears:
            if gear.name == name:
                return gear
        return None
    
    @classmethod
    def create_tricycle(
        cls,
        nose_x: float = -0.30,
        main_x: float = 0.05,
        main_y: float = 0.20,
        nose_z: float = 0.10,
        main_z: float = 0.12,
        nose_steerable: bool = True,
        main_brakes: bool = True,
        strut_stiffness: float = 5000.0,
        tire_stiffness: float = 20000.0,
    ) -> 'LandingGearSet':
        """
        Create tricycle gear configuration.
        
        Args:
            nose_x: Nose gear x position (negative = forward) [m]
            main_x: Main gear x position [m]
            main_y: Main gear y distance from centerline [m]
            nose_z: Nose gear z position (positive = below CG) [m]
            main_z: Main gear z position [m]
            nose_steerable: Whether nose gear is steerable
            main_brakes: Whether main gears have brakes
            strut_stiffness: Strut spring constant [N/m]
            tire_stiffness: Tire spring constant [N/m]
            
        Returns:
            Configured LandingGearSet
        """
        nose = LandingGearParameters(
            name='nose',
            x_b=nose_x,
            y_b=0.0,
            z_b=nose_z,
            strut_stiffness=strut_stiffness * 0.5,  # Lighter loaded
            tire_stiffness=tire_stiffness * 0.5,
            is_steerable=nose_steerable,
            has_brake=False,
        )
        
        main_left = LandingGearParameters(
            name='main_left',
            x_b=main_x,
            y_b=-main_y,
            z_b=main_z,
            strut_stiffness=strut_stiffness,
            tire_stiffness=tire_stiffness,
            is_steerable=False,
            has_brake=main_brakes,
        )
        
        main_right = LandingGearParameters(
            name='main_right',
            x_b=main_x,
            y_b=main_y,
            z_b=main_z,
            strut_stiffness=strut_stiffness,
            tire_stiffness=tire_stiffness,
            is_steerable=False,
            has_brake=main_brakes,
        )
        
        return cls(gears=[nose, main_left, main_right], gear_type=GearType.TRICYCLE)
    
    @classmethod
    def create_taildragger(
        cls,
        main_x: float = -0.05,
        main_y: float = 0.20,
        tail_x: float = 0.50,
        main_z: float = 0.12,
        tail_z: float = 0.05,
        tail_steerable: bool = True,
        main_brakes: bool = True,
        strut_stiffness: float = 5000.0,
        tire_stiffness: float = 20000.0,
    ) -> 'LandingGearSet':
        """
        Create taildragger gear configuration.
        
        Args:
            main_x: Main gear x position [m]
            main_y: Main gear y distance from centerline [m]
            tail_x: Tail wheel x position (positive = aft) [m]
            main_z: Main gear z position [m]
            tail_z: Tail wheel z position [m]
            tail_steerable: Whether tail wheel is steerable
            main_brakes: Whether main gears have brakes
            
        Returns:
            Configured LandingGearSet
        """
        main_left = LandingGearParameters(
            name='main_left',
            x_b=main_x,
            y_b=-main_y,
            z_b=main_z,
            strut_stiffness=strut_stiffness,
            tire_stiffness=tire_stiffness,
            is_steerable=False,
            has_brake=main_brakes,
        )
        
        main_right = LandingGearParameters(
            name='main_right',
            x_b=main_x,
            y_b=main_y,
            z_b=main_z,
            strut_stiffness=strut_stiffness,
            tire_stiffness=tire_stiffness,
            is_steerable=False,
            has_brake=main_brakes,
        )
        
        tail = LandingGearParameters(
            name='tail',
            x_b=tail_x,
            y_b=0.0,
            z_b=tail_z,
            strut_stiffness=strut_stiffness * 0.3,
            tire_stiffness=tire_stiffness * 0.3,
            tire_radius=0.025,  # Smaller tail wheel
            is_steerable=tail_steerable,
            has_brake=False,
        )
        
        return cls(gears=[main_left, main_right, tail], gear_type=GearType.TAILDRAGGER)
    
    @classmethod
    def create_flying_wing_skids(
        cls,
        center_x: float = 0.0,
        wingtip_x: float = 0.10,
        wingtip_y: float = 0.60,
        center_z: float = 0.08,
        wingtip_z: float = 0.05,
        skid_stiffness: float = 3000.0,
    ) -> 'LandingGearSet':
        """
        Create flying wing skid configuration.
        
        Belly skid with wingtip skids for ground handling.
        No wheels - sliding friction only.
        
        Args:
            center_x: Center skid x position [m]
            wingtip_x: Wingtip skid x position [m]
            wingtip_y: Wingtip skid y distance from centerline [m]
            center_z: Center skid z position [m]
            wingtip_z: Wingtip skid z position [m]
            skid_stiffness: Skid spring constant [N/m]
            
        Returns:
            Configured LandingGearSet
        """
        # Skids have high friction, no rolling
        center = LandingGearParameters(
            name='center_skid',
            x_b=center_x,
            y_b=0.0,
            z_b=center_z,
            strut_stiffness=skid_stiffness,
            strut_damping=300.0,
            strut_length=0.02,  # Very short - essentially rigid
            tire_radius=0.01,   # Thin skid plate
            tire_stiffness=skid_stiffness * 2,
            mu_rolling=0.4,     # Sliding friction (no wheels)
            mu_braking=0.4,
            mu_static=0.5,
            is_steerable=False,
            has_brake=False,
        )
        
        left_skid = LandingGearParameters(
            name='left_skid',
            x_b=wingtip_x,
            y_b=-wingtip_y,
            z_b=wingtip_z,
            strut_stiffness=skid_stiffness * 0.5,
            strut_damping=200.0,
            strut_length=0.02,
            tire_radius=0.01,
            tire_stiffness=skid_stiffness,
            mu_rolling=0.4,
            mu_braking=0.4,
            mu_static=0.5,
            is_steerable=False,
            has_brake=False,
        )
        
        right_skid = LandingGearParameters(
            name='right_skid',
            x_b=wingtip_x,
            y_b=wingtip_y,
            z_b=wingtip_z,
            strut_stiffness=skid_stiffness * 0.5,
            strut_damping=200.0,
            strut_length=0.02,
            tire_radius=0.01,
            tire_stiffness=skid_stiffness,
            mu_rolling=0.4,
            mu_braking=0.4,
            mu_static=0.5,
            is_steerable=False,
            has_brake=False,
        )
        
        return cls(gears=[center, left_skid, right_skid], gear_type=GearType.FLYING_WING)


@dataclass
class GearContactState:
    """
    State of a single gear leg contact with ground.
    
    Attributes:
        is_on_ground: Whether gear is in contact with ground
        compression: Strut compression [m] (positive = compressed)
        compression_rate: Rate of compression [m/s]
        normal_force: Normal (vertical) ground reaction [N]
        friction_force: Friction force vector [Fx, Fy] in Earth frame [N]
        contact_point_earth: Contact point in Earth frame [x, y, z]
        slip_velocity: Slip velocity at contact [m/s]
    """
    is_on_ground: bool = False
    compression: float = 0.0
    compression_rate: float = 0.0
    normal_force: float = 0.0
    friction_force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    contact_point_earth: np.ndarray = field(default_factory=lambda: np.zeros(3))
    slip_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


class LandingGearModel:
    """
    Computes ground reaction forces for landing gear.
    
    Models spring-damper strut with tire, and tire friction forces.
    Supports multiple gear legs with individual force/moment calculation.
    
    Example:
        >>> gear_set = LandingGearSet.create_tricycle(...)
        >>> model = LandingGearModel(gear_set)
        >>> forces = model.compute_total_forces(
        ...     position_earth=np.array([0, 0, -0.05]),  # 5cm above ground
        ...     velocity_body=np.array([10, 0, 0.5]),    # Moving forward, descending
        ...     euler_angles=np.array([0, 0.05, 0]),     # Slight pitch up
        ...     angular_rates=np.array([0, 0, 0]),
        ...     ground_altitude=0.0,
        ...     brake_command=0.0,
        ...     steering_angle=0.0,
        ... )
    """
    
    def __init__(self, gear_set: LandingGearSet):
        """
        Initialize landing gear model.
        
        Args:
            gear_set: Landing gear configuration
        """
        self.gear_set = gear_set
    
    def _rotation_matrix_body_to_earth(
        self, 
        phi: float, 
        theta: float, 
        psi: float
    ) -> np.ndarray:
        """Compute rotation matrix from body to Earth frame."""
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        return np.array([
            [cthe*cpsi, sphi*sthe*cpsi - cphi*spsi, cphi*sthe*cpsi + sphi*spsi],
            [cthe*spsi, sphi*sthe*spsi + cphi*cpsi, cphi*sthe*spsi - sphi*cpsi],
            [-sthe,     sphi*cthe,                  cphi*cthe                 ],
        ])
    
    def _compute_gear_position_earth(
        self,
        gear: LandingGearParameters,
        cg_position_earth: np.ndarray,
        R_be: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gear contact point position in Earth frame.
        
        Args:
            gear: Gear parameters
            cg_position_earth: CG position in Earth frame [m]
            R_be: Rotation matrix body to Earth
            
        Returns:
            Gear bottom position in Earth frame [m]
        """
        # Gear attachment point relative to CG (body frame)
        r_attach_body = gear.position_body
        
        # Add strut length and tire radius (in body z direction = down)
        r_contact_body = r_attach_body + np.array([0, 0, gear.total_length])
        
        # Transform to Earth frame
        r_contact_earth = cg_position_earth + R_be @ r_contact_body
        
        return r_contact_earth
    
    def _compute_gear_velocity_earth(
        self,
        gear: LandingGearParameters,
        velocity_body: np.ndarray,
        angular_rates: np.ndarray,
        R_be: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gear contact point velocity in Earth frame.
        
        Includes contribution from angular rotation about CG.
        
        Args:
            gear: Gear parameters  
            velocity_body: CG velocity in body frame [m/s]
            angular_rates: Angular rates [p, q, r] in body frame [rad/s]
            R_be: Rotation matrix body to Earth
            
        Returns:
            Gear contact point velocity in Earth frame [m/s]
        """
        # Gear contact point relative to CG (body frame)
        r_contact_body = gear.position_body + np.array([0, 0, gear.total_length])
        
        # Velocity contribution from rotation: v_rot = omega × r
        omega_body = angular_rates
        v_rotation_body = np.cross(omega_body, r_contact_body)
        
        # Total velocity at contact point (body frame)
        v_contact_body = velocity_body + v_rotation_body
        
        # Transform to Earth frame
        v_contact_earth = R_be @ v_contact_body
        
        return v_contact_earth
    
    def compute_single_gear_forces(
        self,
        gear: LandingGearParameters,
        cg_position_earth: np.ndarray,
        velocity_body: np.ndarray,
        euler_angles: np.ndarray,
        angular_rates: np.ndarray,
        ground_altitude: float = 0.0,
        brake_command: float = 0.0,
        steering_angle: float = 0.0,
        surface: Union[SurfaceType, SurfaceProperties, None] = None,
    ) -> Tuple[GearContactState, np.ndarray, np.ndarray]:
        """
        Compute forces and moments from a single gear leg.
        
        Args:
            gear: Gear parameters
            cg_position_earth: CG position in Earth frame [x_e, y_e, z_e] (NED)
            velocity_body: CG velocity in body frame [u, v, w] [m/s]
            euler_angles: [phi, theta, psi] [rad]
            angular_rates: [p, q, r] [rad/s]
            ground_altitude: Ground elevation in Earth frame [m] (typically 0)
            brake_command: Brake command [0-1]
            steering_angle: Steering angle [rad] (for steerable gears)
            surface: Surface type or custom properties (None = use gear defaults)
            
        Returns:
            Tuple of:
                - GearContactState: Contact state info
                - force_body: Force on aircraft in body frame [N]
                - moment_body: Moment on aircraft about CG in body frame [N·m]
        """
        phi, theta, psi = euler_angles
        R_be = self._rotation_matrix_body_to_earth(phi, theta, psi)
        R_eb = R_be.T
        
        # Get gear contact point position and velocity in Earth frame
        pos_earth = self._compute_gear_position_earth(gear, cg_position_earth, R_be)
        vel_earth = self._compute_gear_velocity_earth(gear, velocity_body, angular_rates, R_be)
        
        # Ground contact check (NED: z positive down, ground at z = -ground_altitude)
        # Actually in NED: altitude = -z_e, so ground is at z_e = -ground_altitude
        # If ground_altitude = 0, ground is at z_e = 0
        # Gear touches ground when pos_earth[2] >= -ground_altitude
        # For simplicity: ground plane at z_e = 0 (sea level)
        z_ground = -ground_altitude  # z coordinate of ground in NED
        z_gear = pos_earth[2]
        
        # Compression: how much the gear is pushed into the ground
        # Positive compression means gear is below ground level
        penetration = z_gear - z_ground  # Positive when below ground
        
        # Initialize contact state
        state = GearContactState()
        state.contact_point_earth = pos_earth.copy()
        
        # Check if on ground
        if penetration <= 0:
            # Not touching ground
            state.is_on_ground = False
            return state, np.zeros(3), np.zeros(3)
        
        state.is_on_ground = True
        state.compression = min(penetration, gear.strut_max_compression + gear.tire_radius)
        
        # Vertical (normal) force: spring-damper
        # Compression rate: rate of change of penetration
        compression_rate = vel_earth[2]  # Positive = moving down = compressing more
        state.compression_rate = compression_rate
        
        # Combined spring-damper force (series combination)
        k_eff = gear.combined_stiffness
        c_eff = gear.combined_damping
        
        # Get surface properties (use gear defaults if no surface specified)
        surf_props = get_surface_properties(surface)
        
        # Add extra rolling resistance from soft surfaces
        softness_resistance = 0.0
        if surf_props is not None and surf_props.softness > 0:
            # Soft surface adds resistance proportional to normal force
            softness_resistance = surf_props.softness * 0.1  # Up to 10% extra for very soft
        
        F_normal = k_eff * state.compression - c_eff * compression_rate
        
        # Normal force can only push (no suction)
        F_normal = max(0.0, F_normal)
        state.normal_force = F_normal
        
        # If no normal force, no friction either
        if F_normal < 0.01:
            return state, np.zeros(3), np.zeros(3)
        
        # Determine friction coefficients (surface overrides gear defaults)
        if surf_props is not None:
            mu_rolling_base = surf_props.mu_rolling
            mu_braking_eff = surf_props.mu_braking
            mu_static_eff = surf_props.mu_static
        else:
            mu_rolling_base = gear.mu_rolling
            mu_braking_eff = gear.mu_braking
            mu_static_eff = gear.mu_static
        
        # Add softness resistance to rolling friction
        mu_rolling_total = mu_rolling_base + softness_resistance
        
        # Friction forces (in Earth frame horizontal plane)
        # Slip velocity in Earth x-y plane
        v_slip_x = vel_earth[0]  # North velocity
        v_slip_y = vel_earth[1]  # East velocity
        state.slip_velocity = np.array([v_slip_x, v_slip_y])
        
        # Longitudinal direction (aircraft forward in Earth frame)
        # Forward direction in Earth frame
        forward_earth = R_be @ np.array([1, 0, 0])
        forward_horiz = np.array([forward_earth[0], forward_earth[1]])
        forward_norm = np.linalg.norm(forward_horiz)
        if forward_norm > 1e-6:
            forward_horiz = forward_horiz / forward_norm
        else:
            forward_horiz = np.array([1, 0])
        
        # Right direction (perpendicular to forward in horizontal plane)
        right_horiz = np.array([forward_horiz[1], -forward_horiz[0]])
        
        # Apply steering if applicable
        if gear.is_steerable and abs(steering_angle) > 1e-6:
            steer_angle = np.clip(steering_angle, 
                                  -np.radians(gear.max_steer_angle_deg),
                                  np.radians(gear.max_steer_angle_deg))
            cos_s = np.cos(steer_angle)
            sin_s = np.sin(steer_angle)
            forward_horiz_new = cos_s * forward_horiz + sin_s * right_horiz
            right_horiz = -sin_s * forward_horiz + cos_s * right_horiz
            forward_horiz = forward_horiz_new
        
        # Decompose slip velocity into longitudinal and lateral
        v_long = np.dot(state.slip_velocity, forward_horiz)  # Forward slip
        v_lat = np.dot(state.slip_velocity, right_horiz)     # Lateral slip
        
        # Longitudinal friction (rolling + braking)
        if gear.has_brake and brake_command > 0:
            # Braking friction
            mu_long = mu_rolling_total + brake_command * (mu_braking_eff - mu_rolling_total)
        else:
            # Just rolling resistance
            mu_long = mu_rolling_total
        
        # Longitudinal force opposes forward motion
        F_long = -np.sign(v_long) * mu_long * F_normal
        
        # Lateral friction (tire slip model)
        # Simple linear slip model with saturation
        slip_stiffness = 1000.0  # [N/(m/s)] - cornering stiffness
        F_lat_linear = -slip_stiffness * v_lat
        F_lat_max = mu_static_eff * F_normal
        F_lat = np.clip(F_lat_linear, -F_lat_max, F_lat_max)
        
        # Total friction force in Earth horizontal plane
        F_friction_horiz = F_long * forward_horiz + F_lat * right_horiz
        state.friction_force = F_friction_horiz
        
        # Total force in Earth frame (friction in x-y, normal in z)
        # Normal force is upward (negative z in NED)
        F_earth = np.array([F_friction_horiz[0], F_friction_horiz[1], -F_normal])
        
        # Transform to body frame
        F_body = R_eb @ F_earth
        
        # Moment about CG from force at contact point
        # r = contact point relative to CG in body frame
        r_contact_body = gear.position_body + np.array([0, 0, gear.total_length])
        M_body = np.cross(r_contact_body, F_body)
        
        return state, F_body, M_body
    
    def compute_total_forces(
        self,
        cg_position_earth: np.ndarray,
        velocity_body: np.ndarray,
        euler_angles: np.ndarray,
        angular_rates: np.ndarray,
        ground_altitude: float = 0.0,
        brake_command: float = 0.0,
        steering_angle: float = 0.0,
        surface: Union[SurfaceType, SurfaceProperties, None] = None,
    ) -> Dict:
        """
        Compute total ground reaction forces from all gear legs.
        
        Args:
            cg_position_earth: CG position in Earth frame [x_e, y_e, z_e] (NED)
            velocity_body: CG velocity in body frame [u, v, w] [m/s]
            euler_angles: [phi, theta, psi] [rad]
            angular_rates: [p, q, r] [rad/s]
            ground_altitude: Ground elevation [m]
            brake_command: Brake command [0-1] (applied to all braked gears)
            steering_angle: Steering angle [rad] (applied to steerable gears)
            surface: Surface type or custom properties (None = use gear defaults)
            
        Returns:
            Dict with:
                - force_body: Total force [Fx, Fy, Fz] in body frame [N]
                - moment_body: Total moment [Mx, My, Mz] about CG in body frame [N·m]
                - any_on_ground: Whether any gear is touching ground
                - gear_states: Dict of GearContactState per gear name
                - normal_force_total: Total normal force [N]
                - surface_used: Surface properties used (or None)
        """
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        gear_states = {}
        any_on_ground = False
        total_normal = 0.0
        
        # Resolve surface once for all gears
        surf_props = get_surface_properties(surface)
        
        for gear in self.gear_set.gears:
            state, F_body, M_body = self.compute_single_gear_forces(
                gear=gear,
                cg_position_earth=cg_position_earth,
                velocity_body=velocity_body,
                euler_angles=euler_angles,
                angular_rates=angular_rates,
                ground_altitude=ground_altitude,
                brake_command=brake_command if gear.has_brake else 0.0,
                steering_angle=steering_angle if gear.is_steerable else 0.0,
                surface=surf_props,  # Pass resolved properties
            )
            
            total_force += F_body
            total_moment += M_body
            gear_states[gear.name] = state
            
            if state.is_on_ground:
                any_on_ground = True
                total_normal += state.normal_force
        
        return {
            'force_body': total_force,
            'moment_body': total_moment,
            'any_on_ground': any_on_ground,
            'gear_states': gear_states,
            'normal_force_total': total_normal,
            'surface_used': surf_props,
        }
    
    def is_on_ground(
        self,
        cg_position_earth: np.ndarray,
        euler_angles: np.ndarray,
        ground_altitude: float = 0.0,
    ) -> bool:
        """
        Quick check if any gear is on ground.
        
        Args:
            cg_position_earth: CG position in Earth frame [m]
            euler_angles: [phi, theta, psi] [rad]
            ground_altitude: Ground elevation [m]
            
        Returns:
            True if any gear is touching ground
        """
        phi, theta, psi = euler_angles
        R_be = self._rotation_matrix_body_to_earth(phi, theta, psi)
        z_ground = -ground_altitude
        
        for gear in self.gear_set.gears:
            pos_earth = self._compute_gear_position_earth(gear, cg_position_earth, R_be)
            if pos_earth[2] >= z_ground:
                return True
        
        return False


# =============================================================================
# Gear Presets
# =============================================================================

def create_small_uav_tricycle(
    wheelbase: float = 0.35,
    track: float = 0.25,
    cg_height: float = 0.08,
) -> LandingGearSet:
    """
    Create gear for small UAV (1-5 kg).
    
    Args:
        wheelbase: Nose to main gear distance [m]
        track: Main gear track width [m]
        cg_height: Height of CG above ground [m]
        
    Returns:
        Configured LandingGearSet
    """
    # Nose gear forward of CG by ~30% of wheelbase
    nose_x = -0.3 * wheelbase
    main_x = 0.7 * wheelbase
    
    return LandingGearSet.create_tricycle(
        nose_x=nose_x,
        main_x=main_x,
        main_y=track / 2,
        nose_z=cg_height + 0.02,
        main_z=cg_height,
        strut_stiffness=3000.0,
        tire_stiffness=15000.0,
    )


def create_flying_wing_belly_skid(
    wingspan: float = 1.5,
    cg_height: float = 0.05,
) -> LandingGearSet:
    """
    Create belly skid gear for flying wing.
    
    Args:
        wingspan: Total wingspan [m]
        cg_height: Height of CG above ground at rest [m]
        
    Returns:
        Configured LandingGearSet
    """
    return LandingGearSet.create_flying_wing_skids(
        center_x=0.0,
        wingtip_x=0.05,
        wingtip_y=wingspan * 0.4,  # 80% of semi-span
        center_z=cg_height,
        wingtip_z=cg_height - 0.02,
        skid_stiffness=2000.0,
    )
