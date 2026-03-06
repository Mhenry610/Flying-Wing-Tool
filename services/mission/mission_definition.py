"""
Mission Definition Module

Structured definition of mission phases, waypoints, and complete mission profiles
for flying wing simulation and optimization.

This module provides:
- MissionPhaseType: Enum of all possible flight phases
- Waypoint: 3D position with optional constraints
- MissionPhase: Single phase definition with targets and limits
- MissionProfile: Complete mission with factory methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import json


class MissionPhaseType(Enum):
    """
    Types of mission phases.
    
    Each phase type implies specific autopilot behavior:
    - Ground phases: throttle/brake control, no altitude control
    - Transition phases: rotation, flare have specific pitch targets
    - Flight phases: full 3-axis control with altitude/speed targets
    """
    # Ground operations
    GROUND_IDLE = auto()        # Stationary, systems check
    TAKEOFF_ROLL = auto()       # Accelerating on ground
    LANDING_ROLL = auto()       # Decelerating on ground
    
    # Transition phases
    ROTATION = auto()           # Pitch up for liftoff
    LANDING_FLARE = auto()      # Pitch up for touchdown
    
    # Climb/descent
    CLIMB = auto()              # Gaining altitude
    DESCENT = auto()            # Losing altitude
    APPROACH = auto()           # Final approach to landing
    
    # Level flight
    CRUISE = auto()             # Steady level flight
    LOITER = auto()             # Holding pattern / orbit
    
    # Navigation
    WAYPOINT_TRANSIT = auto()   # Flying to specific waypoint
    
    def is_ground_phase(self) -> bool:
        """Check if this is a ground-based phase."""
        return self in (
            MissionPhaseType.GROUND_IDLE,
            MissionPhaseType.TAKEOFF_ROLL,
            MissionPhaseType.LANDING_ROLL,
        )
    
    def is_transition_phase(self) -> bool:
        """Check if this is a transition phase (rotation/flare)."""
        return self in (
            MissionPhaseType.ROTATION,
            MissionPhaseType.LANDING_FLARE,
        )
    
    def requires_altitude_control(self) -> bool:
        """Check if phase requires active altitude control."""
        return self in (
            MissionPhaseType.CLIMB,
            MissionPhaseType.DESCENT,
            MissionPhaseType.CRUISE,
            MissionPhaseType.LOITER,
            MissionPhaseType.WAYPOINT_TRANSIT,
            MissionPhaseType.APPROACH,
        )


@dataclass
class Waypoint:
    """
    3D waypoint with optional constraints.
    
    Coordinates are in local East-North-Up (ENU) frame relative to
    mission origin (typically runway threshold).
    
    Attributes:
        x: East position [m] (positive = east)
        y: North position [m] (positive = north)
        altitude: Altitude above ground level [m]
        speed: Target airspeed at waypoint [m/s], None = maintain current
        heading: Target heading at waypoint [deg], None = direct to next
        time: Target arrival time [s], None = no constraint
        position_tolerance: Waypoint capture radius [m]
        altitude_tolerance: Altitude capture band [m]
    """
    x: float                              # East [m]
    y: float                              # North [m]
    altitude: float                       # AGL [m]
    speed: Optional[float] = None         # Target speed [m/s]
    heading: Optional[float] = None       # Target heading [deg]
    time: Optional[float] = None          # Arrival time constraint [s]
    position_tolerance: float = 50.0      # Capture radius [m]
    altitude_tolerance: float = 10.0      # Altitude capture band [m]
    
    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate horizontal distance to another waypoint [m]."""
        import math
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx*dx + dy*dy)
    
    def bearing_to(self, other: 'Waypoint') -> float:
        """Calculate bearing to another waypoint [deg, 0=North, 90=East]."""
        import math
        dx = other.x - self.x
        dy = other.y - self.y
        bearing = math.degrees(math.atan2(dx, dy))
        return bearing % 360.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading,
            'time': self.time,
            'position_tolerance': self.position_tolerance,
            'altitude_tolerance': self.altitude_tolerance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Waypoint':
        """Deserialize from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            altitude=data['altitude'],
            speed=data.get('speed'),
            heading=data.get('heading'),
            time=data.get('time'),
            position_tolerance=data.get('position_tolerance', 50.0),
            altitude_tolerance=data.get('altitude_tolerance', 10.0),
        )


@dataclass
class MissionPhase:
    """
    Single mission phase definition.
    
    A phase has a type (climb, cruise, etc.) and defines:
    - End conditions: when to transition to next phase
    - Targets: what the autopilot should maintain during the phase
    - Limits: throttle and control surface constraints
    
    Attributes:
        name: Human-readable phase name
        phase_type: Type of phase (determines autopilot mode)
        
        # End conditions (first one reached triggers phase end)
        duration: Maximum phase duration [s]
        end_altitude: End when reaching this altitude [m]
        end_speed: End when reaching this speed [m/s]
        end_distance: End after traveling this distance [m]
        waypoint: End when reaching this waypoint
        
        # Targets during phase
        target_altitude: Altitude to maintain/reach [m]
        target_speed: Airspeed to maintain [m/s]
        target_heading: Heading to maintain [deg]
        target_climb_rate: Climb rate target [m/s] (positive = up)
        target_descent_rate: Descent rate target [m/s] (positive = down)
        target_bank_angle: Bank angle for turns [deg]
        
        # Control limits
        throttle_min: Minimum throttle [0-1]
        throttle_max: Maximum throttle [0-1]
        pitch_min: Minimum pitch angle [deg]
        pitch_max: Maximum pitch angle [deg]
        bank_max: Maximum bank angle [deg]
    """
    name: str
    phase_type: MissionPhaseType
    
    # End conditions (phase ends when first condition is met)
    duration: Optional[float] = None
    end_altitude: Optional[float] = None
    end_speed: Optional[float] = None
    end_distance: Optional[float] = None
    waypoint: Optional[Waypoint] = None
    
    # Targets during phase
    target_altitude: Optional[float] = None
    target_speed: Optional[float] = None
    target_heading: Optional[float] = None
    target_climb_rate: Optional[float] = None
    target_descent_rate: Optional[float] = None
    target_bank_angle: Optional[float] = None
    
    # Control limits
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    pitch_min: float = -15.0
    pitch_max: float = 20.0
    bank_max: float = 45.0
    
    def has_end_condition(self) -> bool:
        """Check if phase has at least one end condition."""
        return any([
            self.duration is not None,
            self.end_altitude is not None,
            self.end_speed is not None,
            self.end_distance is not None,
            self.waypoint is not None,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            'name': self.name,
            'phase_type': self.phase_type.name,
            'duration': self.duration,
            'end_altitude': self.end_altitude,
            'end_speed': self.end_speed,
            'end_distance': self.end_distance,
            'waypoint': self.waypoint.to_dict() if self.waypoint else None,
            'target_altitude': self.target_altitude,
            'target_speed': self.target_speed,
            'target_heading': self.target_heading,
            'target_climb_rate': self.target_climb_rate,
            'target_descent_rate': self.target_descent_rate,
            'target_bank_angle': self.target_bank_angle,
            'throttle_min': self.throttle_min,
            'throttle_max': self.throttle_max,
            'pitch_min': self.pitch_min,
            'pitch_max': self.pitch_max,
            'bank_max': self.bank_max,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MissionPhase':
        """Deserialize from dictionary."""
        waypoint_data = data.get('waypoint')
        return cls(
            name=data['name'],
            phase_type=MissionPhaseType[data['phase_type']],
            duration=data.get('duration'),
            end_altitude=data.get('end_altitude'),
            end_speed=data.get('end_speed'),
            end_distance=data.get('end_distance'),
            waypoint=Waypoint.from_dict(waypoint_data) if waypoint_data else None,
            target_altitude=data.get('target_altitude'),
            target_speed=data.get('target_speed'),
            target_heading=data.get('target_heading'),
            target_climb_rate=data.get('target_climb_rate'),
            target_descent_rate=data.get('target_descent_rate'),
            target_bank_angle=data.get('target_bank_angle'),
            throttle_min=data.get('throttle_min', 0.0),
            throttle_max=data.get('throttle_max', 1.0),
            pitch_min=data.get('pitch_min', -15.0),
            pitch_max=data.get('pitch_max', 20.0),
            bank_max=data.get('bank_max', 45.0),
        )


@dataclass
class MissionProfile:
    """
    Complete mission definition.
    
    A mission profile contains:
    - Ordered list of phases to execute
    - Initial conditions (position, speed, battery state)
    - Environment conditions (temperature, wind)
    - Ground/runway properties
    - Safety constraints
    
    Factory methods create common mission types:
    - simple_cruise(): Takeoff, climb, cruise, descend, land
    - waypoint_mission(): Multi-waypoint navigation
    - endurance_loiter(): Maximum duration at altitude
    
    Attributes:
        name: Mission name
        phases: Ordered list of mission phases
        
        # Initial conditions
        initial_altitude: Starting altitude AGL [m]
        initial_speed: Starting airspeed [m/s]
        initial_heading: Starting heading [deg]
        initial_SOC: Starting battery state of charge [0-1]
        
        # Environment
        T_ambient: Ambient temperature [C]
        pressure_altitude: Pressure altitude [m]
        wind_speed: Wind speed [m/s]
        wind_direction: Wind from direction [deg]
        
        # Ground properties
        runway_heading: Runway heading [deg]
        runway_elevation: Runway elevation MSL [m]
        runway_length: Available runway length [m]
        surface_friction: Ground friction coefficient
        
        # Safety margins
        min_SOC: Minimum allowed battery SOC
        reserve_time_s: Required reserve flight time [s]
        min_altitude: Minimum altitude for abort [m]
    """
    name: str
    phases: List[MissionPhase] = field(default_factory=list)
    
    # Initial conditions
    initial_altitude: float = 0.0
    initial_speed: float = 0.0
    initial_heading: float = 0.0
    initial_SOC: float = 1.0
    
    # Environment
    T_ambient: float = 15.0             # ISA sea level [C]
    pressure_altitude: float = 0.0      # [m]
    wind_speed: float = 0.0             # [m/s]
    wind_direction: float = 0.0         # Wind FROM this direction [deg]
    
    # Ground properties
    runway_heading: float = 0.0         # [deg]
    runway_elevation: float = 0.0       # [m MSL]
    runway_length: float = 500.0        # [m]
    surface_friction: float = 0.04      # Grass/dirt typical
    
    # Safety margins
    min_SOC: float = 0.2                # 20% reserve
    reserve_time_s: float = 300.0       # 5 minutes reserve
    min_altitude: float = 30.0          # Minimum for abort [m]
    
    def total_phases(self) -> int:
        """Return number of phases."""
        return len(self.phases)
    
    def get_phase_by_name(self, name: str) -> Optional[MissionPhase]:
        """Find phase by name."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None
    
    def get_waypoints(self) -> List[Waypoint]:
        """Extract all waypoints from phases."""
        waypoints = []
        for phase in self.phases:
            if phase.waypoint is not None:
                waypoints.append(phase.waypoint)
        return waypoints
    
    def validate(self) -> List[str]:
        """
        Validate mission profile for common issues.
        
        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []
        
        if not self.phases:
            issues.append("Mission has no phases defined")
            return issues
        
        # Check each phase has end condition (except last)
        for i, phase in enumerate(self.phases[:-1]):
            if not phase.has_end_condition():
                issues.append(f"Phase '{phase.name}' has no end condition")
        
        # Check for reasonable altitude targets
        for phase in self.phases:
            if phase.target_altitude is not None and phase.target_altitude < 0:
                issues.append(f"Phase '{phase.name}' has negative target altitude")
        
        # Check throttle limits
        for phase in self.phases:
            if phase.throttle_min < 0 or phase.throttle_max > 1:
                issues.append(f"Phase '{phase.name}' has invalid throttle limits")
            if phase.throttle_min > phase.throttle_max:
                issues.append(f"Phase '{phase.name}' has throttle_min > throttle_max")
        
        # Check initial SOC
        if self.initial_SOC < self.min_SOC:
            issues.append(f"Initial SOC ({self.initial_SOC}) below minimum ({self.min_SOC})")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'name': self.name,
            'phases': [p.to_dict() for p in self.phases],
            'initial_altitude': self.initial_altitude,
            'initial_speed': self.initial_speed,
            'initial_heading': self.initial_heading,
            'initial_SOC': self.initial_SOC,
            'T_ambient': self.T_ambient,
            'pressure_altitude': self.pressure_altitude,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'runway_heading': self.runway_heading,
            'runway_elevation': self.runway_elevation,
            'runway_length': self.runway_length,
            'surface_friction': self.surface_friction,
            'min_SOC': self.min_SOC,
            'reserve_time_s': self.reserve_time_s,
            'min_altitude': self.min_altitude,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MissionProfile':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            phases=[MissionPhase.from_dict(p) for p in data.get('phases', [])],
            initial_altitude=data.get('initial_altitude', 0.0),
            initial_speed=data.get('initial_speed', 0.0),
            initial_heading=data.get('initial_heading', 0.0),
            initial_SOC=data.get('initial_SOC', 1.0),
            T_ambient=data.get('T_ambient', 15.0),
            pressure_altitude=data.get('pressure_altitude', 0.0),
            wind_speed=data.get('wind_speed', 0.0),
            wind_direction=data.get('wind_direction', 0.0),
            runway_heading=data.get('runway_heading', 0.0),
            runway_elevation=data.get('runway_elevation', 0.0),
            runway_length=data.get('runway_length', 500.0),
            surface_friction=data.get('surface_friction', 0.04),
            min_SOC=data.get('min_SOC', 0.2),
            reserve_time_s=data.get('reserve_time_s', 300.0),
            min_altitude=data.get('min_altitude', 30.0),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MissionProfile':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def simple_cruise(
        cls,
        cruise_altitude: float,
        cruise_speed: float,
        cruise_duration: float,
        name: str = "Simple Cruise",
        climb_rate: float = 3.0,
        descent_rate: float = 2.0,
        takeoff_speed: float = 12.0,
        rotation_speed: float = 10.0,
    ) -> 'MissionProfile':
        """
        Create a simple cruise mission profile.
        
        Phases: Ground Idle -> Takeoff Roll -> Rotation -> Climb -> 
                Cruise -> Descent -> Approach -> Flare -> Landing Roll
        
        Args:
            cruise_altitude: Cruise altitude AGL [m]
            cruise_speed: Cruise airspeed [m/s]
            cruise_duration: Time at cruise altitude [s]
            name: Mission name
            climb_rate: Target climb rate [m/s]
            descent_rate: Target descent rate [m/s]
            takeoff_speed: Liftoff speed [m/s]
            rotation_speed: Speed to begin rotation [m/s]
        
        Returns:
            MissionProfile configured for simple cruise
        """
        phases = [
            # Ground idle - brief pause for systems
            MissionPhase(
                name="Ground Idle",
                phase_type=MissionPhaseType.GROUND_IDLE,
                duration=5.0,
                throttle_max=0.0,
            ),
            
            # Takeoff roll - accelerate to rotation speed
            MissionPhase(
                name="Takeoff Roll",
                phase_type=MissionPhaseType.TAKEOFF_ROLL,
                end_speed=rotation_speed,
                target_heading=0.0,  # Runway heading
                throttle_min=0.8,
                throttle_max=1.0,
            ),
            
            # Rotation - pitch up for liftoff
            MissionPhase(
                name="Rotation",
                phase_type=MissionPhaseType.ROTATION,
                end_altitude=5.0,  # Clear ground
                end_speed=takeoff_speed,
                target_speed=takeoff_speed,
                throttle_max=1.0,
                pitch_min=5.0,
                pitch_max=15.0,
            ),
            
            # Climb to cruise altitude
            MissionPhase(
                name="Climb",
                phase_type=MissionPhaseType.CLIMB,
                end_altitude=cruise_altitude,
                target_climb_rate=climb_rate,
                target_speed=cruise_speed * 0.9,  # Slightly slower during climb
                throttle_min=0.7,
                throttle_max=1.0,
            ),
            
            # Cruise at altitude
            MissionPhase(
                name="Cruise",
                phase_type=MissionPhaseType.CRUISE,
                duration=cruise_duration,
                target_altitude=cruise_altitude,
                target_speed=cruise_speed,
                throttle_min=0.0,
                throttle_max=0.8,
            ),
            
            # Descent to approach altitude
            MissionPhase(
                name="Descent",
                phase_type=MissionPhaseType.DESCENT,
                end_altitude=50.0,  # Pattern altitude
                target_descent_rate=descent_rate,
                target_speed=cruise_speed * 0.8,
                throttle_min=0.0,
                throttle_max=0.5,
            ),
            
            # Final approach
            MissionPhase(
                name="Approach",
                phase_type=MissionPhaseType.APPROACH,
                end_altitude=5.0,  # Flare altitude
                target_descent_rate=1.5,
                target_speed=takeoff_speed * 1.1,  # Approach speed
                throttle_min=0.0,
                throttle_max=0.4,
            ),
            
            # Landing flare
            MissionPhase(
                name="Flare",
                phase_type=MissionPhaseType.LANDING_FLARE,
                end_altitude=0.5,  # Touchdown
                target_descent_rate=0.5,
                throttle_max=0.2,
                pitch_min=0.0,
                pitch_max=10.0,
            ),
            
            # Landing roll
            MissionPhase(
                name="Landing Roll",
                phase_type=MissionPhaseType.LANDING_ROLL,
                end_speed=1.0,  # Nearly stopped
                throttle_max=0.0,
            ),
        ]
        
        return cls(
            name=name,
            phases=phases,
            initial_altitude=0.0,
            initial_speed=0.0,
            initial_heading=0.0,
        )
    
    @classmethod
    def waypoint_mission(
        cls,
        waypoints: List[Waypoint],
        cruise_altitude: float,
        cruise_speed: float,
        name: str = "Waypoint Mission",
        climb_rate: float = 3.0,
        descent_rate: float = 2.0,
        return_to_start: bool = True,
    ) -> 'MissionProfile':
        """
        Create a multi-waypoint navigation mission.
        
        Phases: Takeoff -> Climb -> [Waypoint Transit]* -> Descent -> Land
        
        Args:
            waypoints: List of waypoints to visit
            cruise_altitude: Default cruise altitude [m]
            cruise_speed: Cruise airspeed [m/s]
            name: Mission name
            climb_rate: Target climb rate [m/s]
            descent_rate: Target descent rate [m/s]
            return_to_start: Add return waypoint at (0,0)?
        
        Returns:
            MissionProfile with waypoint navigation phases
        """
        if not waypoints:
            raise ValueError("waypoints list cannot be empty")
        
        phases = []
        
        # Simplified takeoff sequence
        phases.append(MissionPhase(
            name="Takeoff",
            phase_type=MissionPhaseType.TAKEOFF_ROLL,
            end_speed=12.0,
            throttle_max=1.0,
        ))
        
        phases.append(MissionPhase(
            name="Rotation",
            phase_type=MissionPhaseType.ROTATION,
            end_altitude=10.0,
            throttle_max=1.0,
        ))
        
        # Climb to cruise
        phases.append(MissionPhase(
            name="Climb",
            phase_type=MissionPhaseType.CLIMB,
            end_altitude=cruise_altitude,
            target_climb_rate=climb_rate,
            target_speed=cruise_speed * 0.9,
            throttle_max=1.0,
        ))
        
        # Waypoint transit phases
        for i, wp in enumerate(waypoints):
            # Use waypoint altitude or cruise altitude
            wp_altitude = wp.altitude if wp.altitude > 0 else cruise_altitude
            wp_speed = wp.speed if wp.speed else cruise_speed
            
            phases.append(MissionPhase(
                name=f"Waypoint {i+1}",
                phase_type=MissionPhaseType.WAYPOINT_TRANSIT,
                waypoint=wp,
                target_altitude=wp_altitude,
                target_speed=wp_speed,
                throttle_min=0.3,
                throttle_max=0.9,
            ))
        
        # Optional return to start
        if return_to_start:
            return_wp = Waypoint(x=0, y=0, altitude=cruise_altitude)
            phases.append(MissionPhase(
                name="Return",
                phase_type=MissionPhaseType.WAYPOINT_TRANSIT,
                waypoint=return_wp,
                target_altitude=cruise_altitude,
                target_speed=cruise_speed,
            ))
        
        # Descent and landing
        phases.append(MissionPhase(
            name="Descent",
            phase_type=MissionPhaseType.DESCENT,
            end_altitude=30.0,
            target_descent_rate=descent_rate,
            throttle_max=0.4,
        ))
        
        phases.append(MissionPhase(
            name="Approach",
            phase_type=MissionPhaseType.APPROACH,
            end_altitude=3.0,
            target_descent_rate=1.5,
            throttle_max=0.3,
        ))
        
        phases.append(MissionPhase(
            name="Landing",
            phase_type=MissionPhaseType.LANDING_FLARE,
            end_altitude=0.0,
            throttle_max=0.1,
        ))
        
        return cls(
            name=name,
            phases=phases,
        )
    
    @classmethod
    def endurance_loiter(
        cls,
        loiter_altitude: float,
        loiter_speed: float,
        max_duration: float = 3600.0,
        name: str = "Endurance Loiter",
    ) -> 'MissionProfile':
        """
        Create a maximum endurance loiter mission.
        
        Args:
            loiter_altitude: Loiter altitude [m]
            loiter_speed: Optimal endurance speed [m/s]
            max_duration: Maximum loiter time [s]
            name: Mission name
        
        Returns:
            MissionProfile for endurance testing
        """
        phases = [
            MissionPhase(
                name="Takeoff",
                phase_type=MissionPhaseType.TAKEOFF_ROLL,
                end_speed=12.0,
                throttle_max=1.0,
            ),
            MissionPhase(
                name="Climb",
                phase_type=MissionPhaseType.CLIMB,
                end_altitude=loiter_altitude,
                target_climb_rate=2.0,
                throttle_max=1.0,
            ),
            MissionPhase(
                name="Loiter",
                phase_type=MissionPhaseType.LOITER,
                duration=max_duration,
                target_altitude=loiter_altitude,
                target_speed=loiter_speed,
                target_bank_angle=15.0,  # Gentle orbit
                throttle_min=0.2,
                throttle_max=0.6,
            ),
            MissionPhase(
                name="Descent",
                phase_type=MissionPhaseType.DESCENT,
                end_altitude=10.0,
                target_descent_rate=2.0,
                throttle_max=0.3,
            ),
            MissionPhase(
                name="Landing",
                phase_type=MissionPhaseType.LANDING_FLARE,
                end_altitude=0.0,
                throttle_max=0.1,
            ),
        ]
        
        return cls(
            name=name,
            phases=phases,
        )
