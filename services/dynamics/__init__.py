"""Aircraft dynamics models: 6-DOF rigid body, ground effect, landing gear."""

from .aircraft_dynamics import (
    MassProperties,
    AircraftState,
    ControlInputs,
    StateIndex,
    FlyingWingDynamics6DOF,
    rotation_matrix_body_to_earth,
    rotation_matrix_earth_to_body,
    create_simple_aero_model,
)

from .landing_gear import (
    GearType,
    SurfaceType,
    SurfaceProperties,
    SURFACE_PRESETS,
    get_surface_properties,
    LandingGearParameters,
    LandingGearSet,
    GearContactState,
    LandingGearModel,
    create_small_uav_tricycle,
    create_flying_wing_belly_skid,
)

__all__ = [
    # Aircraft dynamics
    "MassProperties",
    "AircraftState",
    "ControlInputs",
    "StateIndex",
    "FlyingWingDynamics6DOF",
    "rotation_matrix_body_to_earth",
    "rotation_matrix_earth_to_body",
    "create_simple_aero_model",
    # Landing gear
    "GearType",
    "SurfaceType",
    "SurfaceProperties",
    "SURFACE_PRESETS",
    "get_surface_properties",
    "LandingGearParameters",
    "LandingGearSet",
    "GearContactState",
    "LandingGearModel",
    "create_small_uav_tricycle",
    "create_flying_wing_belly_skid",
]
