"""
Service layer scaffolding.

Modules planned:
- viewer.py — 3D visualization utilities, CPACS processing for plotting
- step_export.py — STEP export operations (pythonocc-core dependent)
- fixture_export.py — fixture layout generation and export

Source for extraction reference: FlyingWingGeneratorV1-1.py (class CpacsStepperTab methods)
"""

_AERO_MODEL_EXPORTS = {
    "create_dynamics_aero_model",
    "create_simple_flying_wing_aero_model",
}


def __getattr__(name):
    if name in _AERO_MODEL_EXPORTS:
        from services.aero_model import (
            create_dynamics_aero_model,
            create_simple_flying_wing_aero_model,
        )

        exports = {
            "create_dynamics_aero_model": create_dynamics_aero_model,
            "create_simple_flying_wing_aero_model": create_simple_flying_wing_aero_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_AERO_MODEL_EXPORTS)
