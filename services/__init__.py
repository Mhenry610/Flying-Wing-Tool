"""
Service layer scaffolding.

Modules planned:
- viewer.py — 3D visualization utilities, CPACS processing for plotting
- step_export.py — STEP export operations (pythonocc-core dependent)
- fixture_export.py — fixture layout generation and export

Source for extraction reference: FlyingWingGeneratorV1-1.py (class CpacsStepperTab methods)
"""

# Aero model factory for 6-DOF dynamics integration
from services.aero_model import (
    create_dynamics_aero_model,
    create_simple_flying_wing_aero_model,
)