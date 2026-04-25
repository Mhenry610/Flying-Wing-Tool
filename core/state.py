from __future__ import annotations
import json
import dataclasses
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import os

from core.models.project import WingProject
from services.mission.planner import MissionSegment, AeroConfig
from services.mission.motor import MotorProp

# Schema version for project files
# v0: implicit (no version field) - legacy format with elevon_* fields
# v1: BWB format with body_sections and control_surfaces lists
CURRENT_SCHEMA_VERSION = 1

@dataclass
class MissionProfile:
    segments: List[MissionSegment] = field(default_factory=list)
    aero_config: AeroConfig = field(default_factory=lambda: AeroConfig(cl_cruise=0.5))
    motor: MotorProp = field(default_factory=lambda: MotorProp(KV_rpm_per_V=1000.0, Ri_mOhm=100.0, Io_A=0.5))
    payload_mass_kg: float = 0.0
    gui_settings: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> dict:
        return {
            "segments": [dataclasses.asdict(s) for s in self.segments],
            "aero_config": dataclasses.asdict(self.aero_config),
            "motor": {
                "KV_rpm_per_V": self.motor.KV,
                "Ri_mOhm": self.motor.R20 * 1000.0,
                "Io_A": self.motor.Io,
                "V_at_Io": self.motor.V_at_Io
            },
            "payload_mass_kg": self.payload_mass_kg,
            "gui_settings": self.gui_settings,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> MissionProfile:
        segments = [MissionSegment(**s) for s in data.get("segments", [])]
        aero_config = AeroConfig(**data.get("aero_config", {}))
        motor = MotorProp(**data.get("motor", {}))
        return cls(
            segments=segments,
            aero_config=aero_config,
            motor=motor,
            payload_mass_kg=data.get("payload_mass_kg", 0.0),
            gui_settings=data.get("gui_settings", {}),
        )

@dataclass
class AnalysisResults:
    # Map of "Re_Alpha" -> {"CL": ..., "CD": ...} or similar structure
    # For now, we'll store raw polar data dictionaries
    polars: Dict[str, Any] = field(default_factory=dict)
    
    # Computed twist distribution (degrees) matching the wing sections
    twist_distribution: List[float] = field(default_factory=list)
    
    # Center of Gravity location (x-coordinate)
    x_cg: Optional[float] = None
    
    # Performance metrics (cruise/takeoff speeds, CL, CD, etc.)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Structural analysis results (from StructuralAnalysisResult.as_dict())
    structural_analysis: Dict[str, Any] = field(default_factory=dict)

    # Mission simulation results (summary + per-phase stats)
    mission_results: Dict[str, Any] = field(default_factory=dict)

    # Per-tab GUI settings that do not belong in the aerodynamic/structural data.
    gui_settings: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> dict:
        return {
            "polars": self.polars,
            "twist_distribution": self.twist_distribution,
            "x_cg": self.x_cg,
            "performance_metrics": self.performance_metrics,
            "structural_analysis": self.structural_analysis,
            "mission_results": self.mission_results,
            "gui_settings": self.gui_settings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnalysisResults:
        return cls(
            polars=data.get("polars", {}),
            twist_distribution=data.get("twist_distribution", []),
            x_cg=data.get("x_cg"),
            performance_metrics=data.get("performance_metrics", {}),
            structural_analysis=data.get("structural_analysis", {}),
            mission_results=data.get("mission_results", {}),
            gui_settings=data.get("gui_settings", {}),
        )

@dataclass
class Project:
    wing: WingProject = field(default_factory=WingProject)
    mission: MissionProfile = field(default_factory=MissionProfile)
    analysis: AnalysisResults = field(default_factory=AnalysisResults)
    
    def to_dict(self) -> dict:
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "wing": self.wing.as_dict(),
            "mission": self.mission.as_dict(),
            "analysis": self.analysis.as_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Project:
        # Detect schema version
        schema_version = data.get("schema_version", 0)
        
        if schema_version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Project file uses schema version {schema_version}, but this tool only supports "
                f"up to version {CURRENT_SCHEMA_VERSION}. Please update the tool to load this file."
            )
        
        if schema_version < CURRENT_SCHEMA_VERSION:
            logging.warning(
                f"Loading legacy project (schema v{schema_version}). "
                f"Migrating to v{CURRENT_SCHEMA_VERSION}. Save to update the file format."
            )
        
        # Reconstruct WingProject
        wing_data = data.get("wing", {})
        
        # 1. Planform
        from core.models.planform import PlanformGeometry, BodySection, ControlSurface

        pf_data = wing_data.get("planform", {})
        pf_kwargs = {}
        
        valid_pf_fields = {
            f.name for f in dataclasses.fields(PlanformGeometry) if f.init
        } - {"body_sections", "control_surfaces"}
        for k, v in pf_data.items():
            if k in valid_pf_fields:
                if k in ("center_extension_linear", "snap_to_sections"):
                    pf_kwargs[k] = bool(v)
                else:
                    pf_kwargs[k] = v
        
        # Handle body_sections (new in v1, defaults to empty)
        body_sections_data = pf_data.get("body_sections", [])
        body_sections = [BodySection.from_dict(bs) for bs in body_sections_data]
        
        # Handle control_surfaces with v0 -> v1 migration
        control_surfaces_data = pf_data.get("control_surfaces", [])
        if control_surfaces_data:
            # v1 format: use control_surfaces directly
            control_surfaces = [ControlSurface.from_dict(cs) for cs in control_surfaces_data]
        elif schema_version == 0:
            # v0 -> v1 migration: convert legacy elevon_* fields to a single ControlSurface
            elevon_root_span = pf_data.get("elevon_root_span_percent", 40.0)
            elevon_root_chord = pf_data.get("elevon_root_chord_percent", 35.0)
            elevon_tip_chord = pf_data.get("elevon_tip_chord_percent", 25.0)
            
            # Create a default elevon from legacy fields
            control_surfaces = [ControlSurface(
                name="Elevon",
                surface_type="Elevon",
                span_start_percent=elevon_root_span,
                span_end_percent=100.0,
                chord_start_percent=100.0 - elevon_root_chord,  # Convert to xsi (from TE)
                chord_end_percent=100.0,
                hinge_line_percent=100.0 - elevon_root_chord,
                hinge_rel_height=0.0  # Default to bottom hinge
            )]
            logging.info(f"Migrated legacy elevon fields to ControlSurface: {control_surfaces[0].name}")
        else:
            control_surfaces = []
        
        planform = PlanformGeometry(
            **pf_kwargs,
            body_sections=body_sections,
            control_surfaces=control_surfaces
        )

        # 2. Twist Trim
        tt_data = wing_data.get("twist_trim", {})
        from core.models.twist_trim import TwistTrimParameters
        twist_trim = TwistTrimParameters(**tt_data)

        # 3. Airfoil
        af_data = wing_data.get("airfoil", {})
        from core.models.airfoil import AirfoilInterpolation
        airfoil = AirfoilInterpolation(**af_data)

        # 4. WingProject
        wing = WingProject(
            name=wing_data.get("name", "FlyingWingProject"),
            planform=planform,
            twist_trim=twist_trim,
            airfoil=airfoil,
            optimized_twist_deg=wing_data.get("optimized_twist_deg")
        )
        
        mission = MissionProfile.from_dict(data.get("mission", {}))
        analysis = AnalysisResults.from_dict(data.get("analysis", {}))
        
        return cls(wing=wing, mission=mission, analysis=analysis)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> Project:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

