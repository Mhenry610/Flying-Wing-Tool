from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.materials import StructuralMaterial


@dataclass
class BodySection:
    """Defines a cross-section for BWB body blending."""
    y_pos: float = 0.0          # Spanwise location (absolute, meters)
    chord: float = 1.0          # Chord length (meters)
    x_offset: float = 0.0       # Leading edge x-offset (meters)
    z_offset: float = 0.0       # Vertical offset (meters)
    airfoil: str = "NACA0012"   # Airfoil name or file path
    
    def as_dict(self) -> Dict:
        return {
            "y_pos": self.y_pos,
            "chord": self.chord,
            "x_offset": self.x_offset,
            "z_offset": self.z_offset,
            "airfoil": self.airfoil,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> BodySection:
        return cls(**data)


@dataclass
class ControlSurface:
    """Defines a generic control surface (elevon, flap, aileron, etc.).
    
    Chord percentage defines the hinge line position:
    - chord_start_percent: hinge position at span_start (inboard, % of local chord)
    - chord_end_percent: hinge position at span_end (outboard, % of local chord)
    - Surface always extends from hinge to trailing edge (100% chord)
    """
    name: str = "Elevon"
    surface_type: str = "Elevon"        # Elevon, Flap, Aileron
    span_start_percent: float = 40.0    # eta start (% of half-span)
    span_end_percent: float = 100.0     # eta end (% of half-span)
    chord_start_percent: float = 65.0   # hinge at inboard edge (% of local chord)
    chord_end_percent: float = 75.0     # hinge at outboard edge (% of local chord)
    hinge_rel_height: float = 0.0       # CPACS hingeRelHeight: 0=Bottom, 0.5=Center, 1=Top
    
    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "surface_type": self.surface_type,
            "span_start_percent": self.span_start_percent,
            "span_end_percent": self.span_end_percent,
            "chord_start_percent": self.chord_start_percent,
            "chord_end_percent": self.chord_end_percent,
            "hinge_rel_height": self.hinge_rel_height,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> ControlSurface:
        # Handle legacy hinge_line_percent field
        if "hinge_line_percent" in data and "chord_start_percent" not in data:
            data["chord_start_percent"] = data.pop("hinge_line_percent")
            data["chord_end_percent"] = data.get("chord_end_percent", data["chord_start_percent"])
        elif "hinge_line_percent" in data:
            data.pop("hinge_line_percent", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})



@dataclass
class PlanformGeometry:
    wing_area_m2: float = 15.0
    aspect_ratio: float = 7.5
    taper_ratio: float = 0.4
    sweep_le_deg: float = 20.0
    dihedral_deg: float = 2.0
    elevon_root_span_percent: float = 40.0
    elevon_root_chord_percent: float = 35.0
    elevon_tip_chord_percent: float = 25.0
    front_spar_root_percent: float = 20.0
    front_spar_tip_percent: float = 20.0
    rear_spar_root_percent: float = 65.0
    rear_spar_tip_percent: float = 65.0
    rear_spar_span_percent: float = 100.0  # % of half-span the rear spar extends to

    center_chord_extension_percent: float = 0.0
    center_section_span_percent: float = 0.0
    center_extension_linear: bool = False
    snap_to_sections: bool = False  # If True, spars/surfaces snap to existing section positions
    bwb_blend_span_percent: float = 10.0  # % of wing span to blend from BWB chord to wing root chord
    bwb_dihedral_deg: float = 0.0  # Separate dihedral for BWB body sections (typically 0)
    
    # BWB body sections (empty list = standard flying wing, non-empty = BWB mode)
    body_sections: List[BodySection] = field(default_factory=list)
    
    # Generic control surfaces (replaces legacy elevon_* fields when non-empty)
    control_surfaces: List[ControlSurface] = field(default_factory=list)
    
    # Structural analysis configuration
    # Note: Material objects are referenced by name to avoid circular imports
    spar_material_name: str = "Balsa (Medium, 10 lb/ft³)"
    skin_material_name: str = "Balsa (Medium, 10 lb/ft³)"
    rib_material_name: str = "Plywood (Lite-Ply, 1.5mm)"  # Rib material (often plywood)
    rib_count: int = 10                    # Number of ribs per half-span (legacy, now derived from sections)
    spar_thickness_mm: float = 3.0         # Spar plate thickness [mm]
    skin_thickness_mm: float = 1.5         # Skin sheet thickness [mm]
    rib_thickness_mm: float = 3.0          # Rib plate thickness [mm]
    rib_lightening_fraction: float = 0.4   # Fraction of rib material removed (lightening holes)
    lightening_hole_margin_mm: float = 10.0 # Margin around lightening holes [mm]
    lightening_hole_shape: str = "circular" # Shape of lightening holes: "circular", "elliptical"
    factor_of_safety: float = 1.5          # Design factor of safety
    max_tip_deflection_percent: float = 15.0  # Max tip deflection as % of half-span
    max_tip_twist_deg: float = 3.0            # Max tip twist (combined bending+torsion) in degrees
    
    # Grain/fiber orientation (critical for orthotropic materials!)
    spar_grain_spanwise: bool = True       # True = grain runs along span (RECOMMENDED)
    skin_grain_spanwise: bool = True       # True = grain runs along span
    
    # Stringer configuration (longitudinal stiffeners between spars)
    stringer_count: int = 0                  # Number of stringers per skin panel (0 = no stringers)
    stringer_height_mm: float = 10.0         # Stringer height/depth [mm]
    stringer_thickness_mm: float = 1.5       # Stringer web thickness [mm]
    stringer_material_name: str = "Balsa (Medium, 10 lb/ft³)"  # Stringer material
    
    # Advanced buckling options
    skin_boundary_condition: str = "semi_restrained"  # "simply_supported", "semi_restrained", "clamped"
    include_curvature_effect: bool = True    # Account for airfoil curvature in buckling
    post_buckling_enabled: bool = False      # Allow skin panels to buckle and redistribute load to stringers
    spar_cap_width_mm: float = 10.0          # Spar cap width bearing on ribs [mm] (for rib crushing check)

    cache: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def rear_spar_elevon_root_percent(self) -> float:
        return self.elevon_root_chord_percent

    def area(self) -> float:
        return self.wing_area_m2

    def span(self) -> float:
        if "span" not in self.cache:
            self.cache["span"] = (self.aspect_ratio * self.wing_area_m2) ** 0.5
        return self.cache["span"]

    def half_span(self) -> float:
        return 0.5 * self.span()

    def actual_area(self) -> float:
        """Calculates the total wing area including the center chord extension or BWB sections."""
        base_area = self.wing_area_m2
        
        # 1. BWB sections contribution (mutually exclusive with center extension in current implementation)
        if self.body_sections:
            bwb_area_one_side = 0.0
            sorted_sections = sorted(self.body_sections, key=lambda bs: bs.y_pos)
            for i in range(len(sorted_sections) - 1):
                s1 = sorted_sections[i]
                s2 = sorted_sections[i+1]
                dy = abs(s2.y_pos - s1.y_pos)
                avg_c = 0.5 * (s1.chord + s2.chord)
                bwb_area_one_side += avg_c * dy
            return base_area + (2.0 * bwb_area_one_side)

        # 2. Center chord extension contribution
        if self.center_chord_extension_percent <= 0 or self.center_section_span_percent <= 0:
            return base_area
            
        # Calculate added area from extension
        # Both Linear and Cosine blends result in the same integral (0.5 * width * height)
        # Linear: Triangle area = 0.5 * base * height
        # Cosine: Integral of 0.5(1+cos(x)) from 0 to 1 is 0.5
        
        c_root = self.root_chord()
        half_span = self.half_span()
        
        ext_max_chord = c_root * (self.center_chord_extension_percent / 100.0)
        ext_span_width = half_span * (self.center_section_span_percent / 100.0)
        
        # Area added per side = 0.5 * width * height
        added_area_per_side = 0.5 * ext_span_width * ext_max_chord
        
        return base_area + (2.0 * added_area_per_side)

    def actual_span(self) -> float:
        """Calculates the total span including BWB sections."""
        wing_span = self.span()
        if not self.body_sections:
            return wing_span
        
        bwb_outer_y = max([bs.y_pos for bs in self.body_sections], default=0.0)
        return wing_span + (2.0 * bwb_outer_y)

    def actual_aspect_ratio(self) -> float:
        """Calculates the aspect ratio using the actual area and actual span."""
        area = self.actual_area()
        if area <= 0: return 0.0
        return (self.actual_span() ** 2) / area

    def mean_aerodynamic_chord(self) -> float:
        if "mac" not in self.cache:
            root = self.root_chord()
            tip = self.tip_chord()
            self.cache["mac"] = (2.0 / 3.0) * root * (1 + self.taper_ratio + self.taper_ratio ** 2) / (1 + self.taper_ratio)
        return self.cache["mac"]

    def root_chord(self) -> float:
        if "c_root" not in self.cache:
            span = self.span()
            self.cache["c_root"] = 2 * self.wing_area_m2 / (span * (1 + self.taper_ratio))
        return self.cache["c_root"]

    def extended_root_chord(self) -> float:
        """Returns the root chord length including the center extension."""
        base = self.root_chord()
        if self.center_chord_extension_percent > 0:
            return base * (1.0 + self.center_chord_extension_percent / 100.0)
        return base

    def tip_chord(self) -> float:
        return self.root_chord() * self.taper_ratio

    def reset_cache(self) -> None:
        self.cache.clear()

    def as_dict(self) -> Dict:
        return {
            "wing_area_m2": self.wing_area_m2,
            "aspect_ratio": self.aspect_ratio,
            "taper_ratio": self.taper_ratio,
            "sweep_le_deg": self.sweep_le_deg,
            "dihedral_deg": self.dihedral_deg,
            "elevon_root_span_percent": self.elevon_root_span_percent,
            "elevon_root_chord_percent": self.elevon_root_chord_percent,
            "elevon_tip_chord_percent": self.elevon_tip_chord_percent,
            "front_spar_root_percent": self.front_spar_root_percent,
            "front_spar_tip_percent": self.front_spar_tip_percent,
            "rear_spar_root_percent": self.rear_spar_root_percent,
            "rear_spar_tip_percent": self.rear_spar_tip_percent,
            "rear_spar_span_percent": self.rear_spar_span_percent,

            "rear_spar_elevon_root_percent": self.rear_spar_elevon_root_percent,
            "center_chord_extension_percent": self.center_chord_extension_percent,
            "center_section_span_percent": self.center_section_span_percent,
            "center_extension_linear": self.center_extension_linear,
            "snap_to_sections": self.snap_to_sections,
            "bwb_blend_span_percent": self.bwb_blend_span_percent,
            "bwb_dihedral_deg": self.bwb_dihedral_deg,
            "body_sections": [bs.as_dict() for bs in self.body_sections],
            "control_surfaces": [cs.as_dict() for cs in self.control_surfaces],
            # Structural analysis configuration
            "spar_material_name": self.spar_material_name,
            "skin_material_name": self.skin_material_name,
            "rib_count": self.rib_count,
            "spar_thickness_mm": self.spar_thickness_mm,
            "skin_thickness_mm": self.skin_thickness_mm,
            "rib_thickness_mm": self.rib_thickness_mm,
            "rib_lightening_fraction": self.rib_lightening_fraction,
            "lightening_hole_margin_mm": self.lightening_hole_margin_mm,
            "lightening_hole_shape": self.lightening_hole_shape,
            "factor_of_safety": self.factor_of_safety,
            "max_tip_deflection_percent": self.max_tip_deflection_percent,
            "max_tip_twist_deg": self.max_tip_twist_deg,
            "spar_grain_spanwise": self.spar_grain_spanwise,
            "skin_grain_spanwise": self.skin_grain_spanwise,
            # Stringer configuration
            "stringer_count": self.stringer_count,
            "stringer_height_mm": self.stringer_height_mm,
            "stringer_thickness_mm": self.stringer_thickness_mm,
            "stringer_material_name": self.stringer_material_name,
            # Advanced buckling options
            "skin_boundary_condition": self.skin_boundary_condition,
            "include_curvature_effect": self.include_curvature_effect,
            "post_buckling_enabled": self.post_buckling_enabled,
            "spar_cap_width_mm": self.spar_cap_width_mm,
        }
