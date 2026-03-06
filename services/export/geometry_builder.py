# services/export/geometry_builder.py
"""
Direct geometry generation from Project state for STEP export.

This module generates OCC geometry directly from the native Project + SpanwiseSection
data, replacing the CPACS-based pipeline. It is the SINGLE SOURCE OF TRUTH for
3D geometry generation.

Uses profiles.py for 2D shapes (shared with DXF export) to ensure manufacturing
drawings match CAD geometry.

Architecture:
    Project (JSON) -> geometry_builder.py -> step_export.py -> STEP
                            |
                   Uses profiles.py for 2D shapes (shared with DXF)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Ax3, gp_Trsf
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Wire
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopAbs import TopAbs_SOLID

from core.state import Project
from services.geometry import (
    SpanwiseSection,
    AeroSandboxService,
    compute_local_spar_sweep_angles,
    get_spar_xsi_at_section,
)
from services.export.profiles import (
    generate_rib_profile,
    generate_spar_profile,
    generate_separated_spar_profiles,
    generate_elevon_rib_profile,
    get_lightening_hole_geometries,
    get_stringer_slot_polylines,
    RibProfileParams,
    SparProfileParams,
    RibProfile,
    SparProfile,
    ElevonRibProfile,
)

# Import OCC utilities from core (use relative path from services)
try:
    from core.occ_utils.shapes import (
        make_wire_from_points,
        make_face_from_points,
        extrude_prism,
        loft_solid_from_wires,
        scale_shape,
        mirror_y,
        make_compound,
        make_airfoil_wire_spline,
        make_closed_bspline_wire_from_points,
        make_partial_airfoil_wire_spline,
        loft_surface_from_profiles,
        sew_faces_to_solid,
        split_airfoil_upper_lower,
    )
    from core.occ_utils.booleans import cut as bool_cut, fuse as bool_fuse
except ImportError:
    # Fallback to refactor path for compatibility
    from refactor.occ_utils.shapes import (
        make_wire_from_points,
        make_face_from_points,
        extrude_prism,
        loft_solid_from_wires,
        scale_shape,
        mirror_y,
        make_compound,
        make_airfoil_wire_spline,
        make_closed_bspline_wire_from_points,
        make_partial_airfoil_wire_spline,
        loft_surface_from_profiles,
        sew_faces_to_solid,
        split_airfoil_upper_lower,
    )
    from refactor.occ_utils.booleans import cut as bool_cut, fuse as bool_fuse


# ==============================================================================
# Twist Transformation Utilities
# ==============================================================================

def apply_twist_to_points(
    points: List[np.ndarray],
    twist_deg: float,
    pivot_x: float,
    pivot_z: float,
) -> List[np.ndarray]:
    """
    Apply twist rotation to a list of 3D points about the Y-axis.
    
    Twist is applied as rotation about the leading edge (or specified pivot).
    Positive twist = nose up = trailing edge rotates down.
    
    The rotation is in the XZ plane (about Y-axis), which is correct for
    flying wing twist where Y is the spanwise direction.
    
    Args:
        points: List of 3D points as np.ndarray [x, y, z]
        twist_deg: Twist angle in degrees (positive = nose up)
        pivot_x: X coordinate of rotation pivot (typically x_le_m)
        pivot_z: Z coordinate of rotation pivot (typically z_m at LE)
    
    Returns:
        List of rotated 3D points
    """
    if abs(twist_deg) < 1e-6:
        return points  # No rotation needed
    
    twist_rad = math.radians(twist_deg)
    cos_t = math.cos(twist_rad)
    sin_t = math.sin(twist_rad)
    
    rotated = []
    for pt in points:
        # Translate to pivot origin
        dx = pt[0] - pivot_x
        dz = pt[2] - pivot_z
        
        # Rotate about Y-axis (in XZ plane)
        # For nose-up twist (positive angle), TE moves down:
        #   x' = x * cos(θ) + z * sin(θ)
        #   z' = -x * sin(θ) + z * cos(θ)
        x_rot = dx * cos_t + dz * sin_t
        z_rot = -dx * sin_t + dz * cos_t
        
        # Translate back
        rotated.append(np.array([
            pivot_x + x_rot,
            pt[1],  # Y unchanged
            pivot_z + z_rot,
        ]))
    
    return rotated


def apply_twist_to_shape(
    shape: TopoDS_Shape,
    twist_deg: float,
    pivot_x: float,
    pivot_y: float,
    pivot_z: float,
) -> TopoDS_Shape:
    """
    Apply twist rotation to an OCC shape about the Y-axis at a pivot point.
    
    Used for rotating extruded ribs/elevons after they're built in the XZ plane.
    
    Args:
        shape: OCC shape to rotate
        twist_deg: Twist angle in degrees
        pivot_x, pivot_y, pivot_z: Pivot point coordinates (typically LE position)
    
    Returns:
        Rotated shape
    """
    if abs(twist_deg) < 1e-6:
        return shape  # No rotation needed
    
    twist_rad = math.radians(twist_deg)
    
    # Create rotation transformation about Y-axis at pivot point
    pivot = gp_Pnt(pivot_x, pivot_y, pivot_z)
    axis = gp_Ax1(pivot, gp_Dir(0, 1, 0))  # Y-axis
    
    trsf = gp_Trsf()
    trsf.SetRotation(axis, twist_rad)
    
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class WingGeometryConfig:
    """Export configuration options for geometry generation."""
    
    # Component toggles
    include_skin: bool = True
    include_wingbox: bool = True       # Front/rear spar + skin panels between
    include_stringers: bool = True     # Spanwise stiffeners on skin
    include_ribs: bool = True
    include_control_surfaces: bool = True
    
    # Fixture generation (NEW)
    include_fixtures: bool = False
    fixture_material_thickness_mm: float = 6.35
    fixture_slot_clearance_mm: float = 0.15
    fixture_height_mm: float = 50.0
    fixture_add_cradle: bool = True
    fixture_tab_width_mm: float = 15.0
    fixture_tab_spacing_mm: float = 80.0
    
    # Representation options
    skin_representation: str = "surfaces"  # "surfaces" = separate upper/lower
    
    # Assembly options
    mirror_to_full_aircraft: bool = True
    scale_factor: float = 1000.0       # m -> mm for CAD
    
    # Profile generation options
    include_spar_notches: bool = True
    include_stringer_slots: bool = True
    include_lightening_holes: bool = False
    lightening_hole_shape: str = "circular"
    apply_sweep_correction: bool = True
    
    # Rib notch configuration
    spar_notch_clearance_mm: float = 0.1
    stringer_slot_clearance_mm: float = 0.2


# ==============================================================================
# Generated Geometry Container
# ==============================================================================

@dataclass
class GeneratedGeometry:
    """Container for all generated OCC shapes."""
    
    # Skin surfaces (lofted from airfoil sections)
    wing_skin_upper: Optional[TopoDS_Shape] = None
    wing_skin_lower: Optional[TopoDS_Shape] = None
    
    # Wing box components
    wingbox_front_spar: Optional[TopoDS_Shape] = None  # Lofted/extruded spar web
    wingbox_rear_spar: Optional[TopoDS_Shape] = None
    wingbox_skin_upper: Optional[TopoDS_Shape] = None  # Skin between spars only
    wingbox_skin_lower: Optional[TopoDS_Shape] = None
    
    # Internal structure
    ribs: List[TopoDS_Shape] = None           # Main wing ribs
    elevon_ribs: List[TopoDS_Shape] = None    # Control surface ribs (separate pieces)
    stringers: List[TopoDS_Shape] = None      # Spanwise extrusions
    
    # Spars (may be multiple for BWB: body + wing)
    front_spar_profiles: List[SparProfile] = None
    rear_spar_profiles: List[SparProfile] = None
    
    # Control surfaces (movable surfaces only, no hinge brackets)
    control_surfaces: Dict[str, TopoDS_Shape] = None
    
    # Fixture geometry (NEW)
    fixture_profiles: List[Any] = None        # List of FixtureProfile
    cradle_profiles: List[Any] = None         # List of CradleProfile
    fixtures: List[TopoDS_Shape] = None
    cradles: List[TopoDS_Shape] = None
    base_plate: Optional[TopoDS_Shape] = None
    fixture_assembly: Optional[TopoDS_Compound] = None
    
    # Metadata for diagnostics
    rib_profiles: List[RibProfile] = None
    elevon_rib_profiles: List[ElevonRibProfile] = None
    section_count: int = 0
    
    def __post_init__(self):
        if self.ribs is None:
            self.ribs = []
        if self.elevon_ribs is None:
            self.elevon_ribs = []
        if self.stringers is None:
            self.stringers = []
        if self.control_surfaces is None:
            self.control_surfaces = {}
        if self.rib_profiles is None:
            self.rib_profiles = []
        if self.elevon_rib_profiles is None:
            self.elevon_rib_profiles = []
        if self.front_spar_profiles is None:
            self.front_spar_profiles = []
        if self.rear_spar_profiles is None:
            self.rear_spar_profiles = []
        # Fixture fields
        if self.fixture_profiles is None:
            self.fixture_profiles = []
        if self.cradle_profiles is None:
            self.cradle_profiles = []
        if self.fixtures is None:
            self.fixtures = []
        if self.cradles is None:
            self.cradles = []


# ==============================================================================
# Main Entry Point
# ==============================================================================

def build_geometry_from_project(
    project: Project,
    config: WingGeometryConfig = None,
) -> GeneratedGeometry:
    """
    Main entry point - builds all geometry from native Project state.
    
    This is the SINGLE SOURCE OF TRUTH for 3D geometry generation.
    
    Args:
        project: Project containing wing planform, airfoils, structure params
        config: Export configuration options (defaults provided)
    
    Returns:
        GeneratedGeometry with all requested OCC shapes
    """
    if config is None:
        config = WingGeometryConfig()
    
    # Get spanwise sections from geometry service
    service = AeroSandboxService(project)
    sections = service.spanwise_sections()
    
    if len(sections) < 2:
        print(f"[GeometryBuilder] Warning: Need at least 2 sections, got {len(sections)}")
        return GeneratedGeometry(section_count=len(sections))
    
    result = GeneratedGeometry(section_count=len(sections))
    plan = project.wing.planform
    
    # Compute local spar sweep angles for each section (used for slot corrections)
    front_sweep_angles = compute_local_spar_sweep_angles(sections, "front", plan)
    rear_sweep_angles = compute_local_spar_sweep_angles(sections, "rear", plan)
    dihedral_rad = math.radians(plan.dihedral_deg)
    
    # Build components based on config
    if config.include_skin:
        print("[GeometryBuilder] Building skin surfaces...")
        result.wing_skin_upper, result.wing_skin_lower = build_skin_surfaces(
            sections, plan, segmented=True
        )
    
    if config.include_wingbox:
        print("[GeometryBuilder] Building spar geometry...")
        result.front_spar_profiles, result.wingbox_front_spar = build_spar_geometry(
            project, sections, "front", config
        )
        result.rear_spar_profiles, result.wingbox_rear_spar = build_spar_geometry(
            project, sections, "rear", config
        )
        
        print("[GeometryBuilder] Building wingbox skin...")
        result.wingbox_skin_upper, result.wingbox_skin_lower = build_wingbox_skin(
            sections, plan
        )
    
    if config.include_ribs:
        print("[GeometryBuilder] Building rib geometry...")
        result.ribs, result.elevon_ribs, result.rib_profiles, result.elevon_rib_profiles = \
            build_rib_geometry(
                project, sections,
                front_sweep_angles, rear_sweep_angles, dihedral_rad,
                config
            )
    
    if config.include_stringers:
        print("[GeometryBuilder] Building stringer geometry...")
        result.stringers = build_stringer_geometry(sections, plan)
    
    if config.include_control_surfaces:
        print("[GeometryBuilder] Building control surface geometry...")
        result.control_surfaces = build_control_surface_geometry(sections, plan, project)
    
    if config.include_fixtures:
        print("[GeometryBuilder] Building fixture geometry...")
        result.fixtures, result.cradles, result.base_plate, \
        result.fixture_profiles, result.cradle_profiles, result.fixture_assembly = \
            build_fixture_geometry(project, sections, result, config)
    
    return result


# ==============================================================================
# Skin Surface Generation
# ==============================================================================

def build_skin_surfaces(
    sections: List[SpanwiseSection],
    planform,
    segmented: bool = True,
) -> Tuple[Optional[TopoDS_Shape], Optional[TopoDS_Shape]]:
    """
    Build upper and lower skin as separate lofted surfaces.
    
    Surfaces span the full chord (LE to TE).
    For wingbox skin (between spars only), use build_wingbox_skin().
    
    Args:
        sections: List of SpanwiseSection with airfoil and position data
        planform: PlanformGeometry for reference
        segmented: If True, loft segment-by-segment (more robust)
    
    Returns:
        Tuple of (upper_surface, lower_surface) as TopoDS_Shape
    """
    if len(sections) < 2:
        return None, None
    
    if segmented:
        return _build_skin_surfaces_segmented(sections)
    else:
        return _build_skin_surfaces_single_loft(sections)


def _build_skin_surfaces_single_loft(
    sections: List[SpanwiseSection],
) -> Tuple[Optional[TopoDS_Shape], Optional[TopoDS_Shape]]:
    """
    Build skin using a single multi-section loft.
    May fail with dissimilar profiles.
    
    Applies twist rotation to each section's airfoil profile.
    """
    upper_profiles = []
    lower_profiles = []
    
    for section in sections:
        # Get airfoil coordinates in physical dimensions (meters)
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        coords = np.array(section.airfoil.coordinates)  # Nx2 normalized
        chord = section.chord_m
        
        # Scale and position in 3D
        x = coords[:, 0] * chord + section.x_le_m
        z = coords[:, 1] * chord  # Thickness direction
        y = np.full_like(x, section.y_m)  # Spanwise position
        
        # Add dihedral effect to Z
        z = z + section.z_m
        
        points_3d = np.column_stack([x, y, z])
        
        # Apply twist rotation about the leading edge
        if abs(section.twist_deg) > 1e-6:
            points_list = [pt for pt in points_3d]
            points_list = apply_twist_to_points(
                points_list,
                section.twist_deg,
                pivot_x=section.x_le_m,
                pivot_z=section.z_m,
            )
            points_3d = np.array(points_list)
        
        upper, lower = split_airfoil_upper_lower(points_3d)
        
        upper_profiles.append(upper)
        lower_profiles.append(lower)
    
    if len(upper_profiles) < 2:
        return None, None
    
    # Loft surfaces
    upper_surface = loft_surface_from_profiles(upper_profiles, which='upper')
    lower_surface = loft_surface_from_profiles(lower_profiles, which='lower')
    
    return upper_surface, lower_surface


def _build_skin_surfaces_segmented(
    sections: List[SpanwiseSection],
) -> Tuple[Optional[TopoDS_Shape], Optional[TopoDS_Shape]]:
    """
    Build skin by lofting segment-by-segment between adjacent sections.
    
    Each segment is lofted independently, then sewn into a continuous shell.
    This avoids lofting failures from dissimilar profile counts or sharp transitions.
    """
    if len(sections) < 2:
        return None, None
    
    upper_sewing = BRepBuilderAPI_Sewing(1e-6)
    lower_sewing = BRepBuilderAPI_Sewing(1e-6)
    
    for i in range(len(sections) - 1):
        section_a = sections[i]
        section_b = sections[i + 1]
        
        # Get profiles for this segment
        upper_a, lower_a = _get_section_profiles_3d(section_a)
        upper_b, lower_b = _get_section_profiles_3d(section_b)
        
        if upper_a is None or upper_b is None:
            continue
        
        # Loft just these two profiles
        upper_segment = loft_surface_from_profiles(
            [upper_a, upper_b], which='upper'
        )
        lower_segment = loft_surface_from_profiles(
            [lower_a, lower_b], which='lower'
        )
        
        if upper_segment and not upper_segment.IsNull():
            upper_sewing.Add(upper_segment)
        if lower_segment and not lower_segment.IsNull():
            lower_sewing.Add(lower_segment)
    
    # Sew segments into continuous shells
    upper_sewing.Perform()
    lower_sewing.Perform()
    
    upper_result = upper_sewing.SewedShape()
    lower_result = lower_sewing.SewedShape()
    
    return upper_result, lower_result


def _get_section_profiles_3d(
    section: SpanwiseSection,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Get 3D upper and lower airfoil profiles for a section.
    
    Applies twist rotation about the leading edge to match the
    geometric twist at this spanwise station.
    
    Returns:
        Tuple of (upper_points, lower_points) as lists of np.ndarray
    """
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None, None
    
    coords = np.array(section.airfoil.coordinates)
    chord = section.chord_m
    
    # Scale and position in 3D (before twist)
    x = coords[:, 0] * chord + section.x_le_m
    z = coords[:, 1] * chord + section.z_m
    y = np.full_like(x, section.y_m)
    
    points_3d = np.column_stack([x, y, z])
    
    # Apply twist rotation about the leading edge
    if abs(section.twist_deg) > 1e-6:
        points_list = [pt for pt in points_3d]
        points_list = apply_twist_to_points(
            points_list,
            section.twist_deg,
            pivot_x=section.x_le_m,
            pivot_z=section.z_m,
        )
        points_3d = np.array(points_list)
    
    upper, lower = split_airfoil_upper_lower(points_3d)
    
    return upper, lower


# ==============================================================================
# Spar Geometry Generation
# ==============================================================================

def build_spar_geometry(
    project: Project,
    sections: List[SpanwiseSection],
    spar_type: str,  # "front" or "rear"
    config: WingGeometryConfig,
) -> Tuple[List[SparProfile], Optional[TopoDS_Shape]]:
    """
    Build 3D spar solid by extruding the 2D DXF profile.
    
    The 2D profile (from profiles.py) is the SINGLE SOURCE OF TRUTH for spar geometry.
    It includes rib notches, finger joints, and follows the airfoil envelope.
    
    This ensures manufacturing drawings (DXF) match CAD geometry (STEP).
    
    Args:
        project: Project with planform/structural parameters
        sections: List of SpanwiseSection for rib positions
        spar_type: "front" or "rear"
        config: Geometry configuration
    
    Returns:
        Tuple of (spar_profiles, spar_compound)
    """
    plan = project.wing.planform
    
    params = SparProfileParams(
        include_rib_notches=config.include_spar_notches,
        rib_notch_clearance_mm=config.spar_notch_clearance_mm,
        rib_notch_depth_percent=50.0,
    )
    
    # Generate 2D profiles (same as DXF) - may be split for BWB
    spar_profiles = generate_separated_spar_profiles(project, sections, spar_type, params)
    
    print(f"[GeometryBuilder] {spar_type} spar: got {len(spar_profiles)} profiles")
    for i, p in enumerate(spar_profiles):
        print(f"  Profile {i}: region={p.spar_region}, length={p.length_mm:.1f}mm, stations={len(p.station_heights) if p.station_heights else 0}")
    
    if not spar_profiles:
        print(f"[GeometryBuilder] Warning: No spar profiles generated for {spar_type}")
        return [], None
    
    # Build 3D spar by extruding each 2D profile
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    
    success_count = 0
    for profile in spar_profiles:
        spar_solid = _extrude_spar_from_profile(profile, sections, spar_type, plan)
        if spar_solid and not spar_solid.IsNull():
            builder.Add(compound, spar_solid)
            success_count += 1
        else:
            print(f"[GeometryBuilder] Warning: Failed to extrude {spar_type} spar profile (region={profile.spar_region})")
    
    print(f"[GeometryBuilder] {spar_type} spar: extruded {success_count}/{len(spar_profiles)} profiles")
    
    return spar_profiles, compound


def _extrude_spar_from_profile(
    profile: SparProfile,
    sections: List[SpanwiseSection],
    spar_type: str,
    planform,
) -> Optional[TopoDS_Shape]:
    """
    Extrude a 2D spar profile into 3D.
    
    The spar is a flat plate. The 2D profile is in local coordinates:
    - X = spanwise position (along spar length, in mm, starting at 0)
    - Z = height (vertical in spar plane, in mm, normalized to start at 0)
    
    On a swept wing, the spar X position varies along the span following
    the sweep angle. Each point in the profile is positioned at the correct
    spar X based on its spanwise location.
    
    The spar profile is then extruded perpendicular to the sweep line
    (not purely chordwise) by the material thickness.
    
    IMPORTANT for BWB spars:
    - The profile's station_heights contain the LOCAL airfoil Z (without dihedral)
    - The profile's outline is normalized relative to the minimum z_lower in station_heights
    - We must use the sections' dihedral offsets (section.z_m) when placing in 3D
    """
    if profile.outline is None or len(profile.outline) < 3:
        return None
    
    # Profile is in mm, convert to meters
    outline_m = profile.outline / 1000.0
    thickness_m = planform.spar_thickness_mm / 1000.0
    
    if not sections:
        return None
    
    # Sort sections by Y for interpolation
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    
    # Filter sections to this profile's region (body vs wing for BWB)
    # Only apply BWB filtering if there are actual body_sections defined
    is_bwb = hasattr(planform, 'body_sections') and planform.body_sections and len(planform.body_sections) > 0
    
    if is_bwb:
        last_body_section = max(planform.body_sections, key=lambda bs: bs.y_pos)
        body_y_limit = last_body_section.y_pos
    
    if is_bwb and profile.spar_region == "body":
        # Body sections only - up to BWB junction
        filtered_sections = [s for s in sorted_sections if abs(s.y_m) <= body_y_limit + 0.01]
        if len(filtered_sections) < 2:
            filtered_sections = sorted_sections[:2]  # At minimum, use first two sections
    elif is_bwb and profile.spar_region == "wing":
        # Wing sections only - from BWB junction to tip
        filtered_sections = [s for s in sorted_sections if abs(s.y_m) >= body_y_limit - 0.01]
        if len(filtered_sections) < 2:
            filtered_sections = sorted_sections[-2:]  # At minimum, use last two sections
    else:
        # Standard wing (non-BWB) or unknown region: use all sections
        filtered_sections = sorted_sections
    
    # Get root section of this profile's region
    root_section = filtered_sections[0]
    root_y = abs(root_section.y_m)
    
    # Build arrays for spar position and Z interpolation
    # 
    # KEY FIX: Use profile.station_heights for Z values since they match the profile outline
    # But we still need section data for:
    # 1. Spar chordwise X position (to follow sweep)
    # 2. Dihedral Z offset (section.z_m)
    #
    spar_y_positions = []  # Absolute Y positions of sections
    spar_x_positions = []  # Chordwise X position of spar at each section
    section_z_offsets = []  # Dihedral Z offset at each section (section.z_m)
    
    for section in filtered_sections:
        spar_xsi = get_spar_xsi_at_section(section, planform, spar_type)
        spar_x = section.x_le_m + spar_xsi * section.chord_m
        
        spar_y_positions.append(abs(section.y_m))
        spar_x_positions.append(spar_x)
        section_z_offsets.append(section.z_m)  # Dihedral offset only
    
    # Convert to arrays for interpolation
    spar_y_arr = np.array(spar_y_positions)
    spar_x_arr = np.array(spar_x_positions)
    section_z_arr = np.array(section_z_offsets)
    
    # Build arrays for Z interpolation from profile's station_heights
    # These are the LOCAL airfoil Z values used to generate the profile
    if profile.station_heights:
        # Use profile's own station heights (already in correct local coords)
        station_z_lower_mm = np.array([s.z_lower_mm for s in profile.station_heights])
        
        # Compute the normalization offset used when building the profile
        # (this is the z_lower_min that was subtracted in _build_tapered_spar_outline)
        z_lower_min_mm = float(np.min(station_z_lower_mm))
    else:
        # Fallback: empty profile metadata - shouldn't happen
        print(f"[GeometryBuilder] Warning: No station_heights in spar profile")
        return None
    
    # Profile Z normalization: profile Z=0 corresponds to normalized spar lower edge
    profile_z_min = np.min(outline_m[:, 1])
    
    # The offset we need to add back to restore actual airfoil Z
    z_lower_offset_m = z_lower_min_mm / 1000.0
    
    # DEBUG: Log spar positioning info
    if is_bwb and profile.spar_region in ("body", "wing"):
        print(f"[GeometryBuilder] BWB {profile.spar_region} spar {spar_type}:")
        print(f"  root_y={root_y:.4f}m, profile_z_min={profile_z_min:.4f}m")
        print(f"  z_lower_min_mm={z_lower_min_mm:.2f}mm, z_lower_offset_m={z_lower_offset_m:.4f}m")
        print(f"  first section z_offset (dihedral)={section_z_arr[0]:.4f}m")
    else:
        print(f"[GeometryBuilder] Standard wing spar {spar_type}:")
        print(f"  root_y={root_y:.4f}m, profile_z_min={profile_z_min:.4f}m")
        print(f"  z_lower_min_mm={z_lower_min_mm:.2f}mm, z_lower_offset_m={z_lower_offset_m:.4f}m")
    
    # Compute sweep direction for extrusion
    # Use sections at start and end of this profile's region
    if len(filtered_sections) >= 2:
        start_sec = filtered_sections[0]
        end_sec = filtered_sections[-1]
        
        # Get spar positions at start and end
        xsi_start = get_spar_xsi_at_section(start_sec, planform, spar_type)
        xsi_end = get_spar_xsi_at_section(end_sec, planform, spar_type)
        
        x_start = start_sec.x_le_m + xsi_start * start_sec.chord_m
        x_end = end_sec.x_le_m + xsi_end * end_sec.chord_m
        y_start = abs(start_sec.y_m)
        y_end = abs(end_sec.y_m)
        
        # Sweep angle: tan(sweep) = dx/dy
        dy = y_end - y_start
        dx = x_end - x_start
        
        if abs(dy) > 1e-6:
            sweep_rad = math.atan2(dx, dy)  # Note: atan2(dx, dy) gives angle from Y-axis
        else:
            sweep_rad = 0.0
    else:
        sweep_rad = 0.0
    
    # ==========================================================================
    # BUILD SPAR AS TRANSFORMED 2D PROFILE
    # ==========================================================================
    # 
    # The DXF profile's X dimension is the ACTUAL LENGTH along the swept spar
    # (computed with sweep projection in profiles.py). The Z values include
    # dihedral offset at each station. We:
    # 1. Build the face in LOCAL coordinates (YZ plane, X=0)
    # 2. Rotate by sweep angle (around Z axis) to orient along swept spar line
    # 3. Translate to spar root position
    # 4. Extrude perpendicular to face
    
    # ==========================================================================
    # BUILD SPAR SOLID
    # ==========================================================================
    # 
    # Strategy:
    # 1. Build the 2D profile face in the local YZ plane (X=0)
    # 2. Extrude along X by thickness to create a non-skewed solid
    # 3. Center the solid on the midplane
    # 4. Transform (rotate by sweep, translate to root) into 3D position
    # 5. Fix topology to ensure a valid manifold solid
    
    # Build wire in LOCAL coordinates (YZ plane, X=0)
    wire_points_local = []
    for pt in outline_m:
        x_profile, z_profile = pt[0], pt[1]
        local_x = 0.0
        local_y = x_profile
        local_z = (z_profile - profile_z_min) + z_lower_offset_m
        wire_points_local.append(np.array([local_x, local_y, local_z]))
    
    wire = make_wire_from_points(wire_points_local)
    if wire is None:
        print(f"[GeometryBuilder] Warning: Failed to create wire for {spar_type} spar")
        return None
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    if face.IsNull():
        print(f"[GeometryBuilder] Warning: Failed to create face for {spar_type} spar")
        return None
    
    # Extrude along X axis (thickness direction)
    # Front spar: extrude toward TE (+X); Rear spar: toward LE (-X)
    extrusion_mag = thickness_m if spar_type == "front" else -thickness_m
    spar_solid = extrude_prism(face, (extrusion_mag, 0.0, 0.0))
    
    # Center the solid on the face plane
    trsf_center = gp_Trsf()
    trsf_center.SetTranslation(gp_Vec(-0.5 * extrusion_mag, 0.0, 0.0))
    spar_solid = BRepBuilderAPI_Transform(spar_solid, trsf_center, True).Shape()
    
    # Apply sweep rotation and root translation
    trsf_sweep = gp_Trsf()
    trsf_sweep.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), -sweep_rad)
    
    root_spar_x = spar_x_arr[0]
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(root_spar_x, root_y, 0.0))
    
    trsf_combined = gp_Trsf()
    trsf_combined.Multiply(trsf_translate)
    trsf_combined.Multiply(trsf_sweep)
    
    spar_solid = BRepBuilderAPI_Transform(spar_solid, trsf_combined, True).Shape()
    
    # Clean up and ensure valid solid
    from OCC.Core.ShapeFix import ShapeFix_Shape
    try:
        fixer = ShapeFix_Shape(spar_solid)
        fixer.Perform()
        spar_solid = fixer.Shape()
    except Exception:
        pass
        
    return spar_solid


# ==============================================================================
# Rib Geometry Generation
# ==============================================================================

def build_rib_geometry(
    project: Project,
    sections: List[SpanwiseSection],
    front_sweep_angles: List[float],
    rear_sweep_angles: List[float],
    dihedral_rad: float,
    config: WingGeometryConfig,
) -> Tuple[List[TopoDS_Shape], List[TopoDS_Shape], List[RibProfile], List[ElevonRibProfile]]:
    """
    Build 3D rib solids by extruding 2D profiles from profiles.py.
    
    Profiles include:
    - Spar notches (for spar-rib interlocking) - built into outline
    - Stringer slots (for stringers to pass through) - cut in second pass
    - Lightening holes (if enabled) - cut in second pass
    
    Returns:
        Tuple of (main_ribs, elevon_ribs, rib_profiles, elevon_profiles)
    """
    plan = project.wing.planform
    
    main_ribs = []
    elevon_ribs = []
    rib_profiles = []
    elevon_profiles = []
    
    for i, section in enumerate(sections):
        # Build parameters with sweep correction for this section
        params = RibProfileParams(
            include_spar_notches=config.include_spar_notches,
            include_stringer_slots=config.include_stringer_slots and plan.stringer_count > 0,
            include_lightening_holes=config.include_lightening_holes,
            lightening_hole_shape=config.lightening_hole_shape,
            include_elevon_cutout=True,
            spar_notch_clearance_mm=config.spar_notch_clearance_mm,
            stringer_slot_clearance_mm=config.stringer_slot_clearance_mm,
            spar_notch_depth_percent=50.0,
            front_spar_sweep_rad=front_sweep_angles[i] if i < len(front_sweep_angles) else 0.0,
            rear_spar_sweep_rad=rear_sweep_angles[i] if i < len(rear_sweep_angles) else 0.0,
            dihedral_rad=dihedral_rad,
            apply_sweep_correction=config.apply_sweep_correction,
        )
        
        try:
            # Generate 2D profile (same as DXF)
            profile = generate_rib_profile(section, project, params)
            rib_profiles.append(profile)
            
            # Extrude into 3D
            rib_solid = _extrude_rib_profile(profile, section, plan)
            
            # Second pass: cut lightening holes and stringer slots
            if rib_solid and not rib_solid.IsNull():
                rib_solid = _cut_rib_features(rib_solid, profile, section, plan)
            
            if rib_solid and not rib_solid.IsNull():
                main_ribs.append(rib_solid)
            
            # Generate elevon rib if this section has control surface cutout
            if profile.elevon_cutout and profile.elevon_cutout.has_cutout:
                elevon_profile = generate_elevon_rib_profile(
                    section, project, profile.elevon_cutout,
                    max_deflection_deg=params.elevon_max_deflection_deg,
                    hinge_gap_mm=params.elevon_hinge_gap_mm,
                )
                
                if elevon_profile:
                    elevon_profiles.append(elevon_profile)
                    elevon_solid = _extrude_elevon_rib_profile(elevon_profile, section, plan)
                    if elevon_solid and not elevon_solid.IsNull():
                        elevon_ribs.append(elevon_solid)
                        
        except Exception as e:
            print(f"[GeometryBuilder] Rib {i} generation failed: {e}")
            continue
    
    return main_ribs, elevon_ribs, rib_profiles, elevon_profiles


def _cut_rib_features(
    rib_solid: TopoDS_Shape,
    profile: RibProfile,
    section: SpanwiseSection,
    planform,
) -> TopoDS_Shape:
    """
    Cut lightening holes and stringer slots from the extruded rib solid.
    
    This is the second pass after extrusion. The outline already has spar notches
    built in, but lightening holes and stringer slots are internal features that
    must be cut using boolean operations.
    """
    thickness_m = planform.rib_thickness_mm / 1000.0
    
    # Cut lightening holes
    if profile.lightening_holes:
        hole_geoms = get_lightening_hole_geometries(profile.lightening_holes)
        for geom_type, geom_data in hole_geoms:
            try:
                cutter = _create_poly_cutter(geom_type, geom_data, section, thickness_m)
                if cutter and not cutter.IsNull():
                    result = bool_cut(rib_solid, cutter)
                    if result and not result.IsNull():
                        rib_solid = result
            except Exception as e:
                print(f"[GeometryBuilder] Lightening hole cut failed: {e}")
    
    # Cut stringer slots
    if profile.stringer_slots:
        slot_polylines = get_stringer_slot_polylines(profile.stringer_slots)
        for poly in slot_polylines:
            try:
                cutter = _create_poly_cutter("polyline", poly, section, thickness_m)
                if cutter and not cutter.IsNull():
                    result = bool_cut(rib_solid, cutter)
                    if result and not result.IsNull():
                        rib_solid = result
            except Exception as e:
                print(f"[GeometryBuilder] Stringer slot cut failed: {e}")
    
    return rib_solid


def _create_poly_cutter(
    geom_type: str,
    geom_data: Any,
    section: SpanwiseSection,
    thickness_m: float,
) -> Optional[TopoDS_Shape]:
    """Create a cutter from 2D profile geometry (mm) positioned in 3D."""
    pts_3d = []
    
    if geom_type == "circle":
        cx_mm, cz_mm, r_mm = geom_data
        # Create circular approximation
        n_pts = 16
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for a in angles:
            x_local = (cx_mm + r_mm * np.cos(a)) / 1000.0
            z_local = (cz_mm + r_mm * np.sin(a)) / 1000.0
            pts_3d.append(np.array([
                section.x_le_m + x_local,
                section.y_m,
                section.z_m + z_local,
            ]))
    elif geom_type == "polyline":
        # geom_data is Nx2 array of (x, z) points in mm
        for x_mm, z_mm in geom_data:
            pts_3d.append(np.array([
                section.x_le_m + x_mm / 1000.0,
                section.y_m,
                section.z_m + z_mm / 1000.0,
            ]))
    
    if not pts_3d:
        return None
        
    wire = make_wire_from_points(pts_3d)
    if wire is None:
        return None
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    if face.IsNull():
        return None
    
    # Extrude through rib thickness (with margin)
    extrusion_m = thickness_m * 2.0
    try:
        cutter = extrude_prism(face, (0.0, extrusion_m, 0.0))
        # Center the cutter on the rib
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(0, -extrusion_m / 2.0, 0))
        cutter = BRepBuilderAPI_Transform(cutter, trsf, True).Shape()
        return cutter
    except Exception:
        return None



def _extrude_rib_profile(
    profile: RibProfile,
    section: SpanwiseSection,
    planform,
) -> Optional[TopoDS_Shape]:
    """
    Extrude a 2D rib profile into 3D at the section's spanwise location.
    
    Ribs are VERTICAL (XZ plane) as per Phase 2 specification.
    """
    if profile.outline is None or len(profile.outline) < 3:
        return None
    
    # Convert mm to meters
    outline_m = profile.outline / 1000.0
    thickness_m = planform.rib_thickness_mm / 1000.0
    
    # Build 3D points in vertical XZ plane at section Y position
    wire_points = []
    for pt in outline_m:
        x_local, z_local = pt[0], pt[1]
        # Position at section location
        wire_points.append(np.array([
            section.x_le_m + x_local,  # Chordwise from LE
            section.y_m,  # Spanwise position (constant for vertical rib)
            z_local + section.z_m,  # Height with dihedral offset at LE
        ]))
    
    wire = make_wire_from_points(wire_points)
    if wire is None:
        return None
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    if face.IsNull():
        return None
    
    # Extrude along Y (spanwise) by FULL rib thickness
    extrusion_dir = (0.0, thickness_m, 0.0)
    
    try:
        rib_solid = extrude_prism(face, extrusion_dir)
        
        # Center the rib on section.y_m by translating back half thickness
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(0, -thickness_m / 2.0, 0))
        rib_solid = BRepBuilderAPI_Transform(rib_solid, trsf, True).Shape()
        
        # Fix solid topology
        from OCC.Core.ShapeFix import ShapeFix_Shape
        fixer = ShapeFix_Shape(rib_solid)
        fixer.Perform()
        return fixer.Shape()
    except Exception as e:
        print(f"[GeometryBuilder] Rib extrusion failed: {e}")
        return None


def _extrude_elevon_rib_profile(
    profile: ElevonRibProfile,
    section: SpanwiseSection,
    planform,
) -> Optional[TopoDS_Shape]:
    """
    Extrude an elevon rib profile into 3D.
    Similar to main rib but positioned at hinge location.
    """
    if profile.outline is None or len(profile.outline) < 3:
        return None
    
    # Convert mm to meters
    outline_m = profile.outline / 1000.0
    thickness_m = planform.rib_thickness_mm / 1000.0
    
    # Build 3D points
    wire_points = []
    for pt in outline_m:
        x_local, z_local = pt[0], pt[1]
        wire_points.append(np.array([
            section.x_le_m + x_local,
            section.y_m,
            z_local + section.z_m,
        ]))
    
    wire = make_wire_from_points(wire_points)
    if wire is None:
        return None
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    if face.IsNull():
        return None
    
    extrusion_dir = (0.0, thickness_m, 0.0)  # Full thickness
    
    try:
        solid = extrude_prism(face, extrusion_dir)
        
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(0, -thickness_m / 2.0, 0))  # Center on section Y
        solid = BRepBuilderAPI_Transform(solid, trsf, True).Shape()
        
        # Fix solid topology
        from OCC.Core.ShapeFix import ShapeFix_Shape
        fixer = ShapeFix_Shape(solid)
        fixer.Perform()
        return fixer.Shape()
    except Exception as e:
        print(f"[GeometryBuilder] Elevon rib extrusion failed: {e}")
        return None


# ==============================================================================
# Stringer Geometry Generation
# ==============================================================================

def build_stringer_geometry(
    sections: List[SpanwiseSection],
    planform,
) -> List[TopoDS_Shape]:
    """
    Build stringer geometry by lofting cross-sections along the wing span.
    
    Stringers:
    - Run spanwise (parallel to spars)
    - Attach to inner surface of skin (extend inward)
    - Evenly distributed across wingbox width between spars
    - Follow the wing surface contour at each spanwise station
    - Pass through ribs (ribs have matching slots)
    """
    if not hasattr(planform, 'stringer_count') or planform.stringer_count <= 0:
        return []
    
    stringers = []
    
    # Stringer cross-section (rectangular)
    height_m = getattr(planform, 'stringer_height_mm', 10.0) / 1000.0
    thickness_m = getattr(planform, 'stringer_thickness_mm', 1.5) / 1000.0
    
    if len(sections) < 2:
        return []
    
    # Stringer chordwise positions (evenly distributed between spars)
    n_stringers = planform.stringer_count
    
    for stringer_idx in range(n_stringers):
        xsi_fraction = (stringer_idx + 1) / (n_stringers + 1)
        
        # Build stringer for both upper and lower surfaces
        for surface in ['upper', 'lower']:
            stringer = _build_lofted_stringer(
                sections, planform, xsi_fraction, surface,
                height_m, thickness_m
            )
            if stringer and not stringer.IsNull():
                stringers.append(stringer)
    
    return stringers


def _build_lofted_stringer(
    sections: List[SpanwiseSection],
    planform,
    xsi_fraction: float,
    surface: str,
    height_m: float,
    thickness_m: float,
) -> Optional[TopoDS_Shape]:
    """
    Build a single stringer by lofting segment-by-segment between adjacent stations,
    then fusing all segments into a single solid body.
    
    The stringer follows the wing surface contour, accounting for:
    - Chord variation (taper)
    - Sweep (spar position varies with span)
    - Dihedral (z position varies with span)
    - Airfoil shape (surface z varies with chord position)
    
    Uses segment-by-segment lofting for robustness, then fuses into one body.
    """
    if len(sections) < 2:
        return None
    
    half_t = thickness_m / 2.0
    
    # Collect segment solids for fusing
    segments = []
    
    for i in range(len(sections) - 1):
        section_a = sections[i]
        section_b = sections[i + 1]
        
        # Get cross-section wire at station A
        wire_a = _get_stringer_wire_at_section(
            section_a, planform, xsi_fraction, surface, height_m, half_t
        )
        # Get cross-section wire at station B
        wire_b = _get_stringer_wire_at_section(
            section_b, planform, xsi_fraction, surface, height_m, half_t
        )
        
        if wire_a is None or wire_b is None:
            continue
        
        # Loft just these two profiles
        try:
            segment_solid = loft_solid_from_wires([wire_a, wire_b])
            if segment_solid and not segment_solid.IsNull():
                segments.append(segment_solid)
        except Exception as e:
            print(f"[GeometryBuilder] Stringer segment {i} loft failed: {e}")
            continue
    
    if len(segments) == 0:
        return None
    
    # Fuse all segments into a single body
    if len(segments) == 1:
        return segments[0]
    
    # Progressively fuse segments
    fused = segments[0]
    for seg in segments[1:]:
        try:
            fused = bool_fuse(fused, seg)
        except Exception as e:
            print(f"[GeometryBuilder] Stringer fuse failed: {e}")
            # Continue with what we have
    
    return fused


def _get_stringer_wire_at_section(
    section: SpanwiseSection,
    planform,
    xsi_fraction: float,
    surface: str,
    height_m: float,
    half_t: float,
) -> Optional[TopoDS_Wire]:
    """
    Create a rectangular wire for stringer cross-section at a single section.
    """
    # Calculate spar positions at this section (accounts for piece-wise linear paths)
    front_xsi = get_spar_xsi_at_section(section, planform, "front")
    rear_xsi = get_spar_xsi_at_section(section, planform, "rear")
    
    # Stringer position is fraction between front and rear spar
    stringer_xsi = front_xsi + xsi_fraction * (rear_xsi - front_xsi)
    stringer_x = section.x_le_m + stringer_xsi * section.chord_m
    
    # Get skin Z at this position
    z_surface = _get_airfoil_z_at_xsi(section, stringer_xsi, surface)
    if z_surface is None:
        return None
    
    # Add section Z offset (dihedral)
    z_surface += section.z_m
    y = section.y_m
    
    # Create rectangular cross-section at this station
    if surface == 'upper':
        # Extend inward (downward) from upper surface
        rect_pts = [
            np.array([stringer_x - half_t, y, z_surface]),
            np.array([stringer_x + half_t, y, z_surface]),
            np.array([stringer_x + half_t, y, z_surface - height_m]),
            np.array([stringer_x - half_t, y, z_surface - height_m]),
        ]
    else:
        # Extend inward (upward) from lower surface
        rect_pts = [
            np.array([stringer_x - half_t, y, z_surface]),
            np.array([stringer_x + half_t, y, z_surface]),
            np.array([stringer_x + half_t, y, z_surface + height_m]),
            np.array([stringer_x - half_t, y, z_surface + height_m]),
        ]
    
    return make_wire_from_points(rect_pts)


def _get_airfoil_z_at_xsi(
    section: SpanwiseSection,
    xsi: float,
    surface: str = 'upper',
) -> Optional[float]:
    """Get Z coordinate on airfoil surface at normalized chord position.
    
    Accounts for twist by rotating the airfoil coordinates before interpolation.
    """
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None
    
    coords = np.array(section.airfoil.coordinates)
    chord = section.chord_m
    
    # Scale to physical
    x = coords[:, 0] * chord
    z = coords[:, 1] * chord
    
    # Apply twist rotation about the leading edge
    # This ensures stringer attachment points follow the twisted wing surface
    if abs(section.twist_deg) > 1e-6:
        import math
        twist_rad = math.radians(section.twist_deg)
        cos_t = math.cos(twist_rad)
        sin_t = math.sin(twist_rad)
        
        # Pivot at leading edge
        le_idx = np.argmin(x)
        le_x = x[le_idx]
        le_z = z[le_idx]
        
        # Rotate about LE
        dx = x - le_x
        dz = z - le_z
        x = le_x + dx * cos_t + dz * sin_t
        z = le_z - dx * sin_t + dz * cos_t
    
    target_x = xsi * chord
    
    # Find LE index (may have shifted after twist)
    le_idx = np.argmin(x)
    
    if surface == 'upper':
        # Upper surface: indices 0 to le_idx
        x_surf = x[:le_idx + 1]
        z_surf = z[:le_idx + 1]
    else:
        # Lower surface: indices le_idx to end
        x_surf = x[le_idx:]
        z_surf = z[le_idx:]
    
    # Interpolate
    if len(x_surf) < 2:
        return None
    
    # Ensure monotonic for interpolation
    if x_surf[0] > x_surf[-1]:
        x_surf = x_surf[::-1]
        z_surf = z_surf[::-1]
    
    return float(np.interp(target_x, x_surf, z_surf))


# ==============================================================================
# Wing Box Skin Generation
# ==============================================================================

def build_wingbox_skin(
    sections: List[SpanwiseSection],
    planform,
) -> Tuple[Optional[TopoDS_Shape], Optional[TopoDS_Shape]]:
    """
    Build wing box skin panels (between front and rear spars only).
    
    Unlike full skin surfaces, these are trimmed to the spar region
    and represent the structural skin panels.
    
    Lofts segment-by-segment between adjacent sections for robustness.
    """
    if len(sections) < 2:
        return None, None
    
    upper_sewing = BRepBuilderAPI_Sewing(1e-6)
    lower_sewing = BRepBuilderAPI_Sewing(1e-6)
    
    for i in range(len(sections) - 1):
        section_a = sections[i]
        section_b = sections[i + 1]
        
        # Get spar positions at each section
        front_xsi_a = get_spar_xsi_at_section(section_a, planform, "front")
        rear_xsi_a = get_spar_xsi_at_section(section_a, planform, "rear")
        
        front_xsi_b = get_spar_xsi_at_section(section_b, planform, "front")
        rear_xsi_b = get_spar_xsi_at_section(section_b, planform, "rear")

        
        # Extract airfoil segments between spars for each section
        upper_a, lower_a = _extract_airfoil_segment(section_a, front_xsi_a, rear_xsi_a)
        upper_b, lower_b = _extract_airfoil_segment(section_b, front_xsi_b, rear_xsi_b)
        
        if upper_a is None or upper_b is None:
            continue
        
        # Loft just these two profiles
        upper_segment = loft_surface_from_profiles([upper_a, upper_b], which='upper')
        lower_segment = loft_surface_from_profiles([lower_a, lower_b], which='lower')
        
        if upper_segment and not upper_segment.IsNull():
            upper_sewing.Add(upper_segment)
        if lower_segment and not lower_segment.IsNull():
            lower_sewing.Add(lower_segment)
    
    # Sew segments into continuous shells
    upper_sewing.Perform()
    lower_sewing.Perform()
    
    upper_result = upper_sewing.SewedShape()
    lower_result = lower_sewing.SewedShape()
    
    return upper_result, lower_result


def _extract_airfoil_segment(
    section: SpanwiseSection,
    front_xsi: float,
    rear_xsi: float,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Extract airfoil segment between front and rear spar positions.
    
    Applies twist rotation to match the twisted wing surface.
    """
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None, None
    
    coords = np.array(section.airfoil.coordinates)
    chord = section.chord_m
    
    # Scale to physical coordinates
    x_norm = coords[:, 0]
    x_phys = x_norm * chord + section.x_le_m
    z_phys = coords[:, 1] * chord + section.z_m
    y_phys = np.full_like(x_phys, section.y_m)
    
    points_3d = np.column_stack([x_phys, y_phys, z_phys])
    
    # Apply twist rotation about the leading edge
    if abs(section.twist_deg) > 1e-6:
        points_list = [pt for pt in points_3d]
        points_list = apply_twist_to_points(
            points_list,
            section.twist_deg,
            pivot_x=section.x_le_m,
            pivot_z=section.z_m,
        )
        points_3d = np.array(points_list)
    
    # Split into upper/lower
    upper, lower = split_airfoil_upper_lower(points_3d)
    
    # Filter to points between front and rear spar
    front_x = section.x_le_m + front_xsi * chord
    rear_x = section.x_le_m + rear_xsi * chord
    
    def filter_segment(pts):
        filtered = []
        for pt in pts:
            if front_x <= pt[0] <= rear_x:
                filtered.append(pt)
        return filtered if len(filtered) >= 2 else None
    
    upper_seg = filter_segment(upper)
    lower_seg = filter_segment(lower)
    
    return upper_seg, lower_seg


# ==============================================================================
# Control Surface Geometry Generation
# ==============================================================================

def build_control_surface_geometry(
    sections: List[SpanwiseSection],
    planform,
    project: Project = None,
) -> Dict[str, TopoDS_Shape]:
    """
    Build control surface geometry (movable surfaces only).
    
    Each control surface is a separate solid, built by lofting between
    elevon rib profiles. This ensures twist is properly applied.
    
    Does NOT include hinge bracket geometry.
    """
    control_surfaces = {}
    
    if not hasattr(planform, 'control_surfaces') or not planform.control_surfaces:
        return control_surfaces
    
    for cs in planform.control_surfaces:
        cs_solid = _build_single_control_surface(sections, cs, project)
        if cs_solid and not cs_solid.IsNull():
            control_surfaces[cs.name] = cs_solid
    
    return control_surfaces


def _build_single_control_surface(
    sections: List[SpanwiseSection],
    cs,  # ControlSurface dataclass
    project: Project = None,
) -> Optional[TopoDS_Shape]:
    """Build a single control surface solid using sectional lofts.
    
    Uses the aft portion of the airfoil from hinge line to trailing edge.
    Lofts segment-by-segment between adjacent sections for robustness.
    Twist is applied via _extract_cs_airfoil_segment().
    """
    # Get span extent
    eta_start = cs.span_start_percent / 100.0
    eta_end = cs.span_end_percent / 100.0
    
    # Get chord extent (from hinge line to trailing edge)
    hinge_xsi_start = cs.chord_start_percent / 100.0
    hinge_xsi_end = cs.chord_end_percent / 100.0
    
    # Filter sections within control surface span
    cs_sections = [s for s in sections
                   if eta_start <= s.span_fraction <= eta_end]
    
    if len(cs_sections) < 2:
        return None
    
    # Sort by span position
    cs_sections = sorted(cs_sections, key=lambda s: s.y_m)
    
    # Build closed airfoil wires at each section (aft portion only)
    wires = []
    for section in cs_sections:
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        
        # Interpolate hinge position for this section
        if eta_end != eta_start:
            local_eta = (section.span_fraction - eta_start) / (eta_end - eta_start)
        else:
            local_eta = 0.0
        hinge_xsi = hinge_xsi_start + local_eta * (hinge_xsi_end - hinge_xsi_start)
        
        print(f"[CS Wire] Section {section.index}: span_frac={section.span_fraction:.3f}, hinge_xsi={hinge_xsi:.3f}, chord={section.chord_m:.3f}m")
        
        # Get the aft airfoil portion as a closed profile
        wire = _get_cs_wire_at_section(section, hinge_xsi)
        if wire is not None:
            wires.append(wire)
    
    if len(wires) < 2:
        print(f"[GeometryBuilder] Control surface '{cs.name}': Only {len(wires)} valid wires, need 2+")
        return None
    
    # Sectional loft between adjacent wires, then fuse
    segments = []
    for i in range(len(wires) - 1):
        try:
            segment = loft_solid_from_wires([wires[i], wires[i + 1]])
            if segment and not segment.IsNull():
                segments.append(segment)
            else:
                print(f"[GeometryBuilder] Control surface segment {i}->{i+1} loft returned null")
        except Exception as e:
            print(f"[GeometryBuilder] Control surface segment {i}->{i+1} loft failed: {e}")
            continue
    
    if not segments:
        print(f"[GeometryBuilder] Control surface '{cs.name}': No valid segments created")
        return None
    
    # Fuse all segments
    if len(segments) == 1:
        return segments[0]
    
    result = segments[0]
    for seg in segments[1:]:
        fused = bool_fuse(result, seg)
        if fused and not fused.IsNull():
            result = fused
        else:
            result = make_compound([result, seg])
    
    return result


def _get_cs_wire_at_section(
    section: SpanwiseSection,
    hinge_xsi: float,
) -> Optional[TopoDS_Wire]:
    """
    Create a closed wire for the control surface airfoil at a section.
    
    Extracts the aft portion of the airfoil (from hinge to TE) and creates
    a closed wire suitable for lofting. Twist is applied.
    
    Uses spline interpolation for smooth geometry that lofts reliably.
    """
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None
    
    coords = np.array(section.airfoil.coordinates)
    chord = section.chord_m
    
    # Scale to physical coordinates
    x_phys = coords[:, 0] * chord + section.x_le_m
    z_phys = coords[:, 1] * chord + section.z_m
    y_phys = np.full_like(x_phys, section.y_m)
    
    # Apply twist
    twist_rad = math.radians(section.twist_deg) if abs(section.twist_deg) > 1e-6 else 0.0
    if twist_rad != 0:
        cos_t = math.cos(twist_rad)
        sin_t = math.sin(twist_rad)
        
        pivot_x = section.x_le_m
        pivot_z = section.z_m
        
        dx = x_phys - pivot_x
        dz = z_phys - pivot_z
        x_phys = pivot_x + dx * cos_t + dz * sin_t
        z_phys = pivot_z - dx * sin_t + dz * cos_t
    
    # Find hinge X position in twisted world coordinates
    # The hinge is at a fixed chord fraction, so compute its twisted position
    if twist_rad != 0:
        hinge_dx = hinge_xsi * chord
        hinge_x = section.x_le_m + hinge_dx * math.cos(twist_rad)
    else:
        hinge_x = section.x_le_m + hinge_xsi * chord
    
    # Split into upper and lower surfaces
    le_idx = np.argmin(coords[:, 0])
    
    # Upper surface: from TE (index 0) to LE
    x_upper = x_phys[:le_idx + 1]
    z_upper = z_phys[:le_idx + 1]
    y_upper = y_phys[:le_idx + 1]
    
    # Lower surface: from LE to TE
    x_lower = x_phys[le_idx:]
    z_lower = z_phys[le_idx:]
    y_lower = y_phys[le_idx:]
    
    # Filter to aft portion (x >= hinge_x)
    upper_mask = x_upper >= hinge_x - 1e-6
    lower_mask = x_lower >= hinge_x - 1e-6
    
    x_upper_aft = x_upper[upper_mask]
    z_upper_aft = z_upper[upper_mask]
    y_upper_aft = y_upper[upper_mask]
    
    x_lower_aft = x_lower[lower_mask]
    z_lower_aft = z_lower[lower_mask]
    y_lower_aft = y_lower[lower_mask]
    
    if len(x_upper_aft) < 2 or len(x_lower_aft) < 2:
        print(f"[CS Wire] Section {section.index}: Not enough aft points (upper={len(x_upper_aft)}, lower={len(x_lower_aft)})")
        return None
    
    # Build 3D point arrays for upper and lower aft surfaces
    # Upper: goes from TE toward hinge (we want it ordered hinge->TE for consistent winding)
    upper_pts = []
    for x, y, z in zip(reversed(x_upper_aft), reversed(y_upper_aft), reversed(z_upper_aft)):
        upper_pts.append(np.array([x, y, z]))
    
    # Lower: goes from LE toward TE (filtered to aft portion = hinge->TE)
    lower_pts = []
    for x, y, z in zip(x_lower_aft, y_lower_aft, z_lower_aft):
        lower_pts.append(np.array([x, y, z]))
    
    if len(upper_pts) < 2 or len(lower_pts) < 2:
        return None
    
    # Try spline wire with separate upper/lower surfaces (best for lofting)
    wire = make_partial_airfoil_wire_spline(upper_pts, lower_pts)
    if wire is not None:
        return wire
    
    # Fallback: combine into a closed profile and try other methods
    # Upper (hinge->TE) + lower reversed (TE->hinge)
    profile_pts = upper_pts + list(reversed(lower_pts))
    
    if len(profile_pts) < 4:
        print(f"[CS Wire] Section {section.index}: Too few profile points ({len(profile_pts)})")
        return None
    
    # Try closed BSpline
    wire = make_closed_bspline_wire_from_points(profile_pts)
    if wire is not None:
        return wire
    
    # Final fallback to polyline wire
    print(f"[CS Wire] Section {section.index}: All spline methods failed, trying polyline")
    return make_wire_from_points(profile_pts)


def _extract_cs_airfoil_segment(
    section: SpanwiseSection,
    start_xsi: float,
    end_xsi: float,
) -> Optional[List[np.ndarray]]:
    """Extract airfoil segment from start to end chord positions.
    
    Applies twist rotation to match the twisted wing surface.
    """
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None
    
    coords = np.array(section.airfoil.coordinates)
    chord = section.chord_m
    
    x_norm = coords[:, 0]
    
    # Find indices within range
    mask = (x_norm >= start_xsi) & (x_norm <= end_xsi)
    
    if not np.any(mask):
        return None
    
    # Scale to physical coordinates
    x_phys = coords[mask, 0] * chord + section.x_le_m
    z_phys = coords[mask, 1] * chord + section.z_m
    y_phys = np.full_like(x_phys, section.y_m)
    
    points = [np.array([x, y, z]) for x, y, z in zip(x_phys, y_phys, z_phys)]
    
    # Apply twist rotation about the leading edge
    if abs(section.twist_deg) > 1e-6:
        points = apply_twist_to_points(
            points,
            section.twist_deg,
            pivot_x=section.x_le_m,
            pivot_z=section.z_m,
        )
    
    return points


# ==============================================================================
# Fixture Geometry Generation
# ==============================================================================

def build_fixture_geometry(
    project: Project,
    sections: List[SpanwiseSection],
    wing_geometry: GeneratedGeometry,
    config: WingGeometryConfig,
) -> Tuple[List[TopoDS_Shape], List[TopoDS_Shape], Optional[TopoDS_Shape],
           List, List, Optional[TopoDS_Compound]]:
    """
    Build fixture solids using the profile-extrusion methodology.
    
    Fixtures are generated for each spar (front and rear), with optional cradles.
    The wing structure (spars + ribs) is cut from the fixtures to create indexing.
    
    Returns:
        Tuple of (fixtures, cradles, base_plate, fixture_profiles, cradle_profiles, assembly)
    """
    from services.export.profiles import (
        FixtureProfileParams,
        generate_fixture_profile,
        generate_cradle_profile,
    )
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    
    plan = project.wing.planform
    
    params = FixtureProfileParams(
        material_thickness_mm=config.fixture_material_thickness_mm,
        slot_clearance_mm=config.fixture_slot_clearance_mm,
        fixture_height_mm=config.fixture_height_mm,
        tab_width_mm=config.fixture_tab_width_mm,
        tab_spacing_mm=config.fixture_tab_spacing_mm,
    )
    
    fixtures = []
    cradles = []
    fixture_profiles = []
    cradle_profiles = []
    
    # For BWB projects, split sections into body and wing regions
    # This generates separate fixtures for each region
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    
    section_groups = []
    if plan.body_sections:
        # Find the last body section position
        last_body_section = max(plan.body_sections, key=lambda bs: bs.y_pos)
        body_y_limit = last_body_section.y_pos
        
        # Split sections into body and wing groups
        body_sections = [s for s in sorted_sections if abs(s.y_m) <= body_y_limit + 1e-6]
        wing_sections = [s for s in sorted_sections if abs(s.y_m) >= body_y_limit - 1e-6]
        
        if len(body_sections) >= 2:
            section_groups.append(("body", body_sections))
        if len(wing_sections) >= 2:
            section_groups.append(("wing", wing_sections))
        
        print(f"[GeometryBuilder] BWB fixture split: body={len(body_sections)} sections, wing={len(wing_sections)} sections")
    else:
        # Non-BWB: single group with all sections
        section_groups.append(("wing", sorted_sections))
    
    # Generate fixtures for each region
    for region_name, region_sections in section_groups:
        for spar_type in ["front", "rear"]:
            for fixture_side in ["front", "rear"]:
                try:
                    profile = generate_fixture_profile(
                        project, region_sections, spar_type, fixture_side, params
                    )
                    fixture_profiles.append(profile)
                    
                    solid = _extrude_fixture_from_profile(
                        profile, region_sections, spar_type, plan, params
                    )
                    if solid and not solid.IsNull():
                        fixtures.append(solid)
                        print(f"[GeometryBuilder] Fixture {region_name}/{spar_type}/{fixture_side}: OK")
                    else:
                        print(f"[GeometryBuilder] Fixture {region_name}/{spar_type}/{fixture_side}: failed to extrude")
                except Exception as e:
                    print(f"[GeometryBuilder] Fixture {region_name}/{spar_type}/{fixture_side} failed: {e}")
            
            # Generate cradle for this spar in this region
            if config.fixture_add_cradle:
                try:
                    cradle_profile = generate_cradle_profile(
                        project, region_sections, spar_type, params
                    )
                    cradle_profiles.append(cradle_profile)
                    
                    cradle_solid = _extrude_cradle_from_profile(
                        cradle_profile, region_sections, spar_type, plan, params
                    )
                    if cradle_solid and not cradle_solid.IsNull():
                        cradles.append(cradle_solid)
                        print(f"[GeometryBuilder] Cradle {region_name}/{spar_type}: OK")
                except Exception as e:
                    print(f"[GeometryBuilder] Cradle {region_name}/{spar_type} failed: {e}")
    
    # Cut wing structure from fixtures
    # Spars cut FULL DEPTH (spars pass through fixtures completely)
    # Ribs cut HALF DEPTH (for interlocking construction)
    
    spar_cutting_shapes = []
    if wing_geometry.wingbox_front_spar and not wing_geometry.wingbox_front_spar.IsNull():
        spar_cutting_shapes.append(wing_geometry.wingbox_front_spar)
    if wing_geometry.wingbox_rear_spar and not wing_geometry.wingbox_rear_spar.IsNull():
        spar_cutting_shapes.append(wing_geometry.wingbox_rear_spar)
    
    # Cut spars from fixtures (full depth)
    if spar_cutting_shapes:
        spar_cutter = make_compound(spar_cutting_shapes)
        cut_fixtures = []
        for fx in fixtures:
            try:
                cut_result = bool_cut(fx, spar_cutter)
                cut_fixtures.append(cut_result if cut_result and not cut_result.IsNull() else fx)
            except Exception:
                cut_fixtures.append(fx)
        fixtures = cut_fixtures
    
    # Cut ribs from fixtures (full depth for now)
    # TODO: Implement half-thickness cutting for interlocking construction
    # The half-lap joint approach requires more sophisticated geometry analysis
    # to determine the correct clipping direction based on fixture orientation.
    
    if wing_geometry.ribs:
        cut_fixtures = []
        for fx_idx, fx in enumerate(fixtures):
            if fx is None or fx.IsNull():
                cut_fixtures.append(fx)
                continue
            
            # Get this fixture's bounding box for quick overlap check
            fx_bbox = Bnd_Box()
            brepbndlib.Add(fx, fx_bbox)
            if fx_bbox.IsVoid():
                cut_fixtures.append(fx)
                continue
            
            current_fx = fx
            for rib in wing_geometry.ribs:
                if rib is None or rib.IsNull():
                    continue
                
                # Quick bounding box check
                rib_bbox = Bnd_Box()
                brepbndlib.Add(rib, rib_bbox)
                if fx_bbox.IsOut(rib_bbox):
                    continue
                
                # Cut the rib from the fixture (full depth)
                try:
                    cut_result = bool_cut(current_fx, rib)
                    if cut_result and not cut_result.IsNull():
                        current_fx = cut_result
                except Exception:
                    pass
            
            cut_fixtures.append(current_fx)
        fixtures = cut_fixtures
    
    # Build base plate with slots
    base_plate = _build_slotted_base_plate(fixtures, cradles, params)
    
    # Assemble final compound
    all_shapes = fixtures + cradles
    if base_plate and not base_plate.IsNull():
        all_shapes.append(base_plate)
    
    assembly = make_compound(all_shapes) if all_shapes else None
    
    return fixtures, cradles, base_plate, fixture_profiles, cradle_profiles, assembly


def _extrude_fixture_from_profile(
    profile,  # FixtureProfile
    sections: List[SpanwiseSection],
    spar_type: str,
    planform,
    params,
) -> Optional[TopoDS_Shape]:
    """
    Extrude a 2D fixture profile into 3D.
    
    The fixture is built using actual 3D spar positions at EACH station:
    1. Get spar positions at all sections (in world coordinates)
    2. Build fixture polygon using all station heights (follows actual airfoil surface)
    3. Extrude perpendicular to the spar plane by material thickness
    4. Position based on fixture_side (front or rear of spar)
    """
    if profile.outline is None or len(profile.outline) < 3:
        return None
    
    if not profile.station_heights or len(profile.station_heights) < 2:
        return None
    
    mat_thick_m = params.material_thickness_mm / 1000.0
    clearance_m = params.slot_clearance_mm / 1000.0
    spar_thick_m = planform.spar_thickness_mm / 1000.0
    spar_half_m = spar_thick_m / 2.0
    
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    if len(sorted_sections) < 2:
        return None
    
    # Build arrays of spar positions at each section
    spar_positions = []  # List of (x, y) tuples in world coords
    for section in sorted_sections:
        spar_xsi = get_spar_xsi_at_section(section, planform, spar_type)
        spar_x = section.x_le_m + spar_xsi * section.chord_m
        spar_y = abs(section.y_m)
        spar_positions.append((spar_x, spar_y))
    
    # Get base Z (same for all stations)
    base_z_m = profile.station_heights[0].z_base_mm / 1000.0
    
    # Build the spar direction vector (root to tip) for normal calculation
    root_spar_x, root_y = spar_positions[0]
    tip_spar_x, tip_y = spar_positions[-1]
    
    spar_dir = np.array([tip_spar_x - root_spar_x, tip_y - root_y, 0.0])
    spar_len = np.linalg.norm(spar_dir)
    if spar_len < 1e-9:
        return None
    spar_dir_unit = spar_dir / spar_len
    
    # Normal direction perpendicular to spar in XY plane
    perp1 = np.array([-spar_dir_unit[1], spar_dir_unit[0], 0.0])
    perp2 = np.array([spar_dir_unit[1], -spar_dir_unit[0], 0.0])
    
    # Choose the perpendicular that points more toward positive X (toward trailing edge)
    if perp1[0] > perp2[0]:
        normal_xy = perp1
    else:
        normal_xy = perp2
    
    # Build fixture polygon vertices using ALL station heights
    # The polygon goes: bottom edge (root to tip) then top edge (tip to root)
    polygon_points = []
    
    # Bottom edge: root to tip at base_z
    for (spar_x, spar_y) in spar_positions:
        polygon_points.append(np.array([spar_x, spar_y, base_z_m]))
    
    # Top edge: tip to root at z_chord for each station
    for i in range(len(profile.station_heights) - 1, -1, -1):
        station = profile.station_heights[i]
        spar_x, spar_y = spar_positions[i]
        z_chord_m = station.z_chord_mm / 1000.0
        polygon_points.append(np.array([spar_x, spar_y, z_chord_m]))
    
    # Build the fixture face from the polygon
    ext_wire = make_wire_from_points(polygon_points)
    if ext_wire is None:
        return None
    
    ext_face = BRepBuilderAPI_MakeFace(ext_wire).Face()
    if ext_face.IsNull():
        return None
    
    # Determine offset and extrusion direction based on fixture_side
    if profile.fixture_side == "front":
        # Front fixture: positioned at spar front face, extrudes toward LE (negative normal)
        face_offset = -(spar_half_m + clearance_m)
        extrude_dir = tuple(-normal_xy * mat_thick_m)
    else:
        # Rear fixture: positioned at spar back face, extrudes toward TE (positive normal)
        face_offset = spar_half_m + clearance_m
        extrude_dir = tuple(normal_xy * mat_thick_m)
    
    # Translate face to correct position (offset from spar centerline)
    offset_vec = normal_xy * face_offset
    trsf_offset = gp_Trsf()
    trsf_offset.SetTranslation(gp_Vec(offset_vec[0], offset_vec[1], 0.0))
    offset_face = BRepBuilderAPI_Transform(ext_face, trsf_offset, True).Shape()
    
    # Extrude the face by material thickness
    fixture_solid = extrude_prism(offset_face, extrude_dir)
    if fixture_solid is None or fixture_solid.IsNull():
        return None
    
    # Add alignment tabs at the bottom of the fixture
    # Tabs are rectangular protrusions that extend below the base plate for alignment
    # Each tab is built as an extruded rectangle following the fixture's orientation
    if profile.tabs:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from core.occ_utils.booleans import fuse as bool_fuse
        
        for tab in profile.tabs:
            # Tab position along the spar line
            tab_fraction = tab.x_along_span_mm / (profile.length_mm if profile.length_mm > 0 else 1.0)
            tab_idx = int(tab_fraction * (len(spar_positions) - 1))
            tab_idx = min(max(0, tab_idx), len(spar_positions) - 1)
            
            # Interpolate between adjacent positions for smoother placement
            if tab_idx < len(spar_positions) - 1:
                local_frac = (tab_fraction * (len(spar_positions) - 1)) - tab_idx
                spar_x = spar_positions[tab_idx][0] * (1 - local_frac) + spar_positions[tab_idx + 1][0] * local_frac
                spar_y = spar_positions[tab_idx][1] * (1 - local_frac) + spar_positions[tab_idx + 1][1] * local_frac
            else:
                spar_x, spar_y = spar_positions[tab_idx]
            
            # Tab dimensions
            tab_width_m = tab.width_mm / 1000.0
            tab_depth_m = tab.depth_mm / 1000.0
            half_width = tab_width_m / 2.0
            
            # Tab center position (at spar line, offset by face_offset)
            tab_base = np.array([spar_x, spar_y, base_z_m]) + offset_vec
            
            # Build tab as an extruded rectangle
            # Rectangle corners at the top of the tab (fixture bottom level)
            # Rectangle is oriented along spar direction and normal direction
            p1 = tab_base - spar_dir_unit * half_width
            p2 = tab_base + spar_dir_unit * half_width
            
            # Create the tab face polygon (rectangle at fixture bottom)
            tab_corners = [
                np.array([p1[0], p1[1], base_z_m]),
                np.array([p2[0], p2[1], base_z_m]),
                np.array([p2[0], p2[1], base_z_m - tab_depth_m]),
                np.array([p1[0], p1[1], base_z_m - tab_depth_m]),
            ]
            
            try:
                tab_wire = make_wire_from_points(tab_corners)
                if tab_wire is None:
                    continue
                    
                tab_face = BRepBuilderAPI_MakeFace(tab_wire).Face()
                if tab_face.IsNull():
                    continue
                
                # Extrude tab in the same direction as the fixture
                tab_solid = extrude_prism(tab_face, extrude_dir)
                if tab_solid and not tab_solid.IsNull():
                    fused = bool_fuse(fixture_solid, tab_solid)
                    if fused and not fused.IsNull():
                        fixture_solid = fused
            except Exception:
                pass  # Skip this tab if it fails
    
    return fixture_solid


def _extrude_cradle_from_profile(
    profile,  # CradleProfile
    sections: List[SpanwiseSection],
    spar_type: str,
    planform,
    params,
) -> Optional[TopoDS_Shape]:
    """
    Extrude a 2D cradle profile into 3D.
    
    The cradle follows the lower wing surface at EACH station and is centered on the spar.
    Uses all station heights for accurate surface following.
    """
    if not profile.station_heights or len(profile.station_heights) < 2:
        return None
    
    spar_thickness_m = planform.spar_thickness_mm / 1000.0
    
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    if len(sorted_sections) < 2:
        return None
    
    # Build arrays of spar positions at each section
    spar_positions = []  # List of (x, y) tuples in world coords
    for section in sorted_sections:
        spar_xsi = get_spar_xsi_at_section(section, planform, spar_type)
        spar_x = section.x_le_m + spar_xsi * section.chord_m
        spar_y = abs(section.y_m)
        spar_positions.append((spar_x, spar_y))
    
    # Get base Z (same for all stations)
    base_z_m = profile.station_heights[0].z_base_mm / 1000.0
    
    # Build the spar direction vector for normal calculation
    root_spar_x, root_y = spar_positions[0]
    tip_spar_x, tip_y = spar_positions[-1]
    
    spar_dir = np.array([tip_spar_x - root_spar_x, tip_y - root_y, 0.0])
    spar_len = np.linalg.norm(spar_dir)
    if spar_len < 1e-9:
        return None
    spar_dir_unit = spar_dir / spar_len
    
    # Normal direction perpendicular to spar in XY plane
    perp1 = np.array([-spar_dir_unit[1], spar_dir_unit[0], 0.0])
    perp2 = np.array([spar_dir_unit[1], -spar_dir_unit[0], 0.0])
    if perp1[0] > perp2[0]:
        normal_xy = perp1
    else:
        normal_xy = perp2
    
    # Build cradle polygon vertices using ALL station heights
    # The polygon goes: bottom edge (root to tip) then top edge (tip to root)
    polygon_points = []
    
    # Bottom edge: root to tip at base_z
    for (spar_x, spar_y) in spar_positions:
        polygon_points.append(np.array([spar_x, spar_y, base_z_m]))
    
    # Top edge: tip to root at z_lower_surface for each station
    for i in range(len(profile.station_heights) - 1, -1, -1):
        station = profile.station_heights[i]
        spar_x, spar_y = spar_positions[i]
        z_lower_m = station.z_lower_surface_mm / 1000.0
        polygon_points.append(np.array([spar_x, spar_y, z_lower_m]))
    
    # Build the cradle face from the polygon
    cradle_wire = make_wire_from_points(polygon_points)
    if cradle_wire is None:
        return None
    
    cradle_face = BRepBuilderAPI_MakeFace(cradle_wire).Face()
    if cradle_face.IsNull():
        return None
    
    # Offset to center on spar (start at -half thickness)
    offset_vec = -normal_xy * (spar_thickness_m / 2.0)
    trsf_offset = gp_Trsf()
    trsf_offset.SetTranslation(gp_Vec(offset_vec[0], offset_vec[1], 0.0))
    offset_face = BRepBuilderAPI_Transform(cradle_face, trsf_offset, True).Shape()
    
    # Extrude by spar thickness (cradle fills the spar slot)
    extrude_dir = tuple(normal_xy * spar_thickness_m)
    cradle_solid = extrude_prism(offset_face, extrude_dir)
    if cradle_solid is None or cradle_solid.IsNull():
        return None
    
    return cradle_solid


def _build_slotted_base_plate(
    fixtures: List[TopoDS_Shape],
    cradles: List[TopoDS_Shape],
    params,
) -> Optional[TopoDS_Shape]:
    """
    Build base plate with slots cut for fixtures and cradles.
    
    The fixtures include alignment tabs at the bottom that extend into the base plate.
    This function creates the base plate and cuts slots for both the fixture bodies
    and their alignment tabs.
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    
    all_shapes = fixtures + cradles
    if not all_shapes:
        return None
    
    # Compute overall bounding box (including tabs that extend below)
    overall_bbox = Bnd_Box()
    for shape in all_shapes:
        if shape and not shape.IsNull():
            brepbndlib.Add(shape, overall_bbox)
    
    if overall_bbox.IsVoid():
        return None
    
    xmin, ymin, zmin, xmax, ymax, zmax = overall_bbox.Get()
    margin = params.base_plate_margin_mm / 1000.0 if hasattr(params, 'base_plate_margin_mm') else 0.02
    mat_thick = params.material_thickness_mm / 1000.0
    
    # Base plate spans from bottom of tabs to just below fixture body
    # The tab depth extends below the fixture body, so zmin includes tabs
    # Base plate thickness = material_thickness
    base_z_top = zmin + mat_thick  # Top of base plate (where fixture body sits)
    base_z_bottom = zmin  # Bottom of base plate (bottom of tabs)
    
    p1 = gp_Pnt(xmin - margin, ymin - margin, base_z_bottom)
    p2 = gp_Pnt(xmax + margin, ymax + margin, base_z_top)
    
    base_plate = BRepPrimAPI_MakeBox(p1, p2).Shape()
    
    # Cut slots for fixtures and tabs passing through
    if base_plate and not base_plate.IsNull():
        fixtures_compound = make_compound(all_shapes)
        if fixtures_compound and not fixtures_compound.IsNull():
            try:
                # Cut fixtures (including tabs) from base plate to create slots
                cut_plate = bool_cut(base_plate, fixtures_compound)
                if cut_plate and not cut_plate.IsNull():
                    base_plate = cut_plate
            except Exception as e:
                print(f"[GeometryBuilder] Base plate slotting failed: {e}")
    
    return base_plate
