# services/export/profiles.py
"""
Shared 2D profile generation for DXF and STEP export.

All profiles are generated in local 2D coordinates (mm):
- Ribs: X = chordwise, Z = thickness direction (airfoil coords)
- Spars: X = spanwise, Z = height direction

3D positioning is handled by the caller (geometry_builder.py for STEP).

This is the SINGLE SOURCE OF TRUTH for manufacturing geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

from core.state import Project
from services.geometry import SpanwiseSection, get_spar_xsi_at_section


# ==============================================================================
# Twist Transformation for 2D Rib Profiles
# ==============================================================================

def apply_twist_to_2d_coords(
    x: np.ndarray,
    z: np.ndarray,
    twist_deg: float,
    pivot_x: float = 0.0,
    pivot_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply twist rotation to 2D rib profile coordinates.
    
    For a rib, twist is a rotation in the XZ plane (about the Y-axis in 3D).
    This represents how the airfoil shape appears when viewed along the span
    at a twisted section.
    
    Positive twist = nose up = trailing edge rotates down (negative Z).
    
    Args:
        x: Chordwise coordinates (mm)
        z: Thickness coordinates (mm)
        twist_deg: Twist angle in degrees (absolute, from section.twist_deg)
        pivot_x: X coordinate of rotation pivot (default 0 = leading edge)
        pivot_z: Z coordinate of rotation pivot (default 0)
    
    Returns:
        Tuple of (x_rotated, z_rotated) arrays
    """
    if abs(twist_deg) < 1e-6:
        return x, z  # No rotation needed
    
    twist_rad = math.radians(twist_deg)
    cos_t = math.cos(twist_rad)
    sin_t = math.sin(twist_rad)
    
    # Translate to pivot origin
    dx = x - pivot_x
    dz = z - pivot_z
    
    # Rotate in XZ plane (about Y-axis)
    # For nose-up twist (positive angle), TE moves down:
    #   x' = x * cos(θ) + z * sin(θ)
    #   z' = -x * sin(θ) + z * cos(θ)
    x_rot = dx * cos_t + dz * sin_t
    z_rot = -dx * sin_t + dz * cos_t
    
    # Translate back
    return x_rot + pivot_x, z_rot + pivot_z


# ==============================================================================
# Parameter Dataclasses
# ==============================================================================

@dataclass
class RibProfileParams:
    """Parameters for rib profile generation."""
    include_spar_notches: bool = True
    include_stringer_slots: bool = True
    include_lightening_holes: bool = False
    include_elevon_cutout: bool = True  # Cut away the elevon region on affected ribs
    spar_notch_clearance_mm: float = 0.1
    spar_notch_depth_percent: float = 50.0  # Notch depth as % of local thickness
    stringer_slot_clearance_mm: float = 0.2
    lightening_hole_margin_mm: float = 10.0
    lightening_hole_shape: str = "circular"  # "circular", "elliptical"
    lightening_hole_corner_radius_mm: float = 3.0
    min_hole_diameter_mm: float = 15.0
    # Tab/slot for spar-rib interlocking
    include_tabs: bool = False
    tab_width_mm: float = 8.0
    tab_depth_mm: float = 5.0
    tab_clearance_mm: float = 0.15
    # Elevon hinge clearance
    elevon_max_deflection_deg: float = 30.0  # Max deflection angle for V-notch clearance
    elevon_hinge_gap_mm: float = 1.0  # Gap between main rib and elevon rib at hinge
    # Sweep correction for slot widths (NEW for DXF accuracy)
    front_spar_sweep_rad: float = 0.0  # Local sweep angle at this rib
    rear_spar_sweep_rad: float = 0.0
    dihedral_rad: float = 0.0  # Wing dihedral angle for slot adjustment
    apply_sweep_correction: bool = True  # Toggle for simpler builds


@dataclass
class SparProfileParams:
    """Parameters for spar plate profile generation."""
    # spar_height_mm removed - now computed from airfoil geometry
    include_rib_notches: bool = True
    rib_notch_clearance_mm: float = 0.1
    rib_notch_depth_percent: float = 45.0
    # Tab/slot for spar-rib interlocking
    include_tabs: bool = False
    tab_width_mm: float = 8.0
    tab_depth_mm: float = 5.0


@dataclass
class ElevonCutoutInfo:
    """Information about elevon cutout in a rib."""
    x_hinge_mm: float      # Chordwise position of hinge line
    z_hinge_upper_mm: float  # Z of upper surface at hinge
    z_hinge_lower_mm: float  # Z of lower surface at hinge
    has_cutout: bool = False  # Whether this rib has an elevon cutout


# ==============================================================================
# Profile Result Dataclasses
# ==============================================================================

@dataclass
class SparNotchInfo:
    """Information about a spar notch in a rib."""
    x_center_mm: float  # Chordwise center position
    z_upper_mm: float   # Upper surface Z at this location
    z_lower_mm: float   # Lower surface Z at this location
    notch_width_mm: float  # Width of the notch (spar thickness + clearance)
    notch_depth_mm: float  # Depth from lower surface upward
    spar_type: str  # "front" or "rear"


@dataclass
class StringerSlotInfo:
    """Information about a stringer slot in a rib."""
    x_center_mm: float  # Chordwise center position
    surface: str  # "upper" or "lower"
    slot_width_mm: float
    slot_height_mm: float
    z_surface_mm: float  # Z-coordinate of the skin surface at this location


@dataclass
class LighteningHoleInfo:
    """Information about a lightening hole."""
    cx_mm: float  # Center X
    cz_mm: float  # Center Z
    size_mm: float  # Diameter for circular, height for elliptical
    shape: str  # "circular", "elliptical"
    width_mm: Optional[float] = None  # Width for elliptical holes (if different from size_mm)
    rotation_deg: float = 0.0  # Rotation of the major axis (deg)



@dataclass
class TabSlotInfo:
    """Information about a tab slot (enclosed hole) in a rib for spar tabs."""
    x_center_mm: float  # Chordwise center position (same as spar position)
    z_center_mm: float  # Height center position
    slot_width_mm: float  # Width of slot (spar thickness + clearance)
    slot_height_mm: float  # Height of slot (tab depth + clearance)
    spar_type: str  # "front" or "rear"


@dataclass
class RibProfile:
    """Generated rib profile with metadata."""
    outline: np.ndarray  # Nx2 closed polyline (mm) - X, Z coordinates
    section_index: int  # Which spanwise section
    span_fraction: float  # Eta position (0=root, 1=tip)
    chord_mm: float  # Local chord length
    spar_notches: List[SparNotchInfo] = field(default_factory=list)
    stringer_slots: List[StringerSlotInfo] = field(default_factory=list)
    lightening_holes: List[LighteningHoleInfo] = field(default_factory=list)
    tab_slots: List[TabSlotInfo] = field(default_factory=list)  # Enclosed slots for spar tabs
    elevon_cutout: Optional[ElevonCutoutInfo] = None  # Elevon cutout info if applicable
    is_body_section: bool = False  # True if in BWB body region
    # For grain direction indicator
    recommended_grain: str = "chordwise"


@dataclass
class ElevonRibProfile:
    """Generated elevon rib profile (the aft piece of a rib in control surface region).
    
    This is the portion of the rib from the hinge line to the trailing edge,
    which forms part of the movable control surface.
    
    The outline includes a V-notch at the hinge line for deflection clearance.
    """
    outline: np.ndarray  # Nx2 closed polyline (mm) - X, Z coordinates
    section_index: int  # Which spanwise section (matches parent rib)
    span_fraction: float  # Eta position (0=root, 1=tip)
    chord_mm: float  # Elevon chord (from hinge to TE)
    hinge_x_mm: float  # X position of hinge line
    hinge_z_mm: float = 0.0  # Z position of hinge axis (pivot point)
    max_deflection_deg: float = 30.0  # Max deflection angle
    control_surface_name: str = "Elevon"  # Name of the control surface
    # For grain direction indicator
    recommended_grain: str = "chordwise"


@dataclass
class RibNotchInfo:
    """Information about a rib notch in a spar."""
    x_along_span_mm: float  # Position along spar (from root)
    notch_width_mm: float  # Width of the notch (rib thickness + clearance)
    notch_depth_mm: float  # Depth from bottom edge upward
    spar_height_at_station_mm: float = 0.0  # Local spar height at this station


@dataclass
class SparStationHeight:
    """Spar height at a specific spanwise station."""
    x_along_span_mm: float  # Position along spar (from root)
    z_upper_mm: float       # Upper surface Z at spar position
    z_lower_mm: float       # Lower surface Z at spar position
    
    @property
    def height_mm(self) -> float:
        """Spar height (from lower to upper surface)."""
        return self.z_upper_mm - self.z_lower_mm


@dataclass
class SparCutLine:
    """A cut line on a spar (for separating body/wing sections in BWB)."""
    x_along_span_mm: float  # Position along spar
    z_lower_mm: float       # Z at lower edge of spar at this position
    z_upper_mm: float       # Z at upper edge of spar at this position
    label: str = ""         # Optional label for the cut line


@dataclass
class SparProfile:
    """Generated spar plate profile with metadata."""
    outline: np.ndarray  # Nx2 closed polyline (mm) - X (spanwise), Z (height)
    spar_type: str  # "front" or "rear"
    spar_region: str = "wing"  # "wing" or "body" (for BWB separation)
    length_mm: float = 0.0  # Total spar length (span)
    station_heights: List[SparStationHeight] = field(default_factory=list)  # Height at each rib station
    rib_notches: List[RibNotchInfo] = field(default_factory=list)
    cut_lines: List[SparCutLine] = field(default_factory=list)  # Cut lines for BWB separation
    junction_profile: Optional['FingerJointProfile'] = None  # Finger joint at BWB junction (if applicable)
    # For grain direction indicator
    recommended_grain: str = "spanwise"
    
    @property
    def height_at_root_mm(self) -> float:
        """Spar height at root (first station)."""
        if self.station_heights:
            return self.station_heights[0].height_mm
        return 0.0
    
    @property
    def height_at_tip_mm(self) -> float:
        """Spar height at tip (last station)."""
        if self.station_heights:
            return self.station_heights[-1].height_mm
        return 0.0
    
    @property
    def min_height_mm(self) -> float:
        """Minimum spar height along span."""
        if self.station_heights:
            return min(s.height_mm for s in self.station_heights)
        return 0.0
    
    @property
    def max_height_mm(self) -> float:
        """Maximum spar height along span."""
        if self.station_heights:
            return max(s.height_mm for s in self.station_heights)
        return 0.0


# ==============================================================================
# Sweep and Dihedral Correction Functions (for DXF accuracy)
# ==============================================================================

def compute_effective_slot_width(
    material_thickness_mm: float,
    local_sweep_rad: float,
    clearance_mm: float = 0.1,
) -> float:
    """
    Compute the slot width needed for a spar passing through at an angle.
    
    On a swept wing, spars pass through ribs at an oblique angle. The effective
    horizontal thickness of the spar as seen by the rib is wider than the 
    material thickness: t_effective = t_material / cos(Λ_local)
    
    Example: A 3mm spar on a 30° swept wing needs a 3.46mm slot (not 3.2mm).
    
    Args:
        material_thickness_mm: Nominal spar plate thickness
        local_sweep_rad: Local spar sweep angle in radians
        clearance_mm: Clearance for slip fit (added to each side)
    
    Returns:
        Effective slot width in mm
    """
    cos_sweep = np.cos(local_sweep_rad)
    
    # Clamp to avoid division by near-zero (very high sweep angles > ~60°)
    cos_sweep = max(cos_sweep, 0.5)
    
    effective_thickness = material_thickness_mm / cos_sweep
    slot_width = effective_thickness + 2 * clearance_mm
    
    return float(slot_width)


def compute_full_slot_dimensions(
    spar_thickness_mm: float,
    sweep_rad: float,
    dihedral_rad: float,
    base_clearance_mm: float = 0.1,
) -> Tuple[float, float]:
    """
    Compute slot dimensions accounting for both sweep and dihedral.
    
    When ribs are vertical but spars are tilted (due to dihedral), the slot 
    must be adjusted. The spar passes through the vertical rib at a compound 
    angle from both sweep and dihedral.
    
    Args:
        spar_thickness_mm: Nominal spar plate thickness
        sweep_rad: Local spar sweep angle in radians
        dihedral_rad: Wing dihedral angle in radians
        base_clearance_mm: Base clearance for slip fit
    
    Returns:
        Tuple of (slot_width_mm, slot_height_adjustment_mm)
        - slot_width_mm: Horizontal slot width
        - slot_height_adjustment_mm: Additional height needed at top/bottom
    """
    # Sweep affects horizontal width
    cos_sweep = max(np.cos(sweep_rad), 0.5)
    width_from_sweep = spar_thickness_mm / cos_sweep
    
    # Dihedral affects both width and height
    cos_dihedral = max(np.cos(dihedral_rad), 0.9)
    sin_dihedral = np.sin(dihedral_rad)
    
    # Combined horizontal width
    slot_width = width_from_sweep / cos_dihedral + 2 * base_clearance_mm
    
    # Height adjustment for tilted spar corners
    # For a snug fit, add small vertical clearance
    height_adjustment = spar_thickness_mm * abs(sin_dihedral)
    
    return float(slot_width), float(height_adjustment)


# ==============================================================================
# BWB Finger Joint Configuration and Generation
# ==============================================================================

@dataclass
class BWBJointConfig:
    """Configuration for BWB body/wing spar junction."""
    joint_type: str = "butt"  # "butt", "finger", "scarf" - butt = straight cut, no interlocking
    finger_count: int = 3       # Number of fingers (odd recommended)
    finger_width_mm: float = 0.0  # 0 = auto-calculate from spar height
    finger_depth_mm: float = 0.0  # 0 = use full material thickness
    clearance_mm: float = 0.1    # Fit clearance per side


@dataclass
class FingerJointProfile:
    """Finger joint geometry for one side of a junction."""
    edge_points: np.ndarray  # Nx2 array of (x, z) points defining the finger edge
    is_male: bool            # True = fingers protrude, False = slots cut in


def generate_finger_joint_edge(
    edge_length_mm: float,
    spar_thickness_mm: float,
    config: BWBJointConfig,
    is_male: bool = True,
) -> FingerJointProfile:
    """
    Generate finger joint edge profile for spar junction.
    
    The finger pattern is symmetric about the edge center for aesthetic
    and structural balance. Uses odd finger count for center finger.
    
    Finger joints provide:
    - Positive alignment during assembly
    - Increased glue surface area
    - Stronger mechanical connection
    - Visual reference for correct orientation
    
    Args:
        edge_length_mm: Height of the spar edge at junction
        spar_thickness_mm: Material thickness (finger depth)
        config: Joint configuration parameters
        is_male: True for protruding fingers, False for receiving slots
    
    Returns:
        FingerJointProfile with edge points
    """
    n_fingers = config.finger_count
    if n_fingers < 1:
        n_fingers = 3  # Default to 3 fingers
    
    # Auto-calculate finger width if not specified
    if config.finger_width_mm <= 0:
        # Divide edge into 2*n_fingers - 1 segments (fingers + spaces)
        total_segments = 2 * n_fingers - 1
        finger_width = edge_length_mm / total_segments
    else:
        finger_width = config.finger_width_mm
    
    # Finger depth (how far fingers protrude)
    if config.finger_depth_mm <= 0:
        finger_depth = spar_thickness_mm
    else:
        finger_depth = config.finger_depth_mm
    
    # Add clearance for fit
    clearance = config.clearance_mm
    
    # Build edge profile from bottom to top
    points = []
    z = 0.0  # Start at bottom of spar edge
    
    # Calculate margins to center the finger pattern
    total_pattern_height = (2 * n_fingers - 1) * finger_width
    bottom_margin = (edge_length_mm - total_pattern_height) / 2
    
    # Start with bottom margin (straight edge)
    points.append((0.0, z))
    z = bottom_margin
    points.append((0.0, z))
    
    # Generate finger pattern
    for i in range(2 * n_fingers - 1):
        is_finger = (i % 2 == 0)  # Alternating: finger, space, finger, ...
        
        if is_male:
            # Male side: fingers protrude at x = finger_depth
            if is_finger:
                # Transition out to finger
                points.append((finger_depth - clearance, z))
                z += finger_width
                points.append((finger_depth - clearance, z))
                # Transition back to edge
                points.append((0.0, z))
            else:
                # Space: stay at edge
                z += finger_width
                points.append((0.0, z))
        else:
            # Female side: slots cut in at x = -finger_depth
            if is_finger:
                # Stay at edge (this is a "finger" on male side, so space here)
                z += finger_width
                points.append((0.0, z))
            else:
                # Cut slot inward
                points.append((-finger_depth - clearance, z))
                z += finger_width
                points.append((-finger_depth - clearance, z))
                points.append((0.0, z))
    
    # Top margin (straight edge)
    z = edge_length_mm
    points.append((0.0, z))
    
    return FingerJointProfile(
        edge_points=np.array(points),
        is_male=is_male,
    )


# ==============================================================================
# Main Profile Generation Functions
# ==============================================================================

def generate_rib_profile(
    section: SpanwiseSection,
    project: Project,
    params: RibProfileParams,
) -> RibProfile:
    """
    Generate complete 2D rib profile with all cutouts.
    
    This is the SINGLE SOURCE OF TRUTH for rib geometry.
    Used by:
    - dxf_export.py for laser cutting
    - geometry_builder.py for 3D extrusion
    
    Args:
        section: SpanwiseSection with airfoil and position data
        project: Project with planform/structural parameters
        params: Generation parameters
    
    Returns:
        RibProfile with closed outline and feature positions
    """
    plan = project.wing.planform
    
    # Get airfoil coordinates in mm
    airfoil = section.airfoil
    if airfoil is None or not hasattr(airfoil, 'coordinates') or airfoil.coordinates is None:
        raise ValueError(f"Section {section.index} has no airfoil coordinates")
    
    coords = np.array(airfoil.coordinates)  # Nx2 normalized
    chord_mm = section.chord_m * 1000
    
    # Scale to physical dimensions
    x_raw = coords[:, 0] * chord_mm
    z_raw = coords[:, 1] * chord_mm
    
    # Apply twist rotation about the leading edge
    # This rotates the airfoil shape in the XZ plane to match the local twist
    # at this spanwise station. The resulting 2D profile is what a vertical
    # rib needs to be cut to in order to match the twisted wing surface.
    if abs(section.twist_deg) > 1e-6:
        # Pivot at leading edge (minimum x, z at LE)
        le_x = x_raw.min()
        le_idx = np.argmin(x_raw)
        le_z = z_raw[le_idx]
        x_raw, z_raw = apply_twist_to_2d_coords(
            x_raw, z_raw,
            section.twist_deg,
            pivot_x=le_x,
            pivot_z=le_z,
        )
    
    # Normalize x to start at 0
    x = x_raw - x_raw.min()
    z = z_raw
    
    # ==========================================================================
    # Calculate spar notch X positions
    # ==========================================================================
    # Spar notches follow straight lines in 3D space.
    # This ensures notches align with the actual spar geometry (which is extruded
    # as straight plates per region in geometry_builder.py).
    #
    # The spar is a vertical plate at a fixed chord percentage. The rib profile
    # has been rotated by twist, so the spar notch must be placed at the X position
    # where the vertical spar plane intersects the twisted rib.
    #
    # Since twist rotates about the LE (x=0), and the spar is vertical (constant X
    # in world frame), the spar X position in the twisted rib's local frame is
    # the same as the untwisted X position. The Z values at that X are what change.
    
    eta = section.span_fraction
    front_xsi = get_spar_xsi_at_section(section, plan, "front")
    rear_xsi = get_spar_xsi_at_section(section, plan, "rear")
    
    # is_body_section is used for tab slots and other regions
    is_body_section = False
    if plan.body_sections:
        last_body_section = max(plan.body_sections, key=lambda bs: bs.y_pos)
        if abs(section.y_m) <= last_body_section.y_pos + 0.001:
            is_body_section = True
    
    front_x = front_xsi * chord_mm
    rear_x = rear_xsi * chord_mm
    
    # Get thickness at spar locations for notch depth
    z_at_front = _get_surface_z_at_x(x, z, front_x)
    z_at_rear = _get_surface_z_at_x(x, z, rear_x)
    
    # Spar plate thickness from planform
    spar_thickness_mm = plan.spar_thickness_mm
    
    # Calculate notch dimensions with sweep correction
    # On a swept wing, spars pass through ribs at an oblique angle, making
    # the effective slot wider than the material thickness
    if params.apply_sweep_correction:
        # Use sweep-corrected slot widths
        front_notch_width = compute_effective_slot_width(
            spar_thickness_mm,
            params.front_spar_sweep_rad,
            params.spar_notch_clearance_mm,
        )
        rear_notch_width = compute_effective_slot_width(
            spar_thickness_mm,
            params.rear_spar_sweep_rad,
            params.spar_notch_clearance_mm,
        )
        
        # Apply dihedral correction if significant
        if abs(params.dihedral_rad) > 0.01:  # More than ~0.5 degrees
            front_width_full, _ = compute_full_slot_dimensions(
                spar_thickness_mm,
                params.front_spar_sweep_rad,
                params.dihedral_rad,
                params.spar_notch_clearance_mm,
            )
            rear_width_full, _ = compute_full_slot_dimensions(
                spar_thickness_mm,
                params.rear_spar_sweep_rad,
                params.dihedral_rad,
                params.spar_notch_clearance_mm,
            )
            front_notch_width = max(front_notch_width, front_width_full)
            rear_notch_width = max(rear_notch_width, rear_width_full)
    else:
        # Simple (uncorrected) slot width
        front_notch_width = spar_thickness_mm + 2 * params.spar_notch_clearance_mm
        rear_notch_width = front_notch_width
    
    # Notch depth based on parameters
    front_thickness = z_at_front['upper'] - z_at_front['lower']
    rear_thickness = z_at_rear['upper'] - z_at_rear['lower']
    front_notch_depth = front_thickness * (params.spar_notch_depth_percent / 100.0)
    rear_notch_depth = rear_thickness * (params.spar_notch_depth_percent / 100.0)
    
    # Build spar notch info
    spar_notches = []
    if params.include_spar_notches:
        spar_notches.append(SparNotchInfo(
            x_center_mm=front_x,
            z_upper_mm=z_at_front['upper'],
            z_lower_mm=z_at_front['lower'],
            notch_width_mm=front_notch_width,
            notch_depth_mm=front_notch_depth,
            spar_type="front"
        ))
        spar_notches.append(SparNotchInfo(
            x_center_mm=rear_x,
            z_upper_mm=z_at_rear['upper'],
            z_lower_mm=z_at_rear['lower'],
            notch_width_mm=rear_notch_width,
            notch_depth_mm=rear_notch_depth,
            spar_type="rear"
        ))
    
    # Build stringer slot info
    stringer_slots = []
    if params.include_stringer_slots and plan.stringer_count > 0:
        stringer_slots = _calculate_stringer_slots(
            x, z,
            front_x, rear_x,
            plan.stringer_count,
            plan.stringer_height_mm,
            plan.stringer_thickness_mm,
            params.stringer_slot_clearance_mm,
        )
    
    # Build lightening holes info with collision detection
    lightening_holes = []
    if params.include_lightening_holes:
        lightening_holes = _calculate_lightening_holes(
            x, z,
            front_x, rear_x,
            params.lightening_hole_margin_mm,
            params.lightening_hole_shape,
            params.min_hole_diameter_mm,
            getattr(plan, 'rib_lightening_fraction', 0.4),
            stringer_slots=stringer_slots,  # Pass stringer slots for collision detection
            collision_clearance_mm=2.0,  # Extra clearance around obstacles
            twist_deg=section.twist_deg,
        )

    
    # Check for elevon cutout - is this rib in a control surface region?
    elevon_cutout = None
    if params.include_elevon_cutout and plan.control_surfaces:
        eta_percent = eta * 100.0  # Convert to percentage
        for cs in plan.control_surfaces:
            if cs.span_start_percent <= eta_percent <= cs.span_end_percent:
                # This rib is in the control surface region
                # Hinge position interpolated between start and end
                t = (eta_percent - cs.span_start_percent) / max(0.001, cs.span_end_percent - cs.span_start_percent)
                hinge_xsi = (cs.chord_start_percent * (1 - t) + cs.chord_end_percent * t) / 100.0
                hinge_x = hinge_xsi * chord_mm
                
                z_at_hinge = _get_surface_z_at_x(x, z, hinge_x)
                elevon_cutout = ElevonCutoutInfo(
                    x_hinge_mm=hinge_x,
                    z_hinge_upper_mm=z_at_hinge['upper'],
                    z_hinge_lower_mm=z_at_hinge['lower'],
                    has_cutout=True,
                )
                break  # Only one cutout per rib
    
    # Note: is_body_section is already computed above
    
    # Build tab slots for spar tabs (enclosed holes inside the rib)
    tab_slots: List[TabSlotInfo] = []
    if params.include_tabs:
        # Tab slots are placed at spar positions, centered vertically in the rib
        # The slot is slightly larger than the spar tab for clearance
        slot_width = spar_thickness_mm + 2 * params.tab_clearance_mm
        slot_height = params.tab_depth_mm + 2 * params.tab_clearance_mm
        
        # Front spar tab slot
        front_z_center = (z_at_front['upper'] + z_at_front['lower']) / 2.0
        tab_slots.append(TabSlotInfo(
            x_center_mm=front_x,
            z_center_mm=front_z_center,
            slot_width_mm=slot_width,
            slot_height_mm=slot_height,
            spar_type="front",
        ))
        
        # Rear spar tab slot
        rear_z_center = (z_at_rear['upper'] + z_at_rear['lower']) / 2.0
        tab_slots.append(TabSlotInfo(
            x_center_mm=rear_x,
            z_center_mm=rear_z_center,
            slot_width_mm=slot_width,
            slot_height_mm=slot_height,
            spar_type="rear",
        ))
    
    # Build the outline with all features (including elevon cutout)
    # Note: When using tabs, we DON'T include spar notches (they're mutually exclusive)
    outline = _build_rib_outline_with_features(
        x, z,
        spar_notches=spar_notches if params.include_spar_notches and not params.include_tabs else [],
        stringer_slots=stringer_slots if params.include_stringer_slots else [],
        elevon_cutout=elevon_cutout,
        include_tabs=params.include_tabs,
        tab_width_mm=params.tab_width_mm,
        tab_depth_mm=params.tab_depth_mm,
    )
    
    return RibProfile(
        outline=outline,
        section_index=section.index,
        span_fraction=section.span_fraction,
        chord_mm=chord_mm,
        spar_notches=spar_notches if not params.include_tabs else [],  # Clear if using tabs
        stringer_slots=stringer_slots,
        lightening_holes=lightening_holes,
        tab_slots=tab_slots,  # New field for tab slots
        elevon_cutout=elevon_cutout,
        is_body_section=is_body_section,
        recommended_grain="chordwise",
    )


def generate_elevon_rib_profile(
    section: SpanwiseSection,
    project: Project,
    elevon_cutout: ElevonCutoutInfo,
    max_deflection_deg: float = 30.0,
    hinge_gap_mm: float = 1.0,
) -> Optional[ElevonRibProfile]:
    """
    Generate the aft rib piece for control surfaces (elevon ribs).
    
    This is the portion of the rib from the hinge line to the trailing edge,
    which forms part of the movable control surface.
    
    The profile includes a V-notch at the hinge line to provide clearance
    for the specified deflection angle.
    
    Args:
        section: SpanwiseSection with airfoil and position data
        project: Project with planform/structural parameters
        elevon_cutout: ElevonCutoutInfo with hinge position
        max_deflection_deg: Maximum deflection angle (for V-notch sizing)
        hinge_gap_mm: Gap between main rib and elevon rib at hinge
    
    Returns:
        ElevonRibProfile with closed outline including V-notch, or None if not applicable
    """
    if not elevon_cutout or not elevon_cutout.has_cutout:
        return None
    
    plan = project.wing.planform
    
    # Get airfoil coordinates in mm
    airfoil = section.airfoil
    if airfoil is None or not hasattr(airfoil, 'coordinates') or airfoil.coordinates is None:
        return None
    
    coords = np.array(airfoil.coordinates)  # Nx2 normalized
    chord_mm = section.chord_m * 1000
    
    # Scale to physical dimensions
    x_raw = coords[:, 0] * chord_mm
    z_raw = coords[:, 1] * chord_mm
    
    # Apply twist rotation about the leading edge
    # Same twist is applied to elevon ribs for consistency with main ribs
    if abs(section.twist_deg) > 1e-6:
        le_x = x_raw.min()
        le_idx = np.argmin(x_raw)
        le_z = z_raw[le_idx]
        x_raw, z_raw = apply_twist_to_2d_coords(
            x_raw, z_raw,
            section.twist_deg,
            pivot_x=le_x,
            pivot_z=le_z,
        )
    
    # Normalize x to start at 0
    x = x_raw - x_raw.min()
    z = z_raw
    
    # Extract the aft portion (from hinge to trailing edge)
    hinge_x = elevon_cutout.x_hinge_mm
    elevon_x, elevon_z = _extract_aft_airfoil_portion_with_vnotch(
        x, z, hinge_x, max_deflection_deg, hinge_gap_mm
    )
    
    if len(elevon_x) < 3:
        return None
    
    # Find control surface name and get hinge_rel_height
    cs_name = "Elevon"
    hinge_rel_height = 0.5  # Default to center
    eta_percent = section.span_fraction * 100.0
    for cs in plan.control_surfaces:
        if cs.span_start_percent <= eta_percent <= cs.span_end_percent:
            cs_name = cs.name
            hinge_rel_height = cs.hinge_rel_height
            break
    
    # Calculate hinge Z position based on hinge_rel_height
    z_upper = elevon_cutout.z_hinge_upper_mm
    z_lower = elevon_cutout.z_hinge_lower_mm
    hinge_z = z_lower + (z_upper - z_lower) * hinge_rel_height
    
    outline = np.column_stack((elevon_x, elevon_z))
    elevon_chord = chord_mm - hinge_x  # Chord from hinge to TE
    
    return ElevonRibProfile(
        outline=outline,
        section_index=section.index,
        span_fraction=section.span_fraction,
        chord_mm=elevon_chord,
        hinge_x_mm=hinge_x,
        hinge_z_mm=hinge_z,
        max_deflection_deg=max_deflection_deg,
        control_surface_name=cs_name,
        recommended_grain="chordwise",
    )


def generate_spar_profile(
    project: Project,
    sections: List[SpanwiseSection],
    spar_type: str,  # "front" or "rear"
    params: SparProfileParams,
) -> SparProfile:
    """
    Generate complete 2D spar plate profile with rib notches.
    
    The spar height is computed from the actual airfoil geometry at each
    station, creating a tapered profile that follows the wingbox envelope.
    This aligns with the structural model which uses the full wingbox height.
    
    This is the SINGLE SOURCE OF TRUTH for spar geometry.
    Used by:
    - dxf_export.py for laser cutting
    - geometry_builder.py for 3D extrusion
    
    Note: Stringers run parallel to spars (spanwise) and do NOT 
    intersect them. Stringers only pass through ribs.
    
    Args:
        project: Project with planform/structural parameters
        sections: List of SpanwiseSection for rib positions
        spar_type: "front" or "rear"
        params: Generation parameters
    
    Returns:
        SparProfile with tapered outline following airfoil geometry
    """
    plan = project.wing.planform
    
    if not sections:
        raise ValueError("No sections provided for spar generation")
    
    # Sort sections by spanwise position
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    
    # Compute spar positions at each section to get the actual spar line
    # (needed for sweep-corrected distance calculation)
    # Note: Dihedral (Z) affects 3D position but NOT the flat spar profile shape
    spar_positions = []  # List of (y_m, x_m) positions along the spar
    for section in sorted_sections:
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x_m = section.x_le_m + spar_xsi * section.chord_m
        spar_positions.append((abs(section.y_m), spar_x_m))
    
    # Calculate cumulative distance along the spar line (sweep only, not dihedral)
    # The spar is a flat plate - distance is measured in the XY plane
    cumulative_distances = [0.0]  # First station at distance 0
    for i in range(1, len(spar_positions)):
        y_prev, x_prev = spar_positions[i-1]
        y_curr, x_curr = spar_positions[i]
        dy = y_curr - y_prev
        dx = x_curr - x_prev
        segment_length = math.sqrt(dy*dy + dx*dx)  # 2D distance in XY plane
        cumulative_distances.append(cumulative_distances[-1] + segment_length)
    
    # Calculate spar height at each station from airfoil geometry
    # Z values include dihedral offset so the flat laser-cut spar will
    # fit correctly when assembled into a wing with dihedral
    station_heights: List[SparStationHeight] = []
    
    station_idx = 0
    for section in sorted_sections:
        # Get airfoil coordinates
        airfoil = section.airfoil
        if airfoil is None or not hasattr(airfoil, 'coordinates') or airfoil.coordinates is None:
            station_idx += 1
            continue
        
        coords = np.array(airfoil.coordinates)
        chord_mm = section.chord_m * 1000
        
        # Scale to physical dimensions
        x_raw = coords[:, 0] * chord_mm
        z_raw = coords[:, 1] * chord_mm
        
        # Apply twist rotation about the leading edge
        # This ensures the spar upper/lower edges follow the twisted wing surface
        if abs(section.twist_deg) > 1e-6:
            le_x = x_raw.min()
            le_idx = np.argmin(x_raw)
            le_z = z_raw[le_idx]
            x_raw, z_raw = apply_twist_to_2d_coords(
                x_raw, z_raw,
                section.twist_deg,
                pivot_x=le_x,
                pivot_z=le_z,
            )
        
        x = x_raw - x_raw.min()
        z = z_raw
        
        # Determine spar X position based on type
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x = spar_xsi * chord_mm
        
        # Get Z coordinates at spar position (local airfoil coords)
        z_at_spar = _get_surface_z_at_x(x, z, spar_x)
        
        # Add dihedral offset to Z values
        # This makes the spar profile correctly shaped for assembly
        dihedral_offset_mm = section.z_m * 1000  # Convert m to mm
        
        # Use sweep-corrected distance along spar (not just spanwise Y)
        x_along_spar_mm = cumulative_distances[station_idx] * 1000
        station_heights.append(SparStationHeight(
            x_along_span_mm=x_along_spar_mm,
            z_upper_mm=z_at_spar['upper'] + dihedral_offset_mm,
            z_lower_mm=z_at_spar['lower'] + dihedral_offset_mm,
        ))
        station_idx += 1
    
    if len(station_heights) < 2:
        raise ValueError("Need at least 2 valid stations for spar generation")
    
    span_mm = station_heights[-1].x_along_span_mm
    
    # Build rib notches with proper depth based on local spar height
    rib_thickness_mm = plan.rib_thickness_mm
    notch_width = rib_thickness_mm + 2 * params.rib_notch_clearance_mm
    
    rib_notches = []
    if params.include_rib_notches:
        for station in station_heights:
            local_height = station.height_mm
            notch_depth = local_height * (params.rib_notch_depth_percent / 100.0)
            rib_notches.append(RibNotchInfo(
                x_along_span_mm=station.x_along_span_mm,
                notch_width_mm=notch_width,
                notch_depth_mm=notch_depth,
                spar_height_at_station_mm=local_height,
            ))
    
    # Build tapered spar outline following airfoil envelope
    outline = _build_tapered_spar_outline(
        station_heights=station_heights,
        rib_notches=rib_notches if params.include_rib_notches else [],
        include_tabs=params.include_tabs,
        tab_width_mm=params.tab_width_mm,
        tab_depth_mm=params.tab_depth_mm,
    )
    
    return SparProfile(
        outline=outline,
        spar_type=spar_type,
        spar_region="wing",  # Default to wing - use generate_separated_spar_profiles for BWB
        length_mm=span_mm,
        station_heights=station_heights,
        rib_notches=rib_notches,
        recommended_grain="spanwise",
    )


def generate_separated_spar_profiles(
    project: Project,
    sections: List[SpanwiseSection],
    spar_type: str,  # "front" or "rear"
    params: SparProfileParams,
) -> List[SparProfile]:
    """
    Generate spar profile with cut lines for BWB body/wing separation.
    
    For standard flying wings (no body_sections), returns a single spar.
    For BWB aircraft, returns a single spar with cut lines at the last body section.
    The user can cut at these lines to separate body and wing spars if needed.
    
    Args:
        project: Project with planform/structural parameters
        sections: List of SpanwiseSection for rib positions
        spar_type: "front" or "rear"
        params: Generation parameters
    
    Returns:
        List containing single SparProfile (with cut_lines for BWB)
    """
    plan = project.wing.planform
    
    # Generate full spar first
    spar = generate_spar_profile(project, sections, spar_type, params)
    
    # If not BWB, just return the spar as-is
    if not plan.body_sections:
        return [spar]
    
    # BWB - add cut line at the last body section position
    # The last body section marks the boundary between body and wing
    last_body_section = max(plan.body_sections, key=lambda bs: bs.y_pos)
    cut_y_m = last_body_section.y_pos  # Y position in meters
    cut_y_mm = cut_y_m * 1000
    
    print(f"[BWB Spar] Last body section at y={cut_y_m:.3f}m ({cut_y_mm:.1f}mm)")
    
    # Find the spar station_heights entry closest to this Y position
    # and use that station's x_along_span_mm for the cut line
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    first_section_y_mm = abs(sorted_sections[0].y_m) * 1000
    
    # The cut line position in spar coordinates (x_along_span is relative to first section)
    cut_x_along_spar = cut_y_mm - first_section_y_mm
    
    print(f"[BWB Spar Debug] first_section_y={first_section_y_mm:.1f}mm, cut_x={cut_x_along_spar:.1f}mm")
    
    # Find the station height at or near the cut position
    if not spar.station_heights or cut_x_along_spar < 0:
        return [spar]

    x_positions = [s.x_along_span_mm for s in spar.station_heights]
    
    # Find the station that matches or is closest to the cut position
    best_idx = 0
    best_dist = float('inf')
    for i, x_pos in enumerate(x_positions):
        dist = abs(x_pos - cut_x_along_spar)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    # Use the actual station position for the cut line (snap to rib)
    cut_x_final = x_positions[best_idx]
    
    if cut_x_final <= 0 or cut_x_final >= spar.length_mm:
        return [spar]

    # Split into two SparProfile objects
    # 1. Body Part (left)
    body_stations = [s for s in spar.station_heights if s.x_along_span_mm <= cut_x_final + 1e-3]
    body_notches = [n for n in spar.rib_notches if n.x_along_span_mm <= cut_x_final + 1e-3]
    
    body_outline = _build_tapered_spar_outline(
        station_heights=body_stations,
        rib_notches=body_notches,
        include_tabs=params.include_tabs,
        tab_width_mm=params.tab_width_mm,
        tab_depth_mm=params.tab_depth_mm,
    )
    
    body_spar = SparProfile(
        outline=body_outline,
        spar_type=spar_type,
        spar_region="body",
        length_mm=cut_x_final,
        station_heights=body_stations,
        rib_notches=body_notches,
        recommended_grain="spanwise",
    )
    
    # 2. Wing Part (right)
    wing_stations_orig = [s for s in spar.station_heights if s.x_along_span_mm >= cut_x_final - 1e-3]
    wing_notches_orig = [n for n in spar.rib_notches if n.x_along_span_mm >= cut_x_final - 1e-3]
    
    # Shift X coordinates for the wing part to start at 0
    wing_stations = [
        SparStationHeight(
            x_along_span_mm=s.x_along_span_mm - cut_x_final,
            z_upper_mm=s.z_upper_mm,
            z_lower_mm=s.z_lower_mm
        ) for s in wing_stations_orig
    ]
    
    wing_notches = [
        RibNotchInfo(
            x_along_span_mm=n.x_along_span_mm - cut_x_final,
            notch_width_mm=n.notch_width_mm,
            notch_depth_mm=n.notch_depth_mm,
            spar_height_at_station_mm=n.spar_height_at_station_mm
        ) for n in wing_notches_orig
    ]
    
    wing_outline = _build_tapered_spar_outline(
        station_heights=wing_stations,
        rib_notches=wing_notches,
        include_tabs=params.include_tabs,
        tab_width_mm=params.tab_width_mm,
        tab_depth_mm=params.tab_depth_mm,
    )
    
    wing_spar = SparProfile(
        outline=wing_outline,
        spar_type=spar_type,
        spar_region="wing",
        length_mm=spar.length_mm - cut_x_final,
        station_heights=wing_stations,
        rib_notches=wing_notches,
        recommended_grain="spanwise",
    )
    
    # Generate finger joint profiles if BWB joint config specifies finger joints
    bwb_joint_config = getattr(plan, 'bwb_joint_config', None)
    if bwb_joint_config is None:
        # Create default config for backwards compatibility
        bwb_joint_config = BWBJointConfig()
    
    if bwb_joint_config.joint_type == "finger":
        # Get spar height at junction for finger joint sizing
        # Find station height closest to the cut position (before shift)
        junction_station = None
        for s in spar.station_heights:
            if abs(s.x_along_span_mm - cut_x_final) < 1e-3:
                junction_station = s
                break
        
        if junction_station is None and body_stations:
            # Use last body station
            junction_station = body_stations[-1]
        
        if junction_station:
            junction_height = junction_station.height_mm
            
            # Generate finger joint profiles for body (male) and wing (female)
            body_joint = generate_finger_joint_edge(
                edge_length_mm=junction_height,
                spar_thickness_mm=plan.spar_thickness_mm,
                config=bwb_joint_config,
                is_male=True,  # Body spar has protruding fingers
            )
            
            wing_joint = generate_finger_joint_edge(
                edge_length_mm=junction_height,
                spar_thickness_mm=plan.spar_thickness_mm,
                config=bwb_joint_config,
                is_male=False,  # Wing spar has receiving slots
            )
            
            # Store junction profiles in spar objects
            body_spar.junction_profile = body_joint
            wing_spar.junction_profile = wing_joint
            
            # Rebuild outlines with finger joint geometry integrated
            body_outline_with_joint = _build_spar_outline_with_finger_joint(
                body_outline, body_joint, junction_edge="right"
            )
            wing_outline_with_joint = _build_spar_outline_with_finger_joint(
                wing_outline, wing_joint, junction_edge="left"
            )
            
            # Update outlines
            body_spar = SparProfile(
                outline=body_outline_with_joint,
                spar_type=spar_type,
                spar_region="body",
                length_mm=cut_x_final,
                station_heights=body_stations,
                rib_notches=body_notches,
                junction_profile=body_joint,
                recommended_grain="spanwise",
            )
            
            wing_spar = SparProfile(
                outline=wing_outline_with_joint,
                spar_type=spar_type,
                spar_region="wing",
                length_mm=spar.length_mm - cut_x_final,
                station_heights=wing_stations,
                rib_notches=wing_notches,
                junction_profile=wing_joint,
                recommended_grain="spanwise",
            )
            
            print(f"[BWB Spar] Added finger joints at junction (height={junction_height:.1f}mm, {bwb_joint_config.finger_count} fingers)")
    
    print(f"[BWB Spar] Successfully split {spar_type} spar into Body ({body_spar.length_mm:.1f}mm) and Wing ({wing_spar.length_mm:.1f}mm)")
    
    return [body_spar, wing_spar]



# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_surface_z_at_x(x: np.ndarray, z: np.ndarray, x_target: float) -> Dict[str, float]:
    """
    Find upper and lower surface z-coordinates at a given x position.
    
    Airfoil profile is typically ordered: TE upper → LE → TE lower
    or similar. We need to separate upper and lower surfaces.
    
    Returns:
        Dict with 'upper', 'lower' z-values at x_target
    """
    # Find approximate midpoint (leading edge = min x)
    le_idx = np.argmin(x)
    
    # Upper surface: indices 0 to le_idx (going from TE to LE)
    x_upper = x[:le_idx + 1]
    z_upper = z[:le_idx + 1]
    
    # Lower surface: indices le_idx to end (going from LE to TE)
    x_lower = x[le_idx:]
    z_lower = z[le_idx:]
    
    # Interpolate (need to ensure x is monotonically increasing for interp)
    # Upper surface goes from high x to low x, so reverse for interpolation
    if len(x_upper) > 1:
        x_upper_sorted = x_upper[::-1]
        z_upper_sorted = z_upper[::-1]
        z_upper_at_x = float(np.interp(x_target, x_upper_sorted, z_upper_sorted))
    else:
        z_upper_at_x = float(z_upper[0]) if len(z_upper) > 0 else 0.0
    
    # Lower surface
    if len(x_lower) > 1:
        z_lower_at_x = float(np.interp(x_target, x_lower, z_lower))
    else:
        z_lower_at_x = float(z_lower[0]) if len(z_lower) > 0 else 0.0
    
    return {'upper': z_upper_at_x, 'lower': z_lower_at_x}


def _calculate_stringer_slots(
    x: np.ndarray,
    z: np.ndarray,
    front_spar_x: float,
    rear_spar_x: float,
    stringer_count: int,
    stringer_height_mm: float,
    stringer_thickness_mm: float,
    clearance_mm: float,
) -> List[StringerSlotInfo]:
    """
    Calculate stringer slot positions for a rib.
    
    Stringers are evenly distributed across the wingbox width between spars.
    Each stringer position has a slot on both upper and lower surfaces.
    """
    if stringer_count <= 0:
        return []
    
    slots = []
    box_width = rear_spar_x - front_spar_x
    spacing = box_width / (stringer_count + 1)
    
    slot_width = stringer_thickness_mm + 2 * clearance_mm
    slot_height = stringer_height_mm + clearance_mm
    
    for i in range(stringer_count):
        x_center = front_spar_x + spacing * (i + 1)
        z_at_x = _get_surface_z_at_x(x, z, x_center)
        
        # Upper surface slot
        slots.append(StringerSlotInfo(
            x_center_mm=x_center,
            surface="upper",
            slot_width_mm=slot_width,
            slot_height_mm=slot_height,
            z_surface_mm=z_at_x['upper'],
        ))
        
        # Lower surface slot
        slots.append(StringerSlotInfo(
            x_center_mm=x_center,
            surface="lower",
            slot_width_mm=slot_width,
            slot_height_mm=slot_height,
            z_surface_mm=z_at_x['lower'],
        ))
    
    return slots


def _calculate_lightening_holes(
    x: np.ndarray,
    z: np.ndarray,
    front_spar_x: float,
    rear_spar_x: float,
    margin_mm: float,
    shape: str,
    min_hole_diameter_mm: float,
    lightening_fraction: float,
    stringer_slots: Optional[List[StringerSlotInfo]] = None,
    collision_clearance_mm: float = 2.0,
    twist_deg: float = 0.0,
) -> List[LighteningHoleInfo]:
    """
    Calculate lightening hole positions and sizes with collision detection.
    
    Holes are placed in the region between spars, avoiding edges
    and other features by the specified margin.
    
    Collision detection ensures holes don't intersect with:
    - Airfoil surfaces (upper and lower skin)
    - Stringer slots
    
    If collision is detected, holes are shifted first. If shifting isn't
    possible, holes are reduced in size. Holes are only removed if they
    would be smaller than min_hole_diameter_mm after adjustment.
    
    For "circular" shape: Multiple circles are distributed across the available width.
    For "elliptical" shape: A single large ellipse fills the available space.
    The ellipse major axis follows the wing twist if twist_deg is provided.
    
    Args:
        x: Airfoil x-coordinates (chordwise)
        z: Airfoil z-coordinates (thickness direction)
        front_spar_x: Front spar chordwise position
        rear_spar_x: Rear spar chordwise position
        margin_mm: Minimum margin from edges and features
        shape: "circular" or "elliptical"
        min_hole_diameter_mm: Minimum allowable hole size
        lightening_fraction: Target material removal fraction (for reference)
        stringer_slots: List of stringer slots to avoid
        collision_clearance_mm: Extra clearance around obstacles
        twist_deg: Wing twist at this section (to align elliptical major axis)
    """
    # If twisted, we untwist the airfoil and stringer slots to calculate holes
    # in the "straight" chord-aligned frame, then twist the resulting hole centers
    # back. This ensures elliptical holes optimally follow the wing twist.
    is_twisted = abs(twist_deg) > 1e-6
    
    if is_twisted:
        # Pivot is at leading edge (x=0 in local normalized coords)
        # Apply inverse twist to airfoil coordinates
        x, z = apply_twist_to_2d_coords(x, z, -twist_deg, pivot_x=0.0, pivot_z=0.0)
        
        # Also untwist stringer slot centers for collision detection
        if stringer_slots:
            new_slots = []
            for slot in stringer_slots:
                sx_rot, sz_rot = apply_twist_to_2d_coords(
                    np.array([slot.x_center_mm]), 
                    np.array([slot.z_surface_mm]), 
                    -twist_deg, pivot_x=0.0, pivot_z=0.0
                )
                # Note: we also need to adjust z_surface_mm to match untwisted surface
                new_slots.append(StringerSlotInfo(
                    x_center_mm=float(sx_rot[0]),
                    surface=slot.surface,
                    slot_width_mm=slot.slot_width_mm,
                    slot_height_mm=slot.slot_height_mm,
                    z_surface_mm=float(sz_rot[0])
                ))
            stringer_slots = new_slots

    # Available region for holes (between spars with margin)

    x_min = front_spar_x + margin_mm
    x_max = rear_spar_x - margin_mm
    
    available_width = x_max - x_min
    
    if available_width < min_hole_diameter_mm:
        return []  # Not enough horizontal space for holes
    
    # Build a function to get airfoil Z bounds at any X position
    def get_airfoil_bounds_at_x(x_pos: float) -> Tuple[float, float]:
        """Get (z_lower, z_upper) at a given x position."""
        bounds = _get_surface_z_at_x(x, z, x_pos)
        return bounds['lower'], bounds['upper']
    
    # Calculate safe vertical bounds accounting for airfoil curvature
    # Sample multiple points across the hole region to find the most restrictive bounds
    def get_safe_vertical_bounds(hole_x_min: float, hole_x_max: float, 
                                  stringer_slots: Optional[List[StringerSlotInfo]],
                                  clearance: float) -> Tuple[float, float]:
        """
        Get safe Z bounds for a hole spanning from hole_x_min to hole_x_max.
        Accounts for airfoil curvature and stringer intrusion.
        """
        # Sample airfoil at multiple points across the hole width
        n_samples = max(5, int((hole_x_max - hole_x_min) / 5))  # Sample every ~5mm
        sample_xs = np.linspace(hole_x_min, hole_x_max, n_samples)
        
        z_lower_max = float('-inf')  # Most restrictive lower bound (highest lower surface)
        z_upper_min = float('inf')   # Most restrictive upper bound (lowest upper surface)
        
        for sample_x in sample_xs:
            z_lo, z_up = get_airfoil_bounds_at_x(sample_x)
            z_lower_max = max(z_lower_max, z_lo)
            z_upper_min = min(z_upper_min, z_up)
        
        # Apply margin from airfoil surfaces
        z_lower_safe = z_lower_max + margin_mm + clearance
        z_upper_safe = z_upper_min - margin_mm - clearance
        
        # Account for stringer slots that intrude into the rib
        if stringer_slots:
            for slot in stringer_slots:
                # Check if this stringer overlaps with our X range
                slot_x_min = slot.x_center_mm - slot.slot_width_mm / 2
                slot_x_max = slot.x_center_mm + slot.slot_width_mm / 2
                
                if slot_x_max >= hole_x_min and slot_x_min <= hole_x_max:
                    # Stringer overlaps with hole X range
                    if slot.surface == "upper":
                        # Upper stringer slot extends downward from upper surface
                        stringer_bottom = slot.z_surface_mm - slot.slot_height_mm - clearance
                        z_upper_safe = min(z_upper_safe, stringer_bottom)
                    else:  # lower
                        # Lower stringer slot extends upward from lower surface
                        stringer_top = slot.z_surface_mm + slot.slot_height_mm + clearance
                        z_lower_safe = max(z_lower_safe, stringer_top)
        
        return z_lower_safe, z_upper_safe
    
    holes = []
    
    if shape in ("elliptical", "oval"):
        # Single large ellipse filling the available space
        cx = x_min + available_width / 2
        
        # Get safe bounds for the full ellipse width
        z_lower_safe, z_upper_safe = get_safe_vertical_bounds(
            x_min, x_max, stringer_slots, collision_clearance_mm
        )
        
        available_height = z_upper_safe - z_lower_safe
        
        if available_height < min_hole_diameter_mm:
            return []  # Not enough vertical space after accounting for obstacles
        
        cz = (z_lower_safe + z_upper_safe) / 2
        
        holes.append(LighteningHoleInfo(
            cx_mm=cx,
            cz_mm=cz,
            size_mm=available_height,  # Full available height for the ellipse
            shape=shape,
            width_mm=available_width,  # Full available width
        ))
    else:
        # Circular holes: calculate optimal size and count
        # First pass: determine hole size based on simple bounds
        z_min_global = float(z.min())
        z_max_global = float(z.max())
        simple_height = (z_max_global - z_min_global) - 2 * margin_mm
        
        # Rule of thumb: holes should be 60-70% of available height
        hole_size = min(simple_height * 0.65, available_width * 0.4)
        hole_size = max(hole_size, min_hole_diameter_mm)
        
        # Number of holes that fit
        n_holes = int(available_width / (hole_size + margin_mm))
        n_holes = max(1, n_holes)
        
        # Calculate positions
        if n_holes == 1:
            candidate_positions = [x_min + available_width / 2]
        else:
            spacing = available_width / n_holes
            candidate_positions = [x_min + spacing * (i + 0.5) for i in range(n_holes)]
        
        # Process each candidate hole with collision detection
        for cx in candidate_positions:
            # Define hole bounds
            hole_x_min = cx - hole_size / 2
            hole_x_max = cx + hole_size / 2
            
            # Get safe vertical bounds for this specific hole position
            z_lower_safe, z_upper_safe = get_safe_vertical_bounds(
                hole_x_min, hole_x_max, stringer_slots, collision_clearance_mm
            )
            
            available_height = z_upper_safe - z_lower_safe
            
            if available_height < min_hole_diameter_mm:
                # Try shifting the hole horizontally to avoid stringers
                shifted_hole = _try_shift_hole_to_avoid_stringers(
                    cx, hole_size, x_min, x_max, stringer_slots, 
                    x, z, margin_mm, collision_clearance_mm, min_hole_diameter_mm,
                    existing_holes=holes,  # Pass existing holes to avoid overlap
                )
                if shifted_hole is not None:
                    # Double-check no overlap with existing holes
                    if not _hole_overlaps_existing(shifted_hole, holes, margin_mm):
                        holes.append(shifted_hole)
                # Otherwise, skip this hole - not enough space
                continue
            
            # Check if hole needs to be reduced to fit
            actual_size = min(hole_size, available_height)
            
            if actual_size < min_hole_diameter_mm:
                continue  # Skip holes that are too small
            
            cz = (z_lower_safe + z_upper_safe) / 2
            
            candidate_hole = LighteningHoleInfo(
                cx_mm=cx,
                cz_mm=cz,
                size_mm=actual_size,
                shape=shape,
            )
            
            # Check for overlap with existing holes before adding
            if not _hole_overlaps_existing(candidate_hole, holes, margin_mm):
                holes.append(candidate_hole)
    
    # If we untwisted everything at the start, we MUST twist the results back
    if is_twisted:
        for hole in holes:
            # Rotate center back to twisted frame
            cx_rot, cz_rot = apply_twist_to_2d_coords(
                np.array([hole.cx_mm]), np.array([hole.cz_mm]),
                twist_deg, pivot_x=0.0, pivot_z=0.0
            )
            hole.cx_mm = float(cx_rot[0])
            hole.cz_mm = float(cz_rot[0])
            
            # Set rotation to match twist so major axis follows the wing
            if hole.shape in ("elliptical", "oval"):
                hole.rotation_deg = twist_deg
                
    return holes



def _hole_overlaps_existing(
    new_hole: LighteningHoleInfo,
    existing_holes: List[LighteningHoleInfo],
    min_spacing_mm: float,
) -> bool:
    """
    Check if a new hole overlaps with any existing holes.
    
    For circular holes, checks if the distance between centers is less than
    the sum of radii plus minimum spacing.
    
    For elliptical holes, uses bounding box overlap check.
    
    Args:
        new_hole: The hole to check
        existing_holes: List of already-placed holes
        min_spacing_mm: Minimum spacing between holes
    
    Returns:
        True if the new hole overlaps with any existing hole
    """
    if not existing_holes:
        return False
    
    for existing in existing_holes:
        if new_hole.shape == "circular" and existing.shape == "circular":
            # Circle-circle overlap: distance between centers < sum of radii + spacing
            dx = new_hole.cx_mm - existing.cx_mm
            dz = new_hole.cz_mm - existing.cz_mm
            distance = (dx**2 + dz**2) ** 0.5
            min_distance = (new_hole.size_mm / 2) + (existing.size_mm / 2) + min_spacing_mm
            if distance < min_distance:
                return True
        else:
            # Use bounding box overlap for elliptical or mixed cases
            new_half_w = (new_hole.width_mm or new_hole.size_mm) / 2
            new_half_h = new_hole.size_mm / 2
            exist_half_w = (existing.width_mm or existing.size_mm) / 2
            exist_half_h = existing.size_mm / 2
            
            # Check X overlap
            x_overlap = (abs(new_hole.cx_mm - existing.cx_mm) < 
                        (new_half_w + exist_half_w + min_spacing_mm))
            # Check Z overlap
            z_overlap = (abs(new_hole.cz_mm - existing.cz_mm) < 
                        (new_half_h + exist_half_h + min_spacing_mm))
            
            if x_overlap and z_overlap:
                return True
    
    return False


def _try_shift_hole_to_avoid_stringers(
    original_cx: float,
    hole_size: float,
    x_min: float,
    x_max: float,
    stringer_slots: Optional[List[StringerSlotInfo]],
    x: np.ndarray,
    z: np.ndarray,
    margin_mm: float,
    clearance_mm: float,
    min_hole_diameter_mm: float,
    existing_holes: Optional[List[LighteningHoleInfo]] = None,
) -> Optional[LighteningHoleInfo]:
    """
    Try to shift a hole horizontally to avoid stringer collisions and existing holes.
    
    Attempts shifts in both directions, preferring smaller shifts.
    Returns None if no valid position is found.
    
    Args:
        original_cx: Original X center position
        hole_size: Desired hole diameter
        x_min, x_max: Bounds for hole placement
        stringer_slots: Stringer slots to avoid
        x, z: Airfoil coordinates
        margin_mm: Margin from edges
        clearance_mm: Clearance around obstacles
        min_hole_diameter_mm: Minimum allowable hole size
        existing_holes: Already-placed holes to avoid overlapping
    """
    if stringer_slots is None or len(stringer_slots) == 0:
        return None
    
    # Get stringer X positions (they come in pairs for upper/lower)
    stringer_x_positions = sorted(set(s.x_center_mm for s in stringer_slots))
    
    # Find the stringer(s) that the original position conflicts with
    conflicting_stringers = []
    for slot in stringer_slots:
        slot_x_min = slot.x_center_mm - slot.slot_width_mm / 2 - clearance_mm
        slot_x_max = slot.x_center_mm + slot.slot_width_mm / 2 + clearance_mm
        hole_x_min = original_cx - hole_size / 2
        hole_x_max = original_cx + hole_size / 2
        
        if hole_x_max > slot_x_min and hole_x_min < slot_x_max:
            conflicting_stringers.append(slot)
    
    if not conflicting_stringers:
        return None  # No stringer conflict, problem is elsewhere
    
    # Try shifting to positions between stringers
    # Build list of "gaps" between stringers
    gaps = []
    
    # Gap before first stringer
    if stringer_x_positions and stringer_x_positions[0] > x_min + hole_size / 2:
        gap_center = (x_min + stringer_x_positions[0]) / 2
        gaps.append(gap_center)
    
    # Gaps between stringers
    for i in range(len(stringer_x_positions) - 1):
        gap_center = (stringer_x_positions[i] + stringer_x_positions[i + 1]) / 2
        gaps.append(gap_center)
    
    # Gap after last stringer
    if stringer_x_positions and stringer_x_positions[-1] < x_max - hole_size / 2:
        gap_center = (stringer_x_positions[-1] + x_max) / 2
        gaps.append(gap_center)
    
    # Sort gaps by distance from original position (prefer smaller shifts)
    gaps.sort(key=lambda g: abs(g - original_cx))
    
    # Try each gap position
    for gap_cx in gaps:
        # Check bounds
        if gap_cx - hole_size / 2 < x_min or gap_cx + hole_size / 2 > x_max:
            continue
        
        # Check if this position works (no stringer collision)
        hole_x_min = gap_cx - hole_size / 2
        hole_x_max = gap_cx + hole_size / 2
        
        has_collision = False
        for slot in stringer_slots:
            slot_x_min = slot.x_center_mm - slot.slot_width_mm / 2 - clearance_mm
            slot_x_max = slot.x_center_mm + slot.slot_width_mm / 2 + clearance_mm
            if hole_x_max > slot_x_min and hole_x_min < slot_x_max:
                has_collision = True
                break
        
        if has_collision:
            continue
        
        # Check vertical clearance at this position
        z_lower_safe, z_upper_safe = _get_safe_vertical_bounds_simple(
            hole_x_min, hole_x_max, stringer_slots, x, z, margin_mm, clearance_mm
        )
        
        available_height = z_upper_safe - z_lower_safe
        actual_size = min(hole_size, available_height)
        
        if actual_size >= min_hole_diameter_mm:
            cz = (z_lower_safe + z_upper_safe) / 2
            candidate_hole = LighteningHoleInfo(
                cx_mm=gap_cx,
                cz_mm=cz,
                size_mm=actual_size,
                shape="circular",
            )
            
            # Check for overlap with existing holes
            if existing_holes and _hole_overlaps_existing(candidate_hole, existing_holes, margin_mm):
                continue  # This position overlaps with an existing hole, try next gap
            
            return candidate_hole
    
    return None  # No valid position found


def _get_safe_vertical_bounds_simple(
    hole_x_min: float,
    hole_x_max: float,
    stringer_slots: Optional[List[StringerSlotInfo]],
    x: np.ndarray,
    z: np.ndarray,
    margin_mm: float,
    clearance_mm: float,
) -> Tuple[float, float]:
    """
    Simplified version of safe vertical bounds calculation for use in shifting logic.
    """
    # Sample airfoil at multiple points
    n_samples = max(3, int((hole_x_max - hole_x_min) / 10))
    sample_xs = np.linspace(hole_x_min, hole_x_max, n_samples)
    
    z_lower_max = float('-inf')
    z_upper_min = float('inf')
    
    for sample_x in sample_xs:
        bounds = _get_surface_z_at_x(x, z, sample_x)
        z_lower_max = max(z_lower_max, bounds['lower'])
        z_upper_min = min(z_upper_min, bounds['upper'])
    
    z_lower_safe = z_lower_max + margin_mm + clearance_mm
    z_upper_safe = z_upper_min - margin_mm - clearance_mm
    
    # Account for stringers
    if stringer_slots:
        for slot in stringer_slots:
            slot_x_min = slot.x_center_mm - slot.slot_width_mm / 2
            slot_x_max = slot.x_center_mm + slot.slot_width_mm / 2
            
            if slot_x_max >= hole_x_min and slot_x_min <= hole_x_max:
                if slot.surface == "upper":
                    stringer_bottom = slot.z_surface_mm - slot.slot_height_mm - clearance_mm
                    z_upper_safe = min(z_upper_safe, stringer_bottom)
                else:
                    stringer_top = slot.z_surface_mm + slot.slot_height_mm + clearance_mm
                    z_lower_safe = max(z_lower_safe, stringer_top)
    
    return z_lower_safe, z_upper_safe


def _truncate_airfoil_at_hinge(
    x: np.ndarray,
    z: np.ndarray,
    hinge_x_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate airfoil at the hinge line for elevon cutout.
    
    Everything aft of the hinge line is removed. The profile is closed
    with a vertical line at the hinge position.
    
    Args:
        x: X coordinates of airfoil (chordwise)
        z: Z coordinates of airfoil (thickness)
        hinge_x_mm: X position of hinge line
    
    Returns:
        Tuple of (new_x, new_z) arrays with truncated profile
    """
    # Find leading edge
    le_idx = int(np.argmin(x))
    
    # Separate upper and lower surfaces
    x_upper = x[:le_idx + 1]
    z_upper = z[:le_idx + 1]
    x_lower = x[le_idx:]
    z_lower = z[le_idx:]
    
    # Find where hinge intersects upper surface (going from TE to LE)
    # Upper surface x decreases from TE to LE
    upper_truncate_idx = None
    for i in range(len(x_upper) - 1):
        if x_upper[i] >= hinge_x_mm >= x_upper[i + 1]:
            upper_truncate_idx = i
            break
    
    # Find where hinge intersects lower surface (going from LE to TE)
    # Lower surface x increases from LE to TE
    lower_truncate_idx = None
    for i in range(len(x_lower) - 1):
        if x_lower[i] <= hinge_x_mm <= x_lower[i + 1]:
            lower_truncate_idx = i + 1
            break
    
    if upper_truncate_idx is None or lower_truncate_idx is None:
        # Hinge is outside airfoil or at edge - return original
        return x, z
    
    # Interpolate Z at hinge position for upper surface
    z_upper_at_hinge = float(np.interp(hinge_x_mm, x_upper[::-1], z_upper[::-1]))
    
    # Interpolate Z at hinge position for lower surface
    z_lower_at_hinge = float(np.interp(hinge_x_mm, x_lower, z_lower))
    
    # Build truncated profile:
    # Start at hinge on upper surface, go to LE, then to hinge on lower surface
    new_x = []
    new_z = []
    
    # Hinge point on upper surface
    new_x.append(hinge_x_mm)
    new_z.append(z_upper_at_hinge)
    
    # Upper surface from hinge to LE
    for i in range(upper_truncate_idx + 1, len(x_upper)):
        new_x.append(float(x_upper[i]))
        new_z.append(float(z_upper[i]))
    
    # Lower surface from LE to hinge
    for i in range(1, lower_truncate_idx):
        new_x.append(float(x_lower[i]))
        new_z.append(float(z_lower[i]))
    
    # Hinge point on lower surface (closes the profile)
    new_x.append(hinge_x_mm)
    new_z.append(z_lower_at_hinge)
    
    return np.array(new_x), np.array(new_z)


def _extract_aft_airfoil_portion(
    x: np.ndarray,
    z: np.ndarray,
    hinge_x_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the aft portion of an airfoil (from hinge to trailing edge).
    
    This is the complement of _truncate_airfoil_at_hinge - it returns
    the portion that was cut away, for generating elevon ribs.
    
    Args:
        x: X coordinates of airfoil (chordwise)
        z: Z coordinates of airfoil (thickness)
        hinge_x_mm: X position of hinge line
    
    Returns:
        Tuple of (new_x, new_z) arrays with aft portion profile
    """
    # Find leading edge
    le_idx = int(np.argmin(x))
    
    # Separate upper and lower surfaces
    x_upper = x[:le_idx + 1]
    z_upper = z[:le_idx + 1]
    x_lower = x[le_idx:]
    z_lower = z[le_idx:]
    
    # Find where hinge intersects upper surface (going from TE to LE)
    # Upper surface x decreases from TE to LE
    upper_hinge_idx = None
    for i in range(len(x_upper) - 1):
        if x_upper[i] >= hinge_x_mm >= x_upper[i + 1]:
            upper_hinge_idx = i
            break
    
    # Find where hinge intersects lower surface (going from LE to TE)
    # Lower surface x increases from LE to TE
    lower_hinge_idx = None
    for i in range(len(x_lower) - 1):
        if x_lower[i] <= hinge_x_mm <= x_lower[i + 1]:
            lower_hinge_idx = i + 1
            break
    
    if upper_hinge_idx is None or lower_hinge_idx is None:
        # Hinge is outside airfoil - return empty
        return np.array([]), np.array([])
    
    # Interpolate Z at hinge position for both surfaces
    z_upper_at_hinge = float(np.interp(hinge_x_mm, x_upper[::-1], z_upper[::-1]))
    z_lower_at_hinge = float(np.interp(hinge_x_mm, x_lower, z_lower))
    
    # Build aft profile:
    # Start at hinge on upper surface, go to TE, then back on lower surface to hinge
    new_x = []
    new_z = []
    
    # Hinge point on upper surface
    new_x.append(hinge_x_mm)
    new_z.append(z_upper_at_hinge)
    
    # Upper surface from hinge to TE (indices 0 to upper_hinge_idx)
    for i in range(upper_hinge_idx, -1, -1):
        new_x.append(float(x_upper[i]))
        new_z.append(float(z_upper[i]))
    
    # Lower surface from TE to hinge
    for i in range(len(x_lower) - 1, lower_hinge_idx - 1, -1):
        new_x.append(float(x_lower[i]))
        new_z.append(float(z_lower[i]))
    
    # Hinge point on lower surface (closes the profile)
    new_x.append(hinge_x_mm)
    new_z.append(z_lower_at_hinge)
    
    return np.array(new_x), np.array(new_z)


def _extract_aft_airfoil_portion_with_vnotch(
    x: np.ndarray,
    z: np.ndarray,
    hinge_x_mm: float,
    max_deflection_deg: float = 30.0,
    hinge_gap_mm: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the aft portion of an airfoil with a V-notch for deflection clearance.
    
    The V-notch is cut at the hinge line to allow the elevon to rotate
    without interference. The notch angle matches the max deflection angle.
    
    Args:
        x: X coordinates of airfoil (chordwise)
        z: Z coordinates of airfoil (thickness)
        hinge_x_mm: X position of hinge line
        max_deflection_deg: Maximum deflection angle (determines V-notch angle)
        hinge_gap_mm: Gap at the hinge line between main rib and elevon rib
    
    Returns:
        Tuple of (new_x, new_z) arrays with aft portion profile including V-notch
    """
    # Find leading edge and trailing edge X positions
    le_idx = int(np.argmin(x))
    te_x = float(x[0])  # Trailing edge is typically at index 0
    
    # Separate upper and lower surfaces
    x_upper = x[:le_idx + 1]
    z_upper = z[:le_idx + 1]
    x_lower = x[le_idx:]
    z_lower = z[le_idx:]
    
    # Find where hinge intersects surfaces
    upper_hinge_idx = None
    for i in range(len(x_upper) - 1):
        if x_upper[i] >= hinge_x_mm >= x_upper[i + 1]:
            upper_hinge_idx = i
            break
    
    lower_hinge_idx = None
    for i in range(len(x_lower) - 1):
        if x_lower[i] <= hinge_x_mm <= x_lower[i + 1]:
            lower_hinge_idx = i + 1
            break
    
    if upper_hinge_idx is None or lower_hinge_idx is None:
        return np.array([]), np.array([])
    
    # Interpolate Z at hinge position for both surfaces
    z_upper_at_hinge = float(np.interp(hinge_x_mm, x_upper[::-1], z_upper[::-1]))
    z_lower_at_hinge = float(np.interp(hinge_x_mm, x_lower, z_lower))
    
    # Calculate hinge center (pivot point) - typically at 50% thickness
    hinge_z_center = (z_upper_at_hinge + z_lower_at_hinge) / 2.0
    thickness_at_hinge = z_upper_at_hinge - z_lower_at_hinge
    
    # Calculate V-notch geometry
    # The V-notch opens toward the LEADING EDGE (toward the main wing rib)
    angle_rad = math.radians(max_deflection_deg)
    half_thickness = thickness_at_hinge / 2.0
    notch_depth = half_thickness * math.tan(angle_rad) + hinge_gap_mm
    
    # V-notch geometry - apex points toward the main rib (forward/LE direction)
    apex_x = hinge_x_mm + hinge_gap_mm  # Apex closest to main rib
    corner_x = hinge_x_mm + hinge_gap_mm + notch_depth  # Corners are further aft
    
    # CRITICAL: Clamp corner_x to not exceed trailing edge
    corner_x = min(corner_x, te_x - 1.0)  # Keep at least 1mm from TE
    
    # Interpolate Z at the CORNER position (not hinge position) to avoid extrapolation
    # This ensures vertices are clamped to actual airfoil bounds
    z_upper_at_corner = float(np.interp(corner_x, x_upper[::-1], z_upper[::-1]))
    z_lower_at_corner = float(np.interp(corner_x, x_lower, z_lower))
    
    new_x = []
    new_z = []
    
    # Start at apex (front point of the V, at hinge center height)
    new_x.append(apex_x)
    new_z.append(hinge_z_center)
    
    # Upper corner of V-notch - use interpolated Z at corner_x
    new_x.append(corner_x)
    new_z.append(z_upper_at_corner)
    
    # Upper surface from corner to TE - only include points that are AFT of corner_x
    for i in range(upper_hinge_idx, -1, -1):
        if x_upper[i] > corner_x:
            new_x.append(float(x_upper[i]))
            new_z.append(float(z_upper[i]))
    
    # Lower surface from TE back toward corner - only include points AFT of corner_x
    for i in range(len(x_lower) - 1, lower_hinge_idx - 1, -1):
        if x_lower[i] > corner_x:
            new_x.append(float(x_lower[i]))
            new_z.append(float(z_lower[i]))
    
    # Lower corner of V-notch - use interpolated Z at corner_x
    new_x.append(corner_x)
    new_z.append(z_lower_at_corner)
    
    # Profile closes back to apex (implicitly when polyline is closed)
    
    return np.array(new_x), np.array(new_z)


def _build_rib_outline_with_features(
    x: np.ndarray,
    z: np.ndarray,
    spar_notches: List[SparNotchInfo],
    stringer_slots: List[StringerSlotInfo],
    elevon_cutout: Optional[ElevonCutoutInfo] = None,
    include_tabs: bool = False,
    tab_width_mm: float = 8.0,
    tab_depth_mm: float = 5.0,
) -> np.ndarray:
    """
    Build closed rib profile polyline with spar notches and elevon cutout.
    
    The notches are rectangular cutouts from the lower surface where
    the spar plates slot into the rib.
    
    If elevon_cutout is provided, the rib is truncated at the hinge line
    (everything aft of the hinge is cut away).
    
    Stringer slots are separate rectangular cutouts (not part of the main
    outline) and will be drawn as separate polylines in DXF.
    
    Returns:
        Nx2 array of (x, z) points forming closed polyline with notches
    """
    # Apply elevon cutout first - truncate the airfoil at hinge line
    if elevon_cutout and elevon_cutout.has_cutout:
        x, z = _truncate_airfoil_at_hinge(x, z, elevon_cutout.x_hinge_mm)
    
    if len(spar_notches) == 0:
        # No notches - return (possibly truncated) airfoil profile
        return np.column_stack((x, z))
    
    # Find leading edge index
    le_idx = int(np.argmin(x))
    
    # Separate upper and lower surfaces
    # Upper: 0 to le_idx (TE to LE)
    # Lower: le_idx to end (LE to TE)
    x_upper = x[:le_idx + 1].copy()
    z_upper = z[:le_idx + 1].copy()
    x_lower = x[le_idx:].copy()
    z_lower = z[le_idx:].copy()
    
    # Process lower surface to add notches
    # Sort notches by x position for proper insertion
    sorted_notches = sorted(spar_notches, key=lambda n: n.x_center_mm)
    
    # Build new lower surface with notches
    new_lower_points = []
    current_idx = 0
    
    for notch in sorted_notches:
        half_w = notch.notch_width_mm / 2
        notch_left = notch.x_center_mm - half_w
        notch_right = notch.x_center_mm + half_w
        notch_top_z = notch.z_lower_mm + notch.notch_depth_mm
        
        # Add points up to notch start
        while current_idx < len(x_lower) and x_lower[current_idx] < notch_left:
            new_lower_points.append((float(x_lower[current_idx]), float(z_lower[current_idx])))
            current_idx += 1
        
        # Interpolate z at notch left edge on original profile
        z_at_left = float(np.interp(notch_left, x_lower, z_lower))
        
        # Add point at notch left edge on original profile
        new_lower_points.append((notch_left, z_at_left))
        
        # Add notch geometry (going into the rib)
        # Down to notch bottom (original lower surface)
        # But actually, notch is cut FROM lower surface upward
        # So we go: left edge at surface -> left edge at notch top -> right edge at notch top -> right edge at surface
        
        # For a notch cut from below, the geometry is:
        # - Point on lower surface at left edge
        # - Point at (left, z_lower + notch_depth) - inside the rib
        # - Point at (right, z_lower + notch_depth) - inside the rib  
        # - Point on lower surface at right edge
        # But since we're CUTTING, we skip the interior and just connect the two edges
        # at the notch height, creating the rectangular cutout
        
        # The notch is a rectangular indentation from the lower surface upward
        new_lower_points.append((notch_left, notch_top_z))
        new_lower_points.append((notch_right, notch_top_z))
        
        # Interpolate z at notch right edge on original profile
        z_at_right = float(np.interp(notch_right, x_lower, z_lower))
        new_lower_points.append((notch_right, z_at_right))
        
        # Skip points inside notch region
        while current_idx < len(x_lower) and x_lower[current_idx] <= notch_right:
            current_idx += 1
    
    # Add remaining points after last notch
    while current_idx < len(x_lower):
        new_lower_points.append((float(x_lower[current_idx]), float(z_lower[current_idx])))
        current_idx += 1
    
    # Combine upper and modified lower surface
    # Upper goes TE -> LE, lower goes LE -> TE
    # Result should be: TE upper -> LE -> TE lower -> close back to TE upper
    upper_points = [(float(x_upper[i]), float(z_upper[i])) for i in range(len(x_upper))]
    
    # Note: The LE point is shared between upper and lower
    # We've included it in both, so skip first point of lower (it's the LE)
    if new_lower_points and len(upper_points) > 0:
        # Check if first point of lower matches last point of upper (LE)
        if len(new_lower_points) > 1:
            lower_without_le = new_lower_points[1:]
        else:
            lower_without_le = new_lower_points
        all_points = upper_points + lower_without_le
    else:
        all_points = upper_points + new_lower_points
    
    return np.array(all_points)


def _build_spar_outline_with_features(
    length_mm: float,
    height_mm: float,
    rib_notches: List[RibNotchInfo],
    include_tabs: bool = False,
    tab_width_mm: float = 8.0,
    tab_depth_mm: float = 5.0,
) -> np.ndarray:
    """
    Build closed spar profile polyline with rib notches cut from top.
    
    The spar is a rectangle with notches cut from the top edge where
    ribs slot into the spar.
    
    Returns:
        Nx2 array of (x, z) points forming closed polyline with notches
    """
    if len(rib_notches) == 0:
        # No notches - return simple rectangle
        return np.array([
            [0, 0],
            [length_mm, 0],
            [length_mm, height_mm],
            [0, height_mm],
        ])
    
    # Bottom edge (root to tip)
    bottom_points = [(0.0, 0.0), (length_mm, 0.0)]
    
    # Build top edge with notches (tip back to root)
    sorted_notches = sorted(rib_notches, key=lambda n: n.x_along_span_mm, reverse=True)
    
    top_points = []
    
    for notch in sorted_notches:
        half_w = notch.notch_width_mm / 2
        notch_right = notch.x_along_span_mm + half_w
        notch_left = notch.x_along_span_mm - half_w
        notch_bottom = height_mm - notch.notch_depth_mm
        
        # Clamp to spar bounds
        notch_right = min(length_mm, notch_right)
        notch_left = max(0, notch_left)
        
        if notch_left >= notch_right:
            continue
        
        # Add notch geometry (going down into spar)
        top_points.append((notch_right, height_mm))
        top_points.append((notch_right, notch_bottom))
        top_points.append((notch_left, notch_bottom))
        top_points.append((notch_left, height_mm))
    
    # Complete top edge
    top_points.append((0.0, height_mm))
    
    # Combine
    all_points = bottom_points + top_points
    
    return np.array(all_points)



def _build_tapered_spar_outline(
    station_heights: List[SparStationHeight],
    rib_notches: List[RibNotchInfo],
    include_tabs: bool = False,
    tab_width_mm: float = 8.0,
    tab_depth_mm: float = 5.0,
) -> np.ndarray:
    """
    Build closed spar profile that follows the airfoil wingbox envelope.
    
    The spar height varies along the span, matching the actual airfoil
    thickness at the spar location. This creates a tapered profile.
    
    Coordinate system:
    - X = spanwise (0 at root, increasing toward tip)
    - Z = height (lower surface to upper surface of wingbox)
    
    The profile is normalized so Z=0 is at the lowest point of the lower edge.
    
    When include_tabs=True:
    - Instead of rib notches (cuts into spar), we add tab protrusions
    - Tabs extend DOWNWARD from the lower edge at each rib station
    - Tabs will poke through enclosed slots in the ribs
    
    Args:
        station_heights: List of SparStationHeight with Z coordinates at each rib
        rib_notches: List of RibNotchInfo for rib slot positions (used for notches OR tab positions)
        include_tabs: Whether to add interlocking tabs instead of notches
        tab_width_mm: Width of tabs
        tab_depth_mm: Depth of tabs (how far they extend below lower edge)
    
    Returns:
        Nx2 array of (x, z) points forming closed polyline
    """
    if len(station_heights) < 2:
        raise ValueError("Need at least 2 stations to build tapered spar")
    
    # Sort by spanwise position
    sorted_stations = sorted(station_heights, key=lambda s: s.x_along_span_mm)
    
    # Normalize Z coordinates so the minimum lower surface is at Z=0
    z_lower_min = min(s.z_lower_mm for s in sorted_stations)
    
    # Build upper edge (root to tip) - follows upper surface
    upper_points = []
    
    # Interpolate upper Z at any x position
    x_positions = [s.x_along_span_mm for s in sorted_stations]
    z_upper_values = [s.z_upper_mm - z_lower_min for s in sorted_stations]
    
    def interp_upper_z(x_val: float) -> float:
        return float(np.interp(x_val, x_positions, z_upper_values))

    if not include_tabs:
        # NOTCH MODE: Add rib notches cutting INTO the spar from above
        # Sort notches by x position for proper insertion
        sorted_notches = sorted(rib_notches, key=lambda n: n.x_along_span_mm)
        current_notch_idx = 0
        length_mm = sorted_stations[-1].x_along_span_mm
        
        for station in sorted_stations:
            x_station = station.x_along_span_mm
            z_upper = station.z_upper_mm - z_lower_min
            
            # Check if this station has a notch
            matching_notch = None
            if current_notch_idx < len(sorted_notches):
                notch = sorted_notches[current_notch_idx]
                if abs(notch.x_along_span_mm - x_station) < 1e-3:
                    matching_notch = notch
                    current_notch_idx += 1
            
            if matching_notch:
                half_w = matching_notch.notch_width_mm / 2
                # Clamp notch boundaries to spar extent to ensure vertical edges at ends
                n_left = max(0.0, x_station - half_w)
                n_right = min(length_mm, x_station + half_w)
                
                if n_left >= n_right:
                    upper_points.append((x_station, z_upper))
                    continue
                    
                n_bottom = z_upper - matching_notch.notch_depth_mm
                
                # Interpolate upper Z at notch edges for precision
                z_left = interp_upper_z(n_left)
                z_right = interp_upper_z(n_right)
                
                # Add notch points (going left to right along upper edge)
                upper_points.append((n_left, z_left))
                upper_points.append((n_left, n_bottom))
                upper_points.append((n_right, n_bottom))
                upper_points.append((n_right, z_right))
            else:
                upper_points.append((x_station, z_upper))
    else:
        # TAB MODE or original simple upper edge
        for station in sorted_stations:
            z_upper_normalized = station.z_upper_mm - z_lower_min
            upper_points.append((station.x_along_span_mm, z_upper_normalized))
    
    # Interpolate lower Z at any x position
    z_lower_values = [s.z_lower_mm - z_lower_min for s in sorted_stations]
    
    def interp_lower_z(x: float) -> float:
        return float(np.interp(x, x_positions, z_lower_values))
    
    # Build lower edge points (going from tip back to root)
    lower_points = []
    
    # Start at tip
    tip_x = sorted_stations[-1].x_along_span_mm
    tip_z_lower = interp_lower_z(tip_x)
    lower_points.append((tip_x, tip_z_lower))
    
    if include_tabs:
        # TAB MODE: Add tab protrusions extending BELOW the lower edge
        # Tabs are at each rib station position (use rib_notches for positions)
        sorted_notches = sorted(rib_notches, key=lambda n: n.x_along_span_mm, reverse=True) if rib_notches else []
        
        for notch in sorted_notches:
            half_w = tab_width_mm / 2
            tab_right = notch.x_along_span_mm + half_w
            tab_left = notch.x_along_span_mm - half_w
            
            # Clamp to spar bounds
            tab_right = min(tab_right, tip_x)
            tab_left = max(tab_left, 0)
            
            if tab_left >= tab_right:
                continue
            
            # Get lower surface Z at tab edges
            z_at_right = interp_lower_z(tab_right)
            z_at_left = interp_lower_z(tab_left)
            z_at_center = interp_lower_z(notch.x_along_span_mm)
            
            # Tab extends BELOW the lower edge (negative Z)
            tab_bottom = z_at_center - tab_depth_mm
            
            # Add points for tab protrusion (going right to left)
            # Right edge on lower surface -> down to tab bottom -> across -> up to left edge
            lower_points.append((tab_right, z_at_right))
            lower_points.append((tab_right, tab_bottom))
            lower_points.append((tab_left, tab_bottom))
            lower_points.append((tab_left, z_at_left))
    else:
        # NOTCH MODE: Lower edge is now simple since notches moved to upper surface
        for station in reversed(sorted_stations):
            lower_points.append((station.x_along_span_mm, station.z_lower_mm - z_lower_min))

    
    # End at root
    root_z_lower = interp_lower_z(0)
    lower_points.append((0, root_z_lower))
    
    # Combine: upper edge (root to tip) + lower edge (tip to root)
    all_points = upper_points + lower_points
    
    return np.array(all_points)


def _build_spar_outline_with_finger_joint(
    base_outline: np.ndarray,
    joint_profile: 'FingerJointProfile',
    junction_edge: str,  # "left" or "right"
) -> np.ndarray:
    """
    Build spar outline with finger joint geometry integrated at junction edge.
    
    For body spar (junction_edge="right"): fingers protrude from right edge
    For wing spar (junction_edge="left"): slots cut into left edge
    
    Args:
        base_outline: Original Nx2 spar outline
        joint_profile: FingerJointProfile with edge points
        junction_edge: "left" or "right" - which edge to modify
    
    Returns:
        Modified Nx2 outline with finger joint geometry
    """
    if joint_profile is None or len(joint_profile.edge_points) < 2:
        return base_outline
    
    # Convert base outline to list for manipulation
    points = [tuple(pt) for pt in base_outline]
    
    # Find the vertical edge to replace (left or right side of spar)
    x_coords = base_outline[:, 0]
    
    if junction_edge == "right":
        # Right edge: find max X position
        edge_x = float(np.max(x_coords))
        # Find points on the right edge (within tolerance)
        edge_mask = np.abs(x_coords - edge_x) < 1.0
    else:  # left
        # Left edge: find min X position  
        edge_x = float(np.min(x_coords))
        # Find points on the left edge (within tolerance)
        edge_mask = np.abs(x_coords - edge_x) < 1.0
    
    # Get Z range on this edge
    edge_z = base_outline[edge_mask, 1]
    if len(edge_z) == 0:
        return base_outline
    z_min = float(np.min(edge_z))
    z_max = float(np.max(edge_z))
    
    # Transform finger joint points to spar coordinates
    # Joint profile: X is outward from edge, Z is along edge height
    joint_pts = joint_profile.edge_points
    
    # Scale Z to match edge height
    edge_height = z_max - z_min
    joint_height = float(np.max(joint_pts[:, 1]) - np.min(joint_pts[:, 1]))
    if joint_height > 0:
        z_scale = edge_height / joint_height
    else:
        z_scale = 1.0
    
    # Build new edge points from finger joint
    new_edge_points = []
    for jx, jz in joint_pts:
        # Scale and offset Z
        new_z = z_min + jz * z_scale
        # Transform X based on edge side
        if junction_edge == "right":
            new_x = edge_x + jx  # Positive X = outward (male) or inward (female)
        else:  # left
            new_x = edge_x - jx  # Negative X = outward (male) or inward (female)
        new_edge_points.append((new_x, new_z))
    
    # Reconstruct outline with finger joint
    # Strategy: Remove points on the target edge and insert finger joint points
    new_points = []
    in_edge_region = False
    edge_inserted = False
    
    for i, pt in enumerate(points):
        x, z = pt
        on_edge = abs(x - edge_x) < 1.0
        
        if on_edge and not edge_inserted:
            # We're at the edge - insert finger joint points instead
            if junction_edge == "right":
                # For right edge, insert bottom to top
                for edge_pt in new_edge_points:
                    new_points.append(edge_pt)
            else:
                # For left edge, insert top to bottom (reverse order)
                for edge_pt in reversed(new_edge_points):
                    new_points.append(edge_pt)
            edge_inserted = True
            in_edge_region = True
        elif on_edge and in_edge_region:
            # Skip this point - already replaced by finger joint
            continue
        else:
            # Not on edge - keep this point
            new_points.append(pt)
            in_edge_region = False
    
    return np.array(new_points)


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_stringer_slot_polylines(
    stringer_slots: List[StringerSlotInfo],
) -> List[np.ndarray]:
    """
    Convert stringer slots to separate polylines for DXF export.
    
    Each slot is a rectangular cutout that needs to be drawn as a 
    separate closed polyline (not part of the main rib outline).
    
    Returns:
        List of Nx2 arrays, each representing a closed rectangular slot
    """
    polylines = []
    
    for slot in stringer_slots:
        half_w = slot.slot_width_mm / 2
        
        if slot.surface == "upper":
            # Slot extends inward (downward) from upper surface
            z_outer = slot.z_surface_mm
            z_inner = slot.z_surface_mm - slot.slot_height_mm
        else:
            # Slot extends inward (upward) from lower surface
            z_outer = slot.z_surface_mm
            z_inner = slot.z_surface_mm + slot.slot_height_mm
        
        rect = np.array([
            [slot.x_center_mm - half_w, z_outer],
            [slot.x_center_mm - half_w, z_inner],
            [slot.x_center_mm + half_w, z_inner],
            [slot.x_center_mm + half_w, z_outer],
        ])
        polylines.append(rect)
    
    return polylines


def get_tab_slot_polylines(
    tab_slots: List[TabSlotInfo],
) -> List[np.ndarray]:
    """
    Convert tab slots to separate polylines for DXF export.
    
    Each tab slot is an ENCLOSED rectangular hole inside the rib profile
    where spar tabs will pass through. These are drawn as separate closed
    polylines (not part of the main rib outline).
    
    Returns:
        List of Nx2 arrays, each representing a closed rectangular slot
    """
    polylines = []
    
    for slot in tab_slots:
        half_w = slot.slot_width_mm / 2
        half_h = slot.slot_height_mm / 2
        
        # Rectangular slot centered at (x_center, z_center)
        rect = np.array([
            [slot.x_center_mm - half_w, slot.z_center_mm - half_h],
            [slot.x_center_mm - half_w, slot.z_center_mm + half_h],
            [slot.x_center_mm + half_w, slot.z_center_mm + half_h],
            [slot.x_center_mm + half_w, slot.z_center_mm - half_h],
        ])
        polylines.append(rect)
    
    return polylines


def get_lightening_hole_geometries(
    holes: List[LighteningHoleInfo],
    corner_radius_mm: float = 3.0,
    n_arc_points: int = 32,
) -> List[Tuple[str, Any]]:
    """
    Convert lightening holes to DXF-compatible geometry.
    
    Returns:
        List of (type, data) tuples:
        - ("circle", (cx, cz, radius)) for circular holes
        - ("polyline", np.ndarray) for elliptical holes
    """
    geometries = []
    
    for hole in holes:
        if hole.shape == "circular":
            geometries.append(("circle", (hole.cx_mm, hole.cz_mm, hole.size_mm / 2)))
        
        elif hole.shape in ("elliptical", "oval"):
            # Generate elliptical polyline
            # size_mm is the height, width_mm is the width (if provided)
            h_radius = hole.size_mm / 2
            if hole.width_mm is not None:
                w_radius = hole.width_mm / 2
            else:
                # Fallback: use 1.4 aspect ratio if width not provided
                w_radius = h_radius * 1.4
            
            theta = np.linspace(0, 2*np.pi, n_arc_points, endpoint=False)
            pts_x = w_radius * np.cos(theta)
            pts_z = h_radius * np.sin(theta)
            
            # Apply rotation if needed
            if abs(hole.rotation_deg) > 1e-6:
                rot_rad = math.radians(hole.rotation_deg)
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                # Match the rotation logic in apply_twist_to_2d_coords (XZ plane)
                # For nose-up twist (positive angle), TE moves down (negative Z):
                #   x' = x * cos(θ) + z * sin(θ)
                #   z' = -x * sin(θ) + z * cos(θ)
                x_rot = pts_x * cos_r + pts_z * sin_r
                z_rot = -pts_x * sin_r + pts_z * cos_r
                pts_x, pts_z = x_rot, z_rot
                
            points = np.column_stack((hole.cx_mm + pts_x, hole.cz_mm + pts_z))
            geometries.append(("polyline", points))

    
    return geometries


# ==============================================================================
# Grain Direction Indicators
# ==============================================================================

# Grain direction recommendations for different parts
GRAIN_RECOMMENDATIONS = {
    "rib": "chordwise",      # Grain perpendicular to spar notches (stronger against splitting)
    "front_spar": "spanwise", # Grain along span for bending strength
    "rear_spar": "spanwise",
    "skin": "spanwise",       # Primary stress is spanwise bending
}


@dataclass
class GrainIndicator:
    """Grain direction indicator geometry for DXF export."""
    arrow_start: Tuple[float, float]  # (x, z) start of arrow
    arrow_end: Tuple[float, float]    # (x, z) end of arrow
    arrowhead_left: Tuple[float, float]
    arrowhead_right: Tuple[float, float]
    label_position: Tuple[float, float]
    label_text: str


def generate_grain_indicator(
    part_type: str,
    part_width: float,
    part_height: float,
    arrow_length_mm: float = 20.0,
) -> GrainIndicator:
    """
    Generate grain direction indicator for a part.
    
    Args:
        part_type: "rib", "front_spar", "rear_spar", or "skin"
        part_width: Width of the part (X direction) in mm
        part_height: Height of the part (Z direction) in mm
        arrow_length_mm: Length of the direction arrow
    
    Returns:
        GrainIndicator with arrow and label geometry
    """
    import math
    
    direction = GRAIN_RECOMMENDATIONS.get(part_type.lower(), "spanwise")
    
    # Center position
    cx = part_width * 0.5
    cz = part_height * 0.5
    
    # Arrow angle based on direction
    # For ribs: X = chordwise (LE to TE), so "chordwise" grain = horizontal arrow
    # For spars: X = spanwise (root to tip), so "spanwise" grain = horizontal arrow
    # Both chordwise and spanwise mean "along the X axis" in their respective coordinate systems
    if direction == "chordwise" or direction == "spanwise":
        angle_rad = 0  # Horizontal (along X axis)
    else:
        angle_rad = math.pi / 2  # Vertical (along Z axis)
    
    # Arrow endpoints
    half_len = arrow_length_mm / 2
    dx = half_len * math.cos(angle_rad)
    dz = half_len * math.sin(angle_rad)
    
    start = (cx - dx, cz - dz)
    end = (cx + dx, cz + dz)
    
    # Arrowhead
    head_len = arrow_length_mm * 0.2
    head_angle = math.radians(25)
    
    # Left barb
    left_angle = angle_rad + math.pi - head_angle
    left_pt = (end[0] + head_len * math.cos(left_angle),
               end[1] + head_len * math.sin(left_angle))
    
    # Right barb
    right_angle = angle_rad + math.pi + head_angle
    right_pt = (end[0] + head_len * math.cos(right_angle),
                end[1] + head_len * math.sin(right_angle))
    
    # Label position (below the arrow)
    label_pos = (cx, cz - arrow_length_mm * 0.6)
    label_text = f"GRAIN: {direction.upper()}"
    
    return GrainIndicator(
        arrow_start=start,
        arrow_end=end,
        arrowhead_left=left_pt,
        arrowhead_right=right_pt,
        label_position=label_pos,
        label_text=label_text,
    )


# ==============================================================================
# Fixture Profile Dataclasses
# ==============================================================================

@dataclass
class FixtureProfileParams:
    """Parameters for fixture profile generation."""
    material_thickness_mm: float = 6.35
    slot_clearance_mm: float = 0.15
    fixture_height_mm: float = 50.0      # Default height below spar to base plate
    tab_width_mm: float = 15.0
    tab_spacing_mm: float = 80.0
    tab_edge_margin_mm: float = 12.0
    base_plate_margin_mm: float = 20.0


@dataclass
class FixtureStationHeight:
    """Fixture height at a specific spanwise station."""
    x_along_span_mm: float      # Position along fixture (from root)
    z_chord_mm: float           # Chord line Z at spar position (top of fixture)
    z_lower_mm: float           # Lower surface Z at spar position
    z_base_mm: float            # Base plate level Z (bottom of fixture)
    
    @property
    def fixture_height_mm(self) -> float:
        """Height from base plate to chord line."""
        return self.z_chord_mm - self.z_base_mm


@dataclass
class FixtureTabInfo:
    """Information about a tab on a fixture."""
    x_along_span_mm: float      # Position along fixture
    width_mm: float             # Tab width (along span)
    depth_mm: float             # Tab depth (into base plate)


@dataclass
class FixtureProfile:
    """Generated fixture profile with metadata."""
    outline: np.ndarray                     # Nx2 closed polyline (mm) - X (spanwise), Z (height)
    fixture_side: str                       # "front" (toward LE) or "rear" (toward TE)
    spar_type: str                          # "front" or "rear" spar this supports
    length_mm: float                        # Total fixture length (span)
    station_heights: List[FixtureStationHeight] = field(default_factory=list)
    tabs: List[FixtureTabInfo] = field(default_factory=list)
    # Offset from spar centerline (for 3D positioning)
    offset_from_spar_mm: float = 0.0
    recommended_grain: str = "spanwise"


@dataclass
class CradleStationHeight:
    """Cradle height at a specific spanwise station."""
    x_along_span_mm: float      # Position along cradle
    z_lower_surface_mm: float   # Lower wing surface Z (top of cradle)
    z_base_mm: float            # Base plate level Z (bottom of cradle)
    
    @property
    def cradle_height_mm(self) -> float:
        """Height from base plate to lower surface."""
        return self.z_lower_surface_mm - self.z_base_mm


@dataclass
class CradleProfile:
    """Generated cradle profile (follows lower wing surface)."""
    outline: np.ndarray                     # Nx2 closed polyline (mm)
    spar_type: str                          # Which spar centerline this cradle follows
    length_mm: float
    station_heights: List[CradleStationHeight] = field(default_factory=list)
    recommended_grain: str = "spanwise"


# ==============================================================================
# Fixture Profile Generation Functions
# ==============================================================================

def generate_fixture_profile(
    project: Project,
    sections: List[SpanwiseSection],
    spar_type: str,        # "front" or "rear"
    fixture_side: str,     # "front" (toward LE) or "rear" (toward TE)
    params: FixtureProfileParams,
) -> FixtureProfile:
    """
    Generate 2D fixture profile that extends from base plate to chord line.
    
    The fixture is a flat plate positioned parallel to the spar, offset by:
    - Front fixture: spar front face - clearance - material_thickness/2
    - Rear fixture: spar back face + clearance + material_thickness/2
    
    Profile coordinates (same convention as SparProfile):
    - X = spanwise position (along spar line, in mm, from 0)
    - Z = height (from base plate to chord line)
    
    This is the SINGLE SOURCE OF TRUTH for fixture geometry.
    """
    plan = project.wing.planform
    
    if not sections:
        raise ValueError("No sections provided for fixture generation")
    
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    
    # Calculate spar line positions (same as spar profile generation)
    spar_positions = []
    for section in sorted_sections:
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x_m = section.x_le_m + spar_xsi * section.chord_m
        spar_positions.append((abs(section.y_m), spar_x_m))
    
    # Calculate cumulative distance along spar line (accounts for sweep)
    cumulative_distances = [0.0]
    for i in range(1, len(spar_positions)):
        y_prev, x_prev = spar_positions[i-1]
        y_curr, x_curr = spar_positions[i]
        dy = y_curr - y_prev
        dx = x_curr - x_prev
        segment_length = math.sqrt(dy*dy + dx*dx)
        cumulative_distances.append(cumulative_distances[-1] + segment_length)
    
    # Compute global base plate level (lowest spar bottom - fixture height)
    global_min_z_mm = float('inf')
    for section in sorted_sections:
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        coords = np.array(section.airfoil.coordinates)
        chord_mm = section.chord_m * 1000
        z_mm = coords[:, 1] * chord_mm + section.z_m * 1000
        global_min_z_mm = min(global_min_z_mm, z_mm.min())
    
    base_plate_z_mm = global_min_z_mm - params.fixture_height_mm
    
    # Build station heights
    station_heights = []
    for i, section in enumerate(sorted_sections):
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        
        coords = np.array(section.airfoil.coordinates)
        chord_mm = section.chord_m * 1000
        
        # Scale to physical dimensions with dihedral
        x_local = coords[:, 0] * chord_mm
        z_local = coords[:, 1] * chord_mm
        dihedral_offset_mm = section.z_m * 1000
        
        # Apply twist rotation
        if abs(section.twist_deg) > 1e-6:
            le_x = x_local.min()
            le_idx = np.argmin(x_local)
            le_z = z_local[le_idx]
            x_local, z_local = apply_twist_to_2d_coords(
                x_local, z_local, section.twist_deg, pivot_x=le_x, pivot_z=le_z
            )
        
        # Find spar position
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x_local = spar_xsi * chord_mm
        
        # Get Z at spar position
        z_at_spar = _get_surface_z_at_x(x_local, z_local, spar_x_local)
        
        # Chord line Z (upper surface at spar) with dihedral
        z_chord = z_at_spar['upper'] + dihedral_offset_mm
        z_lower = z_at_spar['lower'] + dihedral_offset_mm
        
        x_along_span_mm = cumulative_distances[i] * 1000
        
        station_heights.append(FixtureStationHeight(
            x_along_span_mm=x_along_span_mm,
            z_chord_mm=z_chord,
            z_lower_mm=z_lower,
            z_base_mm=base_plate_z_mm,
        ))
    
    if len(station_heights) < 2:
        raise ValueError("Need at least 2 valid stations for fixture generation")
    
    span_mm = station_heights[-1].x_along_span_mm
    
    # Calculate offset from spar centerline
    spar_half_mm = plan.spar_thickness_mm / 2.0
    if fixture_side == "front":
        # Front fixture: toward leading edge (negative normal direction)
        offset_from_spar_mm = -(spar_half_mm + params.slot_clearance_mm + params.material_thickness_mm / 2.0)
    else:
        # Rear fixture: toward trailing edge (positive normal direction)
        offset_from_spar_mm = spar_half_mm + params.slot_clearance_mm + params.material_thickness_mm / 2.0
    
    # Calculate tab positions
    tabs = _calculate_fixture_tabs(span_mm, params)
    
    # Build outline
    outline = _build_fixture_outline(station_heights, base_plate_z_mm)
    
    return FixtureProfile(
        outline=outline,
        fixture_side=fixture_side,
        spar_type=spar_type,
        length_mm=span_mm,
        station_heights=station_heights,
        tabs=tabs,
        offset_from_spar_mm=offset_from_spar_mm,
        recommended_grain="spanwise",
    )


def generate_cradle_profile(
    project: Project,
    sections: List[SpanwiseSection],
    spar_type: str,
    params: FixtureProfileParams,
) -> CradleProfile:
    """
    Generate 2D cradle profile that follows lower wing surface.
    
    The cradle spans from base plate to lower airfoil surface,
    centered on the spar thickness. It supports the wing during
    construction by cradling the lower surface contour.
    
    Profile coordinates:
    - X = spanwise position (along spar line, in mm)
    - Z = height (base plate to lower surface)
    """
    plan = project.wing.planform
    
    if not sections:
        raise ValueError("No sections provided for cradle generation")
    
    sorted_sections = sorted(sections, key=lambda s: abs(s.y_m))
    
    # Calculate spar positions and cumulative distances (same as fixture)
    spar_positions = []
    for section in sorted_sections:
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x_m = section.x_le_m + spar_xsi * section.chord_m
        spar_positions.append((abs(section.y_m), spar_x_m))
    
    cumulative_distances = [0.0]
    for i in range(1, len(spar_positions)):
        y_prev, x_prev = spar_positions[i-1]
        y_curr, x_curr = spar_positions[i]
        dy = y_curr - y_prev
        dx = x_curr - x_prev
        segment_length = math.sqrt(dy*dy + dx*dx)
        cumulative_distances.append(cumulative_distances[-1] + segment_length)
    
    # Compute base plate level
    global_min_z_mm = float('inf')
    for section in sorted_sections:
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        coords = np.array(section.airfoil.coordinates)
        chord_mm = section.chord_m * 1000
        z_mm = coords[:, 1] * chord_mm + section.z_m * 1000
        global_min_z_mm = min(global_min_z_mm, z_mm.min())
    
    base_plate_z_mm = global_min_z_mm - params.fixture_height_mm
    
    # Build station heights (tracking lower surface)
    station_heights = []
    for i, section in enumerate(sorted_sections):
        if section.airfoil is None or section.airfoil.coordinates is None:
            continue
        
        coords = np.array(section.airfoil.coordinates)
        chord_mm = section.chord_m * 1000
        
        x_local = coords[:, 0] * chord_mm
        z_local = coords[:, 1] * chord_mm
        dihedral_offset_mm = section.z_m * 1000
        
        # Apply twist
        if abs(section.twist_deg) > 1e-6:
            le_x = x_local.min()
            le_idx = np.argmin(x_local)
            le_z = z_local[le_idx]
            x_local, z_local = apply_twist_to_2d_coords(
                x_local, z_local, section.twist_deg, pivot_x=le_x, pivot_z=le_z
            )
        
        # Find lower surface Z at spar position
        spar_xsi = get_spar_xsi_at_section(section, plan, spar_type)
        spar_x_local = spar_xsi * chord_mm
        z_at_spar = _get_surface_z_at_x(x_local, z_local, spar_x_local)
        z_lower = z_at_spar['lower'] + dihedral_offset_mm
        
        x_along_span_mm = cumulative_distances[i] * 1000
        
        station_heights.append(CradleStationHeight(
            x_along_span_mm=x_along_span_mm,
            z_lower_surface_mm=z_lower,
            z_base_mm=base_plate_z_mm,
        ))
    
    if len(station_heights) < 2:
        raise ValueError("Need at least 2 valid stations for cradle generation")
    
    span_mm = station_heights[-1].x_along_span_mm
    
    # Build outline: bottom edge straight, top edge follows lower surface
    outline = _build_cradle_outline(station_heights)
    
    return CradleProfile(
        outline=outline,
        spar_type=spar_type,
        length_mm=span_mm,
        station_heights=station_heights,
        recommended_grain="spanwise",
    )


def _calculate_fixture_tabs(
    span_mm: float,
    params: FixtureProfileParams,
) -> List[FixtureTabInfo]:
    """Calculate tab positions along fixture span."""
    tabs = []
    
    usable = max(0.0, span_mm - 2.0 * params.tab_edge_margin_mm)
    if usable <= 0:
        return tabs
    
    if usable < params.tab_spacing_mm * 1.5:
        # Single centered tab
        tabs.append(FixtureTabInfo(
            x_along_span_mm=span_mm / 2.0,
            width_mm=params.tab_width_mm,
            depth_mm=params.material_thickness_mm + 0.4,
        ))
    else:
        # Distribute tabs at margins and spacing
        start = params.tab_edge_margin_mm
        end = span_mm - params.tab_edge_margin_mm
        n_interior = max(0, int(np.floor((end - start) / params.tab_spacing_mm)) - 1)
        interior_spacing = (end - start) / (n_interior + 1)
        
        for i in range(n_interior + 2):
            x_pos = start + i * interior_spacing
            tabs.append(FixtureTabInfo(
                x_along_span_mm=x_pos,
                width_mm=params.tab_width_mm,
                depth_mm=params.material_thickness_mm + 0.4,
            ))
    
    return tabs


def _build_fixture_outline(
    station_heights: List[FixtureStationHeight],
    base_z_mm: float,
) -> np.ndarray:
    """Build closed fixture outline from station heights."""
    if not station_heights:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    points = []
    
    # Bottom edge (left to right at base plate level)
    points.append((0.0, base_z_mm))
    points.append((station_heights[-1].x_along_span_mm, base_z_mm))
    
    # Right edge (up to chord line)
    points.append((station_heights[-1].x_along_span_mm, station_heights[-1].z_chord_mm))
    
    # Top edge (right to left, following chord line)
    for station in reversed(station_heights):
        points.append((station.x_along_span_mm, station.z_chord_mm))
    
    # Left edge closes back to start (implicit)
    
    return np.array(points)


def _build_cradle_outline(
    station_heights: List[CradleStationHeight],
) -> np.ndarray:
    """Build closed cradle outline from station heights."""
    if not station_heights:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    base_z_mm = station_heights[0].z_base_mm
    points = []
    
    # Bottom edge (left to right at base plate level)
    points.append((0.0, base_z_mm))
    points.append((station_heights[-1].x_along_span_mm, base_z_mm))
    
    # Right edge (up to lower surface)
    points.append((station_heights[-1].x_along_span_mm, station_heights[-1].z_lower_surface_mm))
    
    # Top edge (right to left, following lower surface contour)
    for station in reversed(station_heights):
        points.append((station.x_along_span_mm, station.z_lower_surface_mm))
    
    return np.array(points)

