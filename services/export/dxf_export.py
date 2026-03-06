# services/export/dxf_export.py
"""
2D DXF export for laser-cut ribs and spar plates.
Generates manufacturing-ready profiles for hobbyist construction.

Includes advanced nesting with:
- Rotational fitting (±5° in 1° steps)
- Finger joint splitting for oversized parts
- Bottom-up sheet filling
"""
from __future__ import annotations
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Callable, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.state import Project
    from services.export.profiles import SparProfile

try:
    import ezdxf
    from ezdxf import units
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    ezdxf = None

from services.export.viewer import ProcessedCpacs


# ==============================================================================
# Constants for rotation fitting
# ==============================================================================
_MAX_ROTATION_DEG = 5.0   # Fixed maximum rotation angle
_ROTATION_STEP_DEG = 1.0  # Fixed rotation step size


# ==============================================================================
# Enums and Dataclasses for Advanced Nesting
# ==============================================================================

class SplitUserChoice(Enum):
    """User's response to split prompt."""
    YES_THIS_ONLY = "yes_this"      # Split this part only
    YES_TO_ALL = "yes_all"          # Split this and all future
    NO_SKIP = "no_skip"             # Skip this part
    CANCEL = "cancel"               # Abort entire nesting


@dataclass
class NestablePart:
    """A part prepared for nesting with bounds and optional rotation."""
    source_path: str                      # Path to DXF file
    doc: Any                              # ezdxf document
    width: float                          # Bounding box width (after rotation)
    height: float                         # Bounding box height (after rotation)
    offset_x: float                       # Offset to normalize origin
    offset_y: float                       # Offset to normalize origin
    rotation_deg: float = 0.0             # Applied rotation
    rotation_center: Tuple[float, float] = (0.0, 0.0)  # Center of rotation
    thickness_mm: float = 0.0             # Material thickness for grouping
    part_name: str = ""                   # Display name (e.g., "Front Spar Piece 1")
    part_type: str = "rib"                # "rib", "spar", "elevon_rib"
    spar_profile: Optional[Any] = None    # SparProfile for splitting
    is_split_piece: bool = False
    split_piece_index: int = 0            # 0, 1, 2... for recursive splits


@dataclass
class NestingResult:
    """Result of the nesting operation."""
    success: bool
    output_files: List[str] = field(default_factory=list)
    placed_count: int = 0
    total_parts: int = 0
    skipped_parts: List[str] = field(default_factory=list)
    split_count: int = 0
    error_message: str = ""


@dataclass
class RibExportParams:
    """Configuration for rib DXF export."""
    spar_plate_thickness_mm: float = 3.0      # Spar thickness for notch width
    notch_clearance_mm: float = 0.1           # Slip fit clearance
    add_lightening_holes: bool = False        # Add weight reduction holes
    lightening_hole_margin_mm: float = 10.0   # Edge margin for holes
    add_spar_tabs: bool = True                # Add interlocking tabs
    tab_width_mm: float = 5.0                 # Width of tabs
    label_ribs: bool = True                   # Engrave rib IDs
    text_height_mm: float = 3.0               # Label text height


@dataclass  
class SparExportParams:
    """Configuration for spar plate DXF export."""
    spar_height_mm: float = 20.0              # Constant spar plate height
    rib_thickness_mm: float = 3.0             # Rib thickness for notch width
    notch_clearance_mm: float = 0.1           # Slip fit clearance
    notch_depth_percent: float = 45.0         # Notch depth as % of spar height
    add_rib_tabs: bool = True                 # Add interlocking tabs
    tab_depth_mm: float = 3.0                 # Tab depth past notch
    label_spars: bool = True                  # Add spar labels


@dataclass
class GridNestingParams:
    """Configuration for basic grid nesting layout."""
    sheet_width_mm: float = 600.0             # Sheet stock width
    sheet_height_mm: float = 300.0            # Sheet stock height
    part_spacing_mm: float = 5.0              # Gap between parts
    margin_mm: float = 10.0                   # Sheet edge margin
    
    # Rotation fitting (fixed at ±5°, 1° steps - not exposed to UI)
    allow_rotation: bool = True
    
    # Finger joint splitting
    allow_splitting: bool = True
    notch_avoidance_margin_mm: float = 20.0   # Min distance from rib notches
    split_joint_clearance_mm: float = 0.15    # Finger joint fit clearance


@dataclass
class PartInfo:
    """Info about a part for nesting, including material thickness."""
    file_path: str                            # Path to individual part DXF
    thickness_mm: float                       # Material thickness for grouping


def check_spar_height_feasibility(
    processed: ProcessedCpacs,
    spar_height_mm: float,
) -> Tuple[bool, float, str]:
    """
    Validate that spar height fits within minimum airfoil thickness.
    
    Args:
        processed: ProcessedCpacs with rib geometry
        spar_height_mm: Proposed spar plate height [mm]
    
    Returns:
        Tuple of (is_feasible, min_thickness_mm, message)
    """
    if not processed.dihedraled_rib_profiles:
        return (True, float('inf'), "No rib profiles available")
    
    min_thickness_mm = float('inf')
    min_station_eta = 0.0
    
    for i, rib in enumerate(processed.dihedraled_rib_profiles):
        # rib is Nx3 array (x, y, z in meters)
        if rib is None or len(rib) < 3:
            continue
        
        # Get z range (thickness)
        z_coords = rib[:, 2] * 1000  # Convert to mm
        thickness = z_coords.max() - z_coords.min()
        
        if thickness < min_thickness_mm:
            min_thickness_mm = thickness
            min_station_eta = i / max(1, len(processed.dihedraled_rib_profiles) - 1)
    
    if spar_height_mm > min_thickness_mm:
        return (
            False,
            min_thickness_mm,
            f"Spar height {spar_height_mm:.1f}mm exceeds minimum airfoil "
            f"thickness {min_thickness_mm:.1f}mm at eta={min_station_eta:.1%}. "
            f"Spar will protrude through skin! Max safe height: {min_thickness_mm * 0.9:.1f}mm"
        )
    
    return (True, min_thickness_mm, f"OK (min thickness: {min_thickness_mm:.1f}mm)")


def generate_rib_dxf(
    processed: ProcessedCpacs,
    output_dir: str,
    params: RibExportParams,
) -> List[str]:
    """
    Generate DXF files for laser-cut rib profiles with spar notches.
    
    Args:
        processed: ProcessedCpacs with rib geometry
        output_dir: Directory for DXF files
        params: Export configuration
    
    Returns:
        List of generated DXF file paths
    """
    if not EZDXF_AVAILABLE:
        raise ImportError(
            "ezdxf is required for DXF export. Install with: pip install ezdxf"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    for i, rib_3d in enumerate(processed.dihedraled_rib_profiles):
        if rib_3d is None or len(rib_3d) < 3:
            continue
        
        # Convert to mm and extract 2D profile (x, z)
        rib_mm = rib_3d * 1000
        x = rib_mm[:, 0]  # Chordwise
        z = rib_mm[:, 2]  # Thickness direction
        
        # Create DXF document
        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = units.MM
        doc.layers.add(name="CUT", color=1)       # Red = cut
        doc.layers.add(name="ENGRAVE", color=5)   # Blue = engrave
        msp = doc.modelspace()
        
        # Normalize x to start at 0
        x = x - x.min()
        
        # Draw airfoil profile as closed polyline
        points_2d = [(float(xi), float(zi)) for xi, zi in zip(x, z)]
        if len(points_2d) >= 3:
            msp.add_lwpolyline(points_2d, close=True, dxfattribs={"layer": "CUT"})
        
        # TODO: Add spar notches based on spar positions
        # This requires knowing the spar X positions at this rib station
        
        # Add rib label
        if params.label_ribs and len(x) > 0:
            chord = x.max() - x.min()
            z_mid = (z.max() + z.min()) / 2
            label = f"R{i + 1}"
            try:
                text = msp.add_text(
                    label, 
                    height=params.text_height_mm,
                    dxfattribs={"layer": "ENGRAVE"}
                )
                text.set_placement((chord * 0.3, z_mid))
            except Exception:
                pass  # Text placement failed, skip label
        
        # Save individual rib
        filename = os.path.join(output_dir, f"rib_{i + 1:02d}.dxf")
        doc.saveas(filename)
        generated_files.append(filename)
    
    return generated_files


def generate_spar_dxf(
    processed: ProcessedCpacs,
    output_dir: str,
    params: SparExportParams,
    rib_stations_mm: Optional[List[float]] = None,
) -> List[str]:
    """
    Generate DXF files for laser-cut spar plates.
    
    Args:
        processed: ProcessedCpacs with spar geometry
        output_dir: Directory for DXF files
        params: Export configuration
        rib_stations_mm: Spanwise positions of ribs [mm]. If None, inferred from geometry.
    
    Returns:
        List of generated DXF file paths
    
    Raises:
        ValueError: If spar_height_mm exceeds minimum airfoil thickness
    """
    if not EZDXF_AVAILABLE:
        raise ImportError(
            "ezdxf is required for DXF export. Install with: pip install ezdxf"
        )
    
    # Validate spar height
    feasible, min_thick, msg = check_spar_height_feasibility(processed, params.spar_height_mm)
    if not feasible:
        raise ValueError(msg)
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # Infer rib stations if not provided
    if rib_stations_mm is None:
        # Use rib profile Y positions
        rib_stations_mm = []
        for rib in processed.dihedraled_rib_profiles:
            if rib is not None and len(rib) > 0:
                y_mm = rib[0, 1] * 1000  # First point's Y coordinate
                rib_stations_mm.append(y_mm)
        rib_stations_mm.sort()
    
    if len(rib_stations_mm) < 2:
        raise ValueError("Need at least 2 rib stations to generate spar")
    
    spar_length_mm = max(rib_stations_mm) - min(rib_stations_mm)
    origin_y = min(rib_stations_mm)
    
    # Generate front and rear spar plates
    for spar_name in ["front_spar", "rear_spar"]:
        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = units.MM
        doc.layers.add(name="CUT", color=1)
        doc.layers.add(name="ENGRAVE", color=5)
        msp = doc.modelspace()
        
        # Rectangular spar plate outline
        msp.add_lwpolyline([
            (0, 0),
            (spar_length_mm, 0),
            (spar_length_mm, params.spar_height_mm),
            (0, params.spar_height_mm),
        ], close=True, dxfattribs={"layer": "CUT"})
        
        # Add rib notches (from top edge)
        notch_width = params.rib_thickness_mm + params.notch_clearance_mm
        notch_depth = params.spar_height_mm * (params.notch_depth_percent / 100)
        
        for station_mm in rib_stations_mm:
            x_pos = station_mm - origin_y
            
            # Rectangular notch from top
            notch_points = [
                (x_pos - notch_width / 2, params.spar_height_mm),
                (x_pos - notch_width / 2, params.spar_height_mm - notch_depth),
                (x_pos + notch_width / 2, params.spar_height_mm - notch_depth),
                (x_pos + notch_width / 2, params.spar_height_mm),
            ]
            msp.add_lwpolyline(notch_points, close=False, dxfattribs={"layer": "CUT"})

        
        # Add label
        if params.label_spars:
            label = spar_name.replace("_", " ").title()
            try:
                text = msp.add_text(
                    label,
                    height=3.0,
                    dxfattribs={"layer": "ENGRAVE"}
                )
                text.set_placement((spar_length_mm / 2, params.spar_height_mm - 5))
            except Exception:
                pass
        
        filename = os.path.join(output_dir, f"{spar_name}.dxf")
        doc.saveas(filename)
        generated_files.append(filename)
    
    return generated_files


def generate_nested_layout(
    part_files: List[str],
    output_path: str,
    params: GridNestingParams,
) -> str:
    """
    Create a simple grid-nested layout of parts on a sheet.
    
    Uses basic left-to-right, bottom-to-top placement.
    For optimal nesting, use external tools like Deepnest.
    
    Args:
        part_files: List of individual part DXF files
        output_path: Output path for nested layout DXF
        params: Nesting configuration
    
    Returns:
        Path to nested layout DXF
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for DXF export")
    
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = units.MM
    doc.layers.add(name="SHEET", color=7)  # White
    doc.layers.add(name="CUT", color=1)
    doc.layers.add(name="ENGRAVE", color=5)
    msp = doc.modelspace()
    
    # Draw sheet boundary
    msp.add_lwpolyline([
        (0, 0),
        (params.sheet_width_mm, 0),
        (params.sheet_width_mm, params.sheet_height_mm),
        (0, params.sheet_height_mm),
    ], close=True, dxfattribs={"layer": "SHEET"})
    
    # Load and measure each part
    parts = []
    for part_file in part_files:
        try:
            source_doc = ezdxf.readfile(part_file)
            bounds = _get_entity_bounds(source_doc.modelspace())
            if bounds:
                parts.append({
                    'file': part_file,
                    'doc': source_doc,
                    'width': bounds['max_x'] - bounds['min_x'],
                    'height': bounds['max_y'] - bounds['min_y'],
                    'offset_x': -bounds['min_x'],
                    'offset_y': -bounds['min_y'],
                })
        except Exception as e:
            print(f"Failed to load {part_file}: {e}")
            continue
    
    # Sort by height (descending) for better packing
    parts.sort(key=lambda p: -p['height'])
    
    # Place parts in grid
    cursor_x = params.margin_mm
    cursor_y = params.margin_mm
    row_height = 0.0
    placed_count = 0
    
    for part in parts:
        part_w = part['width'] + params.part_spacing_mm
        part_h = part['height'] + params.part_spacing_mm
        
        # Check if part fits in current row
        if cursor_x + part_w > params.sheet_width_mm - params.margin_mm:
            # Start new row
            cursor_x = params.margin_mm
            cursor_y += row_height
            row_height = 0.0
        
        # Check if part fits on sheet
        if cursor_y + part_h > params.sheet_height_mm - params.margin_mm:
            print(f"Sheet full. Placed {placed_count}/{len(parts)} parts.")
            break
        
        # Copy entities from part to layout
        _copy_entities_to_layout(
            source_msp=part['doc'].modelspace(),
            target_msp=msp,
            offset_x=cursor_x + part['offset_x'],
            offset_y=cursor_y + part['offset_y'],
        )
        
        cursor_x += part_w
        row_height = max(row_height, part_h)
        placed_count += 1
    
    doc.saveas(output_path)
    return output_path


def _get_entity_bounds(msp) -> Optional[dict]:
    """Calculate bounding box of all entities in modelspace."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for entity in msp:
        try:
            if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'start'):
                # Lines
                min_x = min(min_x, entity.dxf.start.x, entity.dxf.end.x)
                max_x = max(max_x, entity.dxf.start.x, entity.dxf.end.x)
                min_y = min(min_y, entity.dxf.start.y, entity.dxf.end.y)
                max_y = max(max_y, entity.dxf.start.y, entity.dxf.end.y)
            elif hasattr(entity, 'get_points'):
                # Polylines
                for pt in entity.get_points():
                    min_x = min(min_x, pt[0])
                    max_x = max(max_x, pt[0])
                    min_y = min(min_y, pt[1])
                    max_y = max(max_y, pt[1])
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'center'):
                # Circles
                r = entity.dxf.radius
                min_x = min(min_x, entity.dxf.center.x - r)
                max_x = max(max_x, entity.dxf.center.x + r)
                min_y = min(min_y, entity.dxf.center.y - r)
                max_y = max(max_y, entity.dxf.center.y + r)
        except Exception:
            continue
    
    if min_x == float('inf'):
        return None
    
    return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}


def _copy_entities_to_layout(source_msp, target_msp, offset_x: float, offset_y: float):
    """Copy all entities from source to target with offset."""
    from ezdxf.entities import LWPolyline, Circle, Line, Text, MText
    
    for entity in source_msp:
        try:
            if isinstance(entity, LWPolyline):
                points = [(pt[0] + offset_x, pt[1] + offset_y) for pt in entity.get_points()]
                target_msp.add_lwpolyline(
                    points, 
                    close=entity.closed,
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Circle):
                target_msp.add_circle(
                    center=(entity.dxf.center.x + offset_x, entity.dxf.center.y + offset_y),
                    radius=entity.dxf.radius,
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Line):
                target_msp.add_line(
                    start=(entity.dxf.start.x + offset_x, entity.dxf.start.y + offset_y),
                    end=(entity.dxf.end.x + offset_x, entity.dxf.end.y + offset_y),
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Text):
                # Copy text entities (part labels, grain indicators, etc.)
                new_text = target_msp.add_text(
                    entity.dxf.text,
                    height=entity.dxf.height,
                    dxfattribs={"layer": entity.dxf.layer}
                )
                # Get insertion point and apply offset
                insert = entity.dxf.insert
                new_text.set_placement((insert.x + offset_x, insert.y + offset_y))
            elif isinstance(entity, MText):
                # Copy multiline text
                new_mtext = target_msp.add_mtext(
                    entity.text,
                    dxfattribs={"layer": entity.dxf.layer}
                )
                insert = entity.dxf.insert
                new_mtext.dxf.insert = (insert.x + offset_x, insert.y + offset_y)
                new_mtext.dxf.char_height = entity.dxf.char_height
        except Exception:
            continue


def generate_nested_layout_by_thickness(
    parts: List[PartInfo],
    output_base_path: str,
    params: GridNestingParams,
) -> List[str]:
    """
    Create nested layouts grouped by material thickness.
    
    Parts with different thicknesses are placed on separate sheets
    since they require different material stock for CNC cutting.
    
    Args:
        parts: List of PartInfo with file paths and thickness values
        output_base_path: Base output path (e.g., "nested.dxf" -> "nested_3.0mm.dxf")
        params: Nesting configuration
    
    Returns:
        List of generated sheet file paths
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for DXF export")
    
    if not parts:
        return []
    
    # Group parts by thickness
    from collections import defaultdict
    thickness_groups: dict[float, List[str]] = defaultdict(list)
    for part in parts:
        thickness_groups[part.thickness_mm].append(part.file_path)
    
    # Generate output file names based on thickness
    base, ext = os.path.splitext(output_base_path)
    generated_files = []
    
    for thickness_mm, part_files in sorted(thickness_groups.items()):
        # Create filename with thickness suffix
        output_path = f"{base}_{thickness_mm:.1f}mm{ext}"
        
        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = units.MM
        doc.layers.add(name="SHEET", color=7)    # White - sheet boundary
        doc.layers.add(name="CUT", color=1)      # Red - cut lines
        doc.layers.add(name="ENGRAVE", color=5)  # Blue - engrave/labels
        doc.layers.add(name="CUTLINE", color=3)  # Green - BWB cut lines
        msp = doc.modelspace()
        
        # Draw sheet boundary
        msp.add_lwpolyline([
            (0, 0),
            (params.sheet_width_mm, 0),
            (params.sheet_width_mm, params.sheet_height_mm),
            (0, params.sheet_height_mm),
        ], close=True, dxfattribs={"layer": "SHEET"})
        
        # Add thickness label in corner
        try:
            text = msp.add_text(
                f"MATERIAL: {thickness_mm:.1f}mm",
                height=5.0,
                dxfattribs={"layer": "ENGRAVE"}
            )
            text.set_placement((params.margin_mm, params.sheet_height_mm - params.margin_mm - 5))
        except Exception:
            pass
        
        # Load and measure each part
        loaded_parts = []
        for part_file in part_files:
            try:
                source_doc = ezdxf.readfile(part_file)
                bounds = _get_entity_bounds(source_doc.modelspace())
                if bounds:
                    loaded_parts.append({
                        'file': part_file,
                        'doc': source_doc,
                        'width': bounds['max_x'] - bounds['min_x'],
                        'height': bounds['max_y'] - bounds['min_y'],
                        'offset_x': -bounds['min_x'],
                        'offset_y': -bounds['min_y'],
                    })
            except Exception as e:
                print(f"Failed to load {part_file}: {e}")
                continue
        
        # Sort by height (descending) for better packing
        loaded_parts.sort(key=lambda p: -p['height'])
        
        # Place parts in grid
        cursor_x = params.margin_mm
        cursor_y = params.margin_mm
        row_height = 0.0
        placed_count = 0
        
        for part in loaded_parts:
            part_w = part['width'] + params.part_spacing_mm
            part_h = part['height'] + params.part_spacing_mm
            
            # Check if part fits in current row
            if cursor_x + part_w > params.sheet_width_mm - params.margin_mm:
                # Start new row
                cursor_x = params.margin_mm
                cursor_y += row_height
                row_height = 0.0
            
            # Check if part fits on sheet
            if cursor_y + part_h > params.sheet_height_mm - params.margin_mm:
                print(f"Sheet full for {thickness_mm}mm. Placed {placed_count}/{len(loaded_parts)} parts.")
                break
            
            # Copy entities from part to layout
            _copy_entities_to_layout(
                source_msp=part['doc'].modelspace(),
                target_msp=msp,
                offset_x=cursor_x + part['offset_x'],
                offset_y=cursor_y + part['offset_y'],
            )
            
            cursor_x += part_w
            row_height = max(row_height, part_h)
            placed_count += 1
        
        doc.saveas(output_path)
        generated_files.append(output_path)
        print(f"Generated {output_path}: {placed_count} parts at {thickness_mm}mm thickness")
    
    return generated_files


def is_ezdxf_available() -> bool:
    """Check if ezdxf library is available for DXF export."""
    return EZDXF_AVAILABLE


# ==============================================================================
# Rotation Fitting Utilities
# ==============================================================================

def _rotate_point_2d(
    x: float,
    y: float,
    cx: float,
    cy: float,
    angle_rad: float,
) -> Tuple[float, float]:
    """
    Rotate a point (x, y) around center (cx, cy) by angle_rad.
    
    Args:
        x, y: Point coordinates
        cx, cy: Center of rotation
        angle_rad: Rotation angle in radians (positive = CCW)
    
    Returns:
        Tuple of (new_x, new_y)
    """
    dx = x - cx
    dy = y - cy
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_x = cx + dx * cos_a - dy * sin_a
    new_y = cy + dx * sin_a + dy * cos_a
    return new_x, new_y


def _collect_all_entity_points(msp) -> List[Tuple[float, float]]:
    """
    Extract all vertex points from entities in a modelspace.
    
    Handles LWPolyline, Line, Circle, Arc, and Text entities.
    
    Args:
        msp: ezdxf modelspace
    
    Returns:
        List of (x, y) points
    """
    points = []
    
    for entity in msp:
        try:
            entity_type = entity.dxftype()
            
            if entity_type == 'LWPOLYLINE':
                for pt in entity.get_points():
                    points.append((float(pt[0]), float(pt[1])))
            elif entity_type == 'LINE':
                points.append((float(entity.dxf.start.x), float(entity.dxf.start.y)))
                points.append((float(entity.dxf.end.x), float(entity.dxf.end.y)))
            elif entity_type == 'CIRCLE':
                cx, cy = entity.dxf.center.x, entity.dxf.center.y
                r = entity.dxf.radius
                # Add bounding box points for circle
                points.extend([
                    (cx - r, cy),
                    (cx + r, cy),
                    (cx, cy - r),
                    (cx, cy + r),
                ])
            elif entity_type == 'ARC':
                cx, cy = entity.dxf.center.x, entity.dxf.center.y
                r = entity.dxf.radius
                start_angle = math.radians(entity.dxf.start_angle)
                end_angle = math.radians(entity.dxf.end_angle)
                # Add start, end, and center bounding points
                points.append((cx + r * math.cos(start_angle), cy + r * math.sin(start_angle)))
                points.append((cx + r * math.cos(end_angle), cy + r * math.sin(end_angle)))
                # Add cardinal points if within arc
                for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
                    if _angle_in_arc(angle, start_angle, end_angle):
                        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            elif entity_type in ('TEXT', 'MTEXT'):
                # Use insertion point for text
                insert = entity.dxf.insert
                points.append((float(insert.x), float(insert.y)))
        except Exception:
            continue
    
    return points


def _angle_in_arc(angle: float, start: float, end: float) -> bool:
    """Check if angle is within arc from start to end (CCW)."""
    # Normalize angles to [0, 2π)
    angle = angle % (2 * math.pi)
    start = start % (2 * math.pi)
    end = end % (2 * math.pi)
    
    if start <= end:
        return start <= angle <= end
    else:
        # Arc crosses 0
        return angle >= start or angle <= end


def _get_bounds_at_rotation(
    points: List[Tuple[float, float]],
    cx: float,
    cy: float,
    angle_rad: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate axis-aligned bounding box of points after rotation.
    
    Args:
        points: List of (x, y) points
        cx, cy: Center of rotation
        angle_rad: Rotation angle in radians
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if not points:
        return (0, 0, 0, 0)
    
    rotated = [_rotate_point_2d(x, y, cx, cy, angle_rad) for x, y in points]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    
    return (min(xs), min(ys), max(xs), max(ys))


def _try_fit_with_rotation(
    width: float,
    height: float,
    points: List[Tuple[float, float]],
    cx: float,
    cy: float,
    sheet_width: float,
    sheet_height: float,
    margin: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Try to fit a part on the sheet by rotating it.
    
    Tries angles: 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5 degrees.
    
    Args:
        width, height: Original part dimensions
        points: All vertex points of the part
        cx, cy: Center of the part (for rotation)
        sheet_width, sheet_height: Sheet dimensions
        margin: Sheet margin
    
    Returns:
        Tuple of (angle_deg, new_width, new_height, offset_x, offset_y) if fits,
        None if no rotation works
    """
    usable_width = sheet_width - 2 * margin
    usable_height = sheet_height - 2 * margin
    
    # Generate rotation sequence: 0, +1, -1, +2, -2, ...
    angles_deg = [0.0]
    for i in range(1, int(_MAX_ROTATION_DEG / _ROTATION_STEP_DEG) + 1):
        angles_deg.append(i * _ROTATION_STEP_DEG)
        angles_deg.append(-i * _ROTATION_STEP_DEG)
    
    for angle_deg in angles_deg:
        angle_rad = math.radians(angle_deg)
        
        min_x, min_y, max_x, max_y = _get_bounds_at_rotation(points, cx, cy, angle_rad)
        new_width = max_x - min_x
        new_height = max_y - min_y
        
        if new_width <= usable_width and new_height <= usable_height:
            # Part fits! Calculate offset to normalize origin
            offset_x = -min_x
            offset_y = -min_y
            return (angle_deg, new_width, new_height, offset_x, offset_y)
    
    return None


def _copy_entities_to_layout_rotated(
    source_msp,
    target_msp,
    offset_x: float,
    offset_y: float,
    rotation_deg: float = 0.0,
    rotation_center: Tuple[float, float] = (0.0, 0.0),
):
    """
    Copy all entities from source to target with rotation and offset.
    
    Args:
        source_msp: Source modelspace
        target_msp: Target modelspace
        offset_x, offset_y: Translation offset (applied after rotation)
        rotation_deg: Rotation angle in degrees
        rotation_center: Center of rotation (cx, cy)
    """
    from ezdxf.entities import LWPolyline, Circle, Line, Text, MText, Arc
    
    angle_rad = math.radians(rotation_deg)
    cx, cy = rotation_center
    
    for entity in source_msp:
        try:
            if isinstance(entity, LWPolyline):
                # Rotate and translate each point
                new_points = []
                for pt in entity.get_points():
                    rx, ry = _rotate_point_2d(pt[0], pt[1], cx, cy, angle_rad)
                    new_points.append((rx + offset_x, ry + offset_y))
                target_msp.add_lwpolyline(
                    new_points,
                    close=entity.closed,
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Circle):
                # Rotate center
                rx, ry = _rotate_point_2d(
                    entity.dxf.center.x, entity.dxf.center.y, cx, cy, angle_rad
                )
                target_msp.add_circle(
                    center=(rx + offset_x, ry + offset_y),
                    radius=entity.dxf.radius,
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Line):
                # Rotate both endpoints
                sx, sy = _rotate_point_2d(
                    entity.dxf.start.x, entity.dxf.start.y, cx, cy, angle_rad
                )
                ex, ey = _rotate_point_2d(
                    entity.dxf.end.x, entity.dxf.end.y, cx, cy, angle_rad
                )
                target_msp.add_line(
                    start=(sx + offset_x, sy + offset_y),
                    end=(ex + offset_x, ey + offset_y),
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Arc):
                # Rotate center, adjust angles
                rx, ry = _rotate_point_2d(
                    entity.dxf.center.x, entity.dxf.center.y, cx, cy, angle_rad
                )
                target_msp.add_arc(
                    center=(rx + offset_x, ry + offset_y),
                    radius=entity.dxf.radius,
                    start_angle=entity.dxf.start_angle + rotation_deg,
                    end_angle=entity.dxf.end_angle + rotation_deg,
                    dxfattribs={"layer": entity.dxf.layer}
                )
            elif isinstance(entity, Text):
                # Rotate insertion point
                insert = entity.dxf.insert
                rx, ry = _rotate_point_2d(insert.x, insert.y, cx, cy, angle_rad)
                new_text = target_msp.add_text(
                    entity.dxf.text,
                    height=entity.dxf.height,
                    dxfattribs={"layer": entity.dxf.layer}
                )
                new_text.set_placement((rx + offset_x, ry + offset_y))
                # Rotate text angle
                if hasattr(entity.dxf, 'rotation'):
                    new_text.dxf.rotation = entity.dxf.rotation + rotation_deg
            elif isinstance(entity, MText):
                # Rotate insertion point
                insert = entity.dxf.insert
                rx, ry = _rotate_point_2d(insert.x, insert.y, cx, cy, angle_rad)
                new_mtext = target_msp.add_mtext(
                    entity.text,
                    dxfattribs={"layer": entity.dxf.layer}
                )
                new_mtext.dxf.insert = (rx + offset_x, ry + offset_y)
                new_mtext.dxf.char_height = entity.dxf.char_height
        except Exception:
            continue


# ==============================================================================
# Finger Joint Splitting Utilities
# ==============================================================================

def _calculate_finger_count(height_mm: float) -> int:
    """
    Calculate the number of fingers for a finger joint based on spar height.
    
    Rule: 1 finger per 10mm of height, clamped to [3, 11], always odd.
    
    Args:
        height_mm: Height of the spar at the split position
    
    Returns:
        Odd integer finger count in range [3, 11]
    """
    raw_count = int(height_mm / 10.0)
    
    # Clamp to [3, 11]
    count = max(3, min(11, raw_count))
    
    # Ensure odd
    if count % 2 == 0:
        count += 1
        if count > 11:
            count = 11
    
    return count


def _find_safe_split_position(
    spar_profile: 'SparProfile',
    target_length_mm: float,
    notch_avoidance_margin_mm: float = 20.0,
) -> Optional[float]:
    """
    Find a safe position to split a spar, avoiding rib notch locations.
    
    The split position must be at least notch_avoidance_margin_mm away from
    any rib notch to ensure structural integrity and avoid cutting through
    the notch geometry.
    
    Args:
        spar_profile: SparProfile containing rib notch positions
        target_length_mm: Desired length of the first piece (aim for ~sheet width)
        notch_avoidance_margin_mm: Minimum distance from rib notches
    
    Returns:
        X position (along span) for the split, or None if no safe position exists
    """
    from services.export.profiles import SparProfile
    
    if not spar_profile.rib_notches:
        # No rib notches - can split anywhere
        return target_length_mm
    
    # Get all notch positions
    notch_positions = [n.x_along_span_mm for n in spar_profile.rib_notches]
    
    # Build list of forbidden zones (notch position ± margin)
    forbidden_zones = []
    for pos in notch_positions:
        zone_start = pos - notch_avoidance_margin_mm
        zone_end = pos + notch_avoidance_margin_mm
        forbidden_zones.append((zone_start, zone_end))
    
    # Merge overlapping zones
    forbidden_zones.sort()
    merged_zones = []
    for zone in forbidden_zones:
        if merged_zones and zone[0] <= merged_zones[-1][1]:
            # Overlaps with previous zone - extend it
            merged_zones[-1] = (merged_zones[-1][0], max(merged_zones[-1][1], zone[1]))
        else:
            merged_zones.append(zone)
    
    # Try target position first
    is_safe = True
    for zone_start, zone_end in merged_zones:
        if zone_start <= target_length_mm <= zone_end:
            is_safe = False
            break
    
    if is_safe:
        return target_length_mm
    
    # Find nearest safe position (search both directions)
    best_position = None
    min_distance = float('inf')
    
    # Check gaps between forbidden zones
    # Before first zone
    if merged_zones[0][0] > 0:
        candidate = min(target_length_mm, merged_zones[0][0] - 1)
        if candidate > 0:
            dist = abs(candidate - target_length_mm)
            if dist < min_distance:
                min_distance = dist
                best_position = candidate
    
    # Between zones
    for i in range(len(merged_zones) - 1):
        gap_start = merged_zones[i][1]
        gap_end = merged_zones[i + 1][0]
        gap_center = (gap_start + gap_end) / 2
        
        if gap_end - gap_start > 2:  # At least 2mm gap
            # Use center of gap or clamp target to gap
            if gap_start <= target_length_mm <= gap_end:
                best_position = target_length_mm
                min_distance = 0
                break
            else:
                candidate = max(gap_start + 1, min(gap_end - 1, target_length_mm))
                dist = abs(candidate - target_length_mm)
                if dist < min_distance:
                    min_distance = dist
                    best_position = candidate
    
    # After last zone
    if merged_zones[-1][1] < spar_profile.length_mm:
        gap_start = merged_zones[-1][1]
        if target_length_mm > gap_start:
            candidate = max(gap_start + 1, target_length_mm)
            if candidate < spar_profile.length_mm:
                dist = abs(candidate - target_length_mm)
                if dist < min_distance:
                    min_distance = dist
                    best_position = candidate
    
    return best_position


def _interpolate_spar_height_at_x(
    spar_profile: 'SparProfile',
    x_along_span_mm: float,
) -> float:
    """
    Interpolate spar height at a given spanwise position.
    
    Uses linear interpolation between station_heights entries.
    
    Args:
        spar_profile: SparProfile with station_heights
        x_along_span_mm: Position along spar
    
    Returns:
        Interpolated spar height in mm
    """
    if not spar_profile.station_heights:
        return 20.0  # Fallback default
    
    stations = spar_profile.station_heights
    
    # Handle edge cases
    if x_along_span_mm <= stations[0].x_along_span_mm:
        return stations[0].height_mm
    if x_along_span_mm >= stations[-1].x_along_span_mm:
        return stations[-1].height_mm
    
    # Find bracketing stations
    for i in range(len(stations) - 1):
        x0 = stations[i].x_along_span_mm
        x1 = stations[i + 1].x_along_span_mm
        
        if x0 <= x_along_span_mm <= x1:
            # Linear interpolation
            t = (x_along_span_mm - x0) / (x1 - x0) if (x1 - x0) > 0 else 0
            h0 = stations[i].height_mm
            h1 = stations[i + 1].height_mm
            return h0 + t * (h1 - h0)
    
    return stations[-1].height_mm


def _split_spar_at_position(
    spar_profile: 'SparProfile',
    split_x_mm: float,
    clearance_mm: float = 0.15,
    piece_index: int = 0,
) -> Tuple['SparProfile', 'SparProfile']:
    """
    Split a spar profile at a given position with finger joints.
    
    Creates two new SparProfile objects:
    - Left piece: from 0 to split_x_mm (male fingers on right edge)
    - Right piece: from split_x_mm to end (female slots on left edge)
    
    The finger joint geometry is automatically calculated based on the
    spar height at the split position.
    
    Args:
        spar_profile: Original SparProfile to split
        split_x_mm: Position along spar to split
        clearance_mm: Finger joint clearance
        piece_index: Index for naming pieces
    
    Returns:
        Tuple of (left_piece, right_piece) SparProfiles
    """
    from services.export.profiles import (
        SparProfile, SparStationHeight, RibNotchInfo,
        BWBJointConfig, generate_finger_joint_edge
    )
    
    # Get spar height at split position
    height_at_split = _interpolate_spar_height_at_x(spar_profile, split_x_mm)
    
    # Calculate finger count
    finger_count = _calculate_finger_count(height_at_split)
    
    # Create joint config
    joint_config = BWBJointConfig(
        joint_type="finger",
        finger_count=finger_count,
        clearance_mm=clearance_mm,
    )
    
    # Split station heights
    left_stations = []
    right_stations = []
    
    for station in spar_profile.station_heights:
        if station.x_along_span_mm < split_x_mm:
            left_stations.append(station)
        elif station.x_along_span_mm > split_x_mm:
            # Shift X for right piece
            right_stations.append(SparStationHeight(
                x_along_span_mm=station.x_along_span_mm - split_x_mm,
                z_upper_mm=station.z_upper_mm,
                z_lower_mm=station.z_lower_mm,
            ))
    
    # Interpolate station at split point
    if left_stations:
        last_left = left_stations[-1]
        # Interpolate Z values at split
        z_upper = _interpolate_z_at_x(spar_profile, split_x_mm, 'upper')
        z_lower = _interpolate_z_at_x(spar_profile, split_x_mm, 'lower')
        
        left_stations.append(SparStationHeight(
            x_along_span_mm=split_x_mm,
            z_upper_mm=z_upper,
            z_lower_mm=z_lower,
        ))
        
        right_stations.insert(0, SparStationHeight(
            x_along_span_mm=0.0,
            z_upper_mm=z_upper,
            z_lower_mm=z_lower,
        ))
    
    # Split rib notches
    left_notches = [n for n in spar_profile.rib_notches if n.x_along_span_mm < split_x_mm]
    right_notches = [
        RibNotchInfo(
            x_along_span_mm=n.x_along_span_mm - split_x_mm,
            notch_width_mm=n.notch_width_mm,
            notch_depth_mm=n.notch_depth_mm,
            spar_height_at_station_mm=n.spar_height_at_station_mm,
        )
        for n in spar_profile.rib_notches if n.x_along_span_mm > split_x_mm
    ]
    
    # Generate finger joint profiles
    left_joint = generate_finger_joint_edge(
        edge_length_mm=height_at_split,
        spar_thickness_mm=3.0,  # Default, will be overridden by actual value
        config=joint_config,
        is_male=True,
    )
    
    right_joint = generate_finger_joint_edge(
        edge_length_mm=height_at_split,
        spar_thickness_mm=3.0,
        config=joint_config,
        is_male=False,
    )
    
    # Build outlines with finger joints
    left_outline = _build_split_spar_outline(
        left_stations, left_notches, left_joint, "right"
    )
    right_outline = _build_split_spar_outline(
        right_stations, right_notches, right_joint, "left"
    )
    
    left_piece = SparProfile(
        outline=left_outline,
        spar_type=spar_profile.spar_type,
        spar_region=spar_profile.spar_region,
        length_mm=split_x_mm,
        station_heights=left_stations,
        rib_notches=left_notches,
        junction_profile=left_joint,
        recommended_grain="spanwise",
    )
    
    right_piece = SparProfile(
        outline=right_outline,
        spar_type=spar_profile.spar_type,
        spar_region=spar_profile.spar_region,
        length_mm=spar_profile.length_mm - split_x_mm,
        station_heights=right_stations,
        rib_notches=right_notches,
        junction_profile=right_joint,
        recommended_grain="spanwise",
    )
    
    return left_piece, right_piece


def _interpolate_z_at_x(
    spar_profile: 'SparProfile',
    x_along_span_mm: float,
    surface: str,  # 'upper' or 'lower'
) -> float:
    """Interpolate Z coordinate at a spanwise position."""
    stations = spar_profile.station_heights
    
    if not stations:
        return 0.0
    
    if x_along_span_mm <= stations[0].x_along_span_mm:
        return stations[0].z_upper_mm if surface == 'upper' else stations[0].z_lower_mm
    if x_along_span_mm >= stations[-1].x_along_span_mm:
        return stations[-1].z_upper_mm if surface == 'upper' else stations[-1].z_lower_mm
    
    for i in range(len(stations) - 1):
        x0 = stations[i].x_along_span_mm
        x1 = stations[i + 1].x_along_span_mm
        
        if x0 <= x_along_span_mm <= x1:
            t = (x_along_span_mm - x0) / (x1 - x0) if (x1 - x0) > 0 else 0
            if surface == 'upper':
                return stations[i].z_upper_mm + t * (stations[i + 1].z_upper_mm - stations[i].z_upper_mm)
            else:
                return stations[i].z_lower_mm + t * (stations[i + 1].z_lower_mm - stations[i].z_lower_mm)
    
    return stations[-1].z_upper_mm if surface == 'upper' else stations[-1].z_lower_mm


def _build_split_spar_outline(
    station_heights: List,
    rib_notches: List,
    joint_profile: Optional['FingerJointProfile'],
    joint_edge: str,  # 'left' or 'right'
) -> np.ndarray:
    """
    Build spar outline with finger joint on specified edge.
    
    Args:
        station_heights: List of SparStationHeight
        rib_notches: List of RibNotchInfo
        joint_profile: FingerJointProfile for the joint edge
        joint_edge: Which edge has the joint ('left' or 'right')
    
    Returns:
        Nx2 numpy array of outline points
    """
    if not station_heights:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    points = []
    
    # Start from bottom-left, go clockwise
    if joint_edge == 'left' and joint_profile is not None:
        # Left edge has finger joint - start with joint profile
        # Joint points are in local coords (x=0 at edge, z from 0 to height)
        z_offset = station_heights[0].z_lower_mm
        for pt in joint_profile.edge_points:
            points.append((pt[0], pt[1] + z_offset))
    else:
        # Simple left edge
        points.append((0, station_heights[0].z_lower_mm))
        points.append((0, station_heights[0].z_upper_mm))
    
    # Top edge (upper surface) - left to right
    for station in station_heights:
        points.append((station.x_along_span_mm, station.z_upper_mm))
    
    if joint_edge == 'right' and joint_profile is not None:
        # Right edge has finger joint
        x_offset = station_heights[-1].x_along_span_mm
        z_offset = station_heights[-1].z_lower_mm
        # Joint profile goes from bottom to top, we need top to bottom
        for pt in reversed(joint_profile.edge_points):
            points.append((x_offset + pt[0], pt[1] + z_offset))
    else:
        # Simple right edge
        points.append((station_heights[-1].x_along_span_mm, station_heights[-1].z_upper_mm))
        points.append((station_heights[-1].x_along_span_mm, station_heights[-1].z_lower_mm))
    
    # Bottom edge (lower surface) with rib notches - right to left
    # Sort notches by position descending for right-to-left traversal
    sorted_notches = sorted(rib_notches, key=lambda n: -n.x_along_span_mm)
    
    current_x = station_heights[-1].x_along_span_mm
    
    for notch in sorted_notches:
        half_w = notch.notch_width_mm / 2
        notch_right = notch.x_along_span_mm + half_w
        notch_left = notch.x_along_span_mm - half_w
        notch_top = _get_lower_z_at_x(station_heights, notch.x_along_span_mm) + notch.notch_depth_mm
        notch_bottom = _get_lower_z_at_x(station_heights, notch.x_along_span_mm)
        
        # Go to notch right edge at lower surface
        if current_x > notch_right:
            z_at_current = _get_lower_z_at_x(station_heights, current_x)
            z_at_notch_right = _get_lower_z_at_x(station_heights, notch_right)
            points.append((notch_right, z_at_notch_right))
        
        # Notch geometry (rectangular cutout from bottom)
        points.append((notch_right, notch_top))
        points.append((notch_left, notch_top))
        z_at_notch_left = _get_lower_z_at_x(station_heights, notch_left)
        points.append((notch_left, z_at_notch_left))
        
        current_x = notch_left
    
    # Continue to left edge
    if current_x > 0:
        points.append((0, station_heights[0].z_lower_mm))
    
    return np.array(points)


def _get_lower_z_at_x(station_heights: List, x: float) -> float:
    """Get interpolated lower Z at a given X position."""
    if not station_heights:
        return 0.0
    
    if x <= station_heights[0].x_along_span_mm:
        return station_heights[0].z_lower_mm
    if x >= station_heights[-1].x_along_span_mm:
        return station_heights[-1].z_lower_mm
    
    for i in range(len(station_heights) - 1):
        x0 = station_heights[i].x_along_span_mm
        x1 = station_heights[i + 1].x_along_span_mm
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0
            return station_heights[i].z_lower_mm + t * (station_heights[i + 1].z_lower_mm - station_heights[i].z_lower_mm)
    
    return station_heights[-1].z_lower_mm


def _write_spar_profile_to_dxf(
    spar_profile: 'SparProfile',
    output_path: str,
    label: str = "",
) -> str:
    """
    Write a SparProfile to a DXF file.
    
    Args:
        spar_profile: SparProfile to export
        output_path: Output file path
        label: Optional label to add to the part
    
    Returns:
        Path to the written file
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for DXF export")
    
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = units.MM
    doc.layers.add(name="CUT", color=1)
    doc.layers.add(name="ENGRAVE", color=5)
    msp = doc.modelspace()
    
    # Draw outline
    if len(spar_profile.outline) >= 3:
        points = [(float(p[0]), float(p[1])) for p in spar_profile.outline]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "CUT"})
    
    # Add label if provided
    if label:
        # Place label in center of spar
        cx = spar_profile.length_mm / 2
        if spar_profile.station_heights:
            cz = (spar_profile.station_heights[0].z_upper_mm + 
                  spar_profile.station_heights[0].z_lower_mm) / 2
        else:
            cz = 10.0
        
        try:
            text = msp.add_text(
                label,
                height=3.0,
                dxfattribs={"layer": "ENGRAVE"}
            )
            text.set_placement((cx, cz))
        except Exception:
            pass
    
    doc.saveas(output_path)
    return output_path


# ==============================================================================
# Bottom-Up Placement Algorithm (Shelf Nesting)
# ==============================================================================

@dataclass
class PlacedPart:
    """A part that has been placed on the sheet."""
    part: NestablePart
    x: float  # Bottom-left X position on sheet
    y: float  # Bottom-left Y position on sheet


def _place_parts_bottom_up(
    parts: List[NestablePart],
    sheet_width: float,
    sheet_height: float,
    margin: float,
    spacing: float,
) -> Tuple[List[PlacedPart], List[NestablePart]]:
    """
    Place parts on a sheet using bottom-up shelf algorithm.
    
    Parts are placed left-to-right, bottom-to-top. Each row (shelf) has
    height equal to the tallest part in that row.
    
    Note: In DXF, Y=0 is at the bottom. Parts are placed starting from
    the bottom margin and rows stack upward.
    
    Args:
        parts: List of NestablePart to place (should be pre-sorted by height descending)
        sheet_width: Total sheet width
        sheet_height: Total sheet height  
        margin: Sheet edge margin
        spacing: Gap between parts
    
    Returns:
        Tuple of (placed_parts, unplaced_parts)
    """
    placed: List[PlacedPart] = []
    unplaced: List[NestablePart] = []
    
    usable_width = sheet_width - 2 * margin
    usable_height = sheet_height - 2 * margin
    
    # Current position - start at bottom-left
    cursor_x = margin
    cursor_y = margin  # Y=margin is the bottom row
    current_row_height = 0.0
    
    for part in parts:
        part_w = part.width + spacing
        part_h = part.height + spacing
        
        # Check if part is too large for the sheet at all
        if part.width > usable_width or part.height > usable_height:
            unplaced.append(part)
            continue
        
        # Check if part fits in current row (horizontally)
        if cursor_x + part.width > sheet_width - margin:
            # Start new row above current one
            cursor_x = margin
            cursor_y += current_row_height + spacing
            current_row_height = 0.0
        
        # Check if part fits on sheet vertically (new row check)
        if cursor_y + part.height > sheet_height - margin:
            # Part doesn't fit - add to unplaced
            unplaced.append(part)
            continue
        
        # Place the part at current position
        placed.append(PlacedPart(
            part=part,
            x=cursor_x,
            y=cursor_y,
        ))
        
        # Move cursor right for next part
        cursor_x += part.width + spacing
        # Track max height in this row
        current_row_height = max(current_row_height, part.height)
    
    return placed, unplaced


# ==============================================================================
# Main Nesting Function with Rotation and Splitting
# ==============================================================================

def generate_nested_layout_with_fitting(
    part_files: List[PartInfo],
    output_base_path: str,
    params: GridNestingParams,
    spar_profiles: Optional[Dict[str, 'SparProfile']] = None,
    split_callback: Optional[Callable[[str, float, float], SplitUserChoice]] = None,
) -> NestingResult:
    """
    Generate nested layout with rotation fitting and optional splitting.
    
    This is the main entry point for advanced nesting. The algorithm:
    1. Load all parts and calculate bounds
    2. For each part that doesn't fit:
       a. Try rotation (±5° in 1° steps)
       b. If still doesn't fit and splitting is allowed, prompt user and split
       c. Recursively process split pieces
    3. Place all fitting parts using bottom-up shelf algorithm
    4. Group by thickness and generate output files
    
    Args:
        part_files: List of PartInfo with file paths and thicknesses
        output_base_path: Base path for output files
        params: Nesting parameters
        spar_profiles: Optional dict mapping file paths to SparProfile objects
                       (needed for splitting spars with finger joints)
        split_callback: Function called to ask user about splitting.
                        Signature: (part_name, part_length, sheet_width) -> SplitUserChoice
                        If None, splitting is disabled.
    
    Returns:
        NestingResult with success status and output files
    """
    if not EZDXF_AVAILABLE:
        return NestingResult(
            success=False,
            error_message="ezdxf is required for DXF export"
        )
    
    if not part_files:
        return NestingResult(success=True, output_files=[], placed_count=0, total_parts=0)
    
    # Track user's "Yes to All" choice
    split_yes_to_all = False
    split_count = 0
    max_recursion_depth = 5
    
    # Group parts by thickness
    thickness_groups: Dict[float, List[NestablePart]] = defaultdict(list)
    skipped_parts: List[str] = []
    
    def process_part(
        part_info: PartInfo,
        spar_profile: Optional['SparProfile'] = None,
        depth: int = 0,
        piece_suffix: str = "",
    ) -> None:
        """Process a single part, potentially splitting it."""
        nonlocal split_yes_to_all, split_count
        
        if depth > max_recursion_depth:
            skipped_parts.append(f"{part_info.file_path} (max split depth exceeded)")
            return
        
        try:
            source_doc = ezdxf.readfile(part_info.file_path)
            source_msp = source_doc.modelspace()
        except Exception as e:
            skipped_parts.append(f"{part_info.file_path} (load error: {e})")
            return
        
        # Get bounds and all points
        bounds = _get_entity_bounds(source_msp)
        if not bounds:
            skipped_parts.append(f"{part_info.file_path} (no geometry)")
            return
        
        points = _collect_all_entity_points(source_msp)
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        cx = (bounds['min_x'] + bounds['max_x']) / 2
        cy = (bounds['min_y'] + bounds['max_y']) / 2
        
        # Try to fit with rotation
        fit_result = None
        if params.allow_rotation:
            fit_result = _try_fit_with_rotation(
                width, height, points, cx, cy,
                params.sheet_width_mm, params.sheet_height_mm, params.margin_mm
            )
        
        if fit_result is not None:
            # Part fits (possibly with rotation)
            angle_deg, new_width, new_height, offset_x, offset_y = fit_result
            
            part_name = os.path.basename(part_info.file_path)
            if piece_suffix:
                part_name = f"{os.path.splitext(part_name)[0]}{piece_suffix}.dxf"
            
            nestable = NestablePart(
                source_path=part_info.file_path,
                doc=source_doc,
                width=new_width,
                height=new_height,
                offset_x=offset_x,
                offset_y=offset_y,
                rotation_deg=angle_deg,
                rotation_center=(cx, cy),
                thickness_mm=part_info.thickness_mm,
                part_name=part_name,
                spar_profile=spar_profile,
            )
            thickness_groups[part_info.thickness_mm].append(nestable)
            return
        
        # Part doesn't fit even with rotation
        # Check if we can split it
        usable_width = params.sheet_width_mm - 2 * params.margin_mm
        
        if (params.allow_splitting and 
            spar_profile is not None and 
            width > usable_width):
            
            # Determine if we should split
            should_split = False
            
            if split_yes_to_all:
                should_split = True
            elif split_callback is not None:
                part_name = os.path.basename(part_info.file_path)
                choice = split_callback(part_name, width, usable_width)
                
                if choice == SplitUserChoice.YES_THIS_ONLY:
                    should_split = True
                elif choice == SplitUserChoice.YES_TO_ALL:
                    should_split = True
                    split_yes_to_all = True
                elif choice == SplitUserChoice.NO_SKIP:
                    skipped_parts.append(f"{part_info.file_path} (user skipped)")
                    return
                elif choice == SplitUserChoice.CANCEL:
                    # Cancel entire operation - will be handled by caller
                    raise KeyboardInterrupt("User cancelled nesting")
            
            if should_split:
                # Find safe split position
                target_split = usable_width - params.margin_mm
                split_pos = _find_safe_split_position(
                    spar_profile, target_split, params.notch_avoidance_margin_mm
                )
                
                if split_pos is not None and split_pos > 0 and split_pos < spar_profile.length_mm:
                    # Split the spar
                    left_piece, right_piece = _split_spar_at_position(
                        spar_profile, split_pos, params.split_joint_clearance_mm, depth
                    )
                    split_count += 1
                    
                    # Write split pieces to temp files
                    base_name = os.path.splitext(os.path.basename(part_info.file_path))[0]
                    temp_dir = os.path.dirname(part_info.file_path)
                    
                    left_path = os.path.join(temp_dir, f"{base_name}_L{depth + 1}.dxf")
                    right_path = os.path.join(temp_dir, f"{base_name}_R{depth + 1}.dxf")
                    
                    _write_spar_profile_to_dxf(left_piece, left_path, f"{base_name} L{depth + 1}")
                    _write_spar_profile_to_dxf(right_piece, right_path, f"{base_name} R{depth + 1}")
                    
                    # Recursively process split pieces
                    left_info = PartInfo(file_path=left_path, thickness_mm=part_info.thickness_mm)
                    right_info = PartInfo(file_path=right_path, thickness_mm=part_info.thickness_mm)
                    
                    process_part(left_info, left_piece, depth + 1, f"_L{depth + 1}")
                    process_part(right_info, right_piece, depth + 1, f"_R{depth + 1}")
                    return
        
        # Can't fit and didn't split - skip
        skipped_parts.append(f"{part_info.file_path} (doesn't fit)")
    
    # Process all parts
    try:
        for part_info in part_files:
            spar_profile = None
            if spar_profiles:
                spar_profile = spar_profiles.get(part_info.file_path)
            process_part(part_info, spar_profile)
    except KeyboardInterrupt:
        return NestingResult(
            success=False,
            error_message="Nesting cancelled by user"
        )
    
    # Generate output files for each thickness group
    generated_files: List[str] = []
    total_placed = 0
    base, ext = os.path.splitext(output_base_path)
    
    for thickness_mm, parts in sorted(thickness_groups.items()):
        # Sort parts by height (descending) for shelf algorithm
        # This groups parts of similar height together, minimizing wasted vertical space
        parts.sort(key=lambda p: -p.height)
        
        # Place parts
        placed, unplaced = _place_parts_bottom_up(
            parts,
            params.sheet_width_mm,
            params.sheet_height_mm,
            params.margin_mm,
            params.part_spacing_mm,
        )
        
        # Add unplaced to skipped
        for part in unplaced:
            skipped_parts.append(f"{part.part_name} (no room on sheet)")
        
        if not placed:
            continue
        
        # Create output document
        output_path = f"{base}_{thickness_mm:.1f}mm{ext}"
        
        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = units.MM
        doc.layers.add(name="SHEET", color=7)
        doc.layers.add(name="CUT", color=1)
        doc.layers.add(name="ENGRAVE", color=5)
        doc.layers.add(name="CUTLINE", color=3)
        msp = doc.modelspace()
        
        # Draw sheet boundary
        msp.add_lwpolyline([
            (0, 0),
            (params.sheet_width_mm, 0),
            (params.sheet_width_mm, params.sheet_height_mm),
            (0, params.sheet_height_mm),
        ], close=True, dxfattribs={"layer": "SHEET"})
        
        # Add thickness label
        try:
            text = msp.add_text(
                f"MATERIAL: {thickness_mm:.1f}mm",
                height=5.0,
                dxfattribs={"layer": "ENGRAVE"}
            )
            text.set_placement((params.margin_mm, params.sheet_height_mm - params.margin_mm - 5))
        except Exception:
            pass
        
        # Copy each placed part to the sheet
        for placed_part in placed:
            part = placed_part.part
            print(f"  Placing '{part.part_name}' at ({placed_part.x:.1f}, {placed_part.y:.1f}) size {part.width:.1f}x{part.height:.1f}")
            _copy_entities_to_layout_rotated(
                source_msp=part.doc.modelspace(),
                target_msp=msp,
                offset_x=placed_part.x + part.offset_x,
                offset_y=placed_part.y + part.offset_y,
                rotation_deg=part.rotation_deg,
                rotation_center=part.rotation_center,
            )
            total_placed += 1
        
        doc.saveas(output_path)
        generated_files.append(output_path)
        print(f"Generated {output_path}: {len(placed)} parts at {thickness_mm}mm thickness")
    
    return NestingResult(
        success=True,
        output_files=generated_files,
        placed_count=total_placed,
        total_parts=len(part_files),
        skipped_parts=skipped_parts,
        split_count=split_count,
    )
