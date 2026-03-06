"""
Flow5 XML and DAT export for flying wing geometry.

This module generates Flow5-compatible xflplane XML files and airfoil .dat files
from the unified Project state. Flow5 is an open-source VLM/Panel code similar to XFLR5.

Key constraints:
- Airfoil .dat files must have ≤240 points (Flow5 limitation)
- Units are in meters (conversion factors = 1)
- Per-section dihedral (not global wing rotation)
- Airfoil files named: {ProjectName}_sec{N}.dat
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.state import Project
    from services.geometry import SpanwiseSection


# === Constants ===
MAX_AIRFOIL_POINTS = 240
DEFAULT_X_PANELS = 15
DEFAULT_Y_PANELS = 5


# === Airfoil Downsampling ===

def downsample_airfoil(coords: np.ndarray, max_points: int = MAX_AIRFOIL_POINTS) -> np.ndarray:
    """
    Downsample airfoil coordinates using arc-length parameterization.
    
    Preserves overall shape by resampling at uniform arc-length intervals.
    Leading edge and trailing edge regions are preserved.
    
    Args:
        coords: Nx2 array of (x, y) airfoil coordinates
        max_points: Maximum number of points in output (default 240)
    
    Returns:
        Mx2 array where M <= max_points
    """
    if len(coords) <= max_points:
        return coords
    
    # Calculate arc length along airfoil
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_arc = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_arc = cumulative_arc[-1]
    
    if total_arc < 1e-10:
        # Degenerate airfoil - return as-is
        return coords[:max_points]
    
    # Create new uniform arc-length spacing
    new_arc = np.linspace(0, total_arc, max_points)
    
    # Interpolate x and y at new arc lengths
    new_x = np.interp(new_arc, cumulative_arc, coords[:, 0])
    new_y = np.interp(new_arc, cumulative_arc, coords[:, 1])
    
    return np.column_stack([new_x, new_y])


# === Local Dihedral Calculation ===

def compute_local_dihedral_deg(sections: List['SpanwiseSection']) -> List[float]:
    """
    Compute local dihedral angle at each section in degrees.
    
    Flow5 uses per-section dihedral rather than global wing rotation.
    Dihedral is computed as atan2(Δz, Δy) between adjacent sections.
    
    Args:
        sections: List of SpanwiseSection from geometry service
    
    Returns:
        List of dihedral angles in degrees, one per section
    """
    if len(sections) < 2:
        return [0.0] * len(sections)
    
    dihedrals = []
    for i in range(len(sections)):
        if i < len(sections) - 1:
            # Forward difference
            dz = sections[i + 1].z_m - sections[i].z_m
            dy = sections[i + 1].y_m - sections[i].y_m
        else:
            # Last section: use backward difference
            dz = sections[i].z_m - sections[i - 1].z_m
            dy = sections[i].y_m - sections[i - 1].y_m
        
        if abs(dy) > 1e-6:
            dihedral_rad = math.atan2(dz, dy)
        else:
            dihedral_rad = 0.0
        
        dihedrals.append(math.degrees(dihedral_rad))
    
    return dihedrals


# === XML Generation ===

def _format_float(value: float, decimals: int = 3) -> str:
    """Format float for XML output."""
    return f"{value:.{decimals}f}"


def generate_flow5_xml(
    project: 'Project',
    model_name: Optional[str] = None,
    x_panels: int = DEFAULT_X_PANELS,
    y_panels: int = DEFAULT_Y_PANELS,
) -> ET.Element:
    """
    Generate Flow5 xflplane XML from Project state.
    
    Args:
        project: Unified Project containing wing geometry
        model_name: Name for the plane (defaults to project.wing.name)
        x_panels: Chordwise panel count per section (default 15)
        y_panels: Spanwise panel count per section (default 5)
    
    Returns:
        ET.Element root node of the xflplane document
    """
    from services.geometry import AeroSandboxService
    
    if model_name is None:
        model_name = project.wing.name or "FlyingWingProject"
    
    # Get spanwise sections
    svc = AeroSandboxService(project)
    sections = svc.spanwise_sections()
    
    if not sections:
        raise ValueError("No sections available. Check project geometry.")
    
    # Compute local dihedral for each section
    local_dihedrals = compute_local_dihedral_deg(sections)
    
    # Build XML structure
    root = ET.Element("xflplane", version="1.0")
    
    # Units block (all factors = 1 for meters)
    units = ET.SubElement(root, "Units")
    ET.SubElement(units, "meter_to_length_unit").text = "1"
    ET.SubElement(units, "m2_to_area_unit").text = "1"
    ET.SubElement(units, "kg_to_mass_unit").text = "1"
    ET.SubElement(units, "ms_to_speed_unit").text = "1"
    ET.SubElement(units, "kgm2_to_inertia_unit").text = "1"
    
    # Plane block
    plane = ET.SubElement(root, "Plane")
    ET.SubElement(plane, "Name").text = model_name
    ET.SubElement(plane, "Description").text = "Generated by Flying Wing Tool"
    
    # Style block
    style = ET.SubElement(plane, "The_Style")
    ET.SubElement(style, "Stipple").text = "SOLID"
    ET.SubElement(style, "PointStyle").text = "NOSYMBOL"
    ET.SubElement(style, "Width").text = "2"
    color = ET.SubElement(style, "Color")
    ET.SubElement(color, "red").text = "200"
    ET.SubElement(color, "green").text = "200"
    ET.SubElement(color, "blue").text = "200"
    ET.SubElement(color, "alpha").text = "255"
    
    # Wing block
    wing = ET.SubElement(plane, "wing")
    ET.SubElement(wing, "Name").text = model_name
    ET.SubElement(wing, "Type").text = "MAINWING"
    
    wing_color = ET.SubElement(wing, "Color")
    ET.SubElement(wing_color, "red").text = "111"
    ET.SubElement(wing_color, "green").text = "131"
    ET.SubElement(wing_color, "blue").text = "157"
    ET.SubElement(wing_color, "alpha").text = "255"
    
    ET.SubElement(wing, "Description").text = ""
    ET.SubElement(wing, "Position").text = "0, 0, 0"
    ET.SubElement(wing, "Tip_Strips").text = "1"
    ET.SubElement(wing, "Rx_angle").text = "0.000"
    ET.SubElement(wing, "Ry_angle").text = "0.000"
    ET.SubElement(wing, "symmetric").text = "true"
    ET.SubElement(wing, "Two_Sided").text = "true"
    ET.SubElement(wing, "Closed_Inner_Side").text = "false"
    ET.SubElement(wing, "AutoInertia").text = "true"
    
    # Inertia block (auto-calculated by Flow5)
    inertia = ET.SubElement(wing, "Inertia")
    ET.SubElement(inertia, "Mass").text = "0.00000"
    ET.SubElement(inertia, "CoG").text = "0, 0, 0"
    ET.SubElement(inertia, "CoG_Ixx").text = "0"
    ET.SubElement(inertia, "CoG_Iyy").text = "0"
    ET.SubElement(inertia, "CoG_Izz").text = "0"
    ET.SubElement(inertia, "CoG_Ixz").text = "0"
    
    # Sections
    sections_elem = ET.SubElement(wing, "Sections")
    
    for i, section in enumerate(sections):
        sec_elem = ET.SubElement(sections_elem, "Section")
        
        ET.SubElement(sec_elem, "y_position").text = _format_float(section.y_m)
        ET.SubElement(sec_elem, "Chord").text = _format_float(section.chord_m)
        ET.SubElement(sec_elem, "xOffset").text = _format_float(section.x_le_m)
        ET.SubElement(sec_elem, "Dihedral").text = _format_float(local_dihedrals[i])
        ET.SubElement(sec_elem, "Twist").text = _format_float(section.twist_deg)
        
        ET.SubElement(sec_elem, "x_number_of_panels").text = str(x_panels)
        ET.SubElement(sec_elem, "x_panel_distribution").text = "COSINE"
        ET.SubElement(sec_elem, "y_number_of_panels").text = str(y_panels)
        ET.SubElement(sec_elem, "y_panel_distribution").text = "COSINE"
        
        # Airfoil reference (matches .dat filename without extension)
        foil_name = f"{model_name}_sec{section.index + 1}"
        ET.SubElement(sec_elem, "Left_Side_FoilName").text = foil_name
        ET.SubElement(sec_elem, "Right_Side_FoilName").text = foil_name
    
    return root


# === DAT File Generation ===

def create_flow5_airfoil_dat_files(
    project: 'Project',
    output_dir: str,
    model_name: Optional[str] = None,
    max_points: int = MAX_AIRFOIL_POINTS,
) -> List[str]:
    """
    Create .dat airfoil files for each section.
    
    Files are named: {model_name}_sec{N}.dat
    Airfoils are downsampled to max_points if needed.
    
    Args:
        project: Unified Project containing wing geometry
        output_dir: Directory to write .dat files
        model_name: Base name for files (defaults to project.wing.name)
        max_points: Maximum points per airfoil (default 240)
    
    Returns:
        List of absolute paths to created .dat files
    """
    from services.geometry import AeroSandboxService
    
    if model_name is None:
        model_name = project.wing.name or "FlyingWingProject"
    
    # Get spanwise sections
    svc = AeroSandboxService(project)
    sections = svc.spanwise_sections()
    
    if not sections:
        raise ValueError("No sections available. Check project geometry.")
    
    os.makedirs(output_dir, exist_ok=True)
    written_files = []
    
    for section in sections:
        # Get airfoil coordinates
        try:
            coords = section.airfoil.coordinates
            if coords is None or len(coords) < 3:
                raise ValueError("Invalid airfoil coordinates")
        except Exception:
            # Fallback: simple diamond airfoil
            coords = np.array([
                [1.0, 0.0],
                [0.5, 0.05],
                [0.0, 0.0],
                [0.5, -0.05],
                [1.0, 0.0],
            ])
        
        # Ensure numpy array
        coords = np.array(coords)
        
        # Downsample if needed
        if len(coords) > max_points:
            coords = downsample_airfoil(coords, max_points)
        
        # Write .dat file
        filename = f"{model_name}_sec{section.index + 1}.dat"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header line: airfoil name (must match XML reference exactly)
            f.write(f"{model_name}_sec{section.index + 1}\n")
            # Coordinate lines: x y (6 decimal places)
            for x, y in coords:
                f.write(f"{float(x):.6f} {float(y):.6f}\n")
        
        written_files.append(os.path.abspath(filepath))
    
    return written_files


# === High-Level Export Function ===

def export_flow5_project(
    project: 'Project',
    output_dir: str,
    model_name: Optional[str] = None,
    max_airfoil_points: int = MAX_AIRFOIL_POINTS,
    x_panels: int = DEFAULT_X_PANELS,
    y_panels: int = DEFAULT_Y_PANELS,
) -> Tuple[str, List[str]]:
    """
    Export complete Flow5 project (XML + airfoil .dat files).
    
    Args:
        project: Unified Project containing wing geometry
        output_dir: Directory for output files
        model_name: Name for the project (defaults to project.wing.name)
        max_airfoil_points: Max points per airfoil file (default 240)
        x_panels: Chordwise panels per section (default 15)
        y_panels: Spanwise panels per section (default 5)
    
    Returns:
        Tuple of (xml_path, [dat_file_paths])
    """
    if model_name is None:
        model_name = project.wing.name or "FlyingWingProject"
    
    # Sanitize model name for filesystem
    safe_name = "".join(c for c in model_name if c.isalnum() or c in "._- ")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate XML
    xml_root = generate_flow5_xml(
        project,
        model_name=safe_name,
        x_panels=x_panels,
        y_panels=y_panels,
    )
    
    # Write XML with proper declaration and DOCTYPE
    xml_path = os.path.join(output_dir, f"{safe_name}.xml")
    
    # Create XML string with declaration and DOCTYPE
    xml_str = ET.tostring(xml_root, encoding='unicode')
    
    # Pretty print using minidom
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")
    
    # Remove the XML declaration added by minidom (we'll add our own with DOCTYPE)
    lines = pretty_xml.split('\n')
    if lines[0].startswith('<?xml'):
        lines = lines[1:]
    
    # Build final XML with proper header
    final_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    final_xml += '<!DOCTYPE flow5>\n'
    final_xml += '\n'.join(line for line in lines if line.strip())
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    # Generate airfoil .dat files
    dat_files = create_flow5_airfoil_dat_files(
        project,
        output_dir,
        model_name=safe_name,
        max_points=max_airfoil_points,
    )
    
    return os.path.abspath(xml_path), dat_files
