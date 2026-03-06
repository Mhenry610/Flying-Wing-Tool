"""
FlightGear aircraft package generator.

Creates a complete FlightGear aircraft package including:
- JSBSim FDM files (via jsbsim_gen)
- aircraft-set.xml configuration
- model.xml wrapper
- model.ac (minimal AC3D 3D model)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.state import Project
from core.jsbsim_gen import (
    JSBSimExportConfig,
    JSBSimExportResult,
    export_jsbsim_project,
)


@dataclass
class FlightGearExportResult:
    """Result of FlightGear package export."""
    package_dir: str
    aircraft_name: str
    set_xml_path: str
    model_xml_path: str
    model_ac_path: str
    jsbsim_result: JSBSimExportResult


def _sanitize_name(name: str) -> str:
    """Sanitize aircraft name for filesystem and FlightGear compatibility."""
    sanitized = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "FlyingWing"


# =============================================================================
# AC3D Model Generator
# =============================================================================

@dataclass
class AC3DVertex:
    x: float
    y: float
    z: float


@dataclass
class AC3DSurface:
    """A polygon surface in AC3D format."""
    vertex_indices: List[int]
    material_index: int = 0


def _generate_wing_vertices(
    half_span: float,
    root_chord: float,
    tip_chord: float,
    sweep_le_deg: float,
    dihedral_deg: float,
    thickness_fraction: float = 0.04,
    x_offset: float = 0.0,
) -> Tuple[List[AC3DVertex], List[AC3DSurface]]:
    """
    Generate vertices and surfaces for a simple slab wing.
    
    FlightGear coordinate system:
    - X: aft (positive toward tail)
    - Y: right (positive toward right wing)
    - Z: up (positive upward)
    
    Args:
        half_span: Half span in meters
        root_chord: Root chord in meters
        tip_chord: Tip chord in meters
        sweep_le_deg: Leading edge sweep in degrees
        dihedral_deg: Dihedral angle in degrees
        thickness_fraction: Thickness as fraction of MAC
        x_offset: X offset to place origin near CG
    
    Returns:
        Tuple of (vertices, surfaces)
    """
    sweep_rad = math.radians(sweep_le_deg)
    dihedral_rad = math.radians(dihedral_deg)
    
    # Calculate mean chord for thickness
    mac = (2.0 / 3.0) * root_chord * (1 + (tip_chord/root_chord) + (tip_chord/root_chord)**2) / (1 + tip_chord/root_chord)
    thickness = mac * thickness_fraction
    half_t = thickness / 2.0
    
    # Spanwise stations
    # y=0 is centerline, positive Y is right wing
    # We'll build right half, then mirror for left half
    
    vertices = []
    
    # Right wing tip (y = +half_span)
    y_tip = half_span
    x_le_tip = half_span * math.tan(sweep_rad) - x_offset
    x_te_tip = x_le_tip + tip_chord
    z_tip = half_span * math.tan(dihedral_rad)
    
    # Right root (y = 0)
    y_root = 0.0
    x_le_root = -x_offset
    x_te_root = root_chord - x_offset
    z_root = 0.0
    
    # Left wing tip (y = -half_span)
    y_ltip = -half_span
    # Same x positions as right tip due to symmetry
    z_ltip = half_span * math.tan(dihedral_rad)  # Same Z (dihedral is symmetric)
    
    # Build vertices: 12 total (4 corners x 3 spanwise stations, top and bottom)
    # Order: left tip, root, right tip for both top and bottom surfaces
    
    # Left tip - leading edge
    vertices.append(AC3DVertex(x_le_tip, y_ltip, z_ltip + half_t))  # 0: top LE
    vertices.append(AC3DVertex(x_le_tip, y_ltip, z_ltip - half_t))  # 1: bot LE
    # Left tip - trailing edge
    vertices.append(AC3DVertex(x_te_tip, y_ltip, z_ltip + half_t))  # 2: top TE
    vertices.append(AC3DVertex(x_te_tip, y_ltip, z_ltip - half_t))  # 3: bot TE
    
    # Root - leading edge
    vertices.append(AC3DVertex(x_le_root, y_root, z_root + half_t))  # 4: top LE
    vertices.append(AC3DVertex(x_le_root, y_root, z_root - half_t))  # 5: bot LE
    # Root - trailing edge
    vertices.append(AC3DVertex(x_te_root, y_root, z_root + half_t))  # 6: top TE
    vertices.append(AC3DVertex(x_te_root, y_root, z_root - half_t))  # 7: bot TE
    
    # Right tip - leading edge
    vertices.append(AC3DVertex(x_le_tip, y_tip, z_tip + half_t))   # 8: top LE
    vertices.append(AC3DVertex(x_le_tip, y_tip, z_tip - half_t))   # 9: bot LE
    # Right tip - trailing edge
    vertices.append(AC3DVertex(x_te_tip, y_tip, z_tip + half_t))   # 10: top TE
    vertices.append(AC3DVertex(x_te_tip, y_tip, z_tip - half_t))   # 11: bot TE
    
    # Build surfaces (quads)
    surfaces = []
    
    # Top surface (left tip -> root -> right tip, looking from above)
    # Left panel top: 0, 4, 6, 2 (CCW when viewed from above)
    surfaces.append(AC3DSurface([0, 2, 6, 4]))
    # Right panel top: 4, 8, 10, 6
    surfaces.append(AC3DSurface([4, 6, 10, 8]))
    
    # Bottom surface (reversed winding)
    # Left panel bottom: 1, 5, 7, 3 -> reversed: 1, 3, 7, 5
    surfaces.append(AC3DSurface([1, 5, 7, 3]))
    # Right panel bottom: 5, 9, 11, 7 -> reversed: 5, 7, 11, 9
    surfaces.append(AC3DSurface([5, 9, 11, 7]))
    
    # Leading edge (front face)
    # Left section: 0, 1, 5, 4
    surfaces.append(AC3DSurface([0, 4, 5, 1]))
    # Right section: 4, 5, 9, 8
    surfaces.append(AC3DSurface([4, 8, 9, 5]))
    
    # Trailing edge (rear face)
    # Left section: 2, 3, 7, 6
    surfaces.append(AC3DSurface([2, 6, 7, 3]))
    # Right section: 6, 7, 11, 10
    surfaces.append(AC3DSurface([6, 10, 11, 7]))
    
    # Left wingtip (end cap)
    surfaces.append(AC3DSurface([0, 1, 3, 2]))
    
    # Right wingtip (end cap)
    surfaces.append(AC3DSurface([8, 10, 11, 9]))
    
    return vertices, surfaces


def generate_ac3d_model(
    project: Project,
    x_cg_offset: float = 0.0,
) -> str:
    """
    Generate AC3D model content for the flying wing.
    
    Args:
        project: The project containing wing geometry
        x_cg_offset: X offset to place model origin near CG
    
    Returns:
        AC3D file content as string
    """
    planform = project.wing.planform
    
    half_span = planform.half_span()
    root_chord = planform.root_chord()
    tip_chord = planform.tip_chord()
    sweep_le_deg = planform.sweep_le_deg
    dihedral_deg = planform.dihedral_deg
    
    # Add center extension if present
    if planform.center_chord_extension_percent > 0:
        root_chord *= (1.0 + planform.center_chord_extension_percent / 100.0)
    
    # Generate wing geometry
    vertices, surfaces = _generate_wing_vertices(
        half_span=half_span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep_le_deg=sweep_le_deg,
        dihedral_deg=dihedral_deg,
        thickness_fraction=0.04,
        x_offset=x_cg_offset,
    )
    
    # Build AC3D file content
    lines = []
    lines.append("AC3Db")
    
    # Material definition (simple gray)
    lines.append('MATERIAL "WingSurface" rgb 0.7 0.7 0.7  amb 0.2 0.2 0.2  emis 0 0 0  spec 0.3 0.3 0.3  shi 32  trans 0')
    
    # World object
    lines.append("OBJECT world")
    lines.append("kids 1")
    
    # Wing mesh object
    lines.append("OBJECT poly")
    lines.append(f'name "wing"')
    lines.append(f"numvert {len(vertices)}")
    
    # Vertex list (convert FG axes X aft, Y right, Z up -> AC3D axes X aft, Y up, Z left)
    for v in vertices:
        lines.append(f"{v.x:.6f} {v.z:.6f} {-v.y:.6f}")
    
    # Surface list
    lines.append(f"numsurf {len(surfaces)}")
    for surf in surfaces:
        lines.append("SURF 0x30")  # Shaded, two-sided
        lines.append(f"mat {surf.material_index}")
        lines.append(f"refs {len(surf.vertex_indices)}")
        for idx in surf.vertex_indices:
            lines.append(f"{idx} 0 0")  # vertex_index u v
    
    lines.append("kids 0")
    
    return "\n".join(lines)


# =============================================================================
# FlightGear XML Generators
# =============================================================================

def generate_set_xml(
    aircraft_name: str,
    description: str = "",
    fdm_name: Optional[str] = None,
) -> str:
    """
    Generate the aircraft-set.xml configuration file.
    
    Args:
        aircraft_name: Name of the aircraft (used for file references)
        description: Human-readable description
        fdm_name: Name of JSBSim FDM file (without .xml), defaults to aircraft_name
    
    Returns:
        XML content as string
    """
    if not fdm_name:
        fdm_name = aircraft_name
    if not description:
        description = f"{aircraft_name} - Flying Wing"
    
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<PropertyList>
  <sim>
    <description>{description}</description>
    <flight-model>jsb</flight-model>
    <aero>{fdm_name}</aero>
    <model>
      <path>Models/model.xml</path>
    </model>

    <view n="0">
      <name>Helicopter</name>
      <type>lookat</type>
      <internal type="bool">false</internal>
      <config>
        <from-model type="bool">true</from-model>
        <from-model-idx type="int">0</from-model-idx>
        <at-model type="bool">true</at-model>
        <at-model-idx type="int">0</at-model-idx>
        <x-offset-m type="double">0</x-offset-m>
        <y-offset-m type="double">3</y-offset-m>
        <z-offset-m type="double">-25</z-offset-m>
        <default-field-of-view-deg type="double">65.0</default-field-of-view-deg>
      </config>
    </view>
  </sim>
</PropertyList>
'''
    return xml_content


def generate_model_xml(ac_filename: str = "model.ac") -> str:
    """
    Generate the model.xml wrapper file.
    
    Args:
        ac_filename: Name of the AC3D model file
    
    Returns:
        XML content as string
    """
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<PropertyList>
  <path>{ac_filename}</path>
  
  <offsets>
    <heading-deg>0</heading-deg>
    <roll-deg>0</roll-deg>
    <pitch-deg>0</pitch-deg>
    <x-m>0</x-m>
    <y-m>0</y-m>
    <z-m>0</z-m>
  </offsets>
  
  <!-- Control surface animations can be added here -->
  
</PropertyList>
'''
    return xml_content


# =============================================================================
# Main Export Function
# =============================================================================

def export_flightgear_package(
    project: Project,
    output_dir: str,
    jsbsim_config: Optional[JSBSimExportConfig] = None,
) -> FlightGearExportResult:
    """
    Export a complete FlightGear aircraft package.
    
    Creates the following structure:
        output_dir/
            AircraftName/
                AircraftName-set.xml
                AircraftName.xml (JSBSim FDM)
                AircraftName_engine.xml
                AircraftName_prop.xml
                Models/
                    model.xml
                    model.ac
    
    Args:
        project: The project to export
        output_dir: Base output directory
        jsbsim_config: Optional JSBSim export configuration
    
    Returns:
        FlightGearExportResult with paths to all generated files
    """
    if jsbsim_config is None:
        jsbsim_config = JSBSimExportConfig()
    
    # Determine aircraft name
    aircraft_name = jsbsim_config.model_name or project.wing.name or "FlyingWing"
    aircraft_name = _sanitize_name(aircraft_name)
    
    # Ensure JSBSim config uses the same name
    jsbsim_config.model_name = aircraft_name
    
    # Create directory structure
    package_dir = os.path.join(output_dir, aircraft_name)
    models_dir = os.path.join(package_dir, "Models")
    engines_dir = os.path.join(package_dir, "Engines")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(engines_dir, exist_ok=True)
    
    # Ensure propulsion file names (JSBSim resolves Engines/ automatically)
    if jsbsim_config.propulsion is not None:
        engine_file = f"{aircraft_name}_engine"
        prop_file = f"{aircraft_name}_prop"
        jsbsim_config.propulsion.engine_file = engine_file
        jsbsim_config.propulsion.propeller_file = prop_file

    # Export JSBSim FDM files to package root
    jsbsim_result = export_jsbsim_project(project, package_dir, jsbsim_config)

    # Copy engine/prop files into Engines/ (JSBSim resolves without path)
    if jsbsim_result.engine_path:
        engine_name = os.path.basename(jsbsim_result.engine_path)
        engine_base = os.path.splitext(engine_name)[0]
        engine_no_ext = os.path.join(os.path.dirname(jsbsim_result.engine_path), engine_base)
        for src in (jsbsim_result.engine_path, engine_no_ext):
            if os.path.exists(src):
                dst = os.path.join(engines_dir, os.path.basename(src))
                with open(src, "rb") as src_handle, open(dst, "wb") as dst_handle:
                    dst_handle.write(src_handle.read())

    if jsbsim_result.propeller_path:
        prop_name = os.path.basename(jsbsim_result.propeller_path)
        prop_base = os.path.splitext(prop_name)[0]
        prop_no_ext = os.path.join(os.path.dirname(jsbsim_result.propeller_path), prop_base)
        for src in (jsbsim_result.propeller_path, prop_no_ext):
            if os.path.exists(src):
                dst = os.path.join(engines_dir, os.path.basename(src))
                with open(src, "rb") as src_handle, open(dst, "wb") as dst_handle:
                    dst_handle.write(src_handle.read())
    
    # Calculate CG offset for model alignment (same logic as jsbsim_gen)
    from services.geometry import AeroSandboxService
    service = AeroSandboxService(project)
    wing = service.build_wing()
    mac = wing.mean_aerodynamic_chord()
    x_np = wing.aerodynamic_center()[0]
    static_margin = project.wing.twist_trim.static_margin_percent
    x_cg = x_np - (static_margin / 100.0) * mac
    
    # Generate AC3D model
    ac3d_content = generate_ac3d_model(project, x_cg_offset=x_cg)
    model_ac_path = os.path.join(models_dir, "model.ac")
    with open(model_ac_path, "w", encoding="utf-8") as f:
        f.write(ac3d_content)
    
    # Generate model.xml wrapper
    model_xml_content = generate_model_xml("model.ac")
    model_xml_path = os.path.join(models_dir, "model.xml")
    with open(model_xml_path, "w", encoding="utf-8") as f:
        f.write(model_xml_content)
    
    # Generate aircraft-set.xml
    description = f"{project.wing.name or 'Flying Wing'} - Generated by Flying Wing Tool"
    set_xml_content = generate_set_xml(
        aircraft_name=aircraft_name,
        description=description,
        fdm_name=aircraft_name,
    )
    set_xml_path = os.path.join(package_dir, f"{aircraft_name}-set.xml")
    with open(set_xml_path, "w", encoding="utf-8") as f:
        f.write(set_xml_content)
    
    return FlightGearExportResult(
        package_dir=os.path.abspath(package_dir),
        aircraft_name=aircraft_name,
        set_xml_path=os.path.abspath(set_xml_path),
        model_xml_path=os.path.abspath(model_xml_path),
        model_ac_path=os.path.abspath(model_ac_path),
        jsbsim_result=jsbsim_result,
    )
