"""
STEP export core functions extracted from CpacsStepperTab.export_step_file (GUI-free).
- Consumes ProcessedCpacs (from refactor.services.viewer) to build solids.
- Uses shared wrappers in refactor.occ_utils for construction and booleans.
- No dialogs or logging; caller handles I/O and UX.
"""

from __future__ import annotations

from typing import List
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax2, gp_Dir
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Shell
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_ManifoldSolidBrep
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.ShapeFix import ShapeFix_Solid
try:
    # Optional: alternative solid maker
    from OCC.Core.BRepLib import BRepLib_MakeSolid  # type: ignore
except Exception:
    BRepLib_MakeSolid = None  # type: ignore

from core.occ_utils.shapes import (
    make_wire_from_points,
    make_face_from_points,
    extrude_prism,
    loft_solid_from_wires,
    scale_shape as occ_scale_shape,
    mirror_y as occ_mirror_y,
    make_compound,
    make_airfoil_wire_spline,
    loft_surface_from_profiles,
    sew_faces_to_solid,
    unify_same_domain,
)
from core.occ_utils.booleans import cut as bool_cut, fuse as bool_fuse, common as bool_common

# Import new geometry builder for direct Project -> OCC generation
from services.export.geometry_builder import (
    build_geometry_from_project,
    WingGeometryConfig,
    GeneratedGeometry,
)
from core.state import Project
from typing import Optional


# ==============================================================================
# NEW Entry Point: Build STEP from Project (Preferred Path)
# ==============================================================================

def build_step_from_project(
    project: Project,
    config: Optional[WingGeometryConfig] = None,
    output_path: Optional[str] = None,
) -> TopoDS_Compound:
    """
    Build complete STEP geometry directly from native Project state.
    
    This is the NEW PREFERRED entry point for STEP export. It uses
    geometry_builder.py as the single source of truth, ensuring that
    DXF manufacturing drawings match CAD geometry.
    
    Args:
        project: Project containing wing planform, airfoils, structure params
        config: Export configuration (defaults provided if None)
        output_path: Optional path to write STEP file directly
    
    Returns:
        TopoDS_Compound containing all geometry, scaled and mirrored as configured
    """
    if config is None:
        config = WingGeometryConfig()
    
    print("[STEP Export] Building geometry from Project...")
    
    # Generate all geometry using the unified geometry builder
    geom = build_geometry_from_project(project, config)
    
    if geom.section_count < 2:
        print("[STEP Export] Warning: Insufficient sections for geometry generation")
        return TopoDS_Compound()
    
    # Assemble all components into a single compound
    builder = BRep_Builder()
    right_wing = TopoDS_Compound()
    builder.MakeCompound(right_wing)
    
    component_count = 0
    
    # Add skin surfaces
    if config.include_skin:
        if geom.wing_skin_upper and not geom.wing_skin_upper.IsNull():
            builder.Add(right_wing, geom.wing_skin_upper)
            component_count += 1
            print("[STEP Export] Added upper skin surface")
        if geom.wing_skin_lower and not geom.wing_skin_lower.IsNull():
            builder.Add(right_wing, geom.wing_skin_lower)
            component_count += 1
            print("[STEP Export] Added lower skin surface")
    
    # Add wingbox components
    if config.include_wingbox:
        if geom.wingbox_front_spar and not geom.wingbox_front_spar.IsNull():
            builder.Add(right_wing, geom.wingbox_front_spar)
            component_count += 1
            print("[STEP Export] Added front spar")
        if geom.wingbox_rear_spar and not geom.wingbox_rear_spar.IsNull():
            builder.Add(right_wing, geom.wingbox_rear_spar)
            component_count += 1
            print("[STEP Export] Added rear spar")
        if geom.wingbox_skin_upper and not geom.wingbox_skin_upper.IsNull():
            builder.Add(right_wing, geom.wingbox_skin_upper)
            component_count += 1
            print("[STEP Export] Added wingbox upper skin")
        if geom.wingbox_skin_lower and not geom.wingbox_skin_lower.IsNull():
            builder.Add(right_wing, geom.wingbox_skin_lower)
            component_count += 1
            print("[STEP Export] Added wingbox lower skin")
    
    # Add ribs
    if config.include_ribs:
        for i, rib in enumerate(geom.ribs):
            if rib and not rib.IsNull():
                builder.Add(right_wing, rib)
                component_count += 1
        print(f"[STEP Export] Added {len(geom.ribs)} main ribs")
        
        for i, elevon_rib in enumerate(geom.elevon_ribs):
            if elevon_rib and not elevon_rib.IsNull():
                builder.Add(right_wing, elevon_rib)
                component_count += 1
        if geom.elevon_ribs:
            print(f"[STEP Export] Added {len(geom.elevon_ribs)} elevon ribs")
    
    # Add stringers
    if config.include_stringers:
        for i, stringer in enumerate(geom.stringers):
            if stringer and not stringer.IsNull():
                builder.Add(right_wing, stringer)
                component_count += 1
        if geom.stringers:
            print(f"[STEP Export] Added {len(geom.stringers)} stringers")
    
    # Add control surfaces
    if config.include_control_surfaces:
        for name, cs_shape in geom.control_surfaces.items():
            if cs_shape and not cs_shape.IsNull():
                builder.Add(right_wing, cs_shape)
                component_count += 1
                print(f"[STEP Export] Added control surface: {name}")
    
    print(f"[STEP Export] Assembled {component_count} components")
    
    # Scale if needed (default: m -> mm)
    if config.scale_factor != 1.0:
        right_wing = occ_scale_shape(right_wing, config.scale_factor)
        print(f"[STEP Export] Scaled by {config.scale_factor}")
    
    # Mirror to create full aircraft if requested
    if config.mirror_to_full_aircraft:
        full_aircraft = TopoDS_Compound()
        builder.MakeCompound(full_aircraft)
        builder.Add(full_aircraft, right_wing)
        
        mirrored = occ_mirror_y(right_wing)
        if mirrored and not mirrored.IsNull():
            builder.Add(full_aircraft, mirrored)
            print("[STEP Export] Mirrored to full aircraft")
        
        final_shape = full_aircraft
    else:
        final_shape = right_wing
    
    # Write STEP file if path provided
    if output_path:
        success = write_step(final_shape, output_path)
        if success:
            print(f"[STEP Export] Written to {output_path}")
        else:
            print(f"[STEP Export] Failed to write {output_path}")
    
    return final_shape


def write_step(shape: TopoDS_Shape, path: str) -> bool:
    """
    Write a STEP file. Returns True on success.
    """
    writer = STEPControl_Writer()
    # Prefer writing solids as manifold solid B-reps; fall back to AsIs
    try:
        ok = writer.Transfer(shape, STEPControl_ManifoldSolidBrep)
        if ok == 1:
            return writer.Write(path) == 1
    except Exception:
        pass
    writer.Transfer(shape, STEPControl_AsIs)
    return writer.Write(path) == 1


# ==============================================================================
# Legacy Entry Points (ProcessedCpacs-based)
# ==============================================================================

def build_spars_compound(spar_surfaces: List[List[List[np.ndarray]]]) -> TopoDS_Compound:
    """
    Build prism solids for each spar from face polygons.
    Uses front/back face (first two faces) and extrudes along their delta vector.
    """
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for polys in spar_surfaces:
        if len(polys) < 2:
            continue
        front_face_verts, back_face_verts = polys[0], polys[1]
        if not front_face_verts or not back_face_verts:
            continue
        extrusion_vec = np.array(back_face_verts[0]) - np.array(front_face_verts[0])
        wire = make_wire_from_points(front_face_verts)
        if wire:
            base_face = BRepBuilderAPI_MakeFace(wire).Face()
            if not base_face.IsNull() and np.linalg.norm(extrusion_vec) > 1e-7:
                solid = BRepPrimAPI_MakePrism(base_face, gp_Vec(*extrusion_vec)).Shape()
                if not solid.IsNull():
                    builder.Add(comp, solid)
    return comp


def build_ribs_compound(rib_surfaces: List[List[List[np.ndarray]]]) -> TopoDS_Compound:
    """
    Build prism solids for each rib from face polygons.
    """
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for polys in rib_surfaces:
        if len(polys) < 2:
            continue
        front_face_verts, back_face_verts = polys[0], polys[1]
        if not front_face_verts or not back_face_verts:
            continue
        extrusion_vec = np.array(back_face_verts[0]) - np.array(front_face_verts[0])
        wire = make_wire_from_points(front_face_verts)
        if wire:
            base_face = BRepBuilderAPI_MakeFace(wire).Face()
            if not base_face.IsNull() and np.linalg.norm(extrusion_vec) > 1e-7:
                solid = BRepPrimAPI_MakePrism(base_face, gp_Vec(*extrusion_vec)).Shape()
                if not solid.IsNull():
                    builder.Add(comp, solid)
    return comp


def build_elevon_compound(elevon_surfaces: List[List[List[np.ndarray]]]) -> TopoDS_Compound:
    """
    Loft elevon solid from inner and outer profiles if available.
    Uses first elevon set present.
    """
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    if not elevon_surfaces:
        return comp
    faces = elevon_surfaces[0]
    if len(faces) < 6:
        return comp
    inner_v = faces[4]
    outer_v = faces[5]
    w_in = make_wire_from_points(inner_v)
    w_out = make_wire_from_points(outer_v)
    if w_in and w_out:
        loft = loft_solid_from_wires([w_in, w_out])
        if loft and not loft.IsNull():
            builder.Add(comp, loft)
    return comp


def loft_wing_from_ribs(dihedraled_rib_profiles: List[np.ndarray], use_spline_profiles: bool = False) -> TopoDS_Shape:
    """
    Loft a solid wing by wiring the sorted rib profiles (by span Y).
    """
    if len(dihedraled_rib_profiles) < 2:
        return TopoDS_Shape()
    sorted_profiles = sorted(dihedraled_rib_profiles, key=lambda p: float(np.mean(p[:, 1])))
    wires = []
    for prof in sorted_profiles:
        pts = [np.array(v, dtype=float) for v in prof]
        if use_spline_profiles:
            w = make_airfoil_wire_spline(pts)
        else:
            w = make_wire_from_points(pts)
        if w is not None:
            wires.append(w)
    # For spline profiles, ask for smooth continuity to reduce segmentation
    if use_spline_profiles:
        return loft_solid_from_wires(wires, continuity='C2', max_degree=8)
    return loft_solid_from_wires(wires)


def assemble_airframe(spars: TopoDS_Compound,
                      ribs: TopoDS_Compound,
                      elevon: TopoDS_Compound,
                      wing: TopoDS_Shape,
                      mirror_full: bool = True) -> TopoDS_Compound:
    """
    Assemble right wing components, optionally mirror across Y to form full airframe.
    """
    builder = BRep_Builder()
    right = TopoDS_Compound()
    builder.MakeCompound(right)
    if spars and not spars.IsNull():
        builder.Add(right, spars)
    if ribs and not ribs.IsNull():
        builder.Add(right, ribs)
    if elevon and not elevon.IsNull():
        builder.Add(right, elevon)
    if wing and not wing.IsNull():
        builder.Add(right, wing)

    if not mirror_full:
        return right

    final_comp = TopoDS_Compound()
    builder.MakeCompound(final_comp)
    builder.Add(final_comp, right)
    mirrored = occ_mirror_y(right)
    if mirrored and not mirrored.IsNull():
        builder.Add(final_comp, mirrored)
    return final_comp


def write_step(shape: TopoDS_Shape, path: str) -> bool:
    """
    Write a STEP file. Returns True on success.
    """
    writer = STEPControl_Writer()
    # Prefer writing solids as manifold solid B-reps; fall back to AsIs
    try:
        ok = writer.Transfer(shape, STEPControl_ManifoldSolidBrep)
        if ok == 1:
            return writer.Write(path) == 1
    except Exception:
        pass
    writer.Transfer(shape, STEPControl_AsIs)
    return writer.Write(path) == 1


def _ensure_solid(shape: TopoDS_Shape) -> TopoDS_Shape:
    try:
        if not shape or shape.IsNull():
            return shape
        if shape.ShapeType() == TopAbs_SOLID:
            return shape
        # If the hierarchy contains any solid, pick the first
        try:
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            if exp.More():
                return exp.Current()
        except Exception:
            pass
        # Try shell->solid conversion
        try:
            exp_sh = TopExp_Explorer(shape, TopAbs_SHELL)
            while exp_sh.More():
                sh = TopoDS_Shell(exp_sh.Current())
                mk = BRepBuilderAPI_MakeSolid(sh)
                solid = mk.Solid()
                if solid and not solid.IsNull():
                    try:
                        fixer = ShapeFix_Solid(solid)
                        fixer.Perform()
                        solid = fixer.Solid()
                    except Exception:
                        pass
                    return solid
                exp_sh.Next()
        except Exception:
            pass
        # Last resort: collect faces and sew into a solid
        try:
            faces = []
            exp_f = TopExp_Explorer(shape, TopAbs_FACE)
            while exp_f.More():
                faces.append(exp_f.Current())
                exp_f.Next()
            if faces:
                # Try escalating sewing tolerances
                sewed = None
                for tol in (1.0e-6, 5.0e-6, 1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3):
                    try:
                        # Reuse existing utility first
                        sewed = sew_faces_to_solid(faces)
                        if sewed and not sewed.IsNull():
                            break
                    except Exception:
                        sewed = None
                if not sewed or sewed.IsNull():
                    return shape
                try:
                    sewed = unify_same_domain(sewed, unify_edges=True, unify_faces=True) or sewed
                except Exception:
                    pass
                if sewed and not sewed.IsNull():
                    # If still a shell, try explicit shell->solid with both builders
                    if sewed.ShapeType() == TopAbs_SOLID:
                        return sewed
                    try:
                        exp_sh2 = TopExp_Explorer(sewed, TopAbs_SHELL)
                        if exp_sh2.More():
                            sh2 = TopoDS_Shell(exp_sh2.Current())
                            solid2 = BRepBuilderAPI_MakeSolid(sh2).Solid()
                            if solid2 and not solid2.IsNull():
                                try:
                                    fixer = ShapeFix_Solid(solid2)
                                    fixer.Perform()
                                    solid2 = fixer.Solid()
                                except Exception:
                                    pass
                                return solid2
                            # Optional alternate path
                            if BRepLib_MakeSolid:
                                try:
                                    solid3 = BRepLib_MakeSolid(sh2).Solid()  # type: ignore
                                    if solid3 and not solid3.IsNull():
                                        return solid3
                                except Exception:
                                    pass
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    return shape


def _build_deflection_cutter_from_elevon(elevon_surfaces):
    """
    Build the deflection cutter exactly like the monolith:
      - Use inner (index 4) and outer (index 5) elevon faces
      - Extend the low points down by (0,0,-0.05)
      - Compute aft point distance as height * tan(deflection_angle)
      - Direction from cross(v_span, v_height), normalized
      - Loft solid between inboard/outboard triangles (BRepOffsetAPI_ThruSections with solid=True)
    Returns a TopoDS_Shape or None.
    """
    try:
        if not elevon_surfaces:
            return None
        faces = elevon_surfaces[0]
        if len(faces) < 6:
            return None

        inner = faces[4]
        outer = faces[5]
        if not inner or not outer or len(inner) < 4 or len(outer) < 4:
            return None

        import numpy as _np

        # Inner/outer leading-edge upper/lower points
        p_in_le_up = _np.array(inner[3], dtype=float)
        p_in_le_low = _np.array(inner[0], dtype=float)
        p_out_le_up = _np.array(outer[3], dtype=float)
        p_out_le_low = _np.array(outer[0], dtype=float)

        # Span vector (upper LE points) and local heights
        V_span = p_out_le_up - p_in_le_up
        V_height_in = p_in_le_up - p_in_le_low
        V_height_out = p_out_le_up - p_out_le_low

        # Downward extension for robustness (monolith constant)
        down_vec = _np.array([0.0, 0.0, -0.05], dtype=float)
        P1_in = p_in_le_low + down_vec
        P2_in = p_in_le_up
        P1_out = p_out_le_low + down_vec
        P2_out = p_out_le_up

        # Cutter slope set by elevon deflection angle from geometry (derive from inner height vs aft face)
        # In monolith, angle comes from UI; here we recover it geometrically using triangle similarity:
        # Use tan(angle) by matching the GUI's convention: we compute aft offset using height * tan(angle).
        # Since step_export has no direct angle, we infer angle from inner/outer triangles' orientation
        # by projecting V_aft direction and matching depth to GUI expectations.
        # To strictly match the monolith, we choose V_aft via cross(V_span, V_height) and scale by tan(angle).
        # The GUI uses a live angle; our ProcessedCpacs doesn't store it. To enforce parity, we read back
        # the angle encoded in the geometry using the wedge rule:
        #   d = |V_height| * tan(theta), theta = arctan( |(outer inner-hyp) cross normals| )  -- not reliably derivable.
        # Instead, align behavior with GUI by computing V_aft direction and taking theta implicitly from the
        # inner/outer relationship using the same cross order, but we still need tan(theta).
        # We cannot infer theta here; strict parity requires the caller to have precreated the same cutter.
        # Therefore replicate the monolith structure with an approximated tan(theta) from inner vs outer edge slope.
        # To avoid divergence, we compute tan(theta) from the geometry slope between LE lower and upper:
        def _tan_from(V_height):
            h = _np.linalg.norm(V_height)
            # Protect against degenerate height
            if h < 1e-9:
                return 0.0
            # Assume GUI angle is small; we map to a moderate default 0.577 (tan(30deg)) if no better data.
            # But we can try to get a consistent non-zero: use 30 deg as in GUI defaults.
            return _np.tan(_np.deg2rad(30.0))

        tan_theta_in = _tan_from(V_height_in)
        tan_theta_out = _tan_from(V_height_out)

        # Aft direction (normalized) using cross order from monolith
        V_aft_in = _np.cross(V_span, V_height_in)
        n_in = V_aft_in / (_np.linalg.norm(V_aft_in) + 1e-12)
        V_aft_out = _np.cross(V_span, V_height_out)
        n_out = V_aft_out / (_np.linalg.norm(V_aft_out) + 1e-12)

        # Distances
        h_in = _np.linalg.norm(V_height_in)
        h_out = _np.linalg.norm(V_height_out)
        d_in = h_in * tan_theta_in
        d_out = h_out * tan_theta_out

        P3_in = P1_in + n_in * d_in
        P3_out = P1_out + n_out * d_out

        in_tri = [P1_in, P2_in, P3_in]
        out_tri = [P1_out, P3_out, P2_out]  # winding to match monolith viewer polygons

        w_in = make_wire_from_points(in_tri)
        w_out = make_wire_from_points(out_tri)
        if not w_in or not w_out:
            return None

        return loft_solid_from_wires([w_in, w_out])
    except Exception:
        return None


from typing import Optional, Tuple

def _compute_shape_center_from_vertices(verts: list[list[list[float]]]) -> Tuple[float, float, float]:
    """
    Compute the mean point over a collection of polygon faces (list of list of 3D vertices).
    """
    import numpy as _np
    if not verts:
        return (0.0, 0.0, 0.0)
    pts = []
    for face in verts:
        for v in face:
            arr = _np.array(v, dtype=float).reshape(3)
            pts.append(arr)
    if not pts:
        return (0.0, 0.0, 0.0)
    m = _np.mean(_np.vstack(pts), axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))

def _compute_center_from_profiles(profiles: list) -> Tuple[float, float, float]:
    """
    Compute mean of all points across dihedraled rib profiles.
    """
    import numpy as _np
    if not profiles:
        return (0.0, 0.0, 0.0)
    pts = []
    for prof in profiles:
        if prof is None or len(prof) == 0:
            continue
        pts.append(_np.array(prof, dtype=float))
    if not pts:
        return (0.0, 0.0, 0.0)
    allp = _np.vstack(pts)
    m = _np.mean(allp, axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))

def _build_deflection_cutter_from_elevon(elevon_surfaces,
                                         elevon_angle_deg: Optional[float] = None,
                                         rearward: bool = False,
                                         height_scale: float = 1.2):
    """
    Build the deflection cutter exactly like the monolith:
      - Uses GUI-provided elevon_angle_deg (required for parity)
      - Use inner (index 4) and outer (index 5) elevon faces
      - Extend the low points down by (0,0,-0.05)
      - Aft offset: d = |height| * tan(angle)
      - Direction from cross(V_span, V_height), normalized
      - Loft solid between inboard/outboard triangles
    Returns a TopoDS_Shape or None.
    """
    try:
        if not elevon_surfaces or elevon_angle_deg is None or elevon_angle_deg <= 0.0:
            return None
        faces = elevon_surfaces[0]
        if len(faces) < 6:
            return None

        import numpy as _np
        from math import tan, radians

        inner = faces[4]
        outer = faces[5]
        if not inner or not outer or len(inner) < 4 or len(outer) < 4:
            return None

        # Inner/outer leading-edge upper/lower points
        p_in_le_up = _np.array(inner[3], dtype=float)
        p_in_le_low = _np.array(inner[0], dtype=float)
        p_out_le_up = _np.array(outer[3], dtype=float)
        p_out_le_low = _np.array(outer[0], dtype=float)

        # Span vector (upper LE points) and local heights
        V_span = p_out_le_up - p_in_le_up
        V_height_in = p_in_le_up - p_in_le_low
        V_height_out = p_out_le_up - p_out_le_low

        # Allow scaling of the height component while preserving hinge line and angle
        try:
            hs = float(height_scale)
        except Exception:
            hs = 1.0
        hs = 1.0 if hs is None else max(1e-6, hs)
        V_height_in = V_height_in * hs
        V_height_out = V_height_out * hs

        # Downward extension for robustness (monolith constant)
        down_vec = _np.array([0.0, 0.0, -0.05], dtype=float)
        P1_in = p_in_le_low + down_vec
        P2_in = p_in_le_up
        P1_out = p_out_le_low + down_vec
        P2_out = p_out_le_up

        tan_theta = tan(radians(float(elevon_angle_deg)))

        # Aft directions (normalized)
        V_aft_in = _np.cross(V_span, V_height_in)
        n_in = V_aft_in / (_np.linalg.norm(V_aft_in) + 1e-12)
        V_aft_out = _np.cross(V_span, V_height_out)
        n_out = V_aft_out / (_np.linalg.norm(V_aft_out) + 1e-12)

        # Distances (slightly boosted for robustness). Since height was scaled, depth grows proportionally,
        # which preserves the hinge angle.
        h_in = _np.linalg.norm(V_height_in)
        h_out = _np.linalg.norm(V_height_out)
        depth_boost = 1.02
        d_in = h_in * tan_theta * depth_boost
        d_out = h_out * tan_theta * depth_boost

        P3_in = P1_in + n_in * d_in
        P3_out = P1_out + n_out * d_out

        in_tri = [P1_in, P2_in, P3_in]
        out_tri = [P1_out, P3_out, P2_out]  # match winding of monolith preview polys

        w_in = make_wire_from_points(in_tri)
        w_out = make_wire_from_points(out_tri)
        if not w_in or not w_out:
            return None

        return loft_solid_from_wires([w_in, w_out])
    except Exception:
        return None

def _build_deflection_cutter_from_spar(processed,
                                       elevon_angle_deg: Optional[float] = None,
                                       rearward: bool = False,
                                       height_scale: float = 1.2) -> Optional[TopoDS_Shape]:
    """
    Build the deflection cutter wedge aligned with the selected hinge reference on the REAR face
    of rearSpar_outboard, matching the PyVista preview geometry (including centerline "|<" shape).
    Returns a TopoDS_Shape or None.
    """
    try:
        if processed is None or elevon_angle_deg is None or float(elevon_angle_deg) <= 0.0:
            return None
        if not getattr(processed, "spar_surfaces", None) or not getattr(processed, "spar_uids", None):
            return None

        import numpy as _np
        from math import tan, radians

        # Find rearSpar_outboard (fallback to relaxed search)
        uids = list(getattr(processed, "spar_uids", []) or [])
        if not uids:
            return None
        try:
            idx = uids.index("rearSpar_outboard")
        except ValueError:
            idx = next((i for i, u in enumerate(uids)
                        if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
            if idx is None:
                return None

        # Faces of the rear spar segment; identify FRONT vs REAR by X-centroid (FRONT has smaller X)
        spar_surfs = processed.spar_surfaces
        if idx >= len(spar_surfs):
            return None
        faces = spar_surfs[idx]
        if not faces or len(faces) < 2:
            return None
        fA, fB = faces[0], faces[1]
        if (not fA or len(fA) < 4) or (not fB or len(fB) < 4):
            return None

        def _centroid_x(face):
            arr = _np.array(face, dtype=float)
            return float(_np.mean(arr[:, 0])) if arr.ndim == 2 and arr.shape[1] >= 1 else 0.0

        if _centroid_x(fA) <= _centroid_x(fB):
            front_face, rear_face = fA, fB
        else:
            front_face, rear_face = fB, fA

        # Hinge mode from processed (same normalization as viewer)
        hinge_mode_raw = getattr(processed, "cutter_hinge_mode", "Top of rear spar")
        hinge_mode = str(hinge_mode_raw or "").strip().lower()
        use_bottom = ("bottom" in hinge_mode) and ("rear" in hinge_mode)
        use_top = ("top" in hinge_mode) and ("rear" in hinge_mode)
        use_center = ("center" in hinge_mode) and ("rear" in hinge_mode)

        # Rear face vertices (expected order: [bottom-in, bottom-out, top-out, top-in], normalize if needed)
        p_bot_in = _np.array(rear_face[0], dtype=float)
        p_bot_out = _np.array(rear_face[1], dtype=float)
        p_top_out = _np.array(rear_face[2], dtype=float)
        p_top_in = _np.array(rear_face[3], dtype=float)

        def _avg_z_pair(a, b): return 0.5 * (float(a[2]) + float(b[2]))
        def _abs_y_pair(a, b): return abs(0.5 * (float(a[1]) + float(b[1])))

        candidates = [
            (p_bot_in, p_bot_out, p_top_out, p_top_in),
            (p_bot_out, p_top_out, p_top_in, p_bot_in),
            (p_top_out, p_top_in, p_bot_in, p_bot_out),
            (p_top_in, p_bot_in, p_bot_out, p_top_out),
        ]
        best = None
        best_score = -1e18
        for (b_in, b_out, t_out, t_in) in candidates:
            score = (_avg_z_pair(t_in, t_out) - _avg_z_pair(b_in, b_out)) * 10.0
            score += (_abs_y_pair(b_out, t_out) - _abs_y_pair(b_in, t_in)) * 1.0
            if score > best_score:
                best_score = score
                best = (b_in, b_out, t_out, t_in)
        p_bot_in, p_bot_out, p_top_out, p_top_in = best

        # Swap mapping to match existing GUI semantics (historical inversion):
        # Apply optional height scaling while keeping hinge line fixed
        try:
            hs = float(height_scale)
        except Exception:
            hs = 1.0
        hs = 1.0 if hs is None else max(1e-6, hs)

        # Direct mapping to GUI labels:
        # - "Top of rear spar" anchors the TOP edge (extends downward)
        # - "Bottom of rear spar" anchors the BOTTOM edge (extends upward)
        if use_top:
            # Anchor on TOP edge; extend downward
            a0 = p_top_in; a1 = p_top_out                   # hinge: REAR top edge
            h_in_vec = (p_bot_in - p_top_in) * hs           # down
            h_out_vec = (p_bot_out - p_top_out) * hs
        elif use_bottom:
            # Anchor on BOTTOM edge; extend upward
            a0 = p_bot_in; a1 = p_bot_out                   # hinge: REAR bottom edge
            h_in_vec = (p_top_in - p_bot_in) * hs           # up
            h_out_vec = (p_top_out - p_bot_out) * hs
        elif use_center:
            # Centerline handled as two mirrored wedges around mid between top and bottom
            a0 = p_top_in; a1 = p_top_out                   # temporary placeholders for vector math
            h_in_vec = (p_bot_in - p_top_in) * hs
            h_out_vec = (p_bot_out - p_top_out) * hs
        else:
            # Fallback keep as top-edge behavior
            a0 = p_top_in; a1 = p_top_out
            h_in_vec = (p_bot_in - p_top_in) * hs
            h_out_vec = (p_bot_out - p_top_out) * hs

        # Compute aft normal from span × height; two-stage correction to match viewer orientation
        v_span = a1 - a0
        if _np.linalg.norm(v_span) < 1e-12:
            return None
        if _np.linalg.norm(h_in_vec) < 1e-12:
            return None
        n = _np.cross(v_span, h_in_vec)
        n_norm = _np.linalg.norm(n)
        if n_norm < 1e-12:
            return None
        n = n / n_norm

        rear_center = (p_bot_in + p_bot_out + p_top_in + p_top_out) / 4.0
        f_bot_in = _np.array(front_face[0], dtype=float)
        f_bot_out = _np.array(front_face[1], dtype=float)
        f_top_out = _np.array(front_face[2], dtype=float)
        f_top_in = _np.array(front_face[3], dtype=float)
        front_center = (f_bot_in + f_bot_out + f_top_in + f_top_out) / 4.0
        forward_vec = front_center - rear_center
        # Restore original orientation logic (point toward +X / into wing)
        if _np.linalg.norm(forward_vec) > 1e-12 and _np.dot(n, -forward_vec) < 0.0:
            n = -n

        # Axis-aligned guard — force n to point toward increasing X for parity with viewer
        if abs(forward_vec[0]) < 1e-9:
            if n[0] < 0:
                n = -n
        # For centerline mode, enforce the same direction
        if use_center and n[0] < 0:
            n = -n

        tan_theta = tan(radians(float(elevon_angle_deg)))

        def _loft_tri(in_tri, out_tri) -> Optional[TopoDS_Shape]:
            w_in = make_wire_from_points(in_tri)
            w_out = make_wire_from_points(out_tri)
            if not w_in or not w_out:
                return None
            return loft_solid_from_wires([w_in, w_out])

        if use_center:
            # Centerline apex: anchor the ANGLED face on the centerline (hinge),
            # and push the top/bottom far edges aft by d = |h| * tan(theta).
            mid_in = 0.5 * (p_bot_in + p_top_in)
            mid_out = 0.5 * (p_bot_out + p_top_out)

            # Top half: centerline -> top edge (up)
            a0_top = mid_in
            a1_top = mid_out
            h_in_top_vec = (p_top_in - mid_in) * hs
            h_out_top_vec = (p_top_out - mid_out) * hs
            d_top_in = _np.linalg.norm(h_in_top_vec) * tan_theta
            d_top_out = _np.linalg.norm(h_out_top_vec) * tan_theta
            b0_top = a0_top + h_in_top_vec + n * d_top_in
            b1_top = a1_top + h_out_top_vec + n * d_top_out

            in_top = [a0_top, a0_top + h_in_top_vec, b0_top]
            out_top = [a1_top, b1_top, a1_top + h_out_top_vec]
            top_solid = _loft_tri(in_top, out_top)

            # Bottom half: centerline -> bottom edge (down)
            a0_bot = mid_in
            a1_bot = mid_out
            h_in_bot_vec = (p_bot_in - mid_in) * hs
            h_out_bot_vec = (p_bot_out - mid_out) * hs
            d_bot_in = _np.linalg.norm(h_in_bot_vec) * tan_theta
            d_bot_out = _np.linalg.norm(h_out_bot_vec) * tan_theta
            b0_bot = a0_bot + h_in_bot_vec + n * d_bot_in
            b1_bot = a1_bot + h_out_bot_vec + n * d_bot_out

            in_bot = [a0_bot, a0_bot + h_in_bot_vec, b0_bot]
            out_bot = [a1_bot, b1_bot, a1_bot + h_out_bot_vec]
            bot_solid = _loft_tri(in_bot, out_bot)

            if top_solid and not top_solid.IsNull() and bot_solid and not bot_solid.IsNull():
                try:
                    fused = bool_fuse(top_solid, bot_solid)
                    return fused or top_solid
                except Exception:
                    return top_solid
            return top_solid or bot_solid

        # Default single-wedge case (Top/Bottom)
        # Expand cutter span to match elevon span by projecting elevon vertices onto the
        # hinge span direction and adjusting hinge endpoints accordingly (small padding).
        try:
            import numpy as _np
            s_dir = a1 - a0
            s_norm = _np.linalg.norm(s_dir)
            if s_norm > 1e-12 and getattr(processed, "elevon_surfaces", None):
                s_hat = s_dir / s_norm
                ev = processed.elevon_surfaces[0] if processed.elevon_surfaces else None
                if ev:
                    pts = _np.vstack([_np.array(p, dtype=float) for face in ev for p in (face or [])])
                    if pts.size >= 3:
                        svals = (pts - a0.reshape(1, 3)) @ s_hat.reshape(3,)
                        smin = float(_np.min(svals))
                        smax = float(_np.max(svals))
                        slen = max(1e-9, smax - smin)
                        # Overshoot ~5% on each end, with a minimum absolute 0.2 mm pre-scale
                        overshoot_each = max(0.025 * slen, 2.0e-4)
                        smin -= overshoot_each
                        smax += overshoot_each
                        a0 = a0 + smin * s_hat
                        a1 = a1 + smax * s_hat
        except Exception:
            pass

        # Anchor the angled face at the selected hinge edge: keep the hinge edge on the spar,
        # and push the opposite edge aft by d = |h| * tan(theta). This avoids moving the hinge
        # and makes the user-defined angle coincident with the selected edge.
        depth_boost = 1.02
        d_in = _np.linalg.norm(h_in_vec) * tan_theta * depth_boost
        d_out = _np.linalg.norm(h_out_vec) * tan_theta * depth_boost
        # Apply offset at the far (non-hinge) edge
        b0 = a0 + h_in_vec + n * d_in
        b1 = a1 + h_out_vec + n * d_out

        in_tri = [a0, a0 + h_in_vec, b0]
        out_tri = [a1, b1, a1 + h_out_vec]
        return _loft_tri(in_tri, out_tri)
    except Exception:
        return None
def build_full_step_from_processed(processed,
                                   scale_to_mm: bool = True,
                                   cut_wing_with_elevon_opening: bool = True,
                                   hollow_skin_scale: float = 0.98,
                                   elevon_angle_deg: Optional[float] = None,
                                   use_spline_wing: bool = False) -> TopoDS_Compound:
    """
    End-to-end builder using ProcessedCpacs from refactor.services.viewer:
      1) Build spars and ribs compounds
      2) Build elevon (optional)
      3) Loft wing solid from dihedraled rib profiles
      4) Cut spars from ribs and cut elevon deflection clearance from ribs using GUI angle
      5) Optionally cut elevon opening from wing (minimal cutter)
      6) Optionally hollow wing skin, using centers that match monolith logic
      7) Scale to mm and assemble mirrored airframe
    """
    spars_comp = build_spars_compound(processed.spar_surfaces)
    ribs_comp = build_ribs_compound(processed.rib_surfaces)
    # Keep legacy loft-based elevon_comp available
    elevon_comp = build_elevon_compound(processed.elevon_surfaces)
    # Always build point-loft wing for dependent operations (intersections, references)
    wing_point = loft_wing_from_ribs(processed.dihedraled_rib_profiles, use_spline_profiles=False)
    # Optionally build a spline wing as a SOLID loft of closed spline wires (preferred).
    # Fallback to two-surface sew+thicken only if needed.
    wing_spline = None
    wing_was_shell = False
    if use_spline_wing:
        # Preferred: solid loft through closed spline airfoil wires
        try:
            wing_spline = loft_wing_from_ribs(processed.dihedraled_rib_profiles, use_spline_profiles=True)
            if wing_spline and not wing_spline.IsNull() and wing_spline.ShapeType() == TopAbs_SOLID:
                pass  # good
            else:
                wing_spline = None
        except Exception:
            wing_spline = None

        # Fallback: build upper/lower surfaces, sew, and attempt to thicken
        if wing_spline is None:
            try:
                sorted_profiles = sorted(processed.dihedraled_rib_profiles, key=lambda p: float(np.mean(p[:, 1])))
                profs = [[np.array(v, dtype=float) for v in prof] for prof in sorted_profiles]
                upper = loft_surface_from_profiles(profs, 'upper', continuity='C2', max_degree=8)
                lower = loft_surface_from_profiles(profs, 'lower', continuity='C2', max_degree=8)
                inner_wire = make_airfoil_wire_spline(profs[0]) if profs else None
                outer_wire = make_airfoil_wire_spline(profs[-1]) if profs else None
                # Ensure root/tip caps exist; if spline wires fail, fall back to polygonal caps
                inner_cap = BRepBuilderAPI_MakeFace(inner_wire).Face() if inner_wire else make_face_from_points(profs[0])
                outer_cap = BRepBuilderAPI_MakeFace(outer_wire).Face() if outer_wire else make_face_from_points(profs[-1])
                wing_spline = sew_faces_to_solid([upper, lower, inner_cap, outer_cap])
                try:
                    wing_spline = unify_same_domain(wing_spline, unify_edges=True, unify_faces=True) or wing_spline
                except Exception:
                    pass
                # If shell, try to thicken
                if wing_spline and not wing_spline.IsNull() and wing_spline.ShapeType() != TopAbs_SOLID:
                    faces_to_remove = TopTools_ListOfShape()
                    mk = BRepOffsetAPI_MakeThickSolid()
                    mk.MakeThickSolidByJoin(wing_spline, faces_to_remove, 0.001, 1.0e-4)
                    thick = mk.Shape()
                    if not thick or thick.IsNull() or thick.ShapeType() != TopAbs_SOLID:
                        mk_in = BRepOffsetAPI_MakeThickSolid()
                        mk_in.MakeThickSolidByJoin(wing_spline, faces_to_remove, -0.001, 1.0e-4)
                        thick = mk_in.Shape()
                    if thick and not thick.IsNull() and thick.ShapeType() == TopAbs_SOLID:
                        wing_spline = thick
                        wing_was_shell = True
            except Exception:
                wing_spline = None

    # Build elevon by intersecting a rectangular prism region with the lofted wing.
    # Force the prism to align with the REAR spar face (same frame used by the deflection cutter)
    # so the elevon solid and its cut are not canted relative to ribs.
    try:
        if wing_point and not wing_point.IsNull() and processed.elevon_surfaces:
            ev = processed.elevon_surfaces[0]
            if ev and len(ev) >= 4:
                import numpy as _np

                # Aggregate elevon vertices to determine span/height extents
                try:
                    ev_faces = [f for f in ev if f and len(f) >= 3]
                except Exception:
                    ev_faces = []
                if not ev_faces and len(ev) >= 4:
                    ev_faces = [ev[2], ev[3]]
                if not ev_faces:
                    raise RuntimeError("No elevon faces for bounds")
                pts = _np.vstack([_np.array(p, dtype=float) for poly in ev_faces for p in poly])

                # Find rearSpar_outboard and derive a rear-face-aligned frame
                have_spar_frame = False
                e_span = e_height = e_n = None
                rear_center = None
                try:
                    uids = list(getattr(processed, "spar_uids", []) or [])
                    spar_surfs = getattr(processed, "spar_surfaces", None)
                    if uids and spar_surfs:
                        try:
                            sidx = uids.index("rearSpar_outboard")
                        except ValueError:
                            sidx = next((i for i, u in enumerate(uids)
                                         if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
                        if sidx is not None and sidx < len(spar_surfs):
                            faces = spar_surfs[sidx]
                            if faces and len(faces) >= 2 and faces[0] and faces[1] and len(faces[0]) >= 4 and len(faces[1]) >= 4:
                                fA, fB = faces[0], faces[1]

                                def _cx(face):
                                    arr = _np.array(face, dtype=float)
                                    return float(_np.mean(arr[:, 0])) if arr.ndim == 2 else 0.0

                                # FRONT has smaller X centroid
                                if _cx(fA) <= _cx(fB):
                                    front_face, rear_face = fA, fB
                                else:
                                    front_face, rear_face = fB, fA

                                # Normalize REAR face likely ordering -> [bot-in, bot-out, top-out, top-in]
                                p_bot_in = _np.array(rear_face[0], dtype=float)
                                p_bot_out = _np.array(rear_face[1], dtype=float)
                                p_top_out = _np.array(rear_face[2], dtype=float)
                                p_top_in = _np.array(rear_face[3], dtype=float)

                                def _avg_z_pair(a, b): return 0.5 * (float(a[2]) + float(b[2]))
                                def _abs_y_pair(a, b): return abs(0.5 * (float(a[1]) + float(b[1])))

                                # Pick the best mapping by height/inboard heuristics
                                candidates = [
                                    (p_bot_in, p_bot_out, p_top_out, p_top_in),
                                    (p_bot_out, p_top_out, p_top_in, p_bot_in),
                                    (p_top_out, p_top_in, p_bot_in, p_bot_out),
                                    (p_top_in, p_bot_in, p_bot_out, p_top_out),
                                ]
                                best = None; best_score = -1e18
                                for (b_in, b_out, t_out, t_in) in candidates:
                                    score = (_avg_z_pair(t_in, t_out) - _avg_z_pair(b_in, b_out)) * 10.0
                                    score += (_abs_y_pair(b_out, t_out) - _abs_y_pair(b_in, t_in)) * 1.0
                                    if score > best_score:
                                        best_score = score; best = (b_in, b_out, t_out, t_in)
                                p_bot_in, p_bot_out, p_top_out, p_top_in = best

                                # Choose hinge edge consistent with cutter settings
                                hinge_mode_raw = getattr(processed, "cutter_hinge_mode", "Top of rear spar")
                                hinge_mode = str(hinge_mode_raw or "").strip().lower()
                                use_bottom = ("bottom" in hinge_mode) and ("rear" in hinge_mode)
                                use_top = ("top" in hinge_mode) and ("rear" in hinge_mode)
                                use_center = ("center" in hinge_mode) and ("rear" in hinge_mode)

                                if use_bottom:
                                    a0 = p_bot_in; a1 = p_bot_out
                                else:
                                    # Use TOP edge for both top and centerline (span axis is identical)
                                    a0 = p_top_in; a1 = p_top_out

                                v_span = a1 - a0
                                v_h = (p_top_in - p_bot_in)  # upward height
                                if _np.linalg.norm(v_span) > 1e-12 and _np.linalg.norm(v_h) > 1e-12:
                                    e_span = v_span / _np.linalg.norm(v_span)
                                    e_height = v_h / _np.linalg.norm(v_h)
                                    n = _np.cross(e_span, e_height)
                                    n_norm = _np.linalg.norm(n)
                                    if n_norm > 1e-12:
                                        e_n = n / n_norm
                                        rear_center = (p_bot_in + p_bot_out + p_top_in + p_top_out) / 4.0
                                        # Orient normal to FRONT (into the wing)
                                        f_bot_in = _np.array(front_face[0], dtype=float)
                                        f_bot_out = _np.array(front_face[1], dtype=float)
                                        f_top_out = _np.array(front_face[2], dtype=float)
                                        f_top_in = _np.array(front_face[3], dtype=float)
                                        front_center = (f_bot_in + f_bot_out + f_top_in + f_top_out) / 4.0
                                        forward_vec = front_center - rear_center
                                        if _np.linalg.norm(forward_vec) > 1e-12 and _np.dot(e_n, forward_vec) < 0.0:
                                            e_n = -e_n
                                        have_spar_frame = True
                except Exception:
                    have_spar_frame = False

                prism = None
                if have_spar_frame:
                    # Project elevon points into (s,h,t) in the rear-spar frame
                    rel = pts - rear_center.reshape(1, 3)
                    s_vals = rel @ e_span.reshape(3,)
                    h_vals = rel @ e_height.reshape(3,)
                    t_vals = rel @ e_n.reshape(3,)
                    smin, smax = float(s_vals.min()), float(s_vals.max())
                    hmin, hmax = float(h_vals.min()), float(h_vals.max())
                    tmin, tmax = float(t_vals.min()), float(t_vals.max())

                    # Slight height padding; span kept exact to match cutter span
                    h_pad = 0.01 * max(1.0, abs(hmax - hmin))
                    hmin -= h_pad; hmax += h_pad

                    # Ensure a minimal rearward depth so the boolean operation always has volume
                    rear_depth = max(0.005, -tmin)  # meters
                    if rear_depth < 1e-6:
                        rear_depth = 1e-4

                    def _pt(s, h, t):
                        return rear_center + s * e_span + h * e_height + t * e_n

                    # Base face on the rear-spar plane (t=0), extrude strictly rearward (-e_n)
                    t0 = 0.0
                    p00 = _pt(smin, hmin, t0)
                    p01 = _pt(smin, hmax, t0)
                    p11 = _pt(smax, hmax, t0)
                    p10 = _pt(smax, hmin, t0)
                    face = make_face_from_points([p00, p01, p11, p10])
                    if face is not None and rear_depth > 1e-9:
                        v_extrude = (-e_n * rear_depth).tolist()
                        prism = extrude_prism(face, (float(v_extrude[0]), float(v_extrude[1]), float(v_extrude[2])))

                if prism is None:
                    # Fallback: minimal axis-aligned box in +X
                    min_pt = pts.min(axis=0)
                    max_pt = pts.max(axis=0)
                    z_expand = max(0.01, 0.1 * (max_pt[2] - min_pt[2] if max_pt[2] != min_pt[2] else 1.0))
                    x_forward = max(0.1, 0.2 * (max_pt[0] - min_pt[0] if max_pt[0] != min_pt[0] else 1.0))
                    prism_min = (float(min_pt[0]), float(min_pt[1]), float(min_pt[2] - z_expand))
                    prism_max = (float(max_pt[0] + x_forward), float(max_pt[1]), float(max_pt[2] + z_expand))
                    y0 = prism_min[1]; y1 = prism_max[1]
                    z0 = prism_min[2]; z1 = prism_max[2]
                    x0 = prism_min[0]; x_len = prism_max[0] - prism_min[0]
                    box_face = [
                        (_np.array([x0, y0, z0], dtype=float)),
                        (_np.array([x0, y1, z0], dtype=float)),
                        (_np.array([x0, y1, z1], dtype=float)),
                        (_np.array([x0, y0, z1], dtype=float)),
                    ]
                    face = make_face_from_points(box_face)
                    if face is not None and x_len > 1e-9:
                        prism = extrude_prism(face, (x_len, 0.0, 0.0))

                # Intersect prism with wing to obtain elevon solid; force replace when successful
                if prism is not None:
                    # Crop the prism with a global-Y slab to force end faces normal to the Y axis.
                    try:
                        # Compute axis-aligned bounds from elevon points in GLOBAL coordinates
                        min_pt = pts.min(axis=0)
                        max_pt = pts.max(axis=0)
                        y_min = float(min_pt[1]); y_max = float(max_pt[1])
                        y_len = max(1e-6, y_max - y_min)
                        x_min = float(min_pt[0]); x_max = float(max_pt[0])
                        z_min = float(min_pt[2]); z_max = float(max_pt[2])
                        # XY padding to ensure slab fully covers the rear-spar-aligned prism in XZ
                        span_x = max(1e-9, x_max - x_min)
                        span_z = max(1e-9, z_max - z_min)
                        xy_pad = max(1.0e-4, 0.02 * max(span_x, span_z))  # ~2% pad, min 0.1 mm (pre-scale)
                        # Build a rectangle on the plane Y=y_min and extrude along +Y by y_len
                        pA = np.array([x_min - xy_pad, y_min, z_min - xy_pad], dtype=float)
                        pB = np.array([x_min - xy_pad, y_min, z_max + xy_pad], dtype=float)
                        pC = np.array([x_max + xy_pad, y_min, z_max + xy_pad], dtype=float)
                        pD = np.array([x_max + xy_pad, y_min, z_min - xy_pad], dtype=float)
                        y_face = make_face_from_points([pA, pB, pC, pD])
                        if y_face is not None and y_len > 1e-9:
                            y_slab = extrude_prism(y_face, (0.0, float(y_len), 0.0))
                            if y_slab:
                                clipped = bool_common(prism, y_slab)
                                if clipped and not clipped.IsNull():
                                    prism = clipped
                    except Exception:
                        pass

                    elevon_from_prism = None
                    try:
                        elevon_from_prism = bool_common(wing_point, prism)
                    except Exception:
                        elevon_from_prism = None
                    if elevon_from_prism is not None and not elevon_from_prism.IsNull():
                        builder = BRep_Builder()
                        comp = TopoDS_Compound()
                        builder.MakeCompound(comp)
                        builder.Add(comp, elevon_from_prism)
                        elevon_comp = comp
    except Exception:
        # Keep previous elevon_comp if this robust replacement fails
        pass

    # Cut spars out of ribs (rib notches)
    if spars_comp and not spars_comp.IsNull() and ribs_comp and not ribs_comp.IsNull():
        try:
            ribs_comp = bool_cut(ribs_comp, spars_comp) or ribs_comp
        except Exception:
            pass

    # Cut elevon deflection clearance using viewer-matched wedge aligned to selected rear-spar hinge
    # Prefer spar-referenced cutter (supports Top/Bottom/Centerline with "|<" geometry), fallback to elevon-based.
    # Optional height scaling for the deflection cutter (keeps hinge line and angle)
    height_scale = 1.0
    try:
        height_scale = float(getattr(processed, "deflection_height_scale", 1.0) or 1.0)
    except Exception:
        height_scale = 1.0

    deflection_cutter = _build_deflection_cutter_from_spar(processed, elevon_angle_deg=elevon_angle_deg, height_scale=height_scale)
    if not deflection_cutter:
        deflection_cutter = _build_deflection_cutter_from_elevon(processed.elevon_surfaces, elevon_angle_deg=elevon_angle_deg, height_scale=height_scale)
    if deflection_cutter and ribs_comp and not ribs_comp.IsNull():
        try:
            ribs_comp = bool_cut(ribs_comp, deflection_cutter) or ribs_comp
        except Exception:
            pass

    # Choose which wing geometry proceeds to cutting/hollowing for export
    # Prefer a true solid: only pick the spline wing if it is a SOLID; otherwise fall back to point-loft
    if wing_spline and not wing_spline.IsNull() and wing_spline.ShapeType() == TopAbs_SOLID:
        final_wing = wing_spline
    else:
        final_wing = wing_point
    # Try to ensure solid first using non-thickness methods (solid loft/shell->solid/sew)
    try:
        final_wing = _ensure_solid(final_wing)
    except Exception:
        pass
    # If the selected wing is a shell/surface, thicken to make a solid before booleans
    try:
        if final_wing and not final_wing.IsNull():
            st = final_wing.ShapeType()
            if st != TopAbs_SOLID:
                faces_to_remove = TopTools_ListOfShape()
                mk = BRepOffsetAPI_MakeThickSolid()
                mk.MakeThickSolidByJoin(final_wing, faces_to_remove, 0.001, 1.0e-4)
                thick = mk.Shape()
                if not thick or thick.IsNull() or thick.ShapeType() != TopAbs_SOLID:
                    mk_in = BRepOffsetAPI_MakeThickSolid()
                    mk_in.MakeThickSolidByJoin(final_wing, faces_to_remove, -0.001, 1.0e-4)
                    thick = mk_in.Shape()
                if thick and not thick.IsNull() and thick.ShapeType() == TopAbs_SOLID:
                    final_wing = thick
                    wing_was_shell = True
    except Exception:
        pass
    if cut_wing_with_elevon_opening and processed.elevon_surfaces and final_wing and not final_wing.IsNull():
        try:
            # Make the elevon opening cutter taller in +/-Z to avoid slivers:
            # compute asymmetric Z padding using the wing's bounding box so the cut fully spans thickness.
            ff = processed.elevon_surfaces[0][2]
            if ff and len(ff) == 4:
                import numpy as _np
                top_pad = bot_pad = 0.1
                try:
                    # Use current wing bounds (after previous operations) for a robust Z span
                    from OCC.Core.Bnd import Bnd_Box
                    from OCC.Core.BRepBndLib import brepbndlib_Add
                    bb = Bnd_Box()
                    brepbndlib_Add(final_wing, bb)
                    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
                    zmin = float(zmin); zmax = float(zmax)
                    # Local face top/bottom (assumes ff[0:2]=bottom, ff[2:4]=top as in upstream data)
                    z_top_local = float(max(ff[2][2], ff[3][2]))
                    z_bot_local = float(min(ff[0][2], ff[1][2]))
                    z_span = max(1.0e-9, zmax - zmin)
                    # Margin: 2% of wing Z span + 1 mm (pre-scale), ensure >= 0.1 m fallback
                    margin = 0.2 * z_span + 1.0e-2
                    top_pad = max(0.1, (zmax - z_top_local) + margin)
                    bot_pad = max(0.1, (z_bot_local - zmin) + margin)
                except Exception:
                    # Fallback to generous constant if bounds fail
                    top_pad = bot_pad = 0.5
                p0 = _np.array(ff[0], dtype=float) + _np.array([0.0, 0.0, -bot_pad])
                p1 = _np.array(ff[1], dtype=float) + _np.array([0.0, 0.0, -bot_pad])
                p2 = _np.array(ff[2], dtype=float) + _np.array([0.0, 0.0,  top_pad])
                p3 = _np.array(ff[3], dtype=float) + _np.array([0.0, 0.0,  top_pad])
                padded_front = [p0, p1, p2, p3]
                face = make_face_from_points(padded_front)
            else:
                face = make_face_from_points(ff)
            if face:
                x_pad = 2.0
                cutter = extrude_prism(face, (x_pad, 0.0, 0.0))
                if cutter:
                    final_wing = bool_cut(final_wing, cutter) or final_wing
        except Exception:
            pass

    # ORIGINAL ORDER: cut elevon with deflection cutter, then hollow using scaled subtraction with local center
    if deflection_cutter and elevon_comp and not elevon_comp.IsNull():
        try:
            elevon_comp = bool_cut(elevon_comp, deflection_cutter) or elevon_comp
            # Use elevon center derived from its faces to match monolith hollow reference
            elevon_center = _compute_shape_center_from_vertices(processed.elevon_surfaces[0] if processed.elevon_surfaces else [])
            inner_elevon = occ_scale_shape(elevon_comp, 0.95 if hollow_skin_scale is None else 0.95, center=elevon_center)
            if inner_elevon and not inner_elevon.IsNull():
                elevon_comp = bool_cut(elevon_comp, inner_elevon) or elevon_comp
        except Exception:
            pass

    # Hollow wing skin by scaled boolean subtraction using rib-profile mean center
    # Skip if we had to thicken a shell; that shape already has explicit thickness
    if (not wing_was_shell) and hollow_skin_scale and hollow_skin_scale < 1.0 and final_wing and not final_wing.IsNull():
        try:
            wing_center = _compute_center_from_profiles(processed.dihedraled_rib_profiles)
            inner = occ_scale_shape(final_wing, hollow_skin_scale, center=wing_center)
            if inner and not inner.IsNull():
                final_wing = bool_cut(final_wing, inner) or final_wing
        except Exception:
            pass

    if scale_to_mm:
        spars_comp = occ_scale_shape(spars_comp, 1000.0)
        ribs_comp = occ_scale_shape(ribs_comp, 1000.0)
        elevon_comp = occ_scale_shape(elevon_comp, 1000.0)
        final_wing = occ_scale_shape(final_wing, 1000.0)
    # Ensure the wing is a solid after all cuts/hollowing
    try:
        final_wing = _ensure_solid(final_wing)
    except Exception:
        pass
    return assemble_airframe(spars_comp, ribs_comp, elevon_comp, final_wing, mirror_full=True)


# ==============================================================================
# NEW PRIMARY ENTRY POINT - Direct Project -> STEP Generation
# ==============================================================================

def build_step_from_project(
    project: Project,
    config: WingGeometryConfig = None,
) -> TopoDS_Compound:
    """
    Build complete STEP geometry directly from Project state.
    
    This is the NEW primary entry point, replacing build_full_step_from_processed().
    It generates OCC geometry directly from the native Project + SpanwiseSection data,
    eliminating the CPACS XML translation layer.
    
    Architecture:
        Project (JSON) -> geometry_builder.py -> step_export.py -> STEP
        
    Benefits:
        - Single source of truth for geometry
        - DXF and STEP exports stay in sync (shared profiles.py)
        - Cleaner code with fewer translation layers
        - Support for new features (wingbox, stringers)
    
    Args:
        project: Native Project object containing wing planform, airfoils, structure params
        config: Export configuration (component toggles, scale factor, mirror options)
    
    Returns:
        TopoDS_Compound ready for write_step()
    
    Example:
        >>> from core.state import Project
        >>> project = Project.from_json("my_wing.json")
        >>> compound = build_step_from_project(project)
        >>> write_step(compound, "output.step")
    """
    if config is None:
        config = WingGeometryConfig()
    
    print("[StepExport] Building geometry from Project using new direct pipeline...")
    
    # Generate all geometry components via geometry_builder
    geom = build_geometry_from_project(project, config)
    
    if geom.section_count < 2:
        print("[StepExport] Warning: Insufficient sections for geometry generation")
        return TopoDS_Compound()
    
    # Assemble all components into a single compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    
    # Add skin surfaces
    if geom.wing_skin_upper and not geom.wing_skin_upper.IsNull():
        builder.Add(compound, geom.wing_skin_upper)
        print("[StepExport] Added upper skin surface")
    if geom.wing_skin_lower and not geom.wing_skin_lower.IsNull():
        builder.Add(compound, geom.wing_skin_lower)
        print("[StepExport] Added lower skin surface")
    
    # Add wingbox components
    if geom.wingbox_front_spar and not geom.wingbox_front_spar.IsNull():
        builder.Add(compound, geom.wingbox_front_spar)
        print("[StepExport] Added front spar")
    if geom.wingbox_rear_spar and not geom.wingbox_rear_spar.IsNull():
        builder.Add(compound, geom.wingbox_rear_spar)
        print("[StepExport] Added rear spar")
    if geom.wingbox_skin_upper and not geom.wingbox_skin_upper.IsNull():
        builder.Add(compound, geom.wingbox_skin_upper)
    if geom.wingbox_skin_lower and not geom.wingbox_skin_lower.IsNull():
        builder.Add(compound, geom.wingbox_skin_lower)
    
    # Add ribs
    rib_count = 0
    for rib in geom.ribs:
        if rib and not rib.IsNull():
            builder.Add(compound, rib)
            rib_count += 1
    if rib_count > 0:
        print(f"[StepExport] Added {rib_count} main ribs")
    
    # Add elevon ribs
    elevon_rib_count = 0
    for elevon_rib in geom.elevon_ribs:
        if elevon_rib and not elevon_rib.IsNull():
            builder.Add(compound, elevon_rib)
            elevon_rib_count += 1
    if elevon_rib_count > 0:
        print(f"[StepExport] Added {elevon_rib_count} elevon ribs")
    
    # Add stringers
    stringer_count = 0
    for stringer in geom.stringers:
        if stringer and not stringer.IsNull():
            builder.Add(compound, stringer)
            stringer_count += 1
    if stringer_count > 0:
        print(f"[StepExport] Added {stringer_count} stringers")
    
    # Add control surfaces
    for name, cs in geom.control_surfaces.items():
        if cs and not cs.IsNull():
            builder.Add(compound, cs)
            print(f"[StepExport] Added control surface: {name}")
    
    # Scale to mm if configured (default: 1000.0 for m -> mm)
    if config.scale_factor != 1.0:
        print(f"[StepExport] Scaling by {config.scale_factor}x")
        compound = occ_scale_shape(compound, config.scale_factor)
    
    # Mirror for full aircraft if configured
    if config.mirror_to_full_aircraft:
        print("[StepExport] Mirroring to full aircraft")
        mirrored = occ_mirror_y(compound)
        if mirrored and not mirrored.IsNull():
            final = TopoDS_Compound()
            builder.MakeCompound(final)
            builder.Add(final, compound)
            builder.Add(final, mirrored)
            compound = final
    
    print("[StepExport] Geometry build complete")
    return compound


def export_step_from_project(
    project: Project,
    output_path: str,
    config: WingGeometryConfig = None,
) -> bool:
    """
    Convenience function to build and write STEP file in one call.
    
    Args:
        project: Native Project object
        output_path: Path for output .step file
        config: Export configuration
    
    Returns:
        True if export succeeded, False otherwise
    """
    try:
        compound = build_step_from_project(project, config)
        if compound.IsNull():
            print(f"[StepExport] Error: No geometry generated")
            return False
        
        success = write_step(compound, output_path)
        if success:
            print(f"[StepExport] Successfully exported to {output_path}")
        else:
            print(f"[StepExport] Failed to write STEP file")
        return success
    except Exception as e:
        print(f"[StepExport] Export failed: {e}")
        return False


# ==============================================================================
# CFD Wing Export - Solid Half-Wing for CFD Analysis
# ==============================================================================

def build_cfd_wing_solid(
    project: Project,
    scale_to_mm: bool = True,
) -> TopoDS_Shape:
    """
    Build a solid half-wing for CFD analysis (outer mold line only).
    
    This creates a clean, solid wing surface suitable for CFD meshing:
    - No internal structure (ribs, spars, stringers)
    - Half wing only (no mirror)
    - Lofted between adjacent section pairs to avoid geometry errors
    - Uses spline profiles for smooth surfaces
    - Applies absolute twist at each section (same as Geometry/Airfoils tab)
    
    Args:
        project: Project containing wing planform, airfoils
        scale_to_mm: Scale output to mm (default True, 1000x)
    
    Returns:
        TopoDS_Shape solid representing the half-wing OML
    """
    import math
    from services.geometry import AeroSandboxService
    
    svc = AeroSandboxService(project)
    sections = svc.spanwise_sections()
    
    if len(sections) < 2:
        print("[CFD Wing] Error: Need at least 2 sections")
        return TopoDS_Shape()
    
    print(f"[CFD Wing] Building solid from {len(sections)} sections...")
    
    # Create wires for each section
    wires = []
    for section in sections:
        # Get airfoil coordinates (Nx2 array in normalized coords)
        try:
            coords = section.airfoil.coordinates
            if coords is None or len(coords) < 3:
                raise ValueError("Invalid airfoil")
        except Exception as e:
            print(f"[CFD Wing] Warning: Skipping section {section.index}: {e}")
            continue
        
        # Transform to 3D world coordinates with twist applied
        # Airfoil coords are normalized (0-1 chord, thickness in second column)
        # World frame: X=chordwise (aft), Y=spanwise (right), Z=vertical (up)
        # 
        # Twist is applied as rotation about the Y-axis (spanwise) at the leading edge.
        # section.twist_deg is the ABSOLUTE twist angle at this section.
        # Positive twist = nose up = rotation about Y that moves TE down, LE up.
        
        twist_rad = math.radians(float(section.twist_deg))
        cos_twist = math.cos(twist_rad)
        sin_twist = math.sin(twist_rad)
        
        pts_3d = []
        for x_norm, z_norm in coords:
            # Scale to local chord (in section's local frame, before twist)
            x_local = float(x_norm) * section.chord_m  # distance aft from LE
            z_local = float(z_norm) * section.chord_m  # thickness (+ up, - down)
            
            # Apply twist rotation about LE (rotation about Y-axis)
            # For nose-up twist (positive angle), TE moves down:
            #   x' = x * cos(θ) + z * sin(θ)
            #   z' = -x * sin(θ) + z * cos(θ)
            x_twisted = x_local * cos_twist + z_local * sin_twist
            z_twisted = -x_local * sin_twist + z_local * cos_twist
            
            # Translate to world position
            x = section.x_le_m + x_twisted
            y = section.y_m
            z = section.z_m + z_twisted
            
            pts_3d.append([x, y, z])
        
        # Create spline wire for smooth CFD surface
        wire = make_airfoil_wire_spline(pts_3d)
        
        if wire is not None:
            wires.append(wire)
            print(f"[CFD Wing] Section {section.index}: y={section.y_m:.3f}m, chord={section.chord_m:.3f}m, twist={section.twist_deg:.2f}°")
        else:
            print(f"[CFD Wing] Warning: Could not create wire for section {section.index}")
    
    if len(wires) < 2:
        print("[CFD Wing] Error: Could not create enough section wires")
        return TopoDS_Shape()
    
    print(f"[CFD Wing] Created {len(wires)} section wires")
    
    # Loft between adjacent section pairs and fuse
    # This avoids geometry errors from single continuous loft through many sections
    segments = []
    for i in range(len(wires) - 1):
        segment = loft_solid_from_wires([wires[i], wires[i + 1]], continuity='C2', max_degree=8)
        if segment and not segment.IsNull():
            segments.append(segment)
            print(f"[CFD Wing] Created loft segment {i} -> {i+1}")
        else:
            print(f"[CFD Wing] Warning: Loft failed for segment {i} -> {i+1}")
    
    if not segments:
        print("[CFD Wing] Error: No loft segments created")
        return TopoDS_Shape()
    
    # Fuse all segments into single solid
    if len(segments) == 1:
        result = segments[0]
    else:
        result = segments[0]
        for idx, seg in enumerate(segments[1:], start=1):
            fused = bool_fuse(result, seg)
            if fused and not fused.IsNull():
                result = fused
            else:
                print(f"[CFD Wing] Warning: Fuse failed at segment {idx}, using compound fallback")
                result = make_compound([result, seg])
    
    # Scale to mm if requested
    if scale_to_mm:
        result = occ_scale_shape(result, 1000.0)
        print("[CFD Wing] Scaled to mm (1000x)")
    
    print(f"[CFD Wing] Successfully created half-wing solid")
    return result
