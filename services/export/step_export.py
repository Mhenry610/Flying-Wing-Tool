"""
STEP export core functions extracted from CpacsStepperTab.export_step_file (GUI-free).

This module provides two export paths:
1. build_full_step_from_processed() - Legacy path consuming ProcessedCpacs
2. build_step_from_project() - NEW path consuming native Project state directly

The new path (2) is preferred and uses geometry_builder.py as the single source
of truth for 3D geometry generation, ensuring DXF and STEP outputs match.

- Uses shared wrappers in core.occ_utils for construction and booleans.
- No dialogs or logging; caller handles I/O and UX.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax2, gp_Dir
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Shell
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_ManifoldSolidBrep
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_COMPOUND
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
    bool_fuse,
    bool_cut,
    bool_common,
)

# Import the new geometry builder
from core.state import Project
from services.export.geometry_builder import (
    build_geometry_from_project,
    WingGeometryConfig,
    GeneratedGeometry,
)


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


# ==============================================================================
# Legacy Entry Point: Build from ProcessedCpacs (Deprecated)
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
        if front_face_verts is None or len(front_face_verts) == 0 or back_face_verts is None or len(back_face_verts) == 0:
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
        if front_face_verts is None or len(front_face_verts) == 0 or back_face_verts is None or len(back_face_verts) == 0:
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
    print(f"[DEBUG] build_elevon_compound called with {len(elevon_surfaces) if elevon_surfaces else 0} surfaces")
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    if not elevon_surfaces:
        print("[DEBUG] No elevon surfaces to build")
        return comp
        
    for i, faces in enumerate(elevon_surfaces):
        print(f"[DEBUG] Processing elevon surface {i}, faces count: {len(faces)}")
        if len(faces) < 6:
            print(f"[WARN] Elevon surface {i} has insufficient faces ({len(faces)})")
            continue
        inner_v = faces[4]
        outer_v = faces[5]
        print(f"[DEBUG] Surface {i} Inner V count: {len(inner_v)}, Outer V count: {len(outer_v)}")
        
        # Debug vertex positions (first point)
        if len(inner_v) > 0:
            print(f"[DEBUG] Surface {i} Inner[0]: {inner_v[0]}")
        if len(outer_v) > 0:
            print(f"[DEBUG] Surface {i} Outer[0]: {outer_v[0]}")

        w_in = make_wire_from_points(inner_v)
        w_out = make_wire_from_points(outer_v)
        if w_in and w_out:
            loft = loft_solid_from_wires([w_in, w_out])
            if loft and not loft.IsNull():
                builder.Add(comp, loft)
                print(f"[DEBUG] Surface {i} loft added successfully")
            else:
                print(f"[WARN] Surface {i} loft generated Null shape")
        else:
            print(f"[WARN] Surface {i} failed to create wires")
    return comp


def loft_wing_from_ribs(dihedraled_rib_profiles: List[np.ndarray], 
                        use_spline_profiles: bool = False,
                        split_at_y: float | None = None,
                        sort_by_y: bool = True) -> TopoDS_Shape:
    """
    Loft a solid wing by wiring the rib profiles.
    If sort_by_y is True (default), sorts profiles by spanwise Y.
    If sort_by_y is False, assumes profiles are already topologically ordered.
    If split_at_y is provided, splits the loft into two segments (inboard/outboard)
    at that Y location and fuses them.
    """
    if len(dihedraled_rib_profiles) < 2:
        return TopoDS_Shape()
    
    sorted_profiles = sorted(dihedraled_rib_profiles, key=lambda p: float(np.mean(p[:, 1]))) if sort_by_y else dihedraled_rib_profiles
    
    # Debug: show profile Y order
    y_positions = [float(np.mean(p[:, 1])) for p in sorted_profiles]
    print(f"[DEBUG] loft_wing_from_ribs: {len(sorted_profiles)} profiles, sort_by_y={sort_by_y}")
    print(f"[DEBUG] Profile Y positions: {[f'{y:.3f}' for y in y_positions]}")
    
    # Helper to loft a subset of profiles
    def _loft_subset(profiles):
        if len(profiles) < 2:
            return None
        wires = []
        for i, prof in enumerate(profiles):
            pts = [np.array(v, dtype=float) for v in prof]
            if use_spline_profiles:
                w = make_airfoil_wire_spline(pts)
            else:
                w = make_wire_from_points(pts)
            if w is not None:
                wires.append(w)
            else:
                print(f"[WARN] Wire creation failed for profile {i} (Y~{float(np.mean(prof[:, 1])):.3f})")
        
        print(f"[DEBUG] Created {len(wires)} wires from {len(profiles)} profiles")
        
        if not wires:
            return None
        
        # Always use ruled=True for linear/flat surfaces (matches viewer appearance)
        # Spline lofting (ruled=False) creates curved surfaces that bulge between sections
        return loft_solid_from_wires(wires, ruled=True)

    # Auto-detect kink points (duplicate Y positions) and split there
    # This is necessary because BRepOffsetAPI_ThruSections cannot handle two profiles at the same Y
    kink_indices = []
    for i in range(1, len(sorted_profiles)):
        y_prev = float(np.mean(sorted_profiles[i-1][:, 1]))
        y_curr = float(np.mean(sorted_profiles[i][:, 1]))
        if abs(y_curr - y_prev) < 1e-4:  # Duplicate Y detected
            kink_indices.append(i)
            print(f"[DEBUG] Auto-detected kink point at index {i} (Y={y_curr:.3f})")
    
    # If kink points detected, split loft at each kink
    if kink_indices:
        print(f"[DEBUG] Splitting loft at {len(kink_indices)} kink point(s)")
        segments = []
        start_idx = 0
        
        # Profile to force as the start of the *next* segment to ensure continuity
        pending_start_cap = None
        
        for kink_idx in kink_indices:
            # Segment from start_idx to kink_idx
            # sorted_profiles[kink_idx] is the SECOND duplicate.
            # So slice [start_idx : kink_idx] ends with sorted_profiles[kink_idx-1] (FIRST duplicate).
            # This is what we want: Keep first, discard second.
            seg = list(sorted_profiles[start_idx:kink_idx])
            
            # If previous segment left us a cap (the shared profile), prepend it
            if pending_start_cap is not None:
                seg.insert(0, pending_start_cap)
            
            if len(seg) >= 2:
                segments.append(seg)
            
            # Setup for next segment:
            # 1. Identify valid bridge profile: The FIRST duplicate (at kink_idx-1)
            pending_start_cap = sorted_profiles[kink_idx-1]
            
            # 2. Skip the SECOND duplicate (at kink_idx)
            # Next segment data effectively starts at kink_idx + 1
            start_idx = kink_idx + 1
            
        # Final segment from last kink to end
        final_seg = list(sorted_profiles[start_idx:])
        
        # Add the final bridge cap if needed
        if pending_start_cap is not None:
             final_seg.insert(0, pending_start_cap)
             
        if len(final_seg) >= 2:
            segments.append(final_seg)
        
        print(f"[DEBUG] Created {len(segments)} loft segments")
        
        # Loft each segment and fuse them
        lofts = []
        for i, seg in enumerate(segments):
            print(f"[DEBUG] Lofting segment {i} with {len(seg)} profiles")
            loft = _loft_subset(seg)
            if loft and not loft.IsNull():
                print(f"[DEBUG] Segment {i} loft SUCCESS - ShapeType: {loft.ShapeType()}")
                lofts.append(loft)
            else:
                print(f"[WARN] Segment {i} loft FAILED (null or empty)")
        
        print(f"[DEBUG] Total successful lofts: {len(lofts)}")
        
        if len(lofts) == 0:
            print("[WARN] No valid loft segments - returning empty shape")
            return TopoDS_Shape()
        elif len(lofts) == 1:
            return lofts[0]
        else:
            # Fuse all segments? No, fusion often fails or creates invalid topology for coincident faces.
            # Return a Compound of aligned solids instead.
            print(f"[DEBUG] Combining {len(lofts)} segments into one Compound (skipping boolean fuse)")
            comp = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(comp)
            for loft in lofts:
                builder.Add(comp, loft)
            return comp

    if split_at_y is not None:
        # Split profiles into inboard and outboard sets
        # Include the profile closest to split_at_y in BOTH sets to ensure continuity
        # Actually, we should find the split index.
        
        # Find index where Y crosses split_at_y
        split_idx = -1
        for i, prof in enumerate(sorted_profiles):
            y_mean = float(np.mean(prof[:, 1]))
            if y_mean >= split_at_y - 1e-5: # Tolerance
                split_idx = i
                break
        
        if split_idx > 0 and split_idx < len(sorted_profiles):
            # Inboard set: 0 to split_idx (inclusive)
            inboard_profs = sorted_profiles[:split_idx+1]
            # Outboard set: split_idx to end (inclusive)
            outboard_profs = sorted_profiles[split_idx:]
            
            loft_in = _loft_subset(inboard_profs)
            loft_out = _loft_subset(outboard_profs)
            
            if loft_in and not loft_in.IsNull() and loft_out and not loft_out.IsNull():
                # Fuse them
                fused = bool_fuse(loft_in, loft_out)
                return fused
            elif loft_in and not loft_in.IsNull():
                return loft_in
            elif loft_out and not loft_out.IsNull():
                return loft_out

    # Fallback to single loft if no split or split failed
    return _loft_subset(sorted_profiles) or TopoDS_Shape()


def assemble_airframe(spars: TopoDS_Compound,
                      ribs: TopoDS_Compound,
                      elevon: TopoDS_Compound,
                      wing: TopoDS_Shape,
                      mirror_full: bool = True) -> TopoDS_Compound:
    """
    Assemble right wing components, optionally mirror across Y to form full airframe.
    """
    print("[DEBUG] assemble_airframe called")
    builder = BRep_Builder()
    right = TopoDS_Compound()
    builder.MakeCompound(right)
    if spars and not spars.IsNull():
        builder.Add(right, spars)
        print(f"[DEBUG] Added spars to compound")
    else:
        print(f"[WARN] Spars NOT added (null or missing)")
    if ribs and not ribs.IsNull():
        builder.Add(right, ribs)
        print(f"[DEBUG] Added ribs to compound")
    else:
        print(f"[WARN] Ribs NOT added (null or missing)")
    if elevon and not elevon.IsNull():
        builder.Add(right, elevon)
        print(f"[DEBUG] Added elevon to compound")
    else:
        print(f"[WARN] Elevon NOT added (null or missing)")
    if wing and not wing.IsNull():
        builder.Add(right, wing)
        print(f"[DEBUG] Added wing to compound, ShapeType: {wing.ShapeType()}")
    else:
        print(f"[WARN] Wing NOT added (null or missing)")

    if not mirror_full:
        return right

    final_comp = TopoDS_Compound()
    builder.MakeCompound(final_comp)
    builder.Add(final_comp, right)
    mirrored = occ_mirror_y(right)
    if mirrored and not mirrored.IsNull():
        builder.Add(final_comp, mirrored)
        print(f"[DEBUG] Added mirrored copy")
    else:
        print(f"[WARN] Mirror failed or null")
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
        # If the hierarchy contains solids, check how many
        try:
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            solids = []
            while exp.More():
                solids.append(exp.Current())
                exp.Next()
            if len(solids) == 1:
                return solids[0]  # Single solid - return it directly
            elif len(solids) > 1:
                # Multiple solids in a Compound - keep as Compound (don't lose any!)
                return shape
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
                                       spar_uid: str,
                                       elevon_angle_deg: Optional[float] = None,
                                       rearward: bool = False,
                                       height_scale: float = 1.2,
                                       elevon_faces: Optional[List] = None) -> Optional[TopoDS_Shape]:
    """
    Build the deflection cutter wedge aligned with the selected hinge reference on the REAR face
    of the specified spar_uid.
    
    elevon_faces: If provided, use these faces for span matching instead of processed.elevon_surfaces[0].
    
    Returns a TopoDS_Shape or None.
    """
    try:
        if processed is None or elevon_angle_deg is None or float(elevon_angle_deg) <= 0.0:
            return None
        if not getattr(processed, "spar_surfaces", None) or not getattr(processed, "spar_uids", None):
            return None
        if not spar_uid:
            return None

        import numpy as _np
        from math import tan, radians

        # Find specific spar
        uids = list(getattr(processed, "spar_uids", []) or [])
        if not uids:
            return None
        try:
            idx = uids.index(spar_uid)
        except ValueError:
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
            # Use specific elevon_faces if provided, otherwise fallback to elevon_surfaces[0]
            ev = elevon_faces if elevon_faces is not None else (
                processed.elevon_surfaces[0] if getattr(processed, "elevon_surfaces", None) else None
            )
            if s_norm > 1e-12 and ev:
                s_hat = s_dir / s_norm
                pts = _np.vstack([_np.array(p, dtype=float) for face in ev for p in (face or [])])
                if pts.size >= 3:
                    svals = (pts - a0.reshape(1, 3)) @ s_hat.reshape(3,)
                    smin = float(_np.min(svals))
                    smax = float(_np.max(svals))
                    # Use exact elevon span - no padding beyond the elevon bounds
                    # This ensures only ribs WITHIN the elevon span are cut
                    a0 = a0 + smin * s_hat
                    a1 = a0 + smax * s_hat
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
                                   use_spline_wing: bool = False,
                                   split_at_y: Optional[float] = None) -> TopoDS_Compound:
    """
    End-to-end builder using ProcessedCpacs from refactor.services.viewer:
      1) Build spars and ribs compounds
      2) Build elevon (optional)
      3) Loft wing solid from dihedraled rib profiles
      4) Cut spars from ribs and cut elevon deflection clearance from ribs using GUI angle
      5) Optionally cut elevon opening from wing (minimal cutter)
      6) Optionally hollow wing skin, using centers that match monolith logic
      7) Scale to mm and assemble mirrored airframe
      
    Args:
        split_at_y: If provided, splits the wing loft into two segments at this spanwise Y location.
    """
    spars_comp = build_spars_compound(processed.spar_surfaces)
    ribs_comp = build_ribs_compound(processed.rib_surfaces)
    # Keep legacy loft-based elevon_comp available
    elevon_comp = build_elevon_compound(processed.elevon_surfaces)
    
    # 1. Wing Solid (Loft)
    use_spline_loft = False
    sort_by_y = True
    profiles_to_loft = processed.dihedraled_rib_profiles
    
    if hasattr(processed, 'loft_profiles') and processed.loft_profiles:
        # Use explicit sections. Do NOT re-sort them (respect topology).
        profiles_to_loft = processed.loft_profiles
        sort_by_y = False
        print(f"[DEBUG] Using {len(profiles_to_loft)} loft profiles (Sections) for wing solid [Preserving Order]")
        
        # Respect user preference for splines
        if use_spline_wing:
             use_spline_loft = True
             print(f"[DEBUG] Spline lofting ENABLED by user preference")
        else:
             print(f"[DEBUG] Spline lofting DISABLED by user preference")
    else:
        # Use ribs. Sort them by Y to be safe.
        print(f"[DEBUG] Using {len(profiles_to_loft)} rib profiles for wing solid [Reference Only]")
    
    wing_point = loft_wing_from_ribs(profiles_to_loft, use_spline_profiles=use_spline_loft, split_at_y=split_at_y, sort_by_y=sort_by_y)
    
    # Optionally build a spline wing as a SOLID loft of closed spline wires (preferred).
    # Fallback to two-surface sew+thicken only if needed.
    wing_spline = None
    wing_was_shell = False
    if use_spline_wing:
        # Preferred: solid loft through closed spline airfoil wires
        try:
            # Pass sort_by_y=sort_by_y (False if using loft_profiles)
            wing_spline = loft_wing_from_ribs(profiles_to_loft, use_spline_profiles=True, split_at_y=split_at_y, sort_by_y=sort_by_y)

            if wing_spline and not wing_spline.IsNull():
                # Fused shape might be a Compound of Solids
                if wing_spline.ShapeType() == TopAbs_SOLID:
                    pass
                elif wing_spline.ShapeType() == TopAbs_COMPOUND:
                    # Verify it contains solids
                    exp = TopExp_Explorer(wing_spline, TopAbs_SOLID)
                    if not exp.More():
                        wing_spline = None
                else:
                    wing_spline = None
            else:
                wing_spline = None
        except Exception:
            wing_spline = None

    # Build elevons by intersecting wing with prisms defined by each elevon surface bounds.
    # This ensures perfect OML match - the elevon IS the cut-out piece of the wing.
    try:
        if wing_point and not wing_point.IsNull() and processed.elevon_surfaces:
            import numpy as _np
            
            builder = BRep_Builder()
            new_elevon_comp = TopoDS_Compound()
            builder.MakeCompound(new_elevon_comp)
            elevon_count = 0
            
            for ev_idx, ev in enumerate(processed.elevon_surfaces):
                if not ev or len(ev) < 4:
                    print(f"[DEBUG] Skipping elevon {ev_idx}: insufficient faces")
                    continue
                    
                # Aggregate elevon vertices to determine span/height/chord extents
                try:
                    ev_faces = [f for f in ev if f is not None and len(f) >= 3]
                except Exception:
                    ev_faces = []
                if not ev_faces and len(ev) >= 4:
                    ev_faces = [ev[2], ev[3]]
                if not ev_faces:
                    print(f"[DEBUG] Skipping elevon {ev_idx}: no valid faces")
                    continue
                    
                pts = _np.vstack([_np.array(p, dtype=float) for poly in ev_faces for p in poly])
                
                # Get the FRONT face (ev[2]) which represents the HINGE LINE
                # Use this as the actual base face for the prism (respects sweep)
                front_face = ev[2] if len(ev) > 2 else None
                back_face = ev[3] if len(ev) > 3 else None  # Trailing edge face
                
                if front_face is None or len(front_face) < 4:
                    print(f"[DEBUG] Skipping elevon {ev_idx}: no valid front face")
                    continue
                
                # Calculate extrusion direction from front to back face
                front_pts = _np.array(front_face, dtype=float)
                if back_face is not None and len(back_face) >= 4:
                    back_pts = _np.array(back_face, dtype=float)
                    # Average direction from front to back
                    extrude_vec = (back_pts.mean(axis=0) - front_pts.mean(axis=0))
                else:
                    # Fallback: extrude in +X direction to trailing edge
                    x_max = float(pts[:, 0].max())
                    x_front = float(front_pts[:, 0].mean())
                    extrude_vec = _np.array([x_max - x_front + 0.1, 0.0, 0.0])
                
                # Add extra length to ensure we capture full TE even with sweep
                extrude_len = _np.linalg.norm(extrude_vec)
                if extrude_len > 1e-9:
                    extrude_dir = extrude_vec / extrude_len
                    extrude_vec = extrude_dir * (extrude_len * 1.5)  # 50% extra to cover swept TE
                
                # Expand front face slightly in Z for thickness tolerance
                z_min = float(pts[:, 2].min())
                z_max = float(pts[:, 2].max())
                z_pad = max(0.05, 0.2 * (z_max - z_min))
                
                # Build prism base from the front face points, expanded in Z
                p0 = front_pts[0].copy(); p0[2] = z_min - z_pad
                p1 = front_pts[1].copy(); p1[2] = z_min - z_pad
                p2 = front_pts[2].copy(); p2[2] = z_max + z_pad
                p3 = front_pts[3].copy(); p3[2] = z_max + z_pad
                
                base_face = make_face_from_points([p0, p1, p2, p3])
                if base_face is None:
                    print(f"[DEBUG] Skipping elevon {ev_idx}: failed to create base face")
                    continue
                    
                # Extrude to create prism
                prism = extrude_prism(base_face, tuple(extrude_vec.tolist()))
                if prism is None or prism.IsNull():
                    print(f"[DEBUG] Skipping elevon {ev_idx}: failed to create prism")
                    continue
                    
                print(f"[DEBUG] Elevon {ev_idx}: prism created from front face")
                
                # Intersect wing with prism to get the cut-out piece
                try:
                    elevon_solid = bool_common(wing_point, prism)
                except Exception as e:
                    print(f"[DEBUG] Elevon {ev_idx}: bool_common failed: {e}")
                    continue
                    
                if elevon_solid is not None and not elevon_solid.IsNull():
                    builder.Add(new_elevon_comp, elevon_solid)
                    elevon_count += 1
                    print(f"[DEBUG] Elevon {ev_idx}: successfully cut from wing")
                else:
                    print(f"[DEBUG] Elevon {ev_idx}: bool_common returned null")
            
            # Only replace if we successfully generated at least one elevon
            if elevon_count > 0:
                elevon_comp = new_elevon_comp
                print(f"[DEBUG] OML intersection: generated {elevon_count} elevon(s)")
            else:
                print(f"[DEBUG] OML intersection: no elevons generated, keeping profile-based fallback")
    except Exception as e:
        # Keep previous elevon_comp if this fails
        print(f"[DEBUG] OML intersection block failed: {e}")

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

    # Retrieve explicit hinge spars which correspond to control surfaces
    spar_uids = getattr(processed, "spar_uids", []) or []
    hinge_spars = [u for u in spar_uids if u.startswith('hingeSpar_')]
    
    # If using legacy mode (no hinge spars), fallback to any rear spar for single elevon
    if not hinge_spars and processed.elevon_surfaces:
        fallback_spar = None
        for spar_name in ["rearSpar_wing", "rearSpar_outboard", "rearSpar_full"]:
            if spar_name in spar_uids:
                fallback_spar = spar_name
                break
        if not fallback_spar:
            fallback_spar = next((u for u in spar_uids if "rear" in u.lower() and "spar" in u.lower()), None)
        if fallback_spar:
            hinge_spars = [fallback_spar]

    # Iterate through ALL hinge spars / control surfaces
    if processed.elevon_surfaces:
        for i, ev_faces in enumerate(processed.elevon_surfaces):
            # Try to match elevon surface to a hinge spar
            current_spar_uid = hinge_spars[i] if i < len(hinge_spars) else (hinge_spars[0] if hinge_spars else None)
            
            # Build cutter for THIS surface
            print(f"[DEBUG] Processing Elevon Surface {i}, Spar: {current_spar_uid}")
            deflection_cutter = _build_deflection_cutter_from_spar(
                processed, current_spar_uid, 
                elevon_angle_deg=elevon_angle_deg, 
                height_scale=height_scale,
                elevon_faces=ev_faces  # Pass THIS elevon's faces for span matching
            )
            # Fallback to elevon-based cutter if spar cutter fails
            if not deflection_cutter:
                 print(f"[DEBUG] Spar cutter failed for {current_spar_uid}, falling back to elevon-based")
                 deflection_cutter = _build_deflection_cutter_from_elevon([ev_faces], elevon_angle_deg=elevon_angle_deg, height_scale=height_scale)

            # Apply cuts if cutter exists
            if deflection_cutter:
                print(f"[DEBUG] Deflection cutter built for surface {i}")
                # 1. Cut ribs
                if ribs_comp and not ribs_comp.IsNull():
                    try:
                        res = bool_cut(ribs_comp, deflection_cutter)
                        if res and not res.IsNull():
                            ribs_comp = res
                            print(f"[DEBUG] Cut ribs for surface {i} Success")
                        else:
                            print(f"[DEBUG] Cut ribs for surface {i} Failed (null result)")
                    except Exception as e:
                        print(f"[DEBUG] Cut ribs for surface {i} Exception: {e}")
                
                # 2. Cut deflection clearance from elevon itself
                # Now safe with proper per-elevon bounds - cutter only removes a wedge
                if elevon_comp and not elevon_comp.IsNull():
                    try:
                        res = bool_cut(elevon_comp, deflection_cutter)
                        if res and not res.IsNull():
                            elevon_comp = res
                            print(f"[DEBUG] Cut elevon_comp for surface {i} Success")
                        else:
                            print(f"[DEBUG] Cut elevon_comp for surface {i} Failed (null result)")
                    except Exception as e:
                        print(f"[DEBUG] Cut elevon_comp for surface {i} Exception: {e}")
            else:
                print(f"[DEBUG] No deflection cutter generated for surface {i}")


    # Choose which wing geometry proceeds to cutting/hollowing for export
    # Prefer a true solid: only pick the spline wing if it is a SOLID or COMPOUND of solids; otherwise fall back to point-loft
    if wing_spline and not wing_spline.IsNull():
        print(f"[DEBUG] wing_spline is valid, ShapeType: {wing_spline.ShapeType()}")
        if wing_spline.ShapeType() == TopAbs_SOLID:
            final_wing = wing_spline
            print("[DEBUG] Using wing_spline (SOLID)")
        elif wing_spline.ShapeType() == TopAbs_COMPOUND:
             # Check if it has solids
            exp = TopExp_Explorer(wing_spline, TopAbs_SOLID)
            if exp.More():
                final_wing = wing_spline
                print("[DEBUG] Using wing_spline (COMPOUND with solids)")
            else:
                final_wing = wing_point
                print("[DEBUG] wing_spline COMPOUND has no solids, falling back to wing_point")
        else:
            final_wing = wing_point
            print(f"[DEBUG] wing_spline is unusual ShapeType {wing_spline.ShapeType()}, falling back to wing_point")
    else:
        final_wing = wing_point
        print("[DEBUG] wing_spline is null/missing, using wing_point")
    
    # Check wing_point validity
    if final_wing and not final_wing.IsNull():
        print(f"[DEBUG] final_wing is valid, ShapeType: {final_wing.ShapeType()}")
    else:
        print("[WARN] final_wing is NULL or empty!")
    
    # Try to ensure solid first using non-thickness methods (solid loft/shell->solid/sew)
    try:
        before_type = final_wing.ShapeType() if final_wing and not final_wing.IsNull() else "NULL"
        final_wing = _ensure_solid(final_wing)
        after_type = final_wing.ShapeType() if final_wing and not final_wing.IsNull() else "NULL"
        print(f"[DEBUG] _ensure_solid: {before_type} -> {after_type}")
    except Exception as e:
        print(f"[WARN] _ensure_solid failed: {e}")
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
        # Loop over ALL elevon surfaces to cut openings
        for ev in processed.elevon_surfaces:
            try:
                # Make the elevon opening cutter taller in +/-Z to avoid slivers:
                # compute asymmetric Z padding using the wing's bounding box so the cut fully spans thickness.
                ff = ev[2] if len(ev) > 2 else None
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
                    # Ensure cutter extends well past the TE
                    # Re-calculate bounds for this surface if needed or use global
                    # For simplicity use very large number or global bounds from before
                    x_len = 10.0 # Arbitrary large length
                    x_pad = max(2.0, x_len * 2.0)
                    cutter = extrude_prism(face, (x_pad, 0.0, 0.0))
                    if cutter:
                        final_wing = bool_cut(final_wing, cutter) or final_wing
            except Exception:
                pass


    # Hollow wing skin by scaled boolean subtraction using rib-profile mean center
    # DISABLED per user request (Step 1663) to avoid "weird hole" artifacts.
    # if (not wing_was_shell) and hollow_skin_scale and hollow_skin_scale < 1.0 and final_wing and not final_wing.IsNull():
    #     try:
    #         wing_center = _compute_center_from_profiles(processed.dihedraled_rib_profiles)
    #         inner = occ_scale_shape(final_wing, hollow_skin_scale, center=wing_center)
    #         if inner and not inner.IsNull():
    #             final_wing = bool_cut(final_wing, inner) or final_wing
    #     except Exception:
    #         pass

    # Trim spars to wing surface (boolean common)
    # This ensures spars do not protrude from the wing OML.
    # We iterate per spar to ensure robustness; if one fails, we keep the original.
    if spars_comp and not spars_comp.IsNull() and final_wing and not final_wing.IsNull():
        try:
            builder = BRep_Builder()
            new_comp = TopoDS_Compound()
            builder.MakeCompound(new_comp)
            
            exp = TopExp_Explorer(spars_comp, TopAbs_SOLID)
            has_spars = False
            while exp.More():
                spar = exp.Current()
                trimmed = None
                try:
                    trimmed = bool_common(spar, final_wing)
                except Exception:
                    pass
                
                if trimmed and not trimmed.IsNull():
                    builder.Add(new_comp, trimmed)
                    has_spars = True
                else:
                    # Fallback: keep original if trim fails or returns empty
                    # (better to have sticking-up spar than no spar)
                    builder.Add(new_comp, spar)
                    has_spars = True
                exp.Next()
                
            if has_spars:
                spars_comp = new_comp
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
    
    # Debug: check all components before assembly
    print(f"[DEBUG] FINAL ASSEMBLY CHECK:")
    print(f"  spars_comp: {spars_comp is not None and not spars_comp.IsNull()}")
    print(f"  ribs_comp: {ribs_comp is not None and not ribs_comp.IsNull()}")
    print(f"  elevon_comp: {elevon_comp is not None and not elevon_comp.IsNull()}")
    print(f"  final_wing: {final_wing is not None and not final_wing.IsNull()}, ShapeType: {final_wing.ShapeType() if final_wing and not final_wing.IsNull() else 'N/A'}")
    
    return assemble_airframe(spars_comp, ribs_comp, elevon_comp, final_wing, mirror_full=True)
