"""
Fixture layout generation core (GUI-free), adapted from CpacsStepperTab.generate_and_export_layout.
- Consumes ProcessedCpacs from refactor.services.viewer
- Uses shared OCC wrappers in refactor.occ_utils
- Caller handles I/O (paths, dialogs) and logging

DEPRECATED: This module uses the old CPACS-based pipeline.
Use services.export.geometry_builder.build_fixture_geometry() instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import numpy as np

from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Trsf
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
# Added imports needed for plane normal extraction parity with monolith
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane

from core.occ_utils.shapes import (
    make_wire_from_points,
    make_face_from_points,
    extrude_prism,
    make_compound,
    scale_shape,
)
from core.occ_utils.booleans import cut as bool_cut, fuse as bool_fuse, common as bool_common

from services.export.viewer import calculate_chord_line_z, chord_z_from_ribs, ProcessedCpacs


@dataclass
class FixtureParams:
    material_thickness_mm: float = 6.35
    slot_clearance_mm: float = 0.15
    add_cradle: bool = True
    # Tabs/slots
    tab_width_mm: float = 15.0
    tab_spacing_mm: float = 80.0
    tab_edge_margin_mm: float = 12.0
    # Base plate
    base_plate_margin_mm: float = 20.0
    # Generated features
    add_tabs: bool = True
    slot_base_plate: bool = True
    # Flat plate mode for hobbyist construction
    spar_is_flat_plate: bool = True           # Use flat plate thickness instead of 3D spar
    spar_plate_thickness_mm: float = 3.0      # Flat spar plate thickness


@dataclass
class FixtureBuildResult:
    fixtures_fused: List[TopoDS_Shape]
    base_plate: TopoDS_Shape
    final_layout: TopoDS_Compound


def _determine_front_spar_uid(processed: ProcessedCpacs) -> Optional[str]:
    """
    Parity with monolith: determine front-most spar by average CPACS xsi across its endpoints.
    Fallbacks:
      1) If spar_avg_xsi missing/empty, prefer CPACS order assumption only if sizes match.
      2) Else geometric: smaller average X over FRONT face vertices.
    """
    # Preferred: explicit CPACS xsi averages from processing
    try:
        if hasattr(processed, "spar_avg_xsi") and processed.spar_avg_xsi:
            # pick UID with minimum average xsi
            uid_min = min(processed.spar_avg_xsi.items(), key=lambda kv: kv[1])[0]
            return uid_min
    except Exception:
        pass

    # Fallback 1: CPACS order assumption if it maps 1:1 with surfaces
    if processed.spar_uids and len(processed.spar_uids) == len(processed.spar_surfaces):
        return processed.spar_uids[0]

    # Fallback 2: geometric average-X of front face vertices
    if not processed.spar_surfaces:
        return None
    averages = []
    for idx, polys in enumerate(processed.spar_surfaces):
        if not polys or not polys[0]:
            continue
        xs = [float(p[0]) for p in polys[0]]
        averages.append((np.mean(xs), idx))
    if not averages:
        return None
    averages.sort()
    return processed.spar_uids[averages[0][1]] if len(processed.spar_uids) == len(processed.spar_surfaces) else None


def _get_lower_surface_z(profile_mm: np.ndarray, x_mm: float) -> float:
    """
    Get Z coordinate of the lower surface of an airfoil profile at a given X.
    Assumes profile_mm contains the full loop of points.
    """
    if len(profile_mm) < 2:
        return 0.0
    
    # Find Leading Edge (min X)
    le_idx = np.argmin(profile_mm[:, 0])
    
    # Split into two branches at LE
    # Branch 1: Start to LE
    branch1 = profile_mm[:le_idx+1]
    # Branch 2: LE to End
    branch2 = profile_mm[le_idx:]
    
    def get_z_at_x(branch, x):
        if len(branch) < 2:
            return float('inf')
        xs = branch[:, 0]
        zs = branch[:, 2]
        # Sort by X for interpolation
        sort_idx = np.argsort(xs)
        return np.interp(x, xs[sort_idx], zs[sort_idx])

    z1 = get_z_at_x(branch1, x_mm)
    z2 = get_z_at_x(branch2, x_mm)
    
    # Return the lower Z (lower surface)
    return min(z1, z2)


def _lower_surface_z_from_ribs(target_pt_mm: Tuple[float, float],
                               sorted_rib_profiles_mm: List[np.ndarray]) -> Optional[float]:
    """
    Interpolate/extrapolate lower surface Z for a spanwise Y using two nearest rib profiles.
    Mirrors chord_z_from_ribs but targets lower surface.
    """
    if len(sorted_rib_profiles_mm) < 2:
        return None
    target_x, target_y = float(target_pt_mm[0]), float(target_pt_mm[1])
    rib_ys = [rib[0][1] for rib in sorted_rib_profiles_mm if len(rib) > 0]
    if not rib_ys:
        return None
    
    idx = int(np.searchsorted(rib_ys, target_y))
    if idx == 0:
        r1, r2 = sorted_rib_profiles_mm[0], sorted_rib_profiles_mm[1]
    elif idx >= len(rib_ys):
        r1, r2 = sorted_rib_profiles_mm[-2], sorted_rib_profiles_mm[-1]
    else:
        r1, r2 = sorted_rib_profiles_mm[idx - 1], sorted_rib_profiles_mm[idx]
        
    z1 = _get_lower_surface_z(r1, target_x)
    z2 = _get_lower_surface_z(r2, target_x)
    
    y1 = r1[0][1] if len(r1) > 0 else 0.0
    y2 = r2[0][1] if len(r2) > 0 else 0.0
    
    if abs(y2 - y1) < 1e-6:
        return z1
        
    frac = (target_y - y1) / (y2 - y1)
    return float(z1 + frac * (z2 - z1))


def _make_rect_prism(center_xy: Tuple[float, float],
                     dir_xy: Tuple[float, float],
                     width: float,
                     length: float,
                     top_z: float,
                     down_depth: float) -> Optional[TopoDS_Shape]:
    """
    Create a rectangular prism aligned with dir_xy at center_xy, width across edge, length along edge.
    Extruded downwards from top_z by down_depth in +(-Z).
    """
    d = np.array(dir_xy, dtype=float).reshape(2)
    n = np.linalg.norm(d)
    if n < 1e-9 or width <= 0 or length <= 0:
        return None
    ex = np.array([d[0]/n, d[1]/n, 0.0])
    ey = np.array([-ex[1], ex[0], 0.0])
    half_w, half_l = width * 0.5, length * 0.5
    cx, cy = float(center_xy[0]), float(center_xy[1])
    c = np.array([cx, cy, float(top_z)], dtype=float)
    c1 = c + ex * half_l + ey * half_w
    c2 = c + ex * half_l - ey * half_w
    c3 = c - ex * half_l - ey * half_w
    c4 = c - ex * half_l + ey * half_w
    wire = make_wire_from_points([c1, c2, c3, c4])
    if not wire:
        return None
    face = BRepBuilderAPI_MakeFace(wire).Face()
    if face.IsNull():
        return None
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, -float(down_depth))).Shape()

def build_fixture_layout(processed: ProcessedCpacs, params: FixtureParams) -> FixtureBuildResult:
    """
    DEPRECATED: Use geometry_builder.build_fixture_geometry() instead.
    
    Verbatim port of monolith CpacsStepperTab.generate_and_export_layout adapted to ProcessedCpacs and refactor utils.
    Implements:
      - Front/rear fixtures from spar front face, extended to chord caps
      - Center-plane offset: inner faces at spar_half + clearance from spar center
      - Thickness-only extrusion along ± plane normal
      - Optional cradle centered across spar thickness
      - Build dihedraled wing cutter (spars+ribs+elevon) and cut from fixtures
      - Tabs with center offset = spar_half + clearance + mat_thick/2 along actual build_dir_xy
      - Base plate slotted by intersecting fused fixtures with padded slab and cutting plate
    """
    warnings.warn(
        "build_fixture_layout() is deprecated. Use geometry_builder.build_fixture_geometry() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Precompute useful values
    mat_thick = float(params.material_thickness_mm)
    clearance = float(params.slot_clearance_mm)

    # Convert viewer geometry to mm for all fixture operations
    def mm(v): return np.array(v, dtype=float) * 1000.0
    def mm_polys(polys): return [[mm(p) for p in poly] for poly in polys]

    # Determine a global base plate level using lowest spar bottom edge Z minus fixture height
    fixture_height = 50.0  # mm
    global_min_z = float('inf')

    # Collect all cutting shapes (spars and ribs) into a list for optimized spatial querying
    cutting_shapes: List[TopoDS_Shape] = []
    
    # Add spars for cutting structure
    for polys in processed.spar_surfaces:
        if len(polys) < 2:
            continue
        front_face_verts, back_face_verts = polys[0], polys[1]
        front_mm = [mm(v) for v in front_face_verts]
        back_mm = [mm(v) for v in back_face_verts]
        # bottom edge from front face indices 0..1
        p1_bottom = front_mm[0]; p2_bottom = front_mm[1]
        local_min = min(float(p1_bottom[2]), float(p2_bottom[2]))
        global_min_z = min(global_min_z, local_min)
        
        extrusion_vec = gp_Vec(*(np.array(back_mm[0]) - np.array(front_mm[0])))
        wire = make_wire_from_points(front_mm)
        if wire:
            face = BRepBuilderAPI_MakeFace(wire).Face()
            if not face.IsNull() and extrusion_vec.Magnitude() > 1e-7:
                solid = BRepPrimAPI_MakePrism(face, extrusion_vec).Shape()
                if solid and not solid.IsNull():
                    cutting_shapes.append(solid)

    # Add ribs
    for polys in processed.rib_surfaces:
        if len(polys) < 2:
            continue
        front_face_verts, back_face_verts = polys[0], polys[1]
        front_mm = [mm(v) for v in front_face_verts]
        back_mm = [mm(v) for v in back_face_verts]
        extrusion_vec = gp_Vec(*(np.array(back_mm[0]) - np.array(front_mm[0])))
        wire = make_wire_from_points(front_mm)
        if wire:
            face = BRepBuilderAPI_MakeFace(wire).Face()
            if not face.IsNull() and extrusion_vec.Magnitude() > 1e-7:
                solid = BRepPrimAPI_MakePrism(face, extrusion_vec).Shape()
                if solid and not solid.IsNull():
                    cutting_shapes.append(solid)

    # Precompute bboxes for cutting shapes
    cutting_bboxes = []
    for shape in cutting_shapes:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        cutting_bboxes.append(bbox)

    # NOTE: Per current requirements, only use spars and ribs to cut fixture geometry.
    # Elevon geometry is intentionally excluded from the cutting compound.

    # Compute base plate top Z
    global_base_plate_level = global_min_z - fixture_height

    # Build fixtures for each spar segment
    fixtures: List[TopoDS_Shape] = []
    spar_data_cache = []  # bottom edges and directions for tabs

    # Identify front-most spar (by smaller xsi) like monolith using processed mapping if available
    front_spar_uid = _determine_front_spar_uid(processed)

    # For each spar surfaces block, create front/rear fixtures and optional cradle
    for idx, polys in enumerate(processed.spar_surfaces):
        if len(polys) < 2:
            continue
        # Seed from FRONT face for fixtures, match cutting structure
        base_face_verts, opp_face_verts = polys[0], polys[1]  # base = front, opp = back
        p1_bottom, p2_bottom = [mm(p) for p in base_face_verts[:2]]
        p1_upper, p2_upper = [mm(p) for p in (base_face_verts[3], base_face_verts[2])]

        # Estimate chord line caps using dihedraled rib profiles directly (no extra dihedral added here)
        ribs_mm = [np.array(r) * 1000.0 for r in processed.dihedraled_rib_profiles]
        z1 = chord_z_from_ribs((p1_bottom[0], p1_bottom[1]), ribs_mm) or p1_upper[2]
        z2 = chord_z_from_ribs((p2_bottom[0], p2_bottom[1]), ribs_mm) or p2_upper[2]

        # Extended fixture polygon from base to chord line (cap Z from dihedraled ribs; no additional y*tan(dihedral))
        extended_p1 = [p1_bottom[0], p1_bottom[1], global_base_plate_level]
        extended_p2 = [p2_bottom[0], p2_bottom[1], global_base_plate_level]
        extended_p3 = [p2_bottom[0], p2_bottom[1], z2]
        extended_p4 = [p1_bottom[0], p1_bottom[1], z1]

        ext_wire = make_wire_from_points([extended_p1, extended_p2, extended_p3, extended_p4])
        if not ext_wire:
            continue
        ext_face = BRepBuilderAPI_MakeFace(ext_wire).Face()
        if ext_face.IsNull():
            continue
        # Derive OCC plane normal from the fixture base face (ground truth)
        base_wire = make_wire_from_points([p for p in [p1_bottom, p2_bottom, p2_upper, p1_upper]])
        if base_wire is None:
            continue
        base_face = BRepBuilderAPI_MakeFace(base_wire).Face()
        if base_face.IsNull():
            continue
        surf = BRepAdaptor_Surface(base_face, True)
        if surf.GetType() != GeomAbs_Plane:
            continue
        pln = surf.Plane()
        n_dir = pln.Axis().Direction()
        n = np.array([n_dir.X(), n_dir.Y(), n_dir.Z()], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            n = n / n_norm

        # Enforce rearward polarity: ensure +n points from FRONT to REAR using opposite face delta
        back_vec = np.array(mm(opp_face_verts[0])) - np.array(mm(base_face_verts[0]))
        if np.dot(n, back_vec / (np.linalg.norm(back_vec) or 1.0)) < 0.0:
            n = -n

        # Thickness parity: use CPACS-provided spar thickness or flat plate mode
        if params.spar_is_flat_plate:
            # Use flat plate thickness for hobbyist construction
            spar_thickness_mm = params.spar_plate_thickness_mm + params.slot_clearance_mm
        else:
            # Use CPACS-provided spar thickness from 3D geometry
            spar_uid = processed.spar_uids[idx] if idx < len(processed.spar_uids) else ""
            spar_thickness_m = float(processed.initial_spar_thicknesses.get(spar_uid, 0.00635))
            spar_thickness_mm = spar_thickness_m * 1000.0
        spar_half = 0.5 * spar_thickness_mm
        
        # Monolith parity: inner-face offset
        auto_face_offset_mm = max(0.0, float(clearance))
        
        # Build explicit FRONT and REAR reference faces
        front_ref_face = ext_face
        tr_to_rear = gp_Trsf()
        tr_to_rear.SetTranslation(gp_Vec(*(n * float(spar_thickness_mm))))
        rear_ref_face = BRepBuilderAPI_Transform(ext_face, tr_to_rear, True).Shape()

        # Front fixture
        tr_front = gp_Trsf()
        tr_front.SetTranslation(gp_Vec(*(-n * float(clearance))))
        front_inner = BRepBuilderAPI_Transform(front_ref_face, tr_front, True).Shape()
        front_extrude = extrude_prism(front_inner, tuple(-n * mat_thick))
        if front_extrude and not front_extrude.IsNull():
            fixtures.append(front_extrude)

        # Rear fixture
        tr_rear = gp_Trsf()
        tr_rear.SetTranslation(gp_Vec(*(n * float(clearance))))
        rear_inner = BRepBuilderAPI_Transform(rear_ref_face, tr_rear, True).Shape()
        rear_extrude = extrude_prism(rear_inner, tuple(n * mat_thick))
        if rear_extrude and not rear_extrude.IsNull():
            fixtures.append(rear_extrude)

        # Cache build directions
        spar_vec = np.array(p2_bottom) - np.array(p1_bottom)
        spar_vec_xy = (spar_vec[0:2] / (np.linalg.norm(spar_vec[0:2]) or 1.0))

        # Optional cradle
        if params.add_cradle:
            cradle_top_points = []
            len_spar = np.linalg.norm(spar_vec)
            num_samples = max(2, int(len_spar / 5.0))
            t_values = np.linspace(0, 1, num_samples)
            
            for t in t_values:
                pt = np.array(p1_bottom) + spar_vec * t
                z_surf = _lower_surface_z_from_ribs((pt[0], pt[1]), ribs_mm)
                if z_surf is None:
                    z_surf = p1_bottom[2] + (p2_bottom[2] - p1_bottom[2]) * t
                cradle_top_points.append([pt[0], pt[1], z_surf])
            
            cradle_p1 = [p1_bottom[0], p1_bottom[1], global_base_plate_level]
            cradle_p2 = [p2_bottom[0], p2_bottom[1], global_base_plate_level]
            wire_points = [cradle_p1, cradle_p2]
            wire_points.extend(reversed(cradle_top_points))
            
            cradle_wire = make_wire_from_points(wire_points)
            if cradle_wire:
                cradle_face = BRepBuilderAPI_MakeFace(cradle_wire).Face()
                if not cradle_face.IsNull():
                    tr_center = gp_Trsf()
                    tr_center.SetTranslation(gp_Vec(*(n * float(spar_thickness_mm / 2.0))))
                    center_ref_face = BRepBuilderAPI_Transform(cradle_face, tr_center, True).Shape()
                    tr_start = gp_Trsf()
                    tr_start.SetTranslation(gp_Vec(*(-n * float(spar_thickness_mm / 2.0))))
                    start_face = BRepBuilderAPI_Transform(center_ref_face, tr_start, True).Shape()
                    cradle_extrude = extrude_prism(start_face, tuple(n * float(spar_thickness_mm)))
                    if cradle_extrude and not cradle_extrude.IsNull():
                        fixtures.append(cradle_extrude)

        spar_data_cache.append({
            'bottom_edge': (p1_bottom, p2_bottom),
            'dir_xy_front': tuple((-n[0], -n[1])),
            'dir_xy_rear': tuple((n[0], n[1])),
            'spar_dir_xy': tuple(spar_vec_xy.tolist()),
            'spar_half': float(spar_half),
            'spar_thickness_mm': float(spar_thickness_mm),
            'front_inner_offset_mm': float(clearance),
            'rear_inner_offset_mm': float(clearance)
        })

    # Cut wing structure from fixtures to create indexing
    indexed_fixtures: List[TopoDS_Shape] = []
    for i, fx in enumerate(fixtures):
        if not fx or fx.IsNull():
            continue
            
        # Optimization: only cut with shapes that intersect the fixture's bounding box
        fx_bbox = Bnd_Box()
        brepbndlib.Add(fx, fx_bbox)
        
        relevant_cutters = []
        for shape, bbox in zip(cutting_shapes, cutting_bboxes):
            if not fx_bbox.IsOut(bbox):
                relevant_cutters.append(shape)
                
        if not relevant_cutters:
            indexed_fixtures.append(fx)
            continue
            
        # Fuse relevant cutters into a single compound for this fixture
        cutter_comp = make_compound(relevant_cutters)
        
        # Perform cut
        cut_result = bool_cut(fx, cutter_comp)
        indexed_fixtures.append(cut_result if cut_result and not cut_result.IsNull() else fx)

    # Generate tabs and fuse to fixtures
    fused_fixtures: List[TopoDS_Shape] = []
    if params.add_tabs:
        n_per_spar = 2  # Do not generate tabs for the cradle; only front and rear fixtures get tabs
        tabs_for_idx: Dict[int, List[TopoDS_Shape]] = {}

        for sidx, data in enumerate(spar_data_cache):
            # Each spar contributes n_per_spar fixtures; assign tabs to those indices
            base_idx = sidx * n_per_spar

            # Spanwise edge vector and unit direction along the fixture edge
            be_p1, be_p2 = data['bottom_edge']
            p1, p2 = np.array(be_p1, dtype=float), np.array(be_p2, dtype=float)
            v = p2 - p1
            edge_len = float(np.linalg.norm(v[:2]))
            if edge_len < 1e-6:
                continue
            dir_xy = (v[:2] / edge_len)

            # Monolith spacing semantics:
            # - Always place a tab near each end (edge margin)
            # - Fill interior at approximately tab_spacing_mm, clamped to usable length
            # - If only one tab fits, place it centered
            usable = max(0.0, edge_len - 2.0 * params.tab_edge_margin_mm)
            if usable <= 0.0:
                continue

            if usable < params.tab_spacing_mm * 1.5:
                # Small span region: 1 centered tab
                offsets_along = [edge_len * 0.5]
            else:
                # Place end tabs at margins, then distribute interior by spacing
                start = params.tab_edge_margin_mm
                end = edge_len - params.tab_edge_margin_mm
                # Compute evenly spaced interior positions including ends
                n_interior = max(0, int(np.floor((end - start) / params.tab_spacing_mm)) - 1)
                interior_spacing = (end - start) / (n_interior + 1)
                offsets_along = [start + i * interior_spacing for i in range(n_interior + 2)]  # includes start, end

            # Generate tabs for the two normal-facing fixtures and optional cradle
            for local_i in range(n_per_spar):
                fixture_index = base_idx + local_i
                if fixture_index >= len(indexed_fixtures):
                    continue
                
                # Determine build direction per fixture (front, rear, or cradle)
                if True:  # Only front/rear fixtures receive tabs; cradle is excluded
                    # Use the exact OCC-normal-derived directions cached when building fixtures
                    raw_build = np.array(data['dir_xy_front'] if local_i == 0 else data['dir_xy_rear'], dtype=float)
                    bn = np.linalg.norm(raw_build)
                    if bn < 1e-9:
                        continue
                    build_dir_xy = raw_build / bn

                    # Tabs: width across equals material thickness for front/rear fixtures
                    width_across = mat_thick

                    # Pull the tab center offset from the SAME offset logic as the parent fixture:
                    # For front fixture, inner face offset magnitude from FRONT face = front_inner_offset_mm
                    # For rear fixture, inner face offset magnitude from REAR face  = rear_inner_offset_mm
                    # Tabs sit on the fixture mid-plane, so add mat_thick/2 along the same outward build_dir_xy.
                    parent_inner_offset = float(data['front_inner_offset_mm'] if local_i == 0 else data['rear_inner_offset_mm'])

                    # Base outward offset equals the parent inner-face offset + half the tab thickness.
                    offset_center = parent_inner_offset + (mat_thick * 0.5)

                    # IMPORTANT: The rear fixture is referenced from the REAR spar face, which is FRONT + n * spar_thickness_mm.
                    # To align its tabs with the actual rear-based slab, include the base REAR shift magnitude (spar_thickness_mm).
                    if local_i == 1:  # rear fixture
                        offset_center += float(data.get('spar_thickness_mm', 0.0))

                    offset_vec = build_dir_xy * offset_center
                else:
                    # No tabs for cradle
                    continue

                for off in offsets_along:
                    center_xy = (p1[:2] + dir_xy * off) + offset_vec
                    tab = _make_rect_prism(
                        center_xy=center_xy,
                        dir_xy=dir_xy,
                        width=width_across,
                        length=params.tab_width_mm,
                        top_z=global_base_plate_level + 0.2,
                        down_depth=mat_thick + 0.4
                    )
                    if tab and not tab.IsNull():
                        tabs_for_idx.setdefault(fixture_index, []).append(tab)

        # Fuse tabs
        for i, fx in enumerate(indexed_fixtures):
            to_fuse = tabs_for_idx.get(i, [])
            fused = fx
            for t in to_fuse:
                fused = bool_fuse(fused, t) if fused else t
            fused_fixtures.append(fused if fused else fx)
    else:
        # If params.add_tabs is False, then fused_fixtures should just be indexed_fixtures
        fused_fixtures = indexed_fixtures

    base_plate: Optional[TopoDS_Shape] = None
    overall_bbox = Bnd_Box()
    for solid in fused_fixtures:
        if solid and not solid.IsNull():
            brepbndlib.Add(solid, overall_bbox)

    # If bbox is void (e.g., boolean ops produced nulls), fall back to raw indexed fixtures bbox
    if overall_bbox.IsVoid():
        for solid in indexed_fixtures:
            if solid and not solid.IsNull():
                brepbndlib.Add(solid, overall_bbox)

    if not overall_bbox.IsVoid():
        xmin, ymin, zmin, xmax, ymax, zmax = overall_bbox.Get()
        margin = params.base_plate_margin_mm

        # Ensure the base plate thickness is EXACTLY the material thickness (no clearance added).
        top_z = global_base_plate_level
        bottom_z = top_z - mat_thick

        p1 = gp_Pnt(xmin - margin, ymin - margin, bottom_z)
        p2 = gp_Pnt(xmax + margin, ymax + margin, top_z)
        candidate_plate = BRepPrimAPI_MakeBox(p1, p2).Shape()

        if candidate_plate and not candidate_plate.IsNull():
            base_plate = candidate_plate

            if params.slot_base_plate:
                # Build a slab slightly larger than the base plate in XY and thicker in Z
                # to robustly intersect all tabs/fixtures passing through the plate span.
                slab_pad_xy = 5.0
                # Make the slab at least the base plate thickness on each side to guarantee full-through intersection
                slab_pad_z = max(2.0, mat_thick)
                slab_p1 = gp_Pnt(xmin - margin - slab_pad_xy, ymin - margin - slab_pad_xy, bottom_z - slab_pad_z)
                slab_p2 = gp_Pnt(xmax + margin + slab_pad_xy, ymax + margin + slab_pad_xy, top_z + slab_pad_z)
                slab = BRepPrimAPI_MakeBox(slab_p1, slab_p2).Shape()

                fixtures_comp = make_compound(fused_fixtures if fused_fixtures else indexed_fixtures)
                # Intersect to isolate only the geometry passing through the plate Z-span
                slice_tool = bool_common(fixtures_comp, slab) if fixtures_comp else None
                negative = slice_tool if (slice_tool and not slice_tool.IsNull()) else fixtures_comp

                # Cut the isolated pass-through geometry from the base plate to produce through-slots
                if negative and not negative.IsNull():
                    cut_plate = bool_cut(base_plate, negative)
                    if cut_plate and not cut_plate.IsNull():
                        base_plate = cut_plate
    # else: keep base_plate as null, final assembly will exclude it gracefully

    # Assemble final layout
    final_layout = make_compound([*fused_fixtures, base_plate] if base_plate and not base_plate.IsNull() else fused_fixtures)

    return FixtureBuildResult(
        fixtures_fused=fused_fixtures,
        base_plate=base_plate,
        final_layout=final_layout
    )


def write_fixture_step(final_layout: TopoDS_Compound, path: str) -> bool:
    writer = STEPControl_Writer()
    writer.Transfer(final_layout, STEPControl_AsIs)
    return writer.Write(path) == 1