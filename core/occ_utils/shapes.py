"""
Thin pythonocc-core wrappers for shape construction used across services.
These mirror the monolith behavior with basic safety checks.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir, gp_Trsf
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Compound, TopoDS_Shell
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_Interpolate


def make_wire_from_points(points: Iterable[Iterable[float]]) -> TopoDS_Wire | None:
    """
    Safe wire creation identical to monolith CpacsStepperTab._make_wire_from_points:
    - Accepts an iterable of 3D points (x, y, z)
    - Skips zero-length edges
    - Returns None if wire cannot be constructed
    """
    pts = [np.array(p, dtype=float).reshape(3) for p in points]
    if len(pts) < 3:
        return None

    wb = BRepBuilderAPI_MakeWire()
    n = len(pts)
    for i in range(n):
        p1 = gp_Pnt(*pts[i])
        p2 = gp_Pnt(*pts[(i + 1) % n])
        if p1.Distance(p2) > 1e-7:
            edge = BRepBuilderAPI_MakeEdge(p1, p2)
            if edge.IsDone():
                wb.Add(edge.Edge())
    if wb.IsDone():
        return wb.Wire()
    return None


def make_face_from_points(points: Iterable[Iterable[float]]):
    wire = make_wire_from_points(points)
    if not wire:
        return None
    face = BRepBuilderAPI_MakeFace(wire).Face()
    return None if face.IsNull() else face


def extrude_prism(face, vec_xyz: Iterable[float]) -> TopoDS_Shape:
    vx, vy, vz = [float(c) for c in vec_xyz]
    return BRepPrimAPI_MakePrism(face, gp_Vec(vx, vy, vz)).Shape()


def loft_solid_from_wires(wires: List[TopoDS_Wire], *, continuity: str | None = None, max_degree: int | None = None, ruled: bool = False) -> TopoDS_Shape:
    wires = [w for w in wires if w is not None]
    if len(wires) < 2:
        return TopoDS_Shape()
    loft = BRepOffsetAPI_ThruSections(True, ruled)  # solid, ruled?
    # Optional smoothness controls (used by spline wing path only)
    if continuity:
        try:
            if continuity.upper() == 'C2':
                loft.SetContinuity(GeomAbs_C2)
            elif continuity.upper() == 'C1':
                loft.SetContinuity(GeomAbs_C1)
            else:
                loft.SetContinuity(GeomAbs_C0)
        except Exception:
            pass
    if isinstance(max_degree, int) and max_degree > 1:
        try:
            loft.SetMaxDegree(int(max_degree))
        except Exception:
            pass
    for w in wires:
        loft.AddWire(w)
    try:
        loft.CheckCompatibility(True)
    except Exception:
        pass
    loft.Build()
    
    # Check if the loft operation succeeded before calling Shape()
    if not loft.IsDone():
        print(f"[WARN] Loft operation failed: IsDone() returned False")
        return TopoDS_Shape()
    
    try:
        result = loft.Shape()
        if result is None or result.IsNull():
            print(f"[WARN] Loft operation returned null shape")
            return TopoDS_Shape()
        return result
    except Exception as e:
        print(f"[WARN] Loft Shape() failed: {e}")
        return TopoDS_Shape()


def scale_shape(shape: TopoDS_Shape, factor: float, center=(0.0, 0.0, 0.0)) -> TopoDS_Shape:
    """
    Uniform scale a shape around a specified center point.
    Matches monolith use where hollowing scales about a local barycenter.
    """
    tr = gp_Trsf()
    cx, cy, cz = [float(c) for c in center]
    tr.SetScale(gp_Pnt(cx, cy, cz), float(factor))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def mirror_y(shape: TopoDS_Shape) -> TopoDS_Shape:
    tr = gp_Trsf()
    tr.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    mirrored = BRepBuilderAPI_Transform(shape, tr, True).Shape()
    
    # Fix mirrored shape topology/orientation to prevent "flipped" solids
    # and weird surface artifacts in CAD exporters.
    fixer = ShapeFix_Shape(mirrored)
    fixer.Perform()
    return fixer.Shape()


def make_compound(shapes: List[TopoDS_Shape]) -> TopoDS_Compound:
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for s in shapes:
        if s and not s.IsNull():
            builder.Add(comp, s)
    return comp


def make_bspline_edge_from_points(points: Iterable[Iterable[float]], degree: int = 3) -> TopoDS_Shape | None:
    """
    Create a BSpline edge from an ordered list of 3D points.
    Uses GeomAPI_PointsToBSpline (open curve). Returns None if it fails.
    """
    pts = [np.array(p, dtype=float).reshape(3) for p in points]
    if len(pts) < 2:
        return None
    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    try:
        builder = GeomAPI_PointsToBSpline(arr, degree, 3)  # use default parameters, open curve
        curve = builder.Curve()
        edge = BRepBuilderAPI_MakeEdge(curve)
        return edge.Edge() if edge.IsDone() else None
    except Exception:
        return None


def _split_airfoil_upper_lower(profile: Iterable[Iterable[float]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split a closed airfoil polyline into upper and lower open polylines.
    - Detect LE (min X) and TE (max X)
    - Build two paths between LE and TE along existing order
    - Classify upper/lower by mean Z
    Returns (upper, lower) as lists of np.ndarray points including LE and TE endpoints once each.
    """
    pts = [np.array(p, dtype=float).reshape(3) for p in profile]
    if len(pts) < 4:
        return pts[:], pts[:]
    xs = np.array([p[0] for p in pts])
    le_idx = int(np.argmin(xs))
    te_idx = int(np.argmax(xs))
    n = len(pts)
    # Path A: LE -> TE following index increasing (wrap if needed)
    if le_idx <= te_idx:
        pathA = pts[le_idx:te_idx + 1]
    else:
        pathA = pts[le_idx:] + pts[:te_idx + 1]
    # Path B: TE -> LE following index increasing (wrap), we then reverse to get LE->TE
    if te_idx <= le_idx:
        tmp = pts[te_idx:le_idx + 1]
    else:
        tmp = pts[te_idx:] + pts[:le_idx + 1]
    pathB = list(reversed(tmp))
    # Remove duplicate interior endpoints if any
    def _dedupe(seq: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in seq:
            if not out or np.linalg.norm(p - out[-1]) > 1e-9:
                out.append(p)
        return out
    pathA = _dedupe(pathA)
    pathB = _dedupe(pathB)
    # Classify by mean Z (upper has higher Z)
    meanA = float(np.mean([p[2] for p in pathA]))
    meanB = float(np.mean([p[2] for p in pathB]))
    if meanA >= meanB:
        upper, lower = pathA, pathB
    else:
        upper, lower = pathB, pathA
    return upper, lower


def make_airfoil_wire_spline(profile: Iterable[Iterable[float]]) -> TopoDS_Wire | None:
    """
    Build a closed wire for an airfoil by:
      - creating two open BSpline edges (upper/lower) from the polyline,
      - stitching them with straight edges at LE and TE.
    """
    upper, lower = _split_airfoil_upper_lower(profile)
    if len(upper) < 2 or len(lower) < 2:
        return None
    # Ensure both curves are oriented LE->TE and share identical LE/TE points (snap)
    def _approx_equal(a, b, tol=1.0e-7):
        return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)) < tol
    # If lower doesn't start at the same LE as upper, try reversing it
    if not _approx_equal(upper[0], lower[0]):
        lower_rev = list(reversed(lower))
        if _approx_equal(upper[0], lower_rev[0]):
            lower = lower_rev
    # Snap LE/TE to the average of upper/lower endpoints to avoid tiny gaps
    le = 0.5 * (np.array(upper[0], dtype=float) + np.array(lower[0], dtype=float))
    te = 0.5 * (np.array(upper[-1], dtype=float) + np.array(lower[-1], dtype=float))
    upper[0] = le.copy(); lower[0] = le.copy()
    upper[-1] = te.copy(); lower[-1] = te.copy()

    e_upper = make_bspline_edge_from_points(upper)
    e_lower = make_bspline_edge_from_points(lower)
    if e_upper is None or e_lower is None:
        return None
    # Build a closed wire by chaining upper (LE->TE) and lower reversed (TE->LE).
    # With snapped identical endpoints, no stitch edges are required.
    try:
        lower_edge = e_lower.Reversed()
    except Exception:
        # Fallback: add tiny stitches if reverse fails
        le_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*le), gp_Pnt(*le))
        te_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*te), gp_Pnt(*te))
        if not (le_edge.IsDone() and te_edge.IsDone()):
            return None
        wb = BRepBuilderAPI_MakeWire()
        for e in (e_upper, te_edge.Edge(), e_lower, le_edge.Edge()):
            wb.Add(e)
        return wb.Wire() if wb.IsDone() else None
    wb = BRepBuilderAPI_MakeWire()
    for e in (e_upper, lower_edge):
        wb.Add(e)
    return wb.Wire() if wb.IsDone() else None


def split_airfoil_upper_lower(profile: Iterable[Iterable[float]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Public wrapper to obtain (upper, lower) open polylines from a closed airfoil profile."""
    return _split_airfoil_upper_lower(profile)


def make_partial_airfoil_wire_spline(
    upper_pts: List[Iterable[float]],
    lower_pts: List[Iterable[float]],
    n_resample: int = 20,
) -> TopoDS_Wire | None:
    """
    Build a closed wire for a partial airfoil (e.g., control surface aft section).
    
    Takes separate upper and lower surface point lists, resamples them to have
    consistent point counts, and creates a closed polyline wire with sharp corners
    at the hinge and trailing edge.
    
    Args:
        upper_pts: Points along upper surface, ordered from hinge to TE
        lower_pts: Points along lower surface, ordered from hinge to TE
        n_resample: Number of points to resample each surface to (default 20)
    
    Returns a closed wire suitable for lofting.
    """
    if len(upper_pts) < 2 or len(lower_pts) < 2:
        return None
    
    upper = [np.array(p, dtype=float).reshape(3) for p in upper_pts]
    lower = [np.array(p, dtype=float).reshape(3) for p in lower_pts]
    
    def _resample_polyline(pts: List[np.ndarray], n: int) -> List[np.ndarray]:
        """Resample a polyline to have exactly n points, preserving endpoints."""
        if len(pts) <= 2:
            return pts
        
        # Compute cumulative arc length
        dists = [0.0]
        for i in range(1, len(pts)):
            dists.append(dists[-1] + np.linalg.norm(pts[i] - pts[i-1]))
        total_len = dists[-1]
        
        if total_len < 1e-9:
            return pts
        
        # Resample at uniform arc length intervals
        result = [pts[0].copy()]
        for i in range(1, n - 1):
            target_dist = (i / (n - 1)) * total_len
            # Find segment containing this distance
            for j in range(1, len(dists)):
                if dists[j] >= target_dist:
                    # Interpolate within segment j-1 to j
                    seg_start = dists[j-1]
                    seg_end = dists[j]
                    t = (target_dist - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
                    pt = pts[j-1] + t * (pts[j] - pts[j-1])
                    result.append(pt)
                    break
        result.append(pts[-1].copy())
        return result
    
    # Resample both surfaces to same point count
    upper_resampled = _resample_polyline(upper, n_resample)
    lower_resampled = _resample_polyline(lower, n_resample)
    
    # Snap endpoints to match exactly
    # Hinge point (start)
    hinge = 0.5 * (upper_resampled[0] + lower_resampled[0])
    upper_resampled[0] = hinge.copy()
    lower_resampled[0] = hinge.copy()
    
    # TE point (end)
    te = 0.5 * (upper_resampled[-1] + lower_resampled[-1])
    upper_resampled[-1] = te.copy()
    lower_resampled[-1] = te.copy()
    
    # Build closed polyline: upper (hinge->TE) + lower reversed (TE->hinge)
    # This creates a sharp V at hinge and sharp corner at TE
    profile = upper_resampled + list(reversed(lower_resampled[:-1]))  # Avoid duplicate TE point
    
    return make_wire_from_points(profile)


def make_closed_bspline_wire_from_points(points: Iterable[Iterable[float]], tol: float = 1.0e-7) -> TopoDS_Wire | None:
    """
    Build a single-edge closed BSpline wire interpolating the given points.
    Uses GeomAPI_Interpolate with periodic=True to get a smooth closed curve.
    Returns None if interpolation fails.
    """
    pts = [np.array(p, dtype=float).reshape(3) for p in points]
    if len(pts) < 3:
        return None
    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    try:
        interp = GeomAPI_Interpolate(arr, True, tol)  # periodic closed curve
        interp.Perform()
        curve = interp.Curve()
        edge = BRepBuilderAPI_MakeEdge(curve)
        if not edge.IsDone():
            return None
        wire_mk = BRepBuilderAPI_MakeWire(edge.Edge())
        return wire_mk.Wire() if wire_mk.IsDone() else None
    except Exception:
        return None


def loft_surface_from_profiles(profiles: List[Iterable[Iterable[float]]], which: str = 'upper',
                               *, continuity: str = 'C2', max_degree: int = 8):
    """
    Build a single lofted surface (not solid) from either the 'upper' or 'lower' open BSpline edge
    extracted from each closed airfoil profile in profiles. Returns a shell/face shape.
    """
    edges = []
    for prof in profiles:
        up, low = _split_airfoil_upper_lower(prof)
        seq = up if which.lower().startswith('u') else low
        e = make_bspline_edge_from_points(seq)
        if e is not None:
            edges.append(e)
    if len(edges) < 2:
        return TopoDS_Shape()
    loft = BRepOffsetAPI_ThruSections(False)  # not solid
    # smoothness
    try:
        loft.SetContinuity(GeomAbs_C2 if (continuity and continuity.upper() == 'C2') else GeomAbs_C1)
        if isinstance(max_degree, int) and max_degree > 1:
            loft.SetMaxDegree(int(max_degree))
    except Exception:
        pass
    for e in edges:
        try:
            w = BRepBuilderAPI_MakeWire(e).Wire()
            loft.AddWire(w)
        except Exception:
            # skip if cannot wrap edge
            continue
    try:
        loft.CheckCompatibility(True)
    except Exception:
        pass
    loft.Build()
    return loft.Shape()


def sew_faces_to_solid(faces: List[TopoDS_Shape]) -> TopoDS_Shape:
    """Sew a list of faces/shells and try to make a solid. Returns sewed shape if solid fails."""
    sewing = BRepBuilderAPI_Sewing(1.0e-6, True, True, True, True)
    for f in faces:
        if f and not f.IsNull():
            sewing.Add(f)
    try:
        sewing.Perform()
        sewed = sewing.SewedShape()
    except Exception:
        # fallback: compound
        return make_compound([f for f in faces if f and not f.IsNull()])
    # find a shell to convert to solid
    try:
        exp = TopExp_Explorer(sewed, TopAbs_SHELL)
        while exp.More():
            shell = TopoDS_Shell(exp.Current())
            solid_builder = BRepBuilderAPI_MakeSolid(shell)
            solid = solid_builder.Solid()
            if solid and not solid.IsNull():
                return solid
            exp.Next()
        return sewed
    except Exception:
        return sewed


def unify_same_domain(shape: TopoDS_Shape, *, unify_edges: bool = True, unify_faces: bool = True) -> TopoDS_Shape:
    """Merge tangent-adjacent faces/edges that share the same underlying geometry."""
    try:
        unifier = ShapeUpgrade_UnifySameDomain(shape, unify_edges, unify_faces)
        unifier.Build()
        res = unifier.Shape()
        return res if res and not res.IsNull() else shape
    except Exception:
        return shape


def bool_fuse(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    Fuse two shapes using boolean union.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    try:
        algo = BRepAlgoAPI_Fuse(shape1, shape2)
        algo.Build()
        if algo.IsDone():
            return algo.Shape()
    except Exception:
        pass
    return TopoDS_Shape()


def bool_cut(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    Cut shape2 from shape1 using boolean cut.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    try:
        algo = BRepAlgoAPI_Cut(shape1, shape2)
        algo.Build()
        if algo.IsDone():
            res = algo.Shape()
            if res and not res.IsNull():
                return res
    except Exception:
        pass
    except Exception:
        pass
    return None # Return None on failure to distinguish from success


def bool_common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    Intersect shape1 and shape2 using boolean common.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    try:
        algo = BRepAlgoAPI_Common(shape1, shape2)
        algo.Build()
        if algo.IsDone():
            res = algo.Shape()
            if res and not res.IsNull():
                return res
    except Exception:
        pass
    return TopoDS_Shape() # Return empty shape on failure
