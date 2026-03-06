"""
PyVista preview utilities for interactive 3D visualization.

Initial scope:
- Build lightweight meshes from ProcessedCpacs polygonal surfaces (spars, ribs, optional elevon)
- Launch an external PyVistaQt BackgroundPlotter window from Tkinter GUI
- No OCC meshing yet; this focuses on surfaces already available from processing

Requirements:
- pyvista (>=0.43)
- pyvistaqt (>=0.11)
- vtk (matching wheel for your Python)

Usage:
    from refactor.services.viewer_pv import open_pyvista_window
    open_pyvista_window(processed, include_elevon=True)
"""

from __future__ import annotations

from typing import List, Iterable, Tuple, Optional
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
except Exception as e:  # Defer import errors to runtime message in caller
    pv = None
    BackgroundPlotter = None

from refactor.services.viewer import ProcessedCpacs  # [refactor.services.viewer.ProcessedCpacs](refactor/services/viewer.py:1)


def _rotate_points_about_axis(points: Iterable[Iterable[float]],
                              origin: Iterable[float],
                              axis_dir: Iterable[float],
                              angle_deg: float) -> List[Tuple[float, float, float]]:
    """
    Rotate a set of points about an axis defined by origin and axis_dir using Rodrigues' formula.
    Returns a new list of rotated (x,y,z) tuples.
    """
    pts = np.array(points, dtype=float)
    o = np.array(origin, dtype=float).reshape(3)
    k = np.array(axis_dir, dtype=float).reshape(3)
    norm = np.linalg.norm(k)
    if norm == 0.0 or pts.size == 0:
        return [tuple(p.tolist()) for p in pts]
    k = k / norm
    theta = np.deg2rad(float(angle_deg))
    # Shift to origin
    v = pts - o
    # Rodrigues: v_rot = v*cosθ + (k×v)*sinθ + k*(k·v)*(1−cosθ)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    kv = np.dot(v, k)  # (N,)
    k_cross_v = np.cross(k, v)  # (N,3)
    v_rot = v * cos_t + k_cross_v * sin_t + np.outer(kv, k) * (1.0 - cos_t)
    out = v_rot + o
    return [tuple(row.tolist()) for row in out]


def _deflect_elevon_bottom_hinge(processed: ProcessedCpacs, angle_deg: float) -> Optional[List[List[List[Iterable[float]]]]]:
    """
    Build a deflected elevon polys structure by rotating elevon surfaces about the
    bottom edge of the REAR face of rearSpar_outboard.
    Returns a polys_group compatible with _build_group_mesh or None on failure.
    """
    try:
        if processed is None or angle_deg == 0.0:
            return None
        if not getattr(processed, "elevon_surfaces", None):
            return None
        elevon_group = processed.elevon_surfaces
        if not elevon_group or not elevon_group[0]:
            return None

        # Find rearSpar_outboard index
        uids = list(getattr(processed, "spar_uids", []) or [])
        if not uids:
            return None
        try:
            idx = uids.index("rearSpar_outboard")
        except ValueError:
            # If exact UID not found, try a relaxed search
            idx = next((i for i, u in enumerate(uids) if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
            if idx is None:
                return None

        # Extract REAR face bottom edge from spar_surfaces[idx]
        # Assumption per design note:
        # polys[1] corresponds to REAR face with vertices ordered:
        # [0]=bottom-inboard, [1]=bottom-outboard, [2]=top-outboard, [3]=top-inboard
        spar_surfs = getattr(processed, "spar_surfaces", None)
        if not spar_surfs or idx >= len(spar_surfs):
            return None
        rear_faces = spar_surfs[idx]
        if not rear_faces or len(rear_faces) < 2:
            return None
        rear_face = rear_faces[1]
        if not rear_face or len(rear_face) < 2:
            return None
        p0 = np.array(rear_face[0], dtype=float)
        p1 = np.array(rear_face[1], dtype=float)
        axis_origin = p0
        axis_dir = p1 - p0
        if np.linalg.norm(axis_dir) == 0.0:
            return None

        # Rotate every vertex in every face of elevon_surfaces[0]
        deflected_group: List[List[List[Iterable[float]]]] = []
        for surface in elevon_group:
            if not surface:
                deflected_group.append([])
                continue
            new_surface: List[List[Iterable[float]]] = []
            for face in surface:
                if not face or len(face) < 1:
                    new_surface.append(face)
                    continue
                rotated = _rotate_points_about_axis(face, axis_origin, axis_dir, angle_deg)
                new_surface.append(rotated)
            deflected_group.append(new_surface)

        return deflected_group
    except Exception:
        return None


def _deflect_elevon_top_hinge(processed: ProcessedCpacs, angle_deg: float) -> Optional[List[List[List[Iterable[float]]]]]:
    """
    Rotate elevon surfaces about the TOP edge of the REAR face (rearSpar_outboard).
    Hinge axis: rear_face[3] (top-inboard) -> rear_face[2] (top-outboard)
    """
    try:
        if processed is None or angle_deg == 0.0:
            return None
        if not getattr(processed, "elevon_surfaces", None):
            return None

        uids = list(getattr(processed, "spar_uids", []) or [])
        if not uids:
            return None
        try:
            idx = uids.index("rearSpar_outboard")
        except ValueError:
            idx = next((i for i, u in enumerate(uids) if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
            if idx is None:
                return None

        spar_surfs = getattr(processed, "spar_surfaces", None)
        if not spar_surfs or idx >= len(spar_surfs):
            return None
        rear_faces = spar_surfs[idx]
        if not rear_faces or len(rear_faces) < 2:
            return None
        rear_face = rear_faces[1]
        if not rear_face or len(rear_face) < 4:
            return None

        # Top edge: vertices [3] -> [2]
        p_top_in = np.array(rear_face[3], dtype=float)
        p_top_out = np.array(rear_face[2], dtype=float)
        axis_origin = p_top_in
        axis_dir = p_top_out - p_top_in
        if np.linalg.norm(axis_dir) == 0.0:
            return None

        deflected_group: List[List[List[Iterable[float]]]] = []
        for surface in processed.elevon_surfaces:
            if not surface:
                deflected_group.append([])
                continue
            new_surface: List[List[Iterable[float]]] = []
            for face in surface:
                if not face:
                    new_surface.append(face)
                    continue
                rotated = _rotate_points_about_axis(face, axis_origin, axis_dir, angle_deg)
                new_surface.append(rotated)
            deflected_group.append(new_surface)
        return deflected_group
    except Exception:
        return None


def _deflect_elevon_centerline_hinge(processed: ProcessedCpacs, angle_deg: float) -> Optional[List[List[List[Iterable[float]]]]]:
    """
    Rotate elevon surfaces about the CENTERLINE of the REAR face (mid between bottom and top edges).
    Axis is built by averaging corresponding bottom and top points along the rear face.
    """
    try:
        if processed is None or angle_deg == 0.0:
            return None
        if not getattr(processed, "elevon_surfaces", None):
            return None

        uids = list(getattr(processed, "spar_uids", []) or [])
        if not uids:
            return None
        try:
            idx = uids.index("rearSpar_outboard")
        except ValueError:
            idx = next((i for i, u in enumerate(uids) if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
            if idx is None:
                return None

        spar_surfs = getattr(processed, "spar_surfaces", None)
        if not spar_surfs or idx >= len(spar_surfs):
            return None
        rear_faces = spar_surfs[idx]
        if not rear_faces or len(rear_faces) < 2:
            return None
        rear_face = rear_faces[1]
        if not rear_face or len(rear_face) < 4:
            return None

        # Bottom edge midpoint and top edge midpoint
        p_bot_in = np.array(rear_face[0], dtype=float)
        p_bot_out = np.array(rear_face[1], dtype=float)
        p_top_out = np.array(rear_face[2], dtype=float)
        p_top_in = np.array(rear_face[3], dtype=float)
        mid_in = 0.5 * (p_bot_in + p_top_in)
        mid_out = 0.5 * (p_bot_out + p_top_out)
        axis_origin = mid_in
        axis_dir = mid_out - mid_in
        if np.linalg.norm(axis_dir) == 0.0:
            return None

        deflected_group: List[List[List[Iterable[float]]]] = []
        for surface in processed.elevon_surfaces:
            if not surface:
                deflected_group.append([])
                continue
            new_surface: List[List[Iterable[float]]] = []
            for face in surface:
                if not face:
                    new_surface.append(face)
                    continue
                rotated = _rotate_points_about_axis(face, axis_origin, axis_dir, angle_deg)
                new_surface.append(rotated)
            deflected_group.append(new_surface)
        return deflected_group
    except Exception:
        return None


def _polyfaces_to_polydata(polys: List[List[Iterable[float]]]) -> Optional[pv.PolyData]:
    """
    Convert a list of polygon faces (each face is an ordered list of XYZ points)
    into a single PyVista PolyData with proper face indexing.

    For non-triangular faces, we rely on PyVista triangulation.
    """
    if pv is None or polys is None or len(polys) == 0:
        return None

    # Accumulate points and faces in PyVista's faces format: [n, id0, id1, ..., id(n-1), n, id0, ...]
    verts: List[Tuple[float, float, float]] = []
    faces: List[int] = []

    # We will not attempt to merge duplicate vertices here for simplicity.
    # This is acceptable for viewer-only use; triangulation will handle faces.
    for face in polys:
        if not face or len(face) < 3:
            continue
        n = len(face)
        start_idx = len(verts)
        for p in face:
            v = tuple(float(c) for c in p)
            if len(v) != 3:
                continue
            verts.append(v)
        faces.extend([n] + list(range(start_idx, start_idx + n)))

    if not verts or not faces:
        return None

    pts = np.array(verts, dtype=float)
    faces_arr = np.array(faces, dtype=np.int64)
    mesh = pv.PolyData(pts, faces_arr)
    try:
        mesh = mesh.triangulate()
    except Exception:
        # If triangulate fails, still return the mesh; PyVista may render polygons directly.
        pass
    return mesh


def _stack_meshes(meshes: List[pv.PolyData]) -> Optional[pv.PolyData]:
    """
    Combine multiple PolyData into one for simplicity.
    """
    if pv is None:
        return None
    meshes = [m for m in meshes if m is not None and not m.is_empty]
    if not meshes:
        return None
    try:
        return pv.merge(meshes)
    except Exception:
        # Fallback: append as a MultiBlock
        try:
            mb = pv.MultiBlock()
            for i, m in enumerate(meshes):
                mb[i] = m
            return mb.combine()
        except Exception:
            return meshes[0] if meshes else None


def _build_group_mesh(polys_group: List[List[List[Iterable[float]]]]) -> Optional[pv.PolyData]:
    """
    polys_group is a list of surfaces; each surface is a list of polygon faces.
    We flatten to a single list of faces and build one PolyData.
    """
    if not polys_group:
        return None
    flat_faces: List[List[Iterable[float]]] = []
    for surf in polys_group:
        if not surf:
            continue
        for face in surf:
            if face and len(face) >= 3:
                flat_faces.append(face)
    return _polyfaces_to_polydata(flat_faces)


def _build_cutter_wedge(processed, elevon_angle_deg: float) -> Optional[pv.PolyData]:
    """
    Build the deflection cutter wedge aligned with the selected hinge reference on the REAR face
    of rearSpar_outboard. Supports "Top of rear spar", "Bottom of rear spar", and "Centerline of rear spar".
    """
    if pv is None:
        return None
    try:
        import numpy as _np
        if elevon_angle_deg <= 0:
            return None
        if not processed or not getattr(processed, "elevon_surfaces", None):
            return None
        if not getattr(processed, "spar_surfaces", None) or not getattr(processed, "spar_uids", None):
            return None

        # Find rearSpar_outboard
        uids = list(processed.spar_uids or [])
        try:
            idx = uids.index("rearSpar_outboard")
        except ValueError:
            idx = next((i for i, u in enumerate(uids) if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower() and "out" in u.lower()), None)
            if idx is None:
                return None

        # Get faces of the rear spar segment
        # Processing may not guarantee order; identify FRONT vs REAR by X-centroid (FRONT has smaller X)
        spar_surfs = processed.spar_surfaces
        if idx >= len(spar_surfs):
            return None
        faces = spar_surfs[idx]
        if not faces or len(faces) < 2:
            return None
        fA, fB = faces[0], faces[1]
        if (not fA or len(fA) < 4) or (not fB or len(fB) < 4):
            return None

        import numpy as _np
        def _centroid_x(face):
            arr = _np.array(face, dtype=float)
            return float(_np.mean(arr[:, 0])) if arr.ndim == 2 and arr.shape[1] >= 1 else 0.0

        # Smaller X -> FRONT; larger X -> REAR
        if _centroid_x(fA) <= _centroid_x(fB):
            front_face, rear_face = fA, fB
        else:
            front_face, rear_face = fB, fA

        # Determine hinge mode from processed (propagated from GUI), fallback to "Top"
        hinge_mode_raw = getattr(processed, "cutter_hinge_mode", "Top of rear spar")
        hinge_mode = str(hinge_mode_raw or "").strip().lower()

        # Rear face vertices (index 1 guaranteed REAR here; we only use rear_face for hinge)
        p_bot_in = _np.array(rear_face[0], dtype=float)  # bottom-inboard (REAR)
        p_bot_out = _np.array(rear_face[1], dtype=float) # bottom-outboard (REAR)
        p_top_out = _np.array(rear_face[2], dtype=float) # top-outboard (REAR)
        p_top_in = _np.array(rear_face[3], dtype=float)  # top-inboard (REAR)

        # Select hinge anchor and height vectors strictly on REAR face to avoid FRONT-face alignment
        use_bottom = ("bottom" in hinge_mode) and ("rear" in hinge_mode)
        use_top = ("top" in hinge_mode) and ("rear" in hinge_mode)
        use_center = ("center" in hinge_mode) and ("rear" in hinge_mode)

        # Normalize REAR face vertex ordering to robustly identify bottom/top and inboard/outboard
        # Expected: [bottom-inboard, bottom-outboard, top-outboard, top-inboard]
        def _avg_z_pair(a, b): return 0.5 * (float(a[2]) + float(b[2]))
        def _abs_y_pair(a, b): return abs(0.5 * (float(a[1]) + float(b[1])))

        candidates = [
            (p_bot_in, p_bot_out, p_top_out, p_top_in),  # as provided
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

        # Optional: height scaling to preview a larger/smaller cutter while preserving hinge and angle
        try:
            height_scale = float(getattr(processed, "deflection_height_scale", 1.0) or 1.0)
        except Exception:
            height_scale = 1.0

        # After normalization, select hinge anchor and height vectors strictly on REAR face
        # Direct mapping to GUI labels
        if use_top:
            # Anchor on TOP edge; extend downward
            a0 = p_top_in; a1 = p_top_out                   # hinge: REAR top edge
            h_in_vec = (p_bot_in - p_top_in) * height_scale # down
            h_out_vec = (p_bot_out - p_top_out) * height_scale
        elif use_bottom:
            # Anchor on BOTTOM edge; extend upward
            a0 = p_bot_in; a1 = p_bot_out                   # hinge: REAR bottom edge
            h_in_vec = (p_top_in - p_bot_in) * height_scale # up
            h_out_vec = (p_top_out - p_bot_out) * height_scale
        elif use_center:
            # For centerline reference, place the triangular cutter on the TOP rear edge
            # and use downward height vectors; mirroring handled below to achieve a "|<" profile.
            a0 = p_top_in                                   # hinge: REAR top edge
            a1 = p_top_out
            h_in_vec = (p_bot_in - p_top_in) * height_scale # down
            h_out_vec = (p_bot_out - p_top_out) * height_scale
        else:
            # Fallback keep as top-edge behavior
            a0 = p_top_in; a1 = p_top_out
            h_in_vec = p_bot_in - p_top_in
            h_out_vec = p_bot_out - p_top_out

        # Compute aft normal from span × height; then apply two-stage correction
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

        # Stage 1: centroid-based forward/aft correction using FRONT vs REAR
        rear_center = (p_bot_in + p_bot_out + p_top_in + p_top_out) / 4.0
        f_bot_in = _np.array(front_face[0], dtype=float)
        f_bot_out = _np.array(front_face[1], dtype=float)
        f_top_out = _np.array(front_face[2], dtype=float)
        f_top_in = _np.array(front_face[3], dtype=float)
        front_center = (f_bot_in + f_bot_out + f_top_in + f_top_out) / 4.0
        forward_vec = front_center - rear_center
        if _np.linalg.norm(forward_vec) > 1e-12 and _np.dot(n, -forward_vec) < 0.0:
            n = -n

        # Stage 2: axis-aligned guard — force n to point toward increasing X (aft) if ambiguous
        if abs(forward_vec[0]) < 1e-9:
            # If forward_vec unreliable in X, use X-sign guard so wedge extrudes toward larger X (aft)
            if n[0] < 0:
                n = -n
        # For centerline mode, enforce aft direction explicitly (so the profile is "|<")
        if use_center and n[0] < 0:
            n = -n

        if use_center:
            # Centerline requirement (anchored-angle logic):
            # - Hinge lies along the REAR face centerline.
            # - Build two triangular wedges anchored on the centerline; push top/bottom far edges aft.
            mid_in = 0.5 * (p_bot_in + p_top_in)
            mid_out = 0.5 * (p_bot_out + p_top_out)

            tan_theta = _np.tan(_np.deg2rad(abs(elevon_angle_deg)))

            # Top half: centerline -> top edge
            a0_top = mid_in
            a1_top = mid_out
            h_in_top_vec = (p_top_in - mid_in) * height_scale
            h_out_top_vec = (p_top_out - mid_out) * height_scale
            h_in_top = _np.linalg.norm(h_in_top_vec)
            h_out_top = _np.linalg.norm(h_out_top_vec)
            b0_top = a0_top + h_in_top_vec + n * (h_in_top * tan_theta)
            b1_top = a1_top + h_out_top_vec + n * (h_out_top * tan_theta)

            # Bottom half: centerline -> bottom edge
            a0_bot = mid_in
            a1_bot = mid_out
            h_in_bot_vec = (p_bot_in - mid_in) * height_scale
            h_out_bot_vec = (p_bot_out - mid_out) * height_scale
            h_in_bot = _np.linalg.norm(h_in_bot_vec)
            h_out_bot = _np.linalg.norm(h_out_bot_vec)
            b0_bot = a0_bot + h_in_bot_vec + n * (h_in_bot * tan_theta)
            b1_bot = a1_bot + h_out_bot_vec + n * (h_out_bot * tan_theta)

            # Assemble faces
            faces_all = []
            in_face_top = [tuple(a0_top), tuple(a0_top + h_in_top_vec), tuple(b0_top)]
            out_face_top = [tuple(a1_top), tuple(b1_top), tuple(a1_top + h_out_top_vec)]
            slanted_face_top = [tuple(a0_top), tuple(a1_top), tuple(b1_top), tuple(b0_top)]
            hinge_face_top = [tuple(a0_top + h_in_top_vec), tuple(b0_top), tuple(b1_top), tuple(a1_top + h_out_top_vec)]
            faces_all.extend([in_face_top, out_face_top, slanted_face_top, hinge_face_top])

            in_face_bot = [tuple(a0_bot), tuple(a0_bot + h_in_bot_vec), tuple(b0_bot)]
            out_face_bot = [tuple(a1_bot), tuple(b1_bot), tuple(a1_bot + h_out_bot_vec)]
            slanted_face_bot = [tuple(a0_bot), tuple(a1_bot), tuple(b1_bot), tuple(b0_bot)]
            hinge_face_bot = [tuple(a0_bot + h_in_bot_vec), tuple(b0_bot), tuple(b1_bot), tuple(a1_bot + h_out_bot_vec)]
            faces_all.extend([in_face_bot, out_face_bot, slanted_face_bot, hinge_face_bot])

            back_face = [tuple(p_bot_in), tuple(p_top_in), tuple(p_top_out), tuple(p_bot_out)]
            faces_all.append(back_face)

            return _polyfaces_to_polydata(faces_all)

        # Default behavior (Top/Bottom modes): single wedge using the current height vectors
        # Anchor the ANGLED face at the selected hinge edge. Keep the hinge edge on the rear face
        # and shift the opposite edge aft by d = |h| * tan(angle).
        h_in = _np.linalg.norm(h_in_vec)
        h_out = _np.linalg.norm(h_out_vec)
        d_in = h_in * _np.tan(_np.deg2rad(abs(elevon_angle_deg)))
        d_out = h_out * _np.tan(_np.deg2rad(abs(elevon_angle_deg)))

        # Offset applied at far (non-hinge) edge
        b0 = a0 + h_in_vec + n * d_in
        b1 = a1 + h_out_vec + n * d_out

        # Build faces relative to chosen hinge on REAR face
        in_face = [tuple(a0), tuple(a0 + h_in_vec), tuple(b0)]
        out_face = [tuple(a1), tuple(b1), tuple(a1 + h_out_vec)]
        # Now the face through the hinge is the angled face
        slanted_face = [tuple(a0), tuple(a1), tuple(b1), tuple(b0)]
        # And the far edge face is the square face between far-edge and its offset
        hinge_face = [tuple(a0 + h_in_vec), tuple(b0), tuple(b1), tuple(a1 + h_out_vec)]
        back_face = [tuple(p_bot_in), tuple(p_top_in), tuple(p_top_out), tuple(p_bot_out)]

        return _polyfaces_to_polydata([in_face, out_face, hinge_face, slanted_face, back_face])
    except Exception:
        return None


def _build_wing_mesh(wing_quads: List[List[Iterable[float]]],
                     dihedraled_rib_profiles: Optional[List[np.ndarray]] = None) -> Optional[pv.PolyData]:
    """
    Build wing mesh for preview.

    Parity-first behavior:
    - If dihedraled_rib_profiles are available (>=2), loft a skin by connecting consecutive
      rib profiles using proportional index mapping (mirrors STEP export loft_wing_from_ribs).
    - Otherwise, fall back to explicit wing_quads provided by the processor.
    """
    if pv is None:
        return None

    # Helper to connect two profiles into quads
    def _connect_profiles(a: np.ndarray, b: np.ndarray) -> List[List[np.ndarray]]:
        quads: List[List[np.ndarray]] = []
        if a is None or b is None:
            return quads
        if a.ndim != 2 or b.ndim != 2:
            return quads
        na, nb = a.shape[0], b.shape[0]
        if na < 3 or nb < 3:
            return quads
        n = min(na, nb)
        ia = (np.linspace(0, na - 1, n)).astype(int)
        ib = (np.linspace(0, nb - 1, n)).astype(int)
        for i in range(n - 1):
            p00 = a[ia[i]]
            p01 = a[ia[i + 1]]
            p10 = b[ib[i]]
            p11 = b[ib[i + 1]]
            quads.append([p00, p10, p11, p01])
        return quads

    # Prefer loft from ribs for parity with STEP export
    if dihedraled_rib_profiles and len(dihedraled_rib_profiles) >= 2:
        profs = [np.array(p, dtype=float) for p in dihedraled_rib_profiles if p is not None and len(p) >= 3]
        if len(profs) >= 2:
            # Ensure inboard->outboard ordering by mean |Y|
            profs = sorted(profs, key=lambda p: float(np.mean(np.abs(p[:, 1]))))
            loft_quads: List[List[np.ndarray]] = []
            for i in range(len(profs) - 1):
                loft_quads.extend(_connect_profiles(profs[i], profs[i + 1]))
            wing_quads = loft_quads

    if not wing_quads:
        return None

    faces: List[List[Iterable[float]]] = []
    for quad in wing_quads:
        if not quad or len(quad) < 3:
            continue
        faces.append([tuple(float(c) for c in p) for p in quad])
    return _polyfaces_to_polydata(faces)


def open_pyvista_window(processed: ProcessedCpacs, include_elevon: bool = True) -> bool:
    """
    Open a PyVista interactive window showing spars, ribs, and optionally elevon.
    Returns True if a window was launched successfully.

    Notes:
    - BackgroundPlotter (Qt) can conflict with Tk/versions of PyVista/PyVistaQt. If it raises,
      fall back to a standard pv.Plotter(show=True) in a standalone window.
    """
    if processed is None or not getattr(processed, "success", False):
        return False
    if pv is None:
        print("PyVista not available. Install with: pip install pyvista vtk")
        return False

    # Build meshes up-front so both code paths can reuse them
    wing_mesh = _build_wing_mesh(processed.wing_vertices)  # semi-transparent skin
    spars_mesh = _build_group_mesh(processed.spar_surfaces)
    ribs_mesh = _build_group_mesh(processed.rib_surfaces)

    # Reuse GUI elevon angle and hinge mode if present on processed; else defaults
    elevon_angle_deg = float(getattr(processed, "elevon_angle_deg", 0.0) or 0.0)
    hinge_mode_raw = getattr(processed, "cutter_hinge_mode", "")
    hinge_mode_norm = str(hinge_mode_raw or "").strip().lower()

    # Elevon: optionally deflect about selected hinge of rear spar
    elevon_mesh = None
    if include_elevon and getattr(processed, "elevon_surfaces", None):
        deflected = None
        if abs(elevon_angle_deg) > 1e-9:
            try:
                if ("bottom" in hinge_mode_norm) and ("rear" in hinge_mode_norm):
                    deflected = _deflect_elevon_bottom_hinge(processed, elevon_angle_deg)
                elif ("top" in hinge_mode_norm) and ("rear" in hinge_mode_norm):
                    deflected = _deflect_elevon_top_hinge(processed, elevon_angle_deg)
                elif ("center" in hinge_mode_norm) and ("rear" in hinge_mode_norm):
                    deflected = _deflect_elevon_centerline_hinge(processed, elevon_angle_deg)
            except Exception:
                deflected = None
        if deflected:
            elevon_mesh = _build_group_mesh(deflected)
        if elevon_mesh is None:
            elevon_mesh = _build_group_mesh(processed.elevon_surfaces)

    cutter_mesh = _build_cutter_wedge(processed, elevon_angle_deg)

    # Helper to populate a plotter consistently
    def _populate(plotter_obj):
        plotter_obj.add_axes()
        try:
            plotter_obj.show_bounds(grid="front", location="outer")
        except Exception:
            pass

        # Wing skin first (semi-transparent), then internal structures
        if wing_mesh is not None:
            plotter_obj.add_mesh(wing_mesh, color="cyan", smooth_shading=True, name="wing", opacity=0.35)

        if spars_mesh is not None:
            plotter_obj.add_mesh(spars_mesh, color="steelblue", smooth_shading=True, name="spars", opacity=0.95)
        if ribs_mesh is not None:
            plotter_obj.add_mesh(ribs_mesh, color="lightgray", smooth_shading=True, name="ribs", opacity=0.95)
        if include_elevon and elevon_mesh is not None:
            plotter_obj.add_mesh(elevon_mesh, color="orange", smooth_shading=True, name="elevon", opacity=0.95)

        # Deflection cutter wedge (visualization)
        if cutter_mesh is not None:
            plotter_obj.add_mesh(cutter_mesh, color="magenta", smooth_shading=True, name="cutter", opacity=0.7)

        # Mirror across Y=0 (apply same opacity behavior to both original and mirrored)
        try:
            def mirror_y(m: pv.PolyData) -> Optional[pv.PolyData]:
                if m is None or m.is_empty:
                    return None
                pts = m.points.copy()
                pts[:, 1] = -pts[:, 1]
                mirrored = pv.PolyData(pts, m.faces.copy())
                return mirrored.triangulate() if not mirrored.is_empty else mirrored

            if wing_mesh is not None:
                wm = mirror_y(wing_mesh)
                if wm is not None:
                    plotter_obj.add_mesh(wm, color="cyan", smooth_shading=True, opacity=0.35, name="wing_mirror")

            if spars_mesh is not None:
                ms = mirror_y(spars_mesh)
                if ms is not None:
                    plotter_obj.add_mesh(ms, color="steelblue", smooth_shading=True, opacity=0.95, name="spars_mirror")
            if ribs_mesh is not None:
                mr = mirror_y(ribs_mesh)
                if mr is not None:
                    plotter_obj.add_mesh(mr, color="lightgray", smooth_shading=True, opacity=0.95, name="ribs_mirror")
            if include_elevon and elevon_mesh is not None:
                me = mirror_y(elevon_mesh)
                if me is not None:
                    plotter_obj.add_mesh(me, color="orange", smooth_shading=True, opacity=0.95, name="elevon_mirror")

            if cutter_mesh is not None:
                cm = mirror_y(cutter_mesh)
                if cm is not None:
                    plotter_obj.add_mesh(cm, color="magenta", smooth_shading=True, opacity=0.7, name="cutter_mirror")
        except Exception:
            pass

        try:
            plotter_obj.camera_position = "iso"
        except Exception:
            pass

    # Preferred: BackgroundPlotter if available and compatible
    if BackgroundPlotter is not None:
        try:
            bp = BackgroundPlotter(title="PyVista Wing Preview", auto_close=False)
            _populate(bp)
            return True
        except Exception:
            # Fallback below
            pass

    # Fallback: standard Plotter in a new window; blocks until closed.
    try:
        pl = pv.Plotter(title="PyVista Wing Preview (fallback)")
        _populate(pl)
        pl.show()  # show() returns when the window is closed
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False
