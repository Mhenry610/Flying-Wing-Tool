"""
3D visualization and CPACS processing core (extracted as pure functions from CpacsStepperTab).
- No direct GUI dependencies. All UI interactions are handled by the caller.
- Numerical behavior mirrors the monolith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import xml.etree.ElementTree as ET


# ---------------------------- Data Structures ----------------------------

@dataclass
class ProcessedCpacs:
    success: bool
    wing_vertices: List[List[np.ndarray]]
    spar_surfaces: List[List[List[np.ndarray]]]
    rib_surfaces: List[List[List[np.ndarray]]]
    elevon_surfaces: List[List[List[np.ndarray]]]
    cutter_surfaces: List[List[np.ndarray]]
    spar_uids: List[str]
    spar_etas: List[float]
    initial_spar_thicknesses: Dict[str, float]
    initial_rib_thickness: float
    dihedraled_rib_profiles: List[np.ndarray]
    flat_elements: Dict[str, np.ndarray]
    # New: average CPACS xsi per spar segment for front-spar determination parity
    spar_avg_xsi: Dict[str, float]


# ---------------------------- Public API ----------------------------

def _rotate_points_about_axis_mat(points: List[np.ndarray],
                                  origin: np.ndarray,
                                  axis_dir: np.ndarray,
                                  angle_deg: float) -> List[np.ndarray]:
    """
    Rodrigues rotation for a list of numpy 3D points around an axis.
    Returns new list of rotated np.ndarray points.
    """
    if points is None or len(points) == 0:
        return []
    o = np.array(origin, dtype=float).reshape(3)
    k = np.array(axis_dir, dtype=float).reshape(3)
    nrm = np.linalg.norm(k)
    if nrm == 0.0 or abs(angle_deg) < 1e-12:
        return [p.copy() for p in points]
    k = k / nrm
    theta = np.deg2rad(float(angle_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out: List[np.ndarray] = []
    for p in points:
        v = np.array(p, dtype=float) - o
        kv = np.dot(k, v)
        k_cross_v = np.cross(k, v)
        v_rot = v * cos_t + k_cross_v * sin_t + k * kv * (1.0 - cos_t)
        out.append(v_rot + o)
    return out


def _deflect_elevon_bottom_hinge_mat(processed: ProcessedCpacs, angle_deg: float) -> Optional[List[List[List[np.ndarray]]]]:
    """
    Create a deflected elevon polys structure for matplotlib preview by rotating
    elevon_surfaces about the bottom edge of the REAR face of rearSpar_outboard.
    Mirrors the hinge logic used in PyVista.
    """
    try:
        if processed is None or not getattr(processed, "elevon_surfaces", None):
            return None
        if abs(angle_deg) < 1e-12:
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
        if not rear_face or len(rear_face) < 2:
            return None
        p0 = np.array(rear_face[0], dtype=float)
        p1 = np.array(rear_face[1], dtype=float)
        axis_origin = p0
        axis_dir = p1 - p0
        if np.linalg.norm(axis_dir) == 0.0:
            return None

        deflected: List[List[List[np.ndarray]]] = []
        for surface in processed.elevon_surfaces:
            new_surface: List[List[np.ndarray]] = []
            for face in surface:
                rotated_face = _rotate_points_about_axis_mat(list(face), axis_origin, axis_dir, angle_deg)
                new_surface.append([np.array(p, dtype=float) for p in rotated_face])
            deflected.append(new_surface)
        return deflected
    except Exception:
        return None


def process_cpacs_data(xml_string: bytes | str,
                       spar_offsets: Dict[str, float] | None = None) -> ProcessedCpacs:
    """
    Pure function equivalent of CpacsStepperTab._process_cpacs_data.
    Inputs:
      - xml_string: CPACS XML as bytes/str
      - spar_offsets: optional per-spar UID offset (m)
    Outputs a ProcessedCpacs instance with all geometry collections.
    """
    if spar_offsets is None:
        spar_offsets = {}

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return ProcessedCpacs(False, [], [], [], [], [], [], [], {}, 0.0, [], {})

    # Helpers from core
    from core.parsing import parse_airfoil_points  
    from core.geom import get_transformation_matrix, apply_transformation, get_structural_points, get_thickness  

    transformed_elements: Dict[str, np.ndarray] = {}
    flat_elements: Dict[str, np.ndarray] = {}
    wing_vertices: List[List[np.ndarray]] = []
    spar_surfaces: List[List[List[np.ndarray]]] = []
    rib_surfaces: List[List[List[np.ndarray]]] = []
    elevon_surfaces: List[List[List[np.ndarray]]] = []
    cutter_surfaces: List[List[np.ndarray]] = []
    spar_uids: List[str] = []
    spar_etas: List[float] = []
    initial_spar_thicknesses: Dict[str, float] = {}
    dihedraled_rib_profiles: List[np.ndarray] = []
    spar_avg_xsi: Dict[str, float] = {}

    # Wing dihedral
    wing_element = root.find('.//wing[@uID="mainWing"]')
    dihedral_deg = 0.0
    if wing_element is not None:
        rot_x_elem = wing_element.find('transformation/rotation/x')
        if rot_x_elem is not None and rot_x_elem.text:
            dihedral_deg = float(rot_x_elem.text)
    tan_dihedral = np.tan(np.deg2rad(dihedral_deg))

    # Airfoils and element-section mapping
    airfoils = {
        e.get('uID'): parse_airfoil_points(e.find('pointList'))
        for e in root.findall('.//wingAirfoil') if e.find('pointList') is not None
    }
    elem_to_sec_map = {
        elem.get('uID'): sec
        for sec in root.findall('.//section') for elem in sec.findall('.//element')
    }

    # Transform elements
    for elem in root.findall('.//element'):
        elem_uid = elem.get('uID')
        section = elem_to_sec_map.get(elem_uid)
        if not section:
            continue
        base_points = airfoils.get(elem.find('airfoilUID').text)
        if base_points is None:
            continue

        section_transform = get_transformation_matrix(section.find('transformation'))
        element_transform = get_transformation_matrix(elem.find('transformation'))

        flat_trans = section_transform @ element_transform
        flat_pts = apply_transformation(base_points, flat_trans)
        flat_elements[elem_uid] = flat_pts

        dihedraled_points = flat_pts.copy()
        dihedraled_points[:, 2] += dihedraled_points[:, 1] * tan_dihedral
        transformed_elements[elem_uid] = dihedraled_points

    # Wing skin as a multi-section loft using CPACS element airfoils positioned in 3D,
    # HARD-constrained at rear spar top/bottom rails per span station.
    segments = root.findall('.//segment')
    wing_segments_uids: List[str] = []
    if segments:
        # Build forward mapping from->to for the main element chain
        seg_map = {}
        for s in segments:
            fe = s.find('fromElementUID')
            te = s.find('toElementUID')
            if fe is None or te is None or fe.text is None or te.text is None:
                continue
            seg_map[fe.text] = te.text

        # Identify starting element(s) (those that never appear as a 'to')
        start_uid_list = list(set(seg_map.keys()) - set(seg_map.values()))
        if start_uid_list:
            current_uid = start_uid_list[0]
            wing_segments_uids.append(current_uid)

            # Collect ordered list of section polylines (already dihedraled) for loft
            section_order: List[np.ndarray] = []
            if current_uid in transformed_elements:
                section_order.append(transformed_elements[current_uid])
            visited = set()
            while current_uid in seg_map and current_uid not in visited:
                visited.add(current_uid)
                next_uid = seg_map[current_uid]
                if next_uid in transformed_elements:
                    section_order.append(transformed_elements[next_uid])
                wing_segments_uids.append(next_uid)
                current_uid = next_uid

            # Build per-section rear spar top/bottom inboard/outboard anchors.
            # We will try to extract anchors matching the number of sections in section_order.
            # If spar_surfaces packs all faces per spar (not per-section), we fallback to heuristics per section.
            per_section_top = []
            per_section_bot = []

            def _extract_rear_face_edges(face: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                rf = np.array(face, dtype=float)
                idx_sorted = np.argsort(rf[:, 2])
                bot_pair = rf[idx_sorted[:2]]
                top_pair = rf[idx_sorted[-2:]]
                bot_in = bot_pair[np.argmin(np.abs(bot_pair[:, 1]))]
                bot_out = bot_pair[1 - np.argmin(np.abs(bot_pair[:, 1]))]
                top_in = top_pair[np.argmin(np.abs(top_pair[:, 1]))]
                top_out = top_pair[1 - np.argmin(np.abs(top_pair[:, 1]))]
                return top_in, top_out, bot_in, bot_out

            # Find rear spar entry
            rear_idx = None
            try:
                rear_idx = (spar_uids or []).index("rearSpar_outboard")
            except Exception:
                for i, u in enumerate(spar_uids or []):
                    if isinstance(u, str) and "rear" in u.lower() and "spar" in u.lower():
                        rear_idx = i
                        break

            if rear_idx is not None and rear_idx < len(spar_surfaces):
                faces = spar_surfaces[rear_idx]  # list of quad faces
                # If we have at least two faces, pick the two with largest X centroid as rear wall candidates
                if faces:
                    # group by approximate span station using |Y| average to create anchors along span
                    spans: Dict[int, List[List[np.ndarray]]] = {}
                    for f in faces:
                        arr = np.array(f, dtype=float)
                        key = int(round(np.mean(np.abs(arr[:, 1])) * 1000))  # bucket by Y
                        spans.setdefault(key, []).append(f)
                    # Order by increasing span (inboard -> outboard)
                    for _, group in sorted(spans.items(), key=lambda kv: kv[0]):
                        # pick rear-ish face in group by max X centroid
                        if not group:
                            continue
                        centroids_x = [float(np.mean(np.array(g, dtype=float)[:, 0])) for g in group]
                        idx = int(np.argmax(centroids_x))
                        rear_face = group[idx]
                        t_in, t_out, b_in, b_out = _extract_rear_face_edges(rear_face)
                        per_section_top.append((t_in, t_out))
                        per_section_bot.append((b_in, b_out))

            # Fallback: if we could not derive anchors, synthesize from sections' Z-extrema as a very weak proxy
            if len(per_section_top) == 0 or len(per_section_bot) == 0:
                for sec in section_order:
                    # choose inboard/outboard by min/max |Y| and use local Z-extrema among a band near the airfoil upper/lower
                    idx_in = int(np.argmin(np.abs(sec[:, 1])))
                    idx_out = int(np.argmax(np.abs(sec[:, 1])))
                    # Upper approx: local max Z within small window around idx_in/out
                    zmax_in = float(sec[idx_in][2])
                    zmax_out = float(sec[idx_out][2])
                    zmin_in = float(sec[idx_in][2])
                    zmin_out = float(sec[idx_out][2])
                    t_in = sec[idx_in].copy(); t_in[2] = zmax_in
                    t_out = sec[idx_out].copy(); t_out[2] = zmax_out
                    b_in = sec[idx_in].copy(); b_in[2] = zmin_in
                    b_out = sec[idx_out].copy(); b_out[2] = zmin_out
                    per_section_top.append((t_in, t_out))
                    per_section_bot.append((b_in, b_out))

            # Ensure per_section arrays length matches section_order by interpolation/extrapolation if needed
            def _ensure_len(arr: List[Tuple[np.ndarray, np.ndarray]], n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
                if len(arr) == n:
                    return arr
                if len(arr) == 0:
                    return [(section_order[i][0], section_order[i][-1]) for i in range(n)]
                # simple resample by linear interpolation over index
                xs = np.linspace(0, len(arr) - 1, n)
                res = []
                for x in xs:
                    i0 = int(np.floor(x)); i1 = min(len(arr) - 1, i0 + 1)
                    t = float(x - i0)
                    p0_in, p0_out = arr[i0]
                    p1_in, p1_out = arr[i1]
                    pin = (1 - t) * np.array(p0_in) + t * np.array(p1_in)
                    pout = (1 - t) * np.array(p0_out) + t * np.array(p1_out)
                    res.append((pin, pout))
                return res

            per_section_top = _ensure_len(per_section_top, len(section_order))
            per_section_bot = _ensure_len(per_section_bot, len(section_order))

            # Connector that pins Z exactly at rail anchors for inboard/outboard sample indices per section,
            # and blends linearly between them along span.
            def _connect_sections_pinned(a: np.ndarray, b: np.ndarray, topA, botA, topB, botB) -> List[List[np.ndarray]]:
                quads: List[List[np.ndarray]] = []
                if a is None or b is None:
                    return quads
                na = int(a.shape[0]) if a.ndim == 2 else 0
                nb = int(b.shape[0]) if b.ndim == 2 else 0
                if na < 3 or nb < 3:
                    return quads
                n = min(na, nb)
                ia = (np.linspace(0, na - 1, n)).astype(int)
                ib = (np.linspace(0, nb - 1, n)).astype(int)

                # Find inboard/outboard indices by |Y|
                a_in_idx = int(np.argmin(np.abs(a[:, 1])))
                a_out_idx = int(np.argmax(np.abs(a[:, 1])))
                b_in_idx = int(np.argmin(np.abs(b[:, 1])))
                b_out_idx = int(np.argmax(np.abs(b[:, 1])))

                # Build linear rails between A and B anchors
                top_in_line = (np.array(topA[0], dtype=float), np.array(topB[0], dtype=float))
                top_out_line = (np.array(topA[1], dtype=float), np.array(topB[1], dtype=float))
                bot_in_line = (np.array(botA[0], dtype=float), np.array(botB[0], dtype=float))
                bot_out_line = (np.array(botA[1], dtype=float), np.array(botB[1], dtype=float))

                # Helper to get Z target for a point p based on its closeness to top/bottom rail at its local span t
                def _span_t(y, y0, y1):
                    denom = max(1e-9, y1 - y0)
                    return float(np.clip((y - y0) / denom, 0.0, 1.0))

                # Span bounds
                y0 = float(min(abs(a[a_in_idx, 1]), abs(b[b_in_idx, 1])))
                y1 = float(max(abs(a[a_out_idx, 1]), abs(b[b_out_idx, 1])))

                def _interp(line, t):
                    return (1 - t) * line[0] + t * line[1]

                def _target_z(p: np.ndarray) -> float:
                    t = _span_t(abs(p[1]), y0, y1)
                    top_in = _interp(top_in_line, t)
                    top_out = _interp(top_out_line, t)
                    bot_in = _interp(bot_in_line, t)
                    bot_out = _interp(bot_out_line, t)
                    # Interpolate along in->out to get rail Z at this span position
                    # Choose closer in/out by |Y|
                    if abs(p[1]) - y0 <= y1 - abs(p[1]):
                        # closer to inboard: mix between top_in/bot_in
                        tz = float(top_in[2]); bz = float(bot_in[2])
                    else:
                        tz = float(top_out[2]); bz = float(bot_out[2])
                    # Choose nearer of top/bottom
                    pz = float(p[2])
                    return tz if abs(pz - tz) <= abs(pz - bz) else bz

                # Build quads; pin exact Z at inboard/outboard indices
                for i in range(n - 1):
                    p00 = a[ia[i]].copy()
                    p01 = a[ia[i + 1]].copy()
                    p10 = b[ib[i]].copy()
                    p11 = b[ib[i + 1]].copy()

                    # Pin exact at inboard/outboard samples
                    for ref_idx, ref in ((a_in_idx, 'a0'), (a_out_idx, 'a1')):
                        if ia[i] == ref_idx:
                            p00[2] = _target_z(p00)
                        if ia[i + 1] == ref_idx:
                            p01[2] = _target_z(p01)
                    for ref_idx, ref in ((b_in_idx, 'b0'), (b_out_idx, 'b1')):
                        if ib[i] == ref_idx:
                            p10[2] = _target_z(p10)
                        if ib[i + 1] == ref_idx:
                            p11[2] = _target_z(p11)

                    # For non-pinned samples, blend 70% toward target Z
                    if ia[i] != a_in_idx and ia[i] != a_out_idx:
                        tz = _target_z(p00); p00[2] = p00[2] + 0.7 * (tz - p00[2])
                    if ia[i + 1] != a_in_idx and ia[i + 1] != a_out_idx:
                        tz = _target_z(p01); p01[2] = p01[2] + 0.7 * (tz - p01[2])
                    if ib[i] != b_in_idx and ib[i] != b_out_idx:
                        tz = _target_z(p10); p10[2] = p10[2] + 0.7 * (tz - p10[2])
                    if ib[i + 1] != b_in_idx and ib[i + 1] != b_out_idx:
                        tz = _target_z(p11); p11[2] = p11[2] + 0.7 * (tz - p11[2])

                    quads.append([p00, p10, p11, p01])
                return quads

            # Build wing_vertices from pinned connections with per-section guides
            for i in range(len(section_order) - 1):
                a = section_order[i]
                b = section_order[i + 1]
                topA = per_section_top[i]
                topB = per_section_top[i + 1]
                botA = per_section_bot[i]
                botB = per_section_bot[i + 1]
                wing_vertices.extend(_connect_sections_pinned(a, b, topA, botA, topB, botB))

    # Structures
    comp_seg = root.find('.//componentSegment')
    first_rib_thickness_found: Optional[float] = None
    if comp_seg is not None:
        from_pts = flat_elements.get(comp_seg.find('fromElementUID').text)
        to_pts = flat_elements.get(comp_seg.find('toElementUID').text)

        spar_positions = root.findall('.//sparPosition')
        if spar_positions:
            spar_etas = sorted(list(set(float(p.find('.//eta').text) for p in spar_positions if p.find('.//eta') is not None)))

        if from_pts is not None and to_pts is not None:
            spar_pos_map = {p.get('uID'): (float(p.find('.//eta').text), float(p.find('.//xsi').text))
                            for p in root.findall('.//sparPosition')}
            spar_avg_xsi: Dict[str, float] = {}

            # Spars
            for spar_seg in root.findall('.//sparSegment'):
                spar_uid = spar_seg.get('uID')
                spar_uids.append(spar_uid)
                initial_spar_thicknesses[spar_uid] = get_thickness(spar_seg)
                thickness = get_thickness(spar_seg)  # read from XML
                spar_offset = float(spar_offsets.get(spar_uid, 0.0))
                pos_uids = [p.text for p in spar_seg.find('sparPositionUIDs')]
                # Compute average CPACS xsi for this spar_uid for front-most determination
                if len(pos_uids) == 2:
                    xsi1 = spar_pos_map.get(pos_uids[0], (0.0, 0.0))[1]
                    xsi2 = spar_pos_map.get(pos_uids[1], (0.0, 0.0))[1]
                    spar_avg_xsi[spar_uid] = (xsi1 + xsi2) * 0.5
                if len(pos_uids) == 2:
                    p1_upper, p1_lower = get_structural_points(*spar_pos_map[pos_uids[0]], from_pts, to_pts, spar_offset)
                    p2_upper, p2_lower = get_structural_points(*spar_pos_map[pos_uids[1]], from_pts, to_pts, spar_offset)
                    center_face = [p1_lower, p2_lower, p2_upper, p1_upper]

                    v_span = p2_lower - p1_lower
                    v_height = p1_upper - p1_lower
                    normal = np.cross(v_span, v_height)
                    if np.linalg.norm(normal) > 1e-9:
                        normal /= np.linalg.norm(normal)
                    else:
                        normal = np.array([0, 1, 0])
                    offset_vec = normal * (thickness / 2.0)
                    front_face = [p + offset_vec for p in center_face]
                    back_face = [p - offset_vec for p in center_face]
                    polys = [front_face, back_face]
                    for i in range(4):
                        polys.append([front_face[i], front_face[(i + 1) % 4], back_face[(i + 1) % 4], back_face[i]])

                    translated_polys = [[p + np.array([0, 0, p[1] * tan_dihedral]) for p in poly] for poly in polys]
                    spar_surfaces.append(translated_polys)

            # Ribs (support both legacy ribPositioning/etaCoordinates and CPACS 3.5 ribExplicitPositioning)
            for rib_def in root.findall('.//ribsDefinition'):
                if first_rib_thickness_found is None:
                    first_rib_thickness_found = get_thickness(rib_def)
                thickness = get_thickness(rib_def)
                half_thickness = thickness / 2.0

                etas_to_process: List[float] = []

                # Legacy: explicit etas under ribPositioning/etaCoordinates
                eta_elements = rib_def.findall('.//etaCoordinates/eta')
                if eta_elements:
                    for eta_elem in eta_elements:
                        if eta_elem is None or not eta_elem.text:
                            continue
                        try:
                            etas_to_process.append(float(eta_elem.text))
                        except Exception:
                            continue
                else:
                    # CPACS 3.5: ribExplicitPositioning with start/end EtaXsi points (same eta for a single rib)
                    rep = rib_def.find('ribExplicitPositioning')
                    if rep is not None:
                        local_etas = set()
                        se = rep.find('startEtaXsiPoint/eta')
                        ee = rep.find('endEtaXsiPoint/eta')
                        try:
                            if se is not None and se.text:
                                local_etas.add(float(se.text))
                        except Exception:
                            pass
                        try:
                            if ee is not None and ee.text:
                                local_etas.add(float(ee.text))
                        except Exception:
                            pass
                        etas_to_process = sorted(local_etas)

                for eta in etas_to_process:
                    center_face = (1 - eta) * from_pts + eta * to_pts
                    front_face = center_face + np.array([0, half_thickness, 0])
                    back_face = center_face - np.array([0, half_thickness, 0])

                    polys = [front_face, back_face]
                    num_verts = len(front_face)
                    for i in range(num_verts):
                        polys.append([front_face[i], front_face[(i + 1) % num_verts],
                                      back_face[(i + 1) % num_verts], back_face[i]])

                    # VERTICAL RIBS: Do NOT apply dihedral rotation to rib surfaces
                    # Physical model aircraft ribs are built vertical (aligned with global Z-axis)
                    # for ease of laser cutting and assembly on flat building boards.
                    # The rib stays in the vertical XZ plane at its spanwise Y position.
                    # Only the wing skin and spars follow the dihedral angle.
                    vertical_polys = polys  # Keep ribs vertical - no dihedral transformation
                    rib_surfaces.append(vertical_polys)

                    # Store VERTICAL rib profile (not dihedraled)
                    # This matches the physical reality of laser-cut ribs
                    vertical_rib_profile = center_face.copy()
                    # Z coordinates stay as-is (no dihedral tilt applied)
                    dihedraled_rib_profiles.append(vertical_rib_profile)
 
            # Elevon (if rearSpar_outboard exists)
            rear_spar_outboard = root.find(".//sparSegment[@uID='rearSpar_outboard']")
            if rear_spar_outboard is not None:
                pos_uids = [p.text for p in rear_spar_outboard.find('sparPositionUIDs')]
                if len(pos_uids) == 2:
                    (eta_in_le, xsi_in_le) = spar_pos_map[pos_uids[0]]
                    (eta_out_le, xsi_out_le) = spar_pos_map[pos_uids[1]]
                    p_in_le_up, p_in_le_low = get_structural_points(eta_in_le, xsi_in_le, from_pts, to_pts)
                    p_in_te_up, p_in_te_low = get_structural_points(eta_in_le, 1.0, from_pts, to_pts)
                    p_out_le_up, p_out_le_low = get_structural_points(eta_out_le, xsi_out_le, from_pts, to_pts)
                    p_out_te_up, p_out_te_low = get_structural_points(eta_out_le, 1.0, from_pts, to_pts)
 
                    spar_thickness = get_thickness(rear_spar_outboard)
                    normal = np.cross(p_out_le_up - p_in_le_up, p_in_le_up - p_in_le_low)
                    if np.linalg.norm(normal) > 1e-9:
                        normal /= np.linalg.norm(normal)
                    offset_vec = normal * (spar_thickness / 2.0)
                    p_in_le_up += offset_vec; p_in_le_low += offset_vec
                    p_out_le_up += offset_vec; p_out_le_low += offset_vec
 
                    top = [p_in_le_up, p_out_le_up, p_out_te_up, p_in_te_up]
                    bottom = [p_in_le_low, p_in_te_low, p_out_te_low, p_out_le_low]
                    front = [p_in_le_low, p_out_le_low, p_out_le_up, p_in_le_up]
                    back = [p_in_te_low, p_out_te_low, p_out_te_up, p_in_te_up]
                    inner = [p_in_le_low, p_in_te_low, p_in_te_up, p_in_le_up]
                    outer = [p_out_le_low, p_out_te_low, p_out_te_up, p_out_le_up]
 
                    flat_elevon_polys = [top, bottom, front, back, inner, outer]
                    translated_elevon_polys = [[p + np.array([0, 0, p[1] * tan_dihedral]) for p in poly]
                                               for poly in flat_elevon_polys]
                    elevon_surfaces.append(translated_elevon_polys)
 
    return ProcessedCpacs(
        success=True,
        wing_vertices=wing_vertices,
        spar_surfaces=spar_surfaces,
        rib_surfaces=rib_surfaces,
        elevon_surfaces=elevon_surfaces,
        cutter_surfaces=cutter_surfaces,
        spar_uids=spar_uids,
        spar_etas=spar_etas,
        initial_spar_thicknesses=initial_spar_thicknesses,
        initial_rib_thickness=first_rib_thickness_found or 0.00635,
        dihedraled_rib_profiles=dihedraled_rib_profiles,
        flat_elements=flat_elements,
        spar_avg_xsi=locals().get('spar_avg_xsi', {})
    )


def apply_overrides_in_xml(cpacs_root: ET.Element,
                           rib_thickness_override: Optional[float],
                           spar_thickness_overrides: Dict[str, float]) -> ET.Element:
    """
    Mutates the provided cpacs_root to apply thickness overrides, mirroring CpacsStepperTab.apply_overrides.
    Returns the same root for chaining.
    """
    if rib_thickness_override is not None:
        for rib_def in cpacs_root.findall('.//ribsDefinition'):
            thickness_elem = rib_def.find('.//ribCrossSection/material/thickness')
            if thickness_elem is not None:
                thickness_elem.text = str(rib_thickness_override)

    for spar_seg in cpacs_root.findall('.//sparSegment'):
        spar_uid = spar_seg.get('uID')
        if spar_uid in spar_thickness_overrides:
            thickness_elem = spar_seg.find('.//sparCrossSection/web1/material/thickness')
            if thickness_elem is not None:
                thickness_elem.text = str(spar_thickness_overrides[spar_uid])

    return cpacs_root


# ---------------------------- Extras to match monolith utilities ----------------------------

def get_matplotlib_elevon_surfaces(processed: ProcessedCpacs,
                                   hinge_mode: Optional[str],
                                   angle_deg: float) -> List[List[List[np.ndarray]]]:
    """
    Public utility for matplotlib preview to retrieve the appropriate elevon polys:
    - Returns deflected polys when hinge_mode indicates bottom-rear-spar and angle is nonzero.
    - Otherwise returns original processed.elevon_surfaces.
    """
    return _maybe_deflected_elevon_for_matplotlib(processed, hinge_mode, angle_deg)

def _maybe_deflected_elevon_for_matplotlib(processed: ProcessedCpacs,
                                            hinge_mode: Optional[str],
                                            angle_deg: float) -> List[List[List[np.ndarray]]]:
    """
    Utility for matplotlib preview to choose between original or deflected elevon polys.
    """
    use_deflect = bool(hinge_mode) and hinge_mode.strip().lower() in ("bottom of rear spar", "bottom", "bottom hinge", "bottom rear spar")
    if use_deflect and abs(angle_deg) > 1e-9:
        poly = _deflect_elevon_bottom_hinge_mat(processed, angle_deg)
        if poly:
            return poly
    return processed.elevon_surfaces
 
 
def _flatten_profile_vertical(
    center_face: np.ndarray, 
    thickness: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten a 3D airfoil profile to a VERTICAL plane (XZ).
    
    Unlike applying dihedral rotation which tilts ribs with the wing,
    this function keeps ribs strictly vertical as built in reality.
    Physical model aircraft ribs are almost always vertical for:
    - Ease of laser cutting (2D profiles)
    - Assembly on flat building boards
    - Simpler jig construction
    - Standard model aircraft practice
    
    The rib profile is projected onto a plane defined by:
    - X-axis: Chordwise (from LE to TE)
    - Z-axis: Vertical (global up)
    - Y-axis: Spanwise (rib thickness direction)
    
    Args:
        center_face: Nx3 array of airfoil points in 3D
        thickness: Rib material thickness (for offset calculation)
    
    Returns:
        Tuple of (flattened_points, normal_vector)
        - flattened_points: Nx3 array with all Y values set to the mean Y position
        - normal_vector: Unit vector [0, 1, 0] pointing in spanwise direction
    """
    if center_face is None or len(center_face) < 3:
        return center_face if center_face is not None else np.zeros((0, 3)), np.array([0, 1, 0])
    
    # Get the spanwise position (Y) of this rib
    y_position = np.mean(center_face[:, 1])
    
    # Project all points onto the vertical plane at this Y position
    # This keeps X and Z unchanged, only flattening Y to a constant
    vertical_profile = center_face.copy()
    vertical_profile[:, 1] = y_position  # Flatten to constant Y
    
    # Normal is purely spanwise (Y direction)
    # For a vertical rib, the normal points along the span
    normal = np.array([0.0, 1.0, 0.0])
    
    return vertical_profile, normal


def calculate_chord_line_z(airfoil_points_mm: np.ndarray, spar_x_mm: float) -> float:
    """
    Equivalent to CpacsStepperTab.calculate_chord_line_z.
    Given a polyline of airfoil points in mm (x, y, z), compute chord line Z at a given X.
    """
    if airfoil_points_mm is None or len(airfoil_points_mm) < 2:
        return 0.0
    pts = np.array(airfoil_points_mm)
    le_point = pts[np.argmin(pts[:, 0])]
    te_point = pts[np.argmax(pts[:, 0])]
    if abs(te_point[0] - le_point[0]) < 1e-6:
        return float(le_point[2])
    t = (spar_x_mm - le_point[0]) / (te_point[0] - le_point[0])
    return float(le_point[2] + t * (te_point[2] - le_point[2]))


def chord_z_from_ribs(target_pt_mm: Tuple[float, float],
                      sorted_rib_profiles_mm: List[np.ndarray]) -> Optional[float]:
    """
    Equivalent to CpacsStepperTab._get_interp_extrap_chord_z.
    Interpolate/extrapolate chord Z for a spanwise Y using two nearest rib profiles.
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
    z1 = calculate_chord_line_z(r1, target_x)
    z2 = calculate_chord_line_z(r2, target_x)
    y1 = r1[0][1] if len(r1) > 0 else 0.0
    y2 = r2[0][1] if len(r2) > 0 else 0.0
    if abs(y2 - y1) < 1e-6:
        return z1
    frac = (target_y - y1) / (y2 - y1)
    return float(z1 + frac * (z2 - z1))
