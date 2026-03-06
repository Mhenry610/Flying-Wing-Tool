"""
Parsing utilities extracted verbatim from FlyingWingGeneratorV1-1.py to maintain 1:1 behavior.

Note:
- GUI messagebox usage is intentionally preserved for now to avoid behavior changes.
- Future refactors may replace GUI side-effects with exceptions or return codes.
"""
from typing import Tuple, Dict, Any, Optional
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tkinter import messagebox  # preserved side-effect for identical behavior
import numpy as np

def parse_wing_data(filepath: str, refine_to: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses the WingData.txt file, handling Geometric Data, Structures, Controls, and Airfoils.

    Parameters:
      refine_to: Optional[int]
        If provided, each parsed airfoil's (x,z) profile will be resampled to this many points
        using a Catmull-Rom spline interpolation. The returned airfoil points for each section
        will be a list of dicts with keys 'x','y','z' (y preserved as 0.0). If refine_to is None,
        the original discrete points are returned unchanged.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except IOError as e:
        messagebox.showerror("File Error", f"Could not read file: {e}")
        return None, None

    data: Dict[str, Any] = {'sections': [], 'airfoils': {}, 'structures': {}, 'controls': {}}

    section_pattern = re.compile(r"#-+#\s*(.*?)\s*#-+#", re.DOTALL)
    sections = section_pattern.split(content)

    for block in sections:
        block = block.strip()
        if not block:
            continue

        lines = block.split('\n')
        header = lines[0].strip()
        body = lines[1:]

        if "Geometric Data" in header:
            current_section = None
            for line in body:
                line = line.strip()
                if line.startswith("Section"):
                    current_section = {'id': int(line.split()[1])}
                    data['sections'].append(current_section)
                elif ':' in line and current_section is not None:
                    key, value = line.split(':', 1)
                    key_map = {
                        "Span position (m)": "span_pos", "Local chord (m)": "chord",
                        "Leading edge offset (m)": "le_offset", "Geometric twist (deg)": "twist"
                    }
                    if key.strip() in key_map:
                        current_section[key_map[key.strip()]] = float(value.strip())

        elif "Structures" in header:
            current_struct = None
            for line in body:
                line = line.strip();
                if not line: continue
                if not line.startswith(' '):
                    current_struct = line
                    data['structures'][current_struct] = {}
                elif ':' in line and current_struct:
                    key, value = line.split(':', 1)
                    data['structures'][current_struct][key.strip()] = value.strip()

        elif "Controls" in header:
            current_control = None
            for line in body:
                stripped_line = line.strip()
                if not stripped_line: continue
                if line.startswith('  ') and not line.startswith('    '):
                    current_control = stripped_line
                    data['controls'][current_control] = {}
                elif line.startswith('    ') and ':' in stripped_line and current_control:
                    key, value = stripped_line.split(':', 1)
                    data['controls'][current_control][key.strip()] = value.strip()

        elif "Airfoil Section" in header:
            match = re.search(r'Airfoil Section (\d+)', header)
            if match:
                sec_id = int(match.group(1))
                points = []
                for line in body:
                    line = line.strip()
                    if not line: continue
                    parts = re.split(r'\s+', line)
                    if len(parts) == 2:
                        try:
                            points.append({'x': float(parts[0]), 'y': 0.0, 'z': float(parts[1])})
                        except ValueError:
                            continue
                if points:
                    # Closed-loop arc-length Catmull-Rom resampling to exactly n_samples points
                    def _closed_loop_cr_resample(pts2d, n_samples):
                        """
                        pts2d: list of (x,z) tuples representing a closed contour in order
                        n_samples: desired number of output points (>= 2)
                        returns: list of (x,z) tuples (length == n_samples)

                        This version preserves the original leading-edge (min-x) and trailing-edge (max-x)
                        coordinates by snapping the nearest resampled samples back to the original
                        LE/TE positions after resampling.
                        """
                        if n_samples <= 2 or len(pts2d) < 2:
                            return pts2d[:]
                        arr = np.array(pts2d, dtype=float)
                        # identify original LE/TE points (by X)
                        xs = arr[:, 0]
                        le_idx = int(np.argmin(xs))
                        te_idx = int(np.argmax(xs))
                        le_pt = tuple(arr[le_idx])
                        te_pt = tuple(arr[te_idx])
                        # Ensure closed by wrapping endpoints (do not duplicate last if identical to first)
                        if not np.allclose(arr[0], arr[-1]):
                            arr = np.vstack([arr, arr[0]])
                        # compute cumulative arc lengths
                        seg_lengths = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
                        cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
                        total_len = cum[-1]
                        if total_len == 0.0:
                            return [tuple(arr[0]) for _ in range(n_samples)]
                        # target parameter values along arc length [0, total_len)
                        targets = np.linspace(0.0, total_len, num=n_samples, endpoint=False)
                        # prepare wrapped unique points (exclude last duplicated point)
                        pts = arr[:-1]
                        m = len(pts)
                        # wrap pts for control point indexing with two extra on each side
                        wrapped = np.vstack([pts[-2:], pts, pts[:2]])
                        out = []
                        # cumulative lengths for locating segments on original pts
                        cum_pts = np.concatenate(([0.0], np.cumsum(np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0)**2, axis=1)))))
                        for s in targets:
                            # find segment where cum_pts[i] <= s < cum_pts[i+1]
                            idx = int(np.searchsorted(cum_pts, s, side='right') - 1)
                            if idx < 0:
                                idx = 0
                            if idx >= m:
                                idx = m - 1
                            seg0 = cum_pts[idx]
                            seg1 = cum_pts[idx+1] if idx+1 < len(cum_pts) else cum_pts[-1]
                            u_seg = 0.0 if seg1 == seg0 else (s - seg0) / (seg1 - seg0)
                            i_wrapped = idx + 2
                            p0 = wrapped[i_wrapped - 1]
                            p1 = wrapped[i_wrapped]
                            p2 = wrapped[i_wrapped + 1]
                            p3 = wrapped[i_wrapped + 2]
                            # Catmull-Rom (centripetal) interpolation formula
                            u = u_seg
                            u2 = u * u
                            u3 = u2 * u
                            pt = 0.5 * ( (2*p1) +
                                         (-p0 + p2) * u +
                                         (2*p0 - 5*p1 + 4*p2 - p3) * u2 +
                                         (-p0 + 3*p1 - 3*p2 + p3) * u3 )
                            out.append((float(pt[0]), float(pt[1])))
                        # Now enforce exact LE/TE anchor positions by snapping nearest resampled points
                        try:
                            out_arr = np.array(out, dtype=float)
                            # find resampled index nearest original LE x and snap
                            le_res_idx = int(np.argmin(np.abs(out_arr[:, 0] - le_pt[0])))
                            out_arr[le_res_idx, 0] = le_pt[0]; out_arr[le_res_idx, 1] = le_pt[1]
                            # Determine TE upper/lower candidates from original points
                            xs_orig = arr[:, 0]
                            te_x_val = float(np.max(xs_orig))
                            te_mask = np.isclose(xs_orig, te_x_val, atol=1e-8)
                            te_candidates = arr[te_mask]
                            te_upper_z = None
                            te_lower_z = None
                            if te_candidates.shape[0] >= 2:
                                # choose highest and lowest z among TE candidates
                                te_upper_z = float(np.max(te_candidates[:, 1]))
                                te_lower_z = float(np.min(te_candidates[:, 1]))
                            else:
                                # fallback: take the TE point and its nearest neighbor (by index) as upper/lower
                                te_idx_local = int(np.argmax(xs_orig))
                                prev_idx = (te_idx_local - 1) % arr.shape[0]
                                next_idx = (te_idx_local + 1) % arr.shape[0]
                                z_vals = [float(arr[prev_idx,1]), float(arr[te_idx_local,1]), float(arr[next_idx,1])]
                                te_upper_z = max(z_vals)
                                te_lower_z = min(z_vals)
                            # find two resampled indices nearest the TE x and assign upper/lower z to enforce a sharp TE
                            te_res_order = np.argsort(np.abs(out_arr[:, 0] - te_x_val))
                            if len(te_res_order) >= 2:
                                te_idx_a, te_idx_b = int(te_res_order[0]), int(te_res_order[1])
                                # assign higher z to upper candidate, lower z to lower candidate
                                if te_upper_z is not None and te_lower_z is not None:
                                    # decide which index is "upper" by comparing current z
                                    if out_arr[te_idx_a,1] >= out_arr[te_idx_b,1]:
                                        out_arr[te_idx_a, 0] = te_x_val; out_arr[te_idx_a, 1] = te_upper_z
                                        out_arr[te_idx_b, 0] = te_x_val; out_arr[te_idx_b, 1] = te_lower_z
                                    else:
                                        out_arr[te_idx_b, 0] = te_x_val; out_arr[te_idx_b, 1] = te_upper_z
                                        out_arr[te_idx_a, 0] = te_x_val; out_arr[te_idx_a, 1] = te_lower_z
                                else:
                                    # fallback: set both to the original TE point
                                    out_arr[te_idx_a, 0] = te_pt[0]; out_arr[te_idx_a, 1] = te_pt[1]
                                    out_arr[te_idx_b, 0] = te_pt[0]; out_arr[te_idx_b, 1] = te_pt[1]
                            else:
                                # single nearest index fallback
                                te_res_idx = int(np.argmin(np.abs(out_arr[:, 0] - te_pt[0])))
                                out_arr[te_res_idx, 0] = te_pt[0]; out_arr[te_res_idx, 1] = te_pt[1]
                            # Ensure exact tuple return
                            return [ (float(x), float(z)) for x, z in out_arr.tolist() ]
                        except Exception:
                            return out
                    # prepare points for resampling: use (x,z) pairs
                    if refine_to and isinstance(refine_to, int) and refine_to > 1:
                        try:
                            xz = [(p['x'], p['z']) for p in points]
                            resampled = _closed_loop_cr_resample(xz, refine_to)
                            data['airfoils'][sec_id] = [{'x': float(x), 'y': 0.0, 'z': float(z)} for x, z in resampled]
                        except Exception:
                            # fallback to original if resampling fails
                            data['airfoils'][sec_id] = points
                    else:
                        data['airfoils'][sec_id] = points

    if not data['sections']:
        return None, "Parsing failed: Could not find 'Geometric Data'."

    return data, "Data parsed successfully."

def prettify_xml(elem) -> str:
    """Returns a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def parse_airfoil_points(point_list_element):
    """Parses the x, y, z coordinates from a <pointList> element."""
    x_str = point_list_element.find('x').text
    y_str = point_list_element.find('y').text
    z_str = point_list_element.find('z').text
    xs = [float(p) for p in x_str.strip().split(';')]
    ys = [float(p) for p in y_str.strip().split(';')]
    zs = [float(p) for p in z_str.strip().split(';')]
    return np.array(list(zip(xs, ys, zs)))