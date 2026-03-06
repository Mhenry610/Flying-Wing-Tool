"""
Geometry helpers extracted verbatim from FlyingWingGeneratorV1-1.py to maintain 1:1 behavior.

Note:
- Signatures and behavior are kept identical for the initial extraction.
"""
import numpy as np

def get_transformation_matrix(transform_element):
    """Creates a 4x4 transformation matrix from a <transformation> element."""
    mat = np.identity(4)
    if transform_element is None: return mat
    # Order: scaling, rotation, translation
    scale = transform_element.find('scaling')
    if scale is not None:
        s = np.array([float(scale.find(c).text) for c in ['x', 'y', 'z']])
        mat = np.array([[s[0], 0, 0, 0], [0, s[1], 0, 0], [0, 0, s[2], 0], [0, 0, 0, 1]]) @ mat
    rot = transform_element.find('rotation')
    if rot is not None:
        r = np.deg2rad([float(rot.find(c).text) for c in ['x', 'y', 'z']])
        cx, sx = np.cos(r[0]), np.sin(r[0]); cy, sy = np.cos(r[1]), np.sin(r[1]); cz, sz = np.cos(r[2]), np.sin(r[2])
        rxm = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
        rym = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
        rzm = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mat = (rzm @ rym @ rxm) @ mat
    trans = transform_element.find('translation')
    if trans is not None:
        t = np.array([float(trans.find(c).text) for c in ['x', 'y', 'z']])
        mat = np.array([[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]) @ mat
    return mat

def apply_transformation(points, matrix):
    """Applies a 4x4 transformation matrix to a set of 3D points."""
    n_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((n_points, 1))])
    transformed_points_h = (matrix @ homogeneous_points.T).T
    return transformed_points_h[:, :3]

def get_structural_points(eta, xsi, from_section_points, to_section_points, spar_offset=0.0):
    """
    Calculates the 3D coordinates on the upper and lower wing surface for a
    given eta/xsi pair, applying an optional inward offset.
    Returns a tuple (upper_point, lower_point).
    """
    interpolated_section = (1 - eta) * from_section_points + eta * to_section_points
    le_index = np.argmin(interpolated_section[:, 0])
    lower_surface = interpolated_section[:le_index+1][::-1]
    upper_surface = interpolated_section[le_index:]
    x_le, x_te = lower_surface[0, 0], upper_surface[-1, 0]
    target_x = x_le + xsi * (x_te - x_le)
    y_upper = np.interp(target_x, upper_surface[:, 0], upper_surface[:, 1])
    z_upper = np.interp(target_x, upper_surface[:, 0], upper_surface[:, 2])
    y_lower = np.interp(target_x, lower_surface[:, 0], lower_surface[:, 1])
    z_lower = np.interp(target_x, lower_surface[:, 0], lower_surface[:, 2])
    upper_point = np.array([target_x, y_upper, z_upper])
    lower_point = np.array([target_x, y_lower, z_lower])
    upper_point[2] -= spar_offset
    lower_point[2] += spar_offset
    return upper_point, lower_point

def get_thickness(element, override=None, default=0.00635):
    """Parses thickness from a CPACS element or returns a default, prioritizing an override."""
    if override is not None: return override
    cross_section = element.find('sparCrossSection') or element.find('ribCrossSection')
    if cross_section is not None:
        material_elem = cross_section.find('.//material')
        if material_elem is not None:
            thickness_elem = material_elem.find('thickness')
            if thickness_elem is not None and thickness_elem.text: return float(thickness_elem.text)
    thickness_elem = element.find('.//thickness')
    if thickness_elem is not None and thickness_elem.text: return float(thickness_elem.text)
    return default