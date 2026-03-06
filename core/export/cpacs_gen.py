"""
CPACS XML generation extracted from FlyingWingGeneratorV1-1.py (Cpacstab methods).

This module provides pure functions mirroring the monolith behavior without side-effects
or GUI dependencies. It allows other layers (GUI, CLI, tests) to generate CPACS XML
by passing in the required inputs.

Important behavior notes preserved from source:
- Airfoil points are reversed and last point duplicated to close the loop if needed.
- Wing geometry and sections/elements/segments mirror the monolith structure.
- Structures are created under a componentSegment:
  * Upper/lower shell skins use the provided thickness_value.
  * Supports spar_segments array for explicit spar definitions.
  * Supports multiple control surfaces with hinge_rel_height.
- Dihedral is encoded on wing/transformation/rotation/x using dihedral_deg.
- Materials include a single dummyMaterial with thickness on skins and spars.

Function signatures intentionally align with scaffold placeholder to remain stable.
"""

from typing import Dict, Any, Iterable, List, Tuple
import xml.etree.ElementTree as ET
import datetime

def _s(v) -> str:
    """Force-safe XML text as string to avoid ElementTree float serialization errors."""
    return str(v)


def create_rib_set(parent: ET.Element, uID: str, etas: Iterable[float],
                   start_spar: str, end_spar: str, ref_uid: str, thickness: str) -> None:
    """Create CPACS 3.5-compliant ribsDefinition entries using ribExplicitPositioning.
    Generates one ribsDefinition per eta value with start/end EtaXsi points.
    """
    etas_list = list(etas)
    if not etas_list:
        return
    unique_sorted_etas = sorted(list(set(etas_list)))

    for idx, eta in enumerate(unique_sorted_etas, start=1):
        rib_uid = f"{uID}_{idx}"
        rib_def = ET.SubElement(parent, 'ribsDefinition', uID=rib_uid)
        ET.SubElement(rib_def, 'name').text = _s(rib_uid)

        rep = ET.SubElement(rib_def, 'ribExplicitPositioning')

        start_pt = ET.SubElement(rep, 'startEtaXsiPoint')
        ET.SubElement(start_pt, 'eta').text = _s(eta)
        ET.SubElement(start_pt, 'xsi').text = _s(0.0)
        ET.SubElement(start_pt, 'referenceUID').text = _s(ref_uid)

        end_pt = ET.SubElement(rep, 'endEtaXsiPoint')
        ET.SubElement(end_pt, 'eta').text = _s(eta)
        ET.SubElement(end_pt, 'xsi').text = _s(1.0)
        ET.SubElement(end_pt, 'referenceUID').text = _s(ref_uid)

        ET.SubElement(rep, 'ribStart').text = _s(start_spar)
        ET.SubElement(rep, 'ribEnd').text = _s(end_spar)

        rib_cs = ET.SubElement(rib_def, 'ribCrossSection')
        mat = ET.SubElement(rib_cs, 'material')
        ET.SubElement(mat, 'materialUID').text = _s("dummyMaterial")
        ET.SubElement(mat, 'thickness').text = _s(thickness)


def create_spar_pos(parent: ET.Element, uID: str, eta: float, xsi: float, ref_uid: str) -> None:
    """Helper to create a sparPosition with eta/xsi."""
    pos = ET.SubElement(parent, 'sparPosition', uID=uID)
    etaxsi = ET.SubElement(pos, 'sparPositionEtaXsi')
    ET.SubElement(etaxsi, 'eta').text = _s(eta)
    ET.SubElement(etaxsi, 'xsi').text = _s(xsi)
    ET.SubElement(etaxsi, 'referenceUID').text = _s(ref_uid)


def create_spar_seg(parent: ET.Element, uID: str, pos_uids: Iterable[str], thickness_value: str) -> None:
    """Helper to create a sparSegment with a single web and given thickness."""
    seg = ET.SubElement(parent, 'sparSegment', uID=uID)
    ET.SubElement(seg, 'name').text = _s(uID)

    uids_node = ET.SubElement(seg, 'sparPositionUIDs')
    for pos_uid in pos_uids:
        ET.SubElement(uids_node, 'sparPositionUID').text = _s(pos_uid)

    cs = ET.SubElement(seg, 'sparCrossSection')
    web = ET.SubElement(cs, 'web1')
    mat = ET.SubElement(web, 'material')
    ET.SubElement(mat, 'materialUID').text = _s("dummyMaterial")
    ET.SubElement(mat, 'thickness').text = _s(thickness_value)
    ET.SubElement(web, 'relPos').text = _s("0.5")
    ET.SubElement(cs, 'rotation').text = _s("90.0")


def _add_profiles_if_any(vehicles: ET.Element, data: Dict[str, Any]) -> None:
    """Add wingAirfoils/profiles if airfoil discretizations exist in data."""
    if not data.get('airfoils'):
        return

    profiles = ET.SubElement(vehicles, 'profiles')
    wing_airfoils = ET.SubElement(profiles, 'wingAirfoils')

    for sec_id, points in data['airfoils'].items():
        # Preserve monolith behavior: reverse and ensure closed loop
        points = list(points)  # shallow copy
        points.reverse()
        if len(points) > 1:
            points[-1] = points[0].copy()

        airfoil_uid = f"airfoil_sec{sec_id}"
        wing_airfoil = ET.SubElement(wing_airfoils, 'wingAirfoil', uID=airfoil_uid)
        ET.SubElement(wing_airfoil, 'name').text = _s(f"Airfoil Section {sec_id}")
        point_list = ET.SubElement(wing_airfoil, 'pointList')
        ET.SubElement(point_list, 'x').text = _s(";".join([_s(p['x']) for p in points]))
        ET.SubElement(point_list, 'y').text = _s(";".join([_s(p['y']) for p in points]))
        ET.SubElement(point_list, 'z').text = _s(";".join([_s(p['z']) for p in points]))


def _add_wing_geometry(model: ET.Element, data: Dict[str, Any], dihedral_deg: str) -> Tuple[ET.Element, ET.Element]:
    """
    Add wing/mainWing node with sections/elements/segments.
    Returns (wing_element, sections_node)
    """
    wings = ET.SubElement(model, 'wings')
    wing = ET.SubElement(wings, 'wing', uID="mainWing", symmetry="x-z-plane")
    ET.SubElement(wing, 'name').text = _s("Main Wing")

    transformation = ET.SubElement(wing, 'transformation')
    rotation = ET.SubElement(transformation, 'rotation')
    ET.SubElement(rotation, 'x').text = _s(dihedral_deg)
    ET.SubElement(rotation, 'y').text = _s("0.0")
    ET.SubElement(rotation, 'z').text = _s("0.0")

    sections_node = ET.SubElement(wing, 'sections')
    for sd in data['sections']:
        sec_id = sd['id']
        section = ET.SubElement(sections_node, 'section', uID=f"wing_sec{sec_id}")
        ET.SubElement(section, 'name').text = _s(f"Wing Section {sec_id}")

        sec_trans = ET.SubElement(section, 'transformation')
        sec_trans_trans = ET.SubElement(sec_trans, 'translation', refType="absLocal")
        ET.SubElement(sec_trans_trans, 'x').text = _s(sd['le_offset'])
        ET.SubElement(sec_trans_trans, 'y').text = _s(sd['span_pos'])
        ET.SubElement(sec_trans_trans, 'z').text = _s(sd.get('z_offset', 0.0))

        elements = ET.SubElement(section, 'elements')
        element = ET.SubElement(elements, 'element', uID=f"wing_sec{sec_id}_el1")
        ET.SubElement(element, 'name').text = _s(f"Element {sec_id}")
        if data.get('airfoils'):
            ET.SubElement(element, 'airfoilUID').text = _s(f"airfoil_sec{sec_id}")

        el_trans = ET.SubElement(element, 'transformation')
        el_trans_scale = ET.SubElement(el_trans, 'scaling')
        ET.SubElement(el_trans_scale, 'x').text = _s(sd['chord'])
        ET.SubElement(el_trans_scale, 'y').text = _s("1.0")
        ET.SubElement(el_trans_scale, 'z').text = _s(sd['chord'])
        el_trans_rot = ET.SubElement(el_trans, 'rotation')
        ET.SubElement(el_trans_rot, 'x').text = _s("0.0")
        ET.SubElement(el_trans_rot, 'y').text = _s(sd['twist'])
        ET.SubElement(el_trans_rot, 'z').text = _s("0.0")

    segments_node = ET.SubElement(wing, 'segments')
    for i in range(len(data['sections']) - 1):
        from_id = data['sections'][i]['id']
        to_id = data['sections'][i + 1]['id']
        segment = ET.SubElement(segments_node, 'segment', uID=f"wing_seg{from_id}")
        ET.SubElement(segment, 'name').text = _s(f"Segment {from_id}-{to_id}")
        ET.SubElement(segment, 'fromElementUID').text = _s(f"wing_sec{from_id}_el1")
        ET.SubElement(segment, 'toElementUID').text = _s(f"wing_sec{to_id}_el1")

    return wing, sections_node


def _add_structures_and_controls(wing: ET.Element, data: Dict[str, Any],
                                 thickness_value: str) -> None:
    """
    Add componentSegment, shells, spars (positions and segments), ribs,
    and control surfaces. Supports multiple control surfaces via data['controls'].
    """
    if not data.get('structures'):
        return

    comp_seg_uid = "mainWingCS"
    comp_segs = ET.SubElement(wing, 'componentSegments')
    comp_seg = ET.SubElement(comp_segs, 'componentSegment', uID=comp_seg_uid)
    ET.SubElement(comp_seg, 'name').text = _s("Main Wing Component Segment")
    ET.SubElement(comp_seg, 'fromElementUID').text = _s(f"wing_sec{data['sections'][0]['id']}_el1")
    ET.SubElement(comp_seg, 'toElementUID').text = _s(f"wing_sec{data['sections'][-1]['id']}_el1")

    structure = ET.SubElement(comp_seg, 'structure')

    # Skins with thickness
    upper_shell = ET.SubElement(structure, 'upperShell', uID="upperShell")
    skin_upper = ET.SubElement(upper_shell, 'skin')
    mat_upper = ET.SubElement(skin_upper, 'material')
    ET.SubElement(mat_upper, 'materialUID').text = _s("dummyMaterial")
    ET.SubElement(mat_upper, 'thickness').text = _s(thickness_value)

    lower_shell = ET.SubElement(structure, 'lowerShell', uID="lowerShell")
    skin_lower = ET.SubElement(lower_shell, 'skin')
    mat_lower = ET.SubElement(skin_lower, 'material')
    ET.SubElement(mat_lower, 'materialUID').text = _s("dummyMaterial")
    ET.SubElement(mat_lower, 'thickness').text = _s(thickness_value)

    # Spar containers
    spars = ET.SubElement(structure, 'spars')
    spar_positions = ET.SubElement(spars, 'sparPositions')
    spar_segments_node = ET.SubElement(spars, 'sparSegments')

    # Use explicit spar_segments if provided
    spar_segments_data = data['structures'].get('spar_segments', [])
    
    if spar_segments_data:
        # Generate spars from explicit segment definitions
        position_uids = {}
        
        for seg in spar_segments_data:
            seg_uid = seg['uid']
            eta_start = seg['eta_start']
            eta_end = seg['eta_end']
            xsi_start = seg['xsi_start']
            xsi_end = seg['xsi_end']
            
            start_pos_uid = f"{seg_uid}_start"
            end_pos_uid = f"{seg_uid}_end"
            
            pos_key_start = (eta_start, xsi_start)
            if pos_key_start not in position_uids:
                create_spar_pos(spar_positions, start_pos_uid, eta_start, xsi_start, comp_seg_uid)
                position_uids[pos_key_start] = start_pos_uid
            else:
                start_pos_uid = position_uids[pos_key_start]
            
            pos_key_end = (eta_end, xsi_end)
            if pos_key_end not in position_uids:
                create_spar_pos(spar_positions, end_pos_uid, eta_end, xsi_end, comp_seg_uid)
                position_uids[pos_key_end] = end_pos_uid
            else:
                end_pos_uid = position_uids[pos_key_end]
            
            create_spar_seg(spar_segments_node, seg_uid, [start_pos_uid, end_pos_uid], thickness_value)
    else:
        # Legacy fallback
        fsd = data['structures'].get('Front Spar', {})
        rsd = data['structures'].get('Rear Spar', {})
        
        fs_root_xsi = float(fsd.get('Root Chord %', 25)) / 100.0
        fs_tip_xsi = float(fsd.get('Tip Chord %', 20)) / 100.0
        create_spar_pos(spar_positions, "frontSparPos_root", 0.0, fs_root_xsi, comp_seg_uid)
        create_spar_pos(spar_positions, "frontSparPos_tip", 1.0, fs_tip_xsi, comp_seg_uid)
        create_spar_seg(spar_segments_node, "frontSpar", ["frontSparPos_root", "frontSparPos_tip"], thickness_value)
        
        mrs_xsi = float(rsd.get('Root Chord %', 75)) / 100.0
        create_spar_pos(spar_positions, "rearSpar_root", 0.0, mrs_xsi, comp_seg_uid)
        create_spar_pos(spar_positions, "rearSpar_tip", 1.0, mrs_xsi, comp_seg_uid)
        create_spar_seg(spar_segments_node, "rearSpar", ["rearSpar_root", "rearSpar_tip"], thickness_value)

    # Ribs
    ribs_defs = ET.SubElement(structure, 'ribsDefinitions')
    total_half_span = data['sections'][-1]['span_pos'] if data['sections'] and data['sections'][-1]['span_pos'] > 0 else 1.0
    
    if 'Rib Etas' in data.get('structures', {}):
        all_etas = sorted(list(data['structures']['Rib Etas']))
    else:
        all_etas = sorted(list(set([(s['span_pos'] / total_half_span) for s in data['sections']])))

    # Determine front/rear spar names for ribs
    front_spar_name = "frontSpar_inner" if spar_segments_data else "frontSpar"
    rear_spar_segments = [s for s in spar_segments_data if 'rear' in s['uid'].lower()] if spar_segments_data else []
    
    controls_data = data.get('controls', {})
    
    if controls_data and rear_spar_segments:
        # Split ribs based on control surface position
        first_cs = list(controls_data.values())[0]
        split_eta = float(first_cs.get('Root position (% of halfspan)', 50)) / 100.0
        
        TOL = 1e-6
        inboard_etas = [eta for eta in all_etas if eta <= split_eta + TOL]
        outboard_etas = [eta for eta in all_etas if eta >= split_eta - TOL]
        
        inboard_rear = rear_spar_segments[0]['uid'] if rear_spar_segments else "rearSpar"
        outboard_rear = rear_spar_segments[-1]['uid'] if len(rear_spar_segments) > 1 else inboard_rear
        
        create_rib_set(ribs_defs, "inboardRibs", inboard_etas, front_spar_name, inboard_rear, comp_seg_uid, thickness_value)
        create_rib_set(ribs_defs, "outboardRibs", outboard_etas, front_spar_name, outboard_rear, comp_seg_uid, thickness_value)
    else:
        rear_spar_name = "rearSpar_full" if spar_segments_data else "rearSpar"
        create_rib_set(ribs_defs, "allRibs", all_etas, front_spar_name, rear_spar_name, comp_seg_uid, thickness_value)

    # Control Surfaces
    if controls_data:
        controls = ET.SubElement(comp_seg, 'controlSurfaces')
        ted = ET.SubElement(controls, 'trailingEdgeDevices')
        
        for cs_name, cs_data in controls_data.items():
            cs_uid = cs_name.lower().replace(' ', '_')
            device = ET.SubElement(ted, 'trailingEdgeDevice', uID=cs_uid)
            ET.SubElement(device, 'name').text = _s(cs_name)
            ET.SubElement(device, 'parentUID').text = _s(comp_seg_uid)
            
            cs_start_eta = float(cs_data.get('Root position (% of halfspan)', 60)) / 100.0
            cs_end_eta = float(cs_data.get('Tip position (% of halfspan)', 100)) / 100.0
            cs_root_xsi = 1.0 - (float(cs_data.get('Root % of local chord', 30)) / 100.0)
            cs_tip_xsi = 1.0 - (float(cs_data.get('Tip % of local chord', 30)) / 100.0)
            hinge_rel_height = float(cs_data.get('hinge_rel_height', 0.5))

            outer_shape = ET.SubElement(device, 'outerShape')

            inner = ET.SubElement(outer_shape, 'innerBorder')
            etaLE_in = ET.SubElement(inner, 'etaLE')
            ET.SubElement(etaLE_in, 'eta').text = _s(cs_start_eta)
            ET.SubElement(etaLE_in, 'referenceUID').text = _s(comp_seg_uid)
            xsiLE_in = ET.SubElement(inner, 'xsiLE')
            ET.SubElement(xsiLE_in, 'xsi').text = _s(cs_root_xsi)
            ET.SubElement(xsiLE_in, 'referenceUID').text = _s(comp_seg_uid)
            etaTE_in = ET.SubElement(inner, 'etaTE')
            ET.SubElement(etaTE_in, 'eta').text = _s(cs_start_eta)
            ET.SubElement(etaTE_in, 'referenceUID').text = _s(comp_seg_uid)

            outer = ET.SubElement(outer_shape, 'outerBorder')
            etaLE_out = ET.SubElement(outer, 'etaLE')
            ET.SubElement(etaLE_out, 'eta').text = _s(cs_end_eta)
            ET.SubElement(etaLE_out, 'referenceUID').text = _s(comp_seg_uid)
            xsiLE_out = ET.SubElement(outer, 'xsiLE')
            ET.SubElement(xsiLE_out, 'xsi').text = _s(cs_tip_xsi)
            ET.SubElement(xsiLE_out, 'referenceUID').text = _s(comp_seg_uid)
            etaTE_out = ET.SubElement(outer, 'etaTE')
            ET.SubElement(etaTE_out, 'eta').text = _s(cs_end_eta)
            ET.SubElement(etaTE_out, 'referenceUID').text = _s(comp_seg_uid)

            path = ET.SubElement(device, 'path')
            steps = ET.SubElement(path, 'steps')
            step1 = ET.SubElement(steps, 'step')
            ET.SubElement(step1, 'controlParameter').text = _s("-1")
            ET.SubElement(step1, 'hingeLineRotation').text = _s("-30")
            step2 = ET.SubElement(steps, 'step')
            ET.SubElement(step2, 'controlParameter').text = _s("1")
            ET.SubElement(step2, 'hingeLineRotation').text = _s("30")
            
            inner_hinge = ET.SubElement(path, 'innerHingePoint')
            ET.SubElement(inner_hinge, 'hingeXsi').text = _s(cs_root_xsi)
            ET.SubElement(inner_hinge, 'hingeRelHeight').text = _s(hinge_rel_height)
            outer_hinge = ET.SubElement(path, 'outerHingePoint')
            ET.SubElement(outer_hinge, 'hingeXsi').text = _s(cs_tip_xsi)
            ET.SubElement(outer_hinge, 'hingeRelHeight').text = _s(hinge_rel_height)


def generate_cpacs_xml(data: Dict[str, Any], model_name: str,
                       thickness_value: str, dihedral_deg: str) -> ET.Element:
    """
    Pure function to create a CPACS XML tree replicating the monolith's generator.

    Parameters:
        data: Parsed wing data dictionary from WingData.txt
        model_name: Base name for uIDs and header name
        thickness_value: String thickness used for skins and spars (e.g., "0.00635")
        dihedral_deg: String dihedral angle in degrees for wing rotation.x (e.g., "2.0")

    Returns:
        The root ET.Element named 'cpacs'
    """
    # Root and header
    cpacs = ET.Element('cpacs')
    cpacs.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
    cpacs.set('xsi:noNamespaceSchemaLocation', "cpacs_schema.xsd")

    header = ET.SubElement(cpacs, 'header')
    ET.SubElement(header, 'name').text = _s(f"{model_name} Model")
    ET.SubElement(header, 'version').text = _s("1.0.0")
    ET.SubElement(header, 'cpacsVersion').text = _s("3.5")
    versionInfos = ET.SubElement(header, 'versionInfos')
    vi = ET.SubElement(versionInfos, 'versionInfo', version="1.0.0")
    ET.SubElement(vi, 'creator').text = _s("UnifiedConverterApp")
    ET.SubElement(vi, 'timestamp').text = _s(datetime.datetime.now().isoformat())
    ET.SubElement(vi, 'description').text = _s(f"Wing model generated from {model_name}.txt")
    ET.SubElement(vi, 'cpacsVersion').text = _s("3.5")

    vehicles = ET.SubElement(cpacs, 'vehicles')
    aircraft = ET.SubElement(vehicles, 'aircraft')
    model = ET.SubElement(aircraft, 'model', uID=model_name)
    ET.SubElement(model, 'name').text = _s(model_name)

    # Profiles (airfoils)
    _add_profiles_if_any(vehicles, data)

    # Wing geometry
    wing, _sections_node = _add_wing_geometry(model, data, dihedral_deg)

    # Structures and controls
    _add_structures_and_controls(wing, data, thickness_value)

    # Materials
    materials = ET.SubElement(vehicles, 'materials')
    mat = ET.SubElement(materials, 'material', uID="dummyMaterial")
    ET.SubElement(mat, 'name').text = _s("Dummy Material")
    ET.SubElement(mat, 'rho').text = _s("1.0")
    iso = ET.SubElement(mat, 'isotropicProperties')
    ET.SubElement(iso, 'E').text = _s("1.0")

    return cpacs