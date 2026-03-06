"""
XFLR5 XML and DAT export (pure functions extracted verbatim from the monolith's Xflr5Tab logic).
No behavior changes; GUI concerns (logging, dialogs) are left to the caller.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List
import os
import xml.etree.ElementTree as ET


def create_xflr5_xml_structure(parsed_data: Dict[str, Any],
                               wing_name: str,
                               wing_type: str,
                               pos_xyz: Tuple[float, float, float],
                               symmetric: bool) -> ET.Element:
    """
    Create an XFLR5 "explane" XML Element tree using the same logic as Xflr5Tab.create_xml_structure().
    Mirrors unit conversions and field names:
      - Units: length_unit_to_meter=0.0254, mass_unit_to_kg=0.453592
      - Section fields in inches with 3 decimals: y_position, Chord, xOffset; Twist in deg; Dihedral fixed 0.000
      - Left_Side_FoilName and Right_Side_FoilName -> Airfoil_Section_{id}
    Returns the root ET.Element of the XFLR5 document (without XML header/doctype).
    """
    root = ET.Element("explane", version="1.0")

    # Units block
    units = ET.SubElement(root, "Units")
    ET.SubElement(units, "length_unit_to_meter").text = "0.0254"
    ET.SubElement(units, "mass_unit_to_kg").text = "0.453592"

    # Plane/Wing block
    plane = ET.SubElement(root, "Plane")
    ET.SubElement(plane, "Name").text = "Converted Plane"

    wing = ET.SubElement(plane, "wing")
    ET.SubElement(wing, "Name").text = str(wing_name)
    ET.SubElement(wing, "Type").text = str(wing_type)

    # Position as provided (no unit conversion specified in monolith UI; it writes raw values)
    px, py, pz = pos_xyz
    ET.SubElement(wing, "Position").text = f"{px}, {py}, {pz}"
    ET.SubElement(wing, "Symetric").text = "true" if symmetric else "false"

    sections = ET.SubElement(wing, "Sections")

    # Sections from parsed data; convert meters -> inches for y, chord, xOffset
    # inches factor used in monolith: 39.3701
    IN_PER_M = 39.3701

    for sd in parsed_data.get('sections', []):
        section = ET.SubElement(sections, "Section")

        y_in = float(sd['span_pos']) * IN_PER_M
        chord_in = float(sd['chord']) * IN_PER_M
        x_off_in = float(sd['le_offset']) * IN_PER_M
        twist_deg = float(sd['twist'])

        ET.SubElement(section, "y_position").text = f"{y_in:.3f}"
        ET.SubElement(section, "Chord").text = f"{chord_in:.3f}"
        ET.SubElement(section, "xOffset").text = f"{x_off_in:.3f}"
        ET.SubElement(section, "Twist").text = f"{twist_deg:.3f}"
        ET.SubElement(section, "Dihedral").text = "0.000"

        airfoil_name = f"Airfoil_Section_{sd['id']}"
        ET.SubElement(section, "Left_Side_FoilName").text = airfoil_name
        ET.SubElement(section, "Right_Side_FoilName").text = airfoil_name

    return root


def create_airfoil_dat_files(parsed_data: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Create .dat files for each airfoil section as in Xflr5Tab.create_dat_files().
    Writes files named 'Airfoil_Section_{sec_id}.dat' into output_dir.
    Each file content:
        First line: 'Airfoil Section {sec_id}'
        Then one line per (x, z) pair with 6 decimals.
    Returns the list of absolute file paths successfully written.
    """
    written_files: List[str] = []

    airfoils = parsed_data.get('airfoils', {})
    if not airfoils:
        return written_files

    os.makedirs(output_dir, exist_ok=True)

    for sec_id, points in airfoils.items():
        filename = f"Airfoil_Section_{sec_id}.dat"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Airfoil Section {sec_id}\n")
            for p in points:
                # Monolith writes x and z only, 6 decimals
                f.write(f"{float(p['x']):.6f} {float(p['z']):.6f}\n")
        written_files.append(os.path.abspath(filepath))

    return written_files