"""
Adapter service to convert Project state to the dictionary format required by cpacs_gen.py.
"""

from typing import Dict, Any, List
import numpy as np
from core.state import Project
from core.naca5_gen import get_naca_points

def project_to_cpacs_data(project: Project) -> Dict[str, Any]:
    """
    Converts the Unified Project state into the dictionary structure expected by cpacs_gen.
    
    Structure expected:
    {
        'sections': [
            {'id': int, 'span_pos': float, 'chord': float, 'le_offset': float, 'twist': float},
            ...
        ],
        'airfoils': {
            sec_id: [{'x': float, 'y': 0.0, 'z': float}, ...],
            ...
        },
        'structures': {
            'Front Spar': {'Root Chord %': float, 'Tip Chord %': float},
            'Rear Spar': {'Root Chord %': float, 'Tip Chord %': float},
            'spar_segments': [...],  # NEW: explicit spar segment definitions
        },
        'controls': {
            '<name>': {
                'Root position (% of halfspan)': float,
                'Root % of local chord': float,
                'Tip % of local chord': float,
                'hinge_rel_height': float  # NEW: CPACS hingeRelHeight
            }
        }
    }
    """
    data: Dict[str, Any] = {
        'sections': [],
        'airfoils': {},
        'structures': {},
        'controls': {}
    }

    # 1. Generate Sections
    from services.geometry import AeroSandboxService
    service = AeroSandboxService(project)
    sb_sections = service.spanwise_sections()
    
    for sec in sb_sections:
        # Calculate z-offset relative to the global wing dihedral
        # We want the LOCAL z in the CPACS section definition to be such that when
        # rotated by the global dihedral, it lands at the absolute Z from geometry.py.
        # z_abs = z_local + y * tan(dihedral)
        # => z_local = z_abs - y * tan(dihedral)
        
        dihedral_rad = np.radians(project.wing.planform.dihedral_deg)
        z_relative = sec.z_m - (sec.y_m * np.tan(dihedral_rad))
        
        data['sections'].append({
            'id': sec.index + 1,
            'span_pos': sec.y_m,
            'chord': sec.chord_m,
            'le_offset': sec.x_le_m,
            'z_offset': z_relative, 
            'twist': sec.twist_deg
        })
        
    # Set rib etas from sections
    rib_etas = [s.span_fraction for s in sb_sections]

    # Add explicit ribs at control surface boundaries if not snapping to sections
    if not project.wing.planform.snap_to_sections and project.wing.planform.control_surfaces:
        for cs in project.wing.planform.control_surfaces:
            rib_etas.append(cs.span_start_percent / 100.0)
            rib_etas.append(cs.span_end_percent / 100.0)
    
    # Ensure unique and sorted
    rib_etas = sorted(list(set(rib_etas)))
    
    data['structures']['Rib Etas'] = rib_etas
    
    # 2. Generate Airfoils - use actual airfoil from each section
    for sec in sb_sections:
        sec_id = sec.index + 1
        try:
            # Get coordinates from the AeroSandbox airfoil object
            coords = sec.airfoil.coordinates
            airfoil_points = [{'x': float(p[0]), 'y': 0.0, 'z': float(p[1])} for p in coords]
        except Exception:
            # Fallback to NACA points if airfoil object doesn't have coordinates
            try:
                pts = get_naca_points(project.wing.airfoil.root_airfoil, n_points=100)
                airfoil_points = [{'x': float(p[0]), 'y': 0.0, 'z': float(p[1])} for p in pts]
            except Exception:
                airfoil_points = [
                    {'x': 1.0, 'y': 0.0, 'z': 0.0},
                    {'x': 0.5, 'y': 0.0, 'z': 0.1},
                    {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    {'x': 0.5, 'y': 0.0, 'z': -0.1},
                    {'x': 1.0, 'y': 0.0, 'z': 0.0}
                ]
        data['airfoils'][sec_id] = airfoil_points

    # 3. Structures (Spars)
    plan = project.wing.planform
    
    # Basic spar definitions (chord percentages)
    data['structures']['Front Spar'] = {
        'Root Chord %': plan.front_spar_root_percent,
        'Tip Chord %': plan.front_spar_tip_percent
    }
    data['structures']['Rear Spar'] = {
        'Root Chord %': plan.rear_spar_root_percent,
        'Tip Chord %': 100.0 - plan.elevon_tip_chord_percent  # Rear spar ends before elevon
    }
    
    # NEW: Explicit spar segments for BWB/multi-segment support
    # Determine if we need split spars (BWB mode or control surfaces defined)
    has_control_surfaces = len(plan.control_surfaces) > 0 or plan.elevon_root_span_percent > 0
    bwb_mode = len(plan.body_sections) > 0
    
    # Calculate BWB eta (fraction of total span that is BWB body)
    bwb_eta = 0.0
    if bwb_mode and sb_sections:
        sorted_bwb = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
        bwb_outer_y = sorted_bwb[-1].y_pos if sorted_bwb else 0.0
        total_y = sb_sections[-1].y_m if sb_sections else 1.0
        bwb_eta = bwb_outer_y / total_y if total_y > 0 else 0.0
    
    spar_segments = []
    
    if bwb_mode and bwb_eta > 0.01:
        # BWB mode: add straight spars for BWB body section
        # Front spar - straight through BWB, then taper on wing
        spar_segments.append({
            'uid': 'frontSpar_bwb',
            'spar_uid': 'FrontSpar',
            'eta_start': 0.0,
            'eta_end': bwb_eta,
            'xsi_start': plan.front_spar_root_percent / 100.0,
            'xsi_end': plan.front_spar_root_percent / 100.0  # Straight in BWB
        })
        spar_segments.append({
            'uid': 'frontSpar_wing',
            'spar_uid': 'FrontSpar',
            'eta_start': bwb_eta,
            'eta_end': 1.0,
            'xsi_start': plan.front_spar_root_percent / 100.0,
            'xsi_end': plan.front_spar_tip_percent / 100.0
        })
        
        # Rear spar - straight through BWB 
        spar_segments.append({
            'uid': 'rearSpar_bwb',
            'spar_uid': 'RearSpar',
            'eta_start': 0.0,
            'eta_end': bwb_eta,
            'xsi_start': plan.rear_spar_root_percent / 100.0,
            'xsi_end': plan.rear_spar_root_percent / 100.0  # Straight in BWB
        })
        
        
        # Wing portion of rear spar - single segment (hinge spars handle control surface attachment)
        spar_segments.append({
            'uid': 'rearSpar_wing',
            'spar_uid': 'RearSpar',
            'eta_start': bwb_eta,
            'eta_end': 1.0,
            'xsi_start': plan.rear_spar_root_percent / 100.0,
            'xsi_end': plan.rear_spar_tip_percent / 100.0 if hasattr(plan, 'rear_spar_tip_percent') else plan.rear_spar_root_percent / 100.0
        })
    elif has_control_surfaces:
        # Non-BWB case with control surfaces - single rear spar (hinge spars handle control surface attachment)
        spar_segments = [
            {
                'uid': 'frontSpar_full',
                'spar_uid': 'FrontSpar',
                'eta_start': 0.0,
                'eta_end': 1.0,
                'xsi_start': plan.front_spar_root_percent / 100.0,
                'xsi_end': plan.front_spar_tip_percent / 100.0
            },
            {
                'uid': 'rearSpar_full',
                'spar_uid': 'RearSpar',
                'eta_start': 0.0,
                'eta_end': 1.0,
                'xsi_start': plan.rear_spar_root_percent / 100.0,
                'xsi_end': plan.rear_spar_tip_percent / 100.0 if hasattr(plan, 'rear_spar_tip_percent') else plan.rear_spar_root_percent / 100.0
            }
        ]
    else:
        # Simple straight spars
        spar_segments = [
            {
                'uid': 'frontSpar_full',
                'spar_uid': 'FrontSpar',
                'eta_start': 0.0,
                'eta_end': 1.0,
                'xsi_start': plan.front_spar_root_percent / 100.0,
                'xsi_end': plan.front_spar_tip_percent / 100.0
            },
            {
                'uid': 'rearSpar_full',
                'spar_uid': 'RearSpar',
                'eta_start': 0.0,
                'eta_end': 1.0,
                'xsi_start': plan.rear_spar_root_percent / 100.0,
                'xsi_end': plan.rear_spar_root_percent / 100.0
            }
        ]
    
    # Add explicit hinge spars for control surfaces
    if plan.control_surfaces:
        for cs in plan.control_surfaces:
            cs_id = cs.name.replace(" ", "_")
            spar_segments.append({
                'uid': f'hingeSpar_{cs_id}',
                'spar_uid': f'hingeSpar_{cs_id}',
                'eta_start': cs.span_start_percent / 100.0,
                'eta_end': cs.span_end_percent / 100.0,
                'xsi_start': cs.chord_start_percent / 100.0,
                'xsi_end': cs.chord_end_percent / 100.0
            })

    data['structures']['spar_segments'] = spar_segments
    
    # 4. Controls (Control Surfaces)
    # Use new control_surfaces list if defined, otherwise fall back to legacy elevon_* fields
    if plan.control_surfaces:
        for cs in plan.control_surfaces:
            data['controls'][cs.name] = {
                'Root position (% of halfspan)': cs.span_start_percent,
                'Tip position (% of halfspan)': cs.span_end_percent,
                'Root % of local chord': 100.0 - cs.chord_start_percent,
                'Tip % of local chord': 100.0 - cs.chord_end_percent,
                'hinge_rel_height': cs.hinge_rel_height,
                'surface_type': cs.surface_type
            }
    else:
        # Legacy fallback: use elevon_* fields
        data['controls']['Elevon'] = {
            'Root position (% of halfspan)': plan.elevon_root_span_percent,
            'Tip position (% of halfspan)': 100.0,
            'Root % of local chord': plan.elevon_root_chord_percent,
            'Tip % of local chord': plan.elevon_tip_chord_percent,
            'hinge_line_percent': 100.0 - plan.elevon_root_chord_percent,
            'hinge_rel_height': 0.0,
            'surface_type': 'Elevon'
        }
    
    return data
