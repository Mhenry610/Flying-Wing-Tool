from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDoubleSpinBox, 
    QLabel, QGroupBox, QTabWidget, QComboBox, QPushButton, QLineEdit,
    QSpinBox, QTableWidget, QTableWidgetItem, QScrollArea, QCheckBox, QMessageBox,
    QHeaderView
)

from core.aircraft import (
    BodyEnvelope,
    BodyObject,
    canard_rc_aircraft_preset,
    conventional_rc_aircraft_preset,
    twin_fin_rc_aircraft_preset,
)
from core.aircraft.references import Axis, SurfaceTransform
from core.aircraft.surfaces import LiftingSurface, SurfaceAnalysisSettings, SurfaceRole, SymmetryMode
from core.state import Project
from core.models.planform import PlanformGeometry, BodySection, ControlSurface
from core.models.twist_trim import TwistTrimParameters
from services.geometry import AeroSandboxService, SpanwiseSection

# --- Planform Tab ---
class PlanformTab(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.inputs: Dict[str, QDoubleSpinBox] = {}
        self.summary_labels: Dict[str, QLabel] = {}
        self._loading = False
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self._build_ui()
        self.update_from_project()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        
        # Scroll Area for inputs if screen is small
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Basic Planform Parameters
        form = QFormLayout()
        self._add_spin(form, "wing_area_m2", "Wing area [m^2]", 0.1, 500.0, 0.1, 3)
        self._add_spin(form, "aspect_ratio", "Aspect ratio", 1.0, 25.0, 0.1)
        self._add_spin(form, "taper_ratio", "Taper ratio", 0.05, 1.0, 0.01)
        self._add_spin(form, "sweep_le_deg", "Sweep LE [deg]", -45.0, 75.0, 0.5)
        self._add_spin(form, "dihedral_deg", "Dihedral [deg]", -10.0, 15.0, 0.1)
        self._add_spin(form, "front_spar_root_percent", "Front spar root [% chord]", 0.0, 100.0, 1.0)
        self._add_spin(form, "front_spar_tip_percent", "Front spar tip [% chord]", 0.0, 100.0, 1.0)
        self._add_spin(form, "rear_spar_root_percent", "Rear spar root [% chord]", 0.0, 100.0, 1.0)
        self._add_spin(form, "rear_spar_tip_percent", "Rear spar tip [% chord]", 0.0, 100.0, 1.0)
        self._add_spin(form, "rear_spar_span_percent", "Rear spar span [% half-span]", 0.0, 100.0, 1.0)

        self._add_spin(form, "center_chord_extension_percent", "Center chord ext [% root]", 0.0, 100.0, 1.0)
        self._add_spin(form, "center_section_span_percent", "Center span [% half-span]", 0.0, 100.0, 1.0)
        
        self.linear_check = QCheckBox("Linear center extension (Incompatible with BWB)")
        self.linear_check.toggled.connect(self._on_linear_toggled)
        form.addRow("", self.linear_check)
        
        self.snap_check = QCheckBox("Snap control surfaces to sections")
        self.snap_check.toggled.connect(self._on_snap_toggled)
        form.addRow("", self.snap_check)
        
        self._add_spin(form, "bwb_blend_span_percent", "BWB blend [% wing span]", 0.0, 50.0, 1.0)
        self._add_spin(form, "bwb_dihedral_deg", "BWB dihedral [deg]", -10.0, 10.0, 0.5)
        content_layout.addLayout(form)
        
        # BWB Body Sections Group
        bwb_group = QGroupBox("BWB Body Sections (optional)")
        bwb_layout = QVBoxLayout(bwb_group)
        self.bwb_table = QTableWidget()
        self.bwb_table.setColumnCount(5)
        self.bwb_table.setHorizontalHeaderLabels(["Y Pos [m]", "Chord [m]", "X Offset [m]", "Z Offset [m]", "Airfoil"])
        self.bwb_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bwb_table.itemChanged.connect(self._on_bwb_table_changed)
        bwb_layout.addWidget(self.bwb_table)
        
        bwb_btn_row = QHBoxLayout()
        bwb_add_btn = QPushButton("Add Section")
        bwb_add_btn.clicked.connect(self._add_bwb_section)
        bwb_btn_row.addWidget(bwb_add_btn)
        bwb_remove_btn = QPushButton("Remove Selected")
        bwb_remove_btn.clicked.connect(self._remove_bwb_section)
        bwb_btn_row.addWidget(bwb_remove_btn)
        bwb_btn_row.addStretch()
        bwb_layout.addLayout(bwb_btn_row)
        content_layout.addWidget(bwb_group)
        
        # Control Surfaces Group
        cs_group = QGroupBox("Control Surfaces")
        cs_layout = QVBoxLayout(cs_group)
        self.cs_table = QTableWidget()
        self.cs_table.setColumnCount(6)
        self.cs_table.setHorizontalHeaderLabels([
            "Name", "Type", "Span Start %", "Span End %", 
            "Chord Inboard %", "Chord Outboard %"
        ])
        self.cs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cs_table.itemChanged.connect(self._on_cs_table_changed)
        cs_layout.addWidget(self.cs_table)
        
        cs_btn_row = QHBoxLayout()
        cs_add_btn = QPushButton("Add Control Surface")
        cs_add_btn.clicked.connect(self._add_control_surface)
        cs_btn_row.addWidget(cs_add_btn)
        cs_remove_btn = QPushButton("Remove Selected")
        cs_remove_btn.clicked.connect(self._remove_control_surface)
        cs_btn_row.addWidget(cs_remove_btn)
        cs_btn_row.addStretch()
        cs_layout.addLayout(cs_btn_row)
        content_layout.addWidget(cs_group)

        scroll.setWidget(content)
        
        # Splitter or Layout: Left Inputs, Right Plot
        h_layout = QHBoxLayout()
        h_layout.addWidget(scroll, 1)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas, 2)
        
        summary_group = QGroupBox("Derived geometry")
        summary_form = QFormLayout(summary_group)
        for key in ["span", "half_span", "root_chord", "tip_chord", "mac", "dihedral", "actual_area", "actual_ar", "x_cg"]:
            self.summary_labels[key] = QLabel()
            label_text = key.replace("_", " ").title().replace("X Cg", "X CG")
            summary_form.addRow(label_text, self.summary_labels[key])
        right_layout.addWidget(summary_group, 0)
        
        h_layout.addLayout(right_layout, 2)
        main_layout.addLayout(h_layout)

    def _add_spin(self, form: QFormLayout, key: str, label: str, min_val: float, max_val: float, step: float, decimals: int = 2):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(lambda value, k=key: self._on_value_changed(k, value))
        form.addRow(label, spin)
        self.inputs[key] = spin

    def _on_linear_toggled(self, checked: bool) -> None:
        if self._loading: return
        self.project.wing.planform.center_extension_linear = checked
        self.project.wing.planform.reset_cache()
        self._update_ui()

    def _on_snap_toggled(self, checked: bool) -> None:
        if self._loading: return
        self.project.wing.planform.snap_to_sections = checked
        self.project.wing.planform.reset_cache()
        self._update_ui()

    def _on_value_changed(self, key: str, value: float) -> None:
        if self._loading: return
        setattr(self.project.wing.planform, key, float(value))
        self.project.wing.planform.reset_cache()
        self._update_ui()

    # --- BWB Body Sections Table Handlers ---
    def _add_bwb_section(self) -> None:
        plan = self.project.wing.planform
        # Add default section at next y position
        next_y = max([s.y_pos for s in plan.body_sections], default=-0.5) + 0.5
        new_section = BodySection(y_pos=next_y, chord=2.0, x_offset=0.0, z_offset=0.0)
        plan.body_sections.append(new_section)
        plan.reset_cache()
        # Clear optimized twist since geometry changed
        self.project.wing.optimized_twist_deg = None
        self._update_bwb_table()
        self._update_ui()

    def _remove_bwb_section(self) -> None:
        row = self.bwb_table.currentRow()
        if row >= 0 and row < len(self.project.wing.planform.body_sections):
            self.project.wing.planform.body_sections.pop(row)
            self.project.wing.planform.reset_cache()
            # Clear optimized twist since geometry changed
            self.project.wing.optimized_twist_deg = None
            self._update_bwb_table()
            self._update_ui()

    def _on_bwb_table_changed(self, item: QTableWidgetItem) -> None:
        if self._loading: return
        row = item.row()
        col = item.column()
        plan = self.project.wing.planform
        
        if row >= len(plan.body_sections):
            return
        
        try:
            section = plan.body_sections[row]
            value = item.text()
            if col == 0:  # Y Pos
                section.y_pos = float(value)
            elif col == 1:  # Chord
                section.chord = float(value)
            elif col == 2:  # X Offset
                section.x_offset = float(value)
            elif col == 3:  # Z Offset
                section.z_offset = float(value)
            elif col == 4:  # Airfoil
                section.airfoil = value
            plan.reset_cache()
            self._update_plot()
        except ValueError:
            pass  # Invalid number input, ignore

    def _update_bwb_table(self) -> None:
        self._loading = True
        plan = self.project.wing.planform
        self.bwb_table.setRowCount(len(plan.body_sections))
        for i, sec in enumerate(plan.body_sections):
            self.bwb_table.setItem(i, 0, QTableWidgetItem(f"{sec.y_pos:.3f}"))
            self.bwb_table.setItem(i, 1, QTableWidgetItem(f"{sec.chord:.3f}"))
            self.bwb_table.setItem(i, 2, QTableWidgetItem(f"{sec.x_offset:.3f}"))
            self.bwb_table.setItem(i, 3, QTableWidgetItem(f"{sec.z_offset:.3f}"))
            self.bwb_table.setItem(i, 4, QTableWidgetItem(sec.airfoil or ""))
        self._loading = False

    # --- Control Surfaces Table Handlers ---
    def _add_control_surface(self) -> None:
        plan = self.project.wing.planform
        new_cs = ControlSurface(
            name=f"Surface{len(plan.control_surfaces)+1}",
            surface_type="Elevon",
            span_start_percent=60.0,
            span_end_percent=100.0,
            hinge_rel_height=0.0
        )
        plan.control_surfaces.append(new_cs)
        self._update_cs_table()
        self._update_ui()

    def _remove_control_surface(self) -> None:
        row = self.cs_table.currentRow()
        if row >= 0 and row < len(self.project.wing.planform.control_surfaces):
            self.project.wing.planform.control_surfaces.pop(row)
            self._update_cs_table()
            self._update_ui()

    def _on_cs_table_changed(self, item: QTableWidgetItem) -> None:
        if self._loading: return
        row = item.row()
        col = item.column()
        plan = self.project.wing.planform
        
        if row >= len(plan.control_surfaces):
            return
        
        try:
            cs = plan.control_surfaces[row]
            value = item.text()
            if col == 0:  # Name
                cs.name = value
            elif col == 1:  # Type
                cs.surface_type = value
            elif col == 2:  # Span Start %
                cs.span_start_percent = float(value)
            elif col == 3:  # Span End %
                cs.span_end_percent = float(value)
            elif col == 4:  # Chord Inboard %
                cs.chord_start_percent = float(value)
            elif col == 5:  # Chord Outboard %
                cs.chord_end_percent = float(value)
            self._update_plot()
        except ValueError:
            pass  # Invalid number input, ignore

    def _update_cs_table(self) -> None:
        self._loading = True
        plan = self.project.wing.planform
        self.cs_table.setRowCount(len(plan.control_surfaces))
        for i, cs in enumerate(plan.control_surfaces):
            self.cs_table.setItem(i, 0, QTableWidgetItem(cs.name))
            self.cs_table.setItem(i, 1, QTableWidgetItem(cs.surface_type))
            self.cs_table.setItem(i, 2, QTableWidgetItem(f"{cs.span_start_percent:.1f}"))
            self.cs_table.setItem(i, 3, QTableWidgetItem(f"{cs.span_end_percent:.1f}"))
            self.cs_table.setItem(i, 4, QTableWidgetItem(f"{cs.chord_start_percent:.1f}"))
            self.cs_table.setItem(i, 5, QTableWidgetItem(f"{cs.chord_end_percent:.1f}"))
        self._loading = False

    def sync_to_project(self) -> None:
        """Push all editable widgets into project state before saving."""
        if self.project is None:
            return
        plan = self.project.wing.planform
        for key, spin in self.inputs.items():
            setattr(plan, key, float(spin.value()))

        plan.center_extension_linear = bool(self.linear_check.isChecked())
        plan.snap_to_sections = bool(self.snap_check.isChecked())

        body_sections: List[BodySection] = []
        for row in range(self.bwb_table.rowCount()):
            try:
                body_sections.append(
                    BodySection(
                        y_pos=float(self.bwb_table.item(row, 0).text()),
                        chord=float(self.bwb_table.item(row, 1).text()),
                        x_offset=float(self.bwb_table.item(row, 2).text()),
                        z_offset=float(self.bwb_table.item(row, 3).text()),
                        airfoil=(self.bwb_table.item(row, 4).text() if self.bwb_table.item(row, 4) else None),
                    )
                )
            except (AttributeError, TypeError, ValueError):
                continue
        plan.body_sections = body_sections

        control_surfaces: List[ControlSurface] = []
        for row in range(self.cs_table.rowCount()):
            try:
                previous = plan.control_surfaces[row] if row < len(plan.control_surfaces) else ControlSurface()
                control_surfaces.append(
                    ControlSurface(
                        name=self.cs_table.item(row, 0).text(),
                        surface_type=self.cs_table.item(row, 1).text(),
                        span_start_percent=float(self.cs_table.item(row, 2).text()),
                        span_end_percent=float(self.cs_table.item(row, 3).text()),
                        chord_start_percent=float(self.cs_table.item(row, 4).text()),
                        chord_end_percent=float(self.cs_table.item(row, 5).text()),
                        hinge_rel_height=previous.hinge_rel_height,
                    )
                )
            except (AttributeError, TypeError, ValueError):
                continue
        plan.control_surfaces = control_surfaces
        plan.reset_cache()

    def update_from_project(self) -> None:
        self._loading = True
        plan = self.project.wing.planform
        for key, spin in self.inputs.items():
            current = getattr(plan, key)
            if abs(spin.value() - current) > 1e-6:
                spin.setValue(float(current))
        self.linear_check.setChecked(plan.center_extension_linear)
        self.snap_check.setChecked(plan.snap_to_sections)
        self._loading = False
        self._update_bwb_table()
        self._update_cs_table()
        self._update_ui()

    def _update_ui(self) -> None:
        plan = self.project.wing.planform
        
        # Get sections to calculate actual geometry including BWB
        service = AeroSandboxService(self.project)
        sections = service.spanwise_sections()
        
        # Calculate actual half-span from outermost section
        if sections:
            actual_half_span = sections[-1].y_m
            actual_span = 2 * actual_half_span
        else:
            actual_half_span = plan.half_span()
            actual_span = plan.span()
        
        self.summary_labels["span"].setText(f"{actual_span:.3f}")
        self.summary_labels["half_span"].setText(f"{actual_half_span:.3f}")
        self.summary_labels["root_chord"].setText(f"{plan.extended_root_chord():.3f}")
        self.summary_labels["tip_chord"].setText(f"{plan.tip_chord():.3f}")
        
        # Calculate Actual MAC using numerical integration
        if sections and plan.actual_area() > 0:
            # Trapezoidal integration of c^2 dy
            integral_c2 = 0.0
            for i in range(len(sections) - 1):
                s1 = sections[i]
                s2 = sections[i+1]
                dy = abs(s2.y_m - s1.y_m)
                # Average c^2 (approx) or Simpson's rule? Linear approx of c is c(y) = a*y + b
                # Integral of (ay+b)^2 = a^2 y^3/3 + ab y^2 + b^2 y
                # Let's just use trapezoidal on c^2 for simplicity: 0.5 * (c1^2 + c2^2) * dy
                avg_c2 = 0.5 * (s1.chord_m**2 + s2.chord_m**2)
                integral_c2 += avg_c2 * dy
            
            # MAC = (2 / S) * Integral(c^2 dy) over half span
            # sections cover half span
            actual_mac = (2.0 / plan.actual_area()) * integral_c2
            self.summary_labels["mac"].setText(f"{actual_mac:.3f}")
        else:
            self.summary_labels["mac"].setText(f"{plan.mean_aerodynamic_chord():.3f}")

        self.summary_labels["dihedral"].setText(f"{plan.dihedral_deg:.2f}")
        self.summary_labels["actual_area"].setText(f"{plan.actual_area():.3f}")
        self.summary_labels["actual_ar"].setText(f"{plan.actual_aspect_ratio():.3f}")
        
        if self.project.analysis.x_cg is not None:
            self.summary_labels["x_cg"].setText(f"{self.project.analysis.x_cg:.3f}")
        else:
            self.summary_labels["x_cg"].setText("-")
            
        self._update_plot()

    def _update_plot(self) -> None:
        service = AeroSandboxService(self.project)
        sections = service.spanwise_sections()
        plan = self.project.wing.planform
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not sections:
            ax.text(0.5, 0.5, "No sections defined", transform=ax.transAxes, ha="center", va="center")
            self.canvas.draw_idle()
            return

        # Extract spar parameters
        front_root = plan.front_spar_root_percent / 100.0
        front_tip = plan.front_spar_tip_percent / 100.0
        rear_root = plan.rear_spar_root_percent / 100.0
        rear_tip = plan.rear_spar_tip_percent / 100.0


        y_half = [sec.y_m for sec in sections]
        x_le = [sec.x_le_m for sec in sections]
        x_te = [sec.x_le_m + sec.chord_m for sec in sections]

        # Plot Wing Outline
        ax.plot(y_half, x_le, color="#1f77b4", linewidth=2)
        ax.plot(y_half, x_te, color="#1f77b4", linewidth=2)
        y_left = [-y for y in y_half]
        ax.plot(y_left, x_le, color="#1f77b4", linewidth=2)
        ax.plot(y_left, x_te, color="#1f77b4", linewidth=2)
        
        # Connect tips
        ax.plot([y_half[-1], y_half[-1]], [x_le[-1], x_te[-1]], color="#1f77b4", linewidth=2)
        ax.plot([y_left[-1], y_left[-1]], [x_le[-1], x_te[-1]], color="#1f77b4", linewidth=2)

        # Determine BWB outer boundary
        bwb_outer_y = 0.0
        if plan.body_sections:
            sorted_bwb = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_bwb[-1].y_pos if sorted_bwb else 0.0
        
        total_half_span = sections[-1].y_m if sections else 1.0
        wing_half_span = total_half_span - bwb_outer_y

        # Helper to interpolate section properties at a given y position
        def _section_at_y(y_target: float):
            """Interpolate section at given y position from sections list."""
            if not sections:
                return (0.0, 0.0, 1.0)  # x_le, chord, y
            if y_target <= sections[0].y_m:
                s = sections[0]
                return (s.x_le_m, s.chord_m, s.y_m)
            if y_target >= sections[-1].y_m:
                s = sections[-1]
                return (s.x_le_m, s.chord_m, s.y_m)
            for i in range(len(sections) - 1):
                s1, s2 = sections[i], sections[i+1]
                if s1.y_m <= y_target <= s2.y_m:
                    t = (y_target - s1.y_m) / (s2.y_m - s1.y_m) if s2.y_m != s1.y_m else 0
                    interp_x_le = s1.x_le_m + t * (s2.x_le_m - s1.x_le_m)
                    interp_chord = s1.chord_m + t * (s2.chord_m - s1.chord_m)
                    return (interp_x_le, interp_chord, y_target)
            return (sections[-1].x_le_m, sections[-1].chord_m, sections[-1].y_m)

        # --- Plot BWB Body Spars (if BWB sections exist) ---
        if plan.body_sections and bwb_outer_y > 0:
            sorted_body = sorted(plan.body_sections, key=lambda b: b.y_pos)
            if sorted_body:
                # Use only first and last points for a single straight member
                s_first = sorted_body[0]
                s_last = sorted_body[-1]
                
                # Calculate absolute X at junction to use for the entire BWB section
                # front
                xsi_f_junc = (plan.front_spar_root_percent * (1 - s_last.y_pos / total_half_span) + 
                              plan.front_spar_tip_percent * (s_last.y_pos / total_half_span)) / 100.0
                xf_junc = s_last.x_offset + s_last.chord * xsi_f_junc
                
                # BWB front spar: straight spanwise line (constant X = junction X)
                ax.plot([s_first.y_pos, s_last.y_pos], [xf_junc, xf_junc], color="#2ca02c", linewidth=1.8, linestyle="-")
                ax.plot([-s_first.y_pos, -s_last.y_pos], [xf_junc, xf_junc], color="#2ca02c", linewidth=1.8, linestyle="-")
                
                # rear
                xsi_r_junc = (plan.rear_spar_root_percent * (1 - s_last.y_pos / total_half_span) + 
                              plan.rear_spar_tip_percent * (s_last.y_pos / total_half_span)) / 100.0
                xr_junc = s_last.x_offset + s_last.chord * xsi_r_junc
                
                # BWB rear spar: straight spanwise line (constant X = junction X)
                ax.plot([s_first.y_pos, s_last.y_pos], [xr_junc, xr_junc], color="#d62728", linewidth=1.6, linestyle="-")
                ax.plot([-s_first.y_pos, -s_last.y_pos], [xr_junc, xr_junc], color="#d62728", linewidth=1.6, linestyle="-")


        # --- Plot Wing Spars (from BWB junction to tip) ---
        # Front Spar on wing portion
        wing_sections = [s for s in sections if s.y_m >= bwb_outer_y]
        if wing_sections:
            wing_front_pts = []
            for sec in wing_sections:
                # Interpolate front spar % along wing span
                wing_local_frac = (sec.y_m - bwb_outer_y) / wing_half_span if wing_half_span > 0 else 0
                frac = front_root + (front_tip - front_root) * wing_local_frac
                x_spar = sec.x_le_m + sec.chord_m * frac
                wing_front_pts.append((sec.y_m, x_spar))
            
            wy = [p[0] for p in wing_front_pts]
            wx = [p[1] for p in wing_front_pts]
            ax.plot(wy, wx, color="#2ca02c", linewidth=1.8, label="Front spar")
            ax.plot([-y for y in wy], wx, color="#2ca02c", linewidth=1.8)

        # --- Rear Spar (independent of control surfaces) ---
        # Rear spar runs from wing root (BWB junction) to rear_spar_span_percent
        rear_spar_end_y = bwb_outer_y + wing_half_span * (plan.rear_spar_span_percent / 100.0)
        
        # Always use interpolated straight line for spars as requested
        # Straight line from start to end
        x_le_start, chord_start, _ = _section_at_y(bwb_outer_y)
        x_le_end, chord_end, _ = _section_at_y(rear_spar_end_y)
        
        # Interpolate percentages at y positions
        frac_start = rear_root + (rear_tip - rear_root) * (bwb_outer_y / total_half_span)
        frac_end = rear_root + (rear_tip - rear_root) * (rear_spar_end_y / total_half_span)
        
        x_spar_start = x_le_start + chord_start * frac_start
        x_spar_end = x_le_end + chord_end * frac_end

        ax.plot([bwb_outer_y, rear_spar_end_y], [x_spar_start, x_spar_end], 
               color="#d62728", linewidth=1.6, label="Rear spar")
        ax.plot([-bwb_outer_y, -rear_spar_end_y], [x_spar_start, x_spar_end], 
               color="#d62728", linewidth=1.6)

        # --- Plot Control Surfaces from control_surfaces list ---
        # Chord percentages: chord_start_percent at span_start, chord_end_percent at span_end
        # Surface always extends to trailing edge (100% chord)
        cs_colors = ["#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8"]
        snap = plan.snap_to_sections
        
        for cs_idx, cs in enumerate(plan.control_surfaces):
            color = cs_colors[cs_idx % len(cs_colors)]
            
            # Span percentages are relative to TOTAL halfspan (including BWB)
            cs_start_y = total_half_span * (cs.span_start_percent / 100.0)
            cs_end_y = total_half_span * (cs.span_end_percent / 100.0)
            chord_inboard = cs.chord_start_percent / 100.0  # hinge at inboard edge
            chord_outboard = cs.chord_end_percent / 100.0   # hinge at outboard edge
            
            cs_points = []
            
            if snap:
                # Snapping logic: find sections closest to the requested start/end y
                y_positions = [s.y_m for s in sections]
                idx_start = np.argmin(np.abs(np.array(y_positions) - cs_start_y))
                idx_end = np.argmin(np.abs(np.array(y_positions) - cs_end_y))
                
                idx_min = min(idx_start, idx_end)
                idx_max = max(idx_start, idx_end)
                
                cs_sections = sections[idx_min : idx_max + 1]
                
                eff_start_y = sections[idx_min].y_m
                eff_end_y = sections[idx_max].y_m
                
                for sec in cs_sections:
                    # Interpolate hinge % along control surface span using actual snapped boundaries
                    span_range = eff_end_y - eff_start_y
                    if span_range > 1e-6:
                        local_frac = (sec.y_m - eff_start_y) / span_range
                    else:
                        local_frac = 0
                    hinge_pct = chord_inboard + (chord_outboard - chord_inboard) * local_frac
                    hinge_x = sec.x_le_m + sec.chord_m * hinge_pct
                    te_x = sec.x_le_m + sec.chord_m  # Always to TE
                    cs_points.append((sec.y_m, hinge_x, te_x))
            else:
                # Straight line - only start and end points (no interior sections)
                # Start boundary
                x_le_s, chord_s, _ = _section_at_y(cs_start_y)
                hinge_x_s = x_le_s + chord_s * chord_inboard
                te_x_s = x_le_s + chord_s  # Always to TE
                cs_points.append((cs_start_y, hinge_x_s, te_x_s))
                
                # End boundary only (straight line)
                x_le_e, chord_e, _ = _section_at_y(cs_end_y)
                hinge_x_e = x_le_e + chord_e * chord_outboard
                te_x_e = x_le_e + chord_e
                cs_points.append((cs_end_y, hinge_x_e, te_x_e))
            
            if cs_points:
                # Sort by y
                cs_points.sort(key=lambda p: p[0])
                
                cs_y = [p[0] for p in cs_points]
                cs_hinge = [p[1] for p in cs_points]
                cs_te = [p[2] for p in cs_points]
                
                # Plot hinge line
                ax.plot(cs_y, cs_hinge, color=color, linewidth=1.4, linestyle="--", label=cs.name)
                ax.plot([-y for y in cs_y], cs_hinge, color=color, linewidth=1.4, linestyle="--")
                
                # Fill control surface area
                ax.fill_between(cs_y, cs_hinge, cs_te, color=color, alpha=0.3)
                ax.fill_between([-y for y in reversed(cs_y)], list(reversed(cs_hinge)), list(reversed(cs_te)), color=color, alpha=0.3)

        # Plot Section Lines (chordwise grey lines)
        for yi, x_front, x_back in zip(y_half, x_le, x_te):
            ax.plot([yi, yi], [x_front, x_back], color="#999999", linewidth=0.6, alpha=0.7)
            ax.plot([-yi, -yi], [x_front, x_back], color="#999999", linewidth=0.6, alpha=0.7)

        # Plot Mean Aerodynamic Chord (MAC) using actual sections
        # Calculate MAC y-position and chord from actual geometry
        if sections and len(sections) >= 2:
            # Numerical integration for MAC y-position: y_mac = integral(y*c dy) / integral(c dy)
            integral_yc = 0.0
            integral_c = 0.0
            for i in range(len(sections) - 1):
                s1, s2 = sections[i], sections[i+1]
                dy = s2.y_m - s1.y_m
                avg_c = 0.5 * (s1.chord_m + s2.chord_m)
                avg_y = 0.5 * (s1.y_m + s2.y_m)
                integral_yc += avg_y * avg_c * dy
                integral_c += avg_c * dy
            
            if integral_c > 0:
                y_mac = integral_yc / integral_c
                # Interpolate section at y_mac
                x_le_mac, chord_mac, _ = _section_at_y(y_mac)
                
                ax.plot([y_mac, y_mac], [x_le_mac, x_le_mac + chord_mac], color="#000000", linewidth=2.0, label="MAC")
                ax.plot([-y_mac, -y_mac], [x_le_mac, x_le_mac + chord_mac], color="#000000", linewidth=2.0)
                ax.text(y_mac, x_le_mac + chord_mac + 0.02 * chord_mac, "MAC", color="#000000", fontsize=8, ha="center")

        # Plot CG Marker if available
        if self.project.analysis.x_cg is not None:
            x_cg = self.project.analysis.x_cg
            # Plot a standard CG symbol (circle with quadrants)
            # Simplified: A distinctive marker
            ax.plot(0, x_cg, marker='o', markersize=10, markeredgecolor='black', markerfacecolor='white', markeredgewidth=1.5, zorder=10, label="CG")
            ax.plot(0, x_cg, marker='+', markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
            ax.text(0, x_cg + 0.05, "CG", ha="center", va="bottom", fontsize=9, fontweight='bold')

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title("Planform View")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.legend(fontsize="small", loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()


# --- Twist & Trim Tab ---
class TwistTrimTab(QWidget):
    analysisRequested = pyqtSignal(str)
    dataChanged = pyqtSignal()

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.inputs: Dict[str, QDoubleSpinBox] = {}
        self.result_labels: Dict[str, QLabel] = {}
        self._loading = False
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self._build_ui()
        self.update_from_project()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        
        h_layout = QHBoxLayout()
        
        # Inputs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        
        self._add_spin(form, "gross_takeoff_weight_kg", "GTW [kg]", 0.1, 500.0, 0.5)
        self._add_spin(form, "cruise_altitude_m", "Cruise Alt [m]", 0.0, 20000.0, 100.0)
        self._add_spin(form, "cm0_root", "Cm0 root", -1.0, 1.0, 0.01, 3)
        self._add_spin(form, "cm0_tip", "Cm0 tip", -1.0, 1.0, 0.01, 3)
        self._add_spin(form, "zero_lift_aoa_root_deg", "Alpha_L0 root", -10.0, 15.0, 0.1)
        self._add_spin(form, "zero_lift_aoa_tip_deg", "Alpha_L0 tip", -10.0, 15.0, 0.1)
        self._add_spin(form, "cl_alpha_root_per_deg", "Cla root", 0.01, 0.2, 0.001, 4)
        self._add_spin(form, "cl_alpha_tip_per_deg", "Cla tip", 0.01, 0.2, 0.001, 4)
        self._add_spin(form, "design_cl", "Design Cl", 0.05, 2.0, 0.01)
        
        self.distribution_box = QComboBox()
        self.distribution_box.addItems(["Bell", "Elliptical"])
        self.distribution_box.currentTextChanged.connect(self._on_dist_changed)
        form.addRow("Lift Dist", self.distribution_box)

        self.structural_spanload_box = QComboBox()
        self.structural_spanload_box.addItems(["VLM", "Blind Hybrid BWB"])
        self.structural_spanload_box.currentTextChanged.connect(self._on_structural_spanload_changed)
        form.addRow("Struct Spanload", self.structural_spanload_box)
        
        self._add_spin(form, "static_margin_percent", "Static Margin %", -5.0, 30.0, 0.5)
        self._add_spin(form, "estimated_cl_max", "Est Cl max", 0.1, 5.0, 0.05)
        self._add_spin(form, "estimated_cl_max_speed", "Est Cl @ Vmax", 0.05, 3.0, 0.05)
        
        scroll.setWidget(content)
        h_layout.addWidget(scroll, 1)
        
        # Plots
        h_layout.addWidget(self.canvas, 2)
        main_layout.addLayout(h_layout)
        
        # Actions
        btn_row = QHBoxLayout()
        est_btn = QPushButton("Estimate from Airfoils")
        est_btn.clicked.connect(self._on_estimate)
        btn_row.addWidget(est_btn)
        
        opt_btn = QPushButton("Optimize Twist")
        opt_btn.clicked.connect(self._on_optimize_twist)
        btn_row.addWidget(opt_btn)
        
        analyze_btn = QPushButton("Run Analysis")
        analyze_btn.clicked.connect(self._on_analyze)
        btn_row.addWidget(analyze_btn)
        
        viz_btn = QPushButton("Visualize 3D Flow")
        viz_btn.clicked.connect(self._on_visualize_flow)
        btn_row.addWidget(viz_btn)
        
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

    def _add_spin(self, form: QFormLayout, key: str, label: str, min_val: float, max_val: float, step: float, decimals: int = 2):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(lambda value, k=key: self._on_value_changed(k, value))
        form.addRow(label, spin)
        self.inputs[key] = spin

    def _on_value_changed(self, key: str, value: float) -> None:
        if self._loading: return
        setattr(self.project.wing.twist_trim, key, float(value))
        self._update_plots()

    def _on_dist_changed(self, text: str) -> None:
        if self._loading: return
        self.project.wing.twist_trim.lift_distribution = text.lower()
        self._update_plots()

    def _on_structural_spanload_changed(self, text: str) -> None:
        if self._loading: return
        if text == "Blind Hybrid BWB":
            self.project.wing.twist_trim.structural_spanload_model = "blind_hybrid_bwb_body"
        else:
            self.project.wing.twist_trim.structural_spanload_model = "vlm"

    def sync_to_project(self) -> None:
        """Push all editable widgets into project state before saving."""
        if self.project is None:
            return
        params = self.project.wing.twist_trim
        for key, spin in self.inputs.items():
            setattr(params, key, float(spin.value()))
        params.lift_distribution = self.distribution_box.currentText().lower()
        params.structural_spanload_model = (
            "blind_hybrid_bwb_body"
            if self.structural_spanload_box.currentText() == "Blind Hybrid BWB"
            else "vlm"
        )

    def _on_estimate(self):
        service = AeroSandboxService(self.project)
        try:
            root_airfoil = self.project.wing.airfoil.root_airfoil
            tip_airfoil = self.project.wing.airfoil.tip_airfoil
            
            root_params = service.analyze_airfoil_parameters(root_airfoil)
            tip_params = service.analyze_airfoil_parameters(tip_airfoil)
            
            self.project.wing.twist_trim.cm0_root = root_params["cm0"]
            self.project.wing.twist_trim.cm0_tip = tip_params["cm0"]
            self.project.wing.twist_trim.zero_lift_aoa_root_deg = root_params["zero_lift_aoa_deg"]
            self.project.wing.twist_trim.zero_lift_aoa_tip_deg = tip_params["zero_lift_aoa_deg"]
            self.project.wing.twist_trim.cl_alpha_root_per_deg = root_params["cl_alpha_per_deg"]
            self.project.wing.twist_trim.cl_alpha_tip_per_deg = tip_params["cl_alpha_per_deg"]
            
            self.update_from_project()
            self.dataChanged.emit()
            QMessageBox.information(self, "Success", "Estimated parameters from airfoils.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Estimation failed: {e}")

    def _on_optimize_twist(self):
        service = AeroSandboxService(self.project)
        try:
            twist = service.calculate_optimized_twist()
            self.project.wing.optimized_twist_deg = twist
            self._update_plots()
            self.dataChanged.emit()
            QMessageBox.information(self, "Success", "Twist optimization complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Optimization failed: {e}")

    def _on_analyze(self):
        """Run full analysis and update all plots including Reynolds and circulation."""
        service = AeroSandboxService(self.project)
        try:
            # Run AeroBuildup to get full results including CG
            results = service.run_aero_buildup()
            
            # Store results in project state
            self.project.analysis.x_cg = results.get("x_cg")
            
            # Update plots
            self._update_plots()
            
            # Notify other tabs (e.g. PlanformTab for CG marker)
            self.dataChanged.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")

    def _on_visualize_flow(self):
        service = AeroSandboxService(self.project)
        try:
            # Show a loading message or wait cursor if possible, but this blocks GUI anyway
            QMessageBox.information(self, "Info", "Running VLM Analysis... Visualization window will open shortly.")
            service.visualize_flow()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed: {e}")

    def update_from_project(self) -> None:
        self._loading = True
        params = self.project.wing.twist_trim
        for key, spin in self.inputs.items():
            current = getattr(params, key)
            if abs(spin.value() - current) > 1e-6:
                spin.setValue(float(current))
        
        idx = 0 if params.lift_distribution == "bell" else 1
        if self.distribution_box.currentIndex() != idx:
            self.distribution_box.setCurrentIndex(idx)

        spanload_idx = 1 if getattr(params, "structural_spanload_model", "vlm") == "blind_hybrid_bwb_body" else 0
        if self.structural_spanload_box.currentIndex() != spanload_idx:
            self.structural_spanload_box.setCurrentIndex(spanload_idx)
            
        self._loading = False
        self._update_plots()

    def _update_plots(self) -> None:
        service = AeroSandboxService(self.project)
        distribution = service.spanwise_distribution()
        self.figure.clear()
        
        if not distribution:
            return

        # Create 2x2 subplot grid
        axes = self.figure.subplots(2, 2)
        ax_twist, ax_cl = axes[0]
        ax_reynolds, ax_gamma = axes[1]
        
        span = [d.section.span_fraction for d in distribution]
        
        # Twist
        ax_twist.plot(span, [d.total_twist_deg for d in distribution], label="Total", color="#1f77b4")
        ax_twist.plot(span, [d.geometric_twist_deg for d in distribution], label="Geometric", linestyle="--", color="#ff7f0e")
        ax_twist.set_ylabel("Twist [deg]")
        ax_twist.legend(fontsize="small")
        ax_twist.grid(True, alpha=0.3)
        ax_twist.set_title("Twist Distribution")
        
        # Cl
        ax_cl.plot(span, [d.cl_design for d in distribution], label="Design Cl", color="#2ca02c")
        ax_cl.plot(span, [d.cl_min_speed for d in distribution], label="Min Speed", linestyle="--", color="#d62728")
        ax_cl.set_ylabel("Cl")
        ax_cl.legend(fontsize="small")
        ax_cl.grid(True, alpha=0.3)
        ax_cl.set_title("Lift Coefficient")
        
        # Reynolds Number
        ax_reynolds.plot(span, [d.reynolds_trim for d in distribution], label="Cruise", color="#9467bd")
        ax_reynolds.plot(span, [d.reynolds_min for d in distribution], label="Min Speed", linestyle="--", color="#8c564b")
        ax_reynolds.set_ylabel("Reynolds Number")
        ax_reynolds.set_xlabel("Span Fraction")
        ax_reynolds.legend(fontsize="small")
        ax_reynolds.grid(True, alpha=0.3)
        ax_reynolds.set_title("Reynolds Number")
        ax_reynolds.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Lift Circulation (Gamma)
        ax_gamma.plot(span, [d.gamma_trim for d in distribution], label="Cruise", color="#e377c2")
        ax_gamma.set_ylabel("Circulation [m²/s]")
        ax_gamma.set_xlabel("Span Fraction")
        ax_gamma.legend(fontsize="small")
        ax_gamma.grid(True, alpha=0.3)
        ax_gamma.set_title("Lift Circulation")
        
        self.figure.tight_layout()
        self.canvas.draw_idle()


# --- Airfoil Tab ---
class AirfoilTab(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self._loading = False
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.table = QTableWidget()
        self._build_ui()
        self.update_from_project()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # Airfoil inputs - grouped by region
        form = QFormLayout()
        
        # BWB Airfoil (used for body sections)
        self.bwb_edit = QLineEdit()
        self.bwb_edit.editingFinished.connect(self._on_change)
        form.addRow("BWB Airfoil", self.bwb_edit)
        
        # Wing Airfoils
        self.root_edit = QLineEdit()
        self.root_edit.editingFinished.connect(self._on_change)
        form.addRow("Wing Root Airfoil", self.root_edit)
        
        self.tip_edit = QLineEdit()
        self.tip_edit.editingFinished.connect(self._on_change)
        form.addRow("Wing Tip Airfoil", self.tip_edit)
        
        layout.addLayout(form)
        
        # Section Details Table
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Index", "Type", "Airfoil", "Span %", "y [m]", "Chord [m]", "Twist [deg]"])
        layout.addWidget(self.table)
        
        # Num Sections control below table
        section_layout = QHBoxLayout()
        section_layout.addWidget(QLabel("Num Sections:"))
        self.sections_spin = QSpinBox()
        self.sections_spin.setRange(2, 100)
        self.sections_spin.valueChanged.connect(self._on_change)
        section_layout.addWidget(self.sections_spin)
        section_layout.addStretch()
        layout.addLayout(section_layout)
        
        layout.addWidget(self.canvas)

    def _on_change(self):
        if self._loading: return
        self.project.wing.airfoil.bwb_airfoil = self.bwb_edit.text()
        self.project.wing.airfoil.root_airfoil = self.root_edit.text()
        self.project.wing.airfoil.tip_airfoil = self.tip_edit.text()
        self.project.wing.airfoil.num_sections = self.sections_spin.value()
        self._update_ui()

    def sync_to_project(self) -> None:
        """Push all editable widgets into project state before saving."""
        if self.project is None:
            return
        self.project.wing.airfoil.bwb_airfoil = self.bwb_edit.text()
        self.project.wing.airfoil.root_airfoil = self.root_edit.text()
        self.project.wing.airfoil.tip_airfoil = self.tip_edit.text()
        self.project.wing.airfoil.num_sections = int(self.sections_spin.value())

    def update_from_project(self):
        self._loading = True
        self.bwb_edit.setText(self.project.wing.airfoil.bwb_airfoil)
        self.root_edit.setText(self.project.wing.airfoil.root_airfoil)
        self.tip_edit.setText(self.project.wing.airfoil.tip_airfoil)
        self.sections_spin.setValue(self.project.wing.airfoil.num_sections)
        self._loading = False
        self._update_ui()

    def _update_ui(self):
        self._update_table()
        self._update_plot()

    def _update_table(self):
        service = AeroSandboxService(self.project)
        sections = service.spanwise_sections()
        
        # Determine BWB outer y position and wing half-span
        plan = self.project.wing.planform
        bwb_outer_y = 0.0
        if plan.body_sections:
            sorted_bwb = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_bwb[-1].y_pos if sorted_bwb else 0.0
        
        wing_half_span = plan.half_span()
        total_half_span = bwb_outer_y + wing_half_span
        wing_tip_y = total_half_span
        
        self.table.setRowCount(len(sections))
        
        for i, sec in enumerate(sections):
            # Determine if BWB or Wing section (strict < so junction is Wing Root)
            is_bwb = sec.y_m < bwb_outer_y
            section_type = "BWB" if is_bwb else "Wing"
            
            # Determine airfoil for this section
            is_bwb_root = abs(sec.y_m) < 1e-6  # Centerline
            is_wing_root = abs(sec.y_m - bwb_outer_y) < 1e-6 and bwb_outer_y > 0  # Junction
            is_wing_tip = abs(sec.y_m - wing_tip_y) < 1e-6  # Tip
            
            if is_bwb and is_bwb_root:
                airfoil_label = f"BWB ({self.project.wing.airfoil.bwb_airfoil})"
            elif not is_bwb and is_wing_root:
                airfoil_label = f"Wing Root ({self.project.wing.airfoil.root_airfoil})"
            elif not is_bwb and is_wing_tip:
                airfoil_label = f"Wing Tip ({self.project.wing.airfoil.tip_airfoil})"
            else:
                airfoil_label = "Interpolated"
            
            self.table.setItem(i, 0, QTableWidgetItem(str(sec.index)))
            
            type_item = QTableWidgetItem(section_type)
            if is_bwb:
                type_item.setBackground(QColor(173, 216, 230))  # Light blue for Type only
            self.table.setItem(i, 1, type_item)
            
            # Airfoil column - no background color
            self.table.setItem(i, 2, QTableWidgetItem(airfoil_label))
            
            self.table.setItem(i, 3, QTableWidgetItem(f"{sec.span_fraction * 100:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{sec.y_m:.3f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{sec.chord_m:.3f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{sec.twist_deg:.2f}"))
        
        self.table.resizeColumnsToContents()

    def _update_plot(self):
        service = AeroSandboxService(self.project)
        sections = service.spanwise_sections()
        
        # Determine BWB outer y position
        plan = self.project.wing.planform
        bwb_outer_y = 0.0
        if plan.body_sections:
            sorted_bwb = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_bwb[-1].y_pos if sorted_bwb else 0.0
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        for section in sections:
            coords = section.airfoil.repanel(n_points_per_side=60).coordinates
            chord = section.chord_m
            theta = -math.radians(section.twist_deg)
            
            x = coords[:, 0] * chord
            z = coords[:, 1] * chord
            
            # Rotate
            x_rot = x * math.cos(theta) - z * math.sin(theta)
            z_rot = x * math.sin(theta) + z * math.cos(theta)
            
            # Translate
            x_global = x_rot + section.x_le_m
            z_global = z_rot + section.z_m
            
            # Color based on section type (strict < so junction is Wing Root)
            is_bwb = section.y_m < bwb_outer_y
            color = "#1f77b4" if is_bwb else "black"  # Blue for BWB, black for wing
            linewidth = 1.0 if is_bwb else 0.5
            alpha = 0.7 if is_bwb else 0.5
            
            ax.plot(x_global, z_global, color=color, alpha=alpha, linewidth=linewidth)
            
        ax.set_aspect("equal")
        ax.set_title("Wing Loft (Side View) - Blue: BWB, Black: Wing")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        self.figure.tight_layout()
        self.canvas.draw_idle()


# --- Performance Tab ---
class PerformanceTab(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.labels: Dict[str, QLabel] = {}
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.last_polars = {}
        
        # State for interactive click annotations
        self._alphas = None
        self._data_by_ax = {}  # {ax: {"cruise": [...], "takeoff": [...]}}
        self._label_by_ax = {}  # {ax: "CL"}
        self._click_artists_by_ax = {}
        self._click_cid = None
        
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Calculate Button & Toggle
        btn_layout = QHBoxLayout()
        calc_btn = QPushButton("Calculate Performance")
        calc_btn.clicked.connect(self._calculate)
        btn_layout.addWidget(calc_btn)
        
        self.force_toggle = QCheckBox("Show Forces/Moments")
        self.force_toggle.toggled.connect(lambda: self._update_plots(self.last_polars))
        btn_layout.addWidget(self.force_toggle)
        
        btn_layout.addStretch()
        content_layout.addLayout(btn_layout)
        
        # Cruise Group
        cruise_group = QGroupBox("Cruise Condition (L = W at Actual CL)")
        cruise_form = QFormLayout(cruise_group)
        self._add_label(cruise_form, "cruise_velocity", "Velocity [m/s]")
        self._add_label(cruise_form, "cruise_alpha", "Angle of Attack [deg]")
        self._add_label(cruise_form, "cruise_cl", "Lift Coefficient (CL)")
        self._add_label(cruise_form, "cruise_cd", "Drag Coefficient (CD)")
        self._add_label(cruise_form, "cruise_pressure_drag_delta_cd", "BWB Pressure Drag dCD")
        self._add_label(cruise_form, "cruise_cm", "Pitching Moment (Cm)")
        self._add_label(cruise_form, "cruise_l_d", "Corrected Lift-to-Drag (L/D)")
        self._add_label(cruise_form, "cruise_l_d_uncorrected", "AeroBuildup L/D")
        content_layout.addWidget(cruise_group)
        
        # Takeoff Group
        takeoff_group = QGroupBox("Takeoff Condition (1.2 x V_stall, L > W)")
        takeoff_form = QFormLayout(takeoff_group)
        self._add_label(takeoff_form, "takeoff_velocity", "Velocity [m/s]")
        self._add_label(takeoff_form, "takeoff_alpha", "Angle of Attack [deg]")
        self._add_label(takeoff_form, "takeoff_cl", "Lift Coefficient (CL)")
        self._add_label(takeoff_form, "takeoff_cd", "Drag Coefficient (CD)")
        self._add_label(takeoff_form, "takeoff_pressure_drag_delta_cd", "BWB Pressure Drag dCD")
        self._add_label(takeoff_form, "takeoff_cm", "Pitching Moment (Cm)")
        self._add_label(takeoff_form, "takeoff_l_d", "Corrected Lift-to-Drag (L/D)")
        self._add_label(takeoff_form, "takeoff_l_d_uncorrected", "AeroBuildup L/D")
        content_layout.addWidget(takeoff_group)
        
        # Graphs Area
        content_layout.addWidget(self.canvas)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _add_label(self, form: QFormLayout, key: str, label_text: str):
        label = QLabel("-")
        form.addRow(label_text, label)
        self.labels[key] = label

    def _calculate(self):
        service = AeroSandboxService(self.project)
        try:
            # 1. Calculate Metrics
            metrics = service.calculate_performance_metrics()
            self.project.analysis.performance_metrics = metrics
            self._update_labels(metrics)
            
            # 2. Calculate Polars for Graphs
            polars = service.calculate_performance_polars()
            self.last_polars = polars
            self._update_plots(polars)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Performance calculation failed: {e}")
            import traceback
            traceback.print_exc()

    def _update_labels(self, metrics: dict):
        for key, label in self.labels.items():
            val = metrics.get(key)
            if val is not None:
                if isinstance(val, (float, int)):
                    label.setText(f"{val:.4f}")
                else:
                    label.setText(str(val))
            else:
                label.setText("-")

    def _update_plots(self, polars: dict):
        self.figure.clear()
        
        # Clear click state
        self._data_by_ax = {}
        self._label_by_ax = {}
        self._click_artists_by_ax = {}
        
        if not polars:
            return

        alphas = polars["alpha"]
        cruise = polars["cruise"]
        takeoff = polars["takeoff"]
        
        # Store alphas for interpolation
        self._alphas = alphas
        
        use_forces = self.force_toggle.isChecked()
        
        # Define what to plot based on toggle
        # (Key, Label, ForceKey, ForceLabel)
        plots_def = [
            ("L_D", "L/D [-]", "L_D", "L/D [-]"),  # L/D is same for both
            ("CL", "CL [-]", "L", "Lift [N]"),
            ("CD", "CD [-]", "D", "Drag [N]"),
            ("CM", "Cm [-]", "M", "Moment [Nm]"),
        ]
        
        axes = self.figure.subplots(2, 2)
        # Flatten for easy iteration
        ax_list = axes.flatten()
        
        for ax, (coef_key, coef_label, force_key, force_label) in zip(ax_list, plots_def):
            key = force_key if use_forces else coef_key
            label = force_label if use_forces else coef_label
            
            # Plot Cruise
            ax.plot(alphas, cruise[key], label="Cruise", color="#1f77b4")
            
            # Plot Takeoff
            ax.plot(alphas, takeoff[key], label="Takeoff", color="#ff7f0e", linestyle="--")
            
            ax.set_xlabel("Alpha [deg]")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small")
            
            # Store data for click handling
            self._data_by_ax[ax] = {"cruise": cruise[key], "takeoff": takeoff[key]}
            self._label_by_ax[ax] = label.split(" ")[0]  # Just "CL", "CD", etc.
            
        self.figure.tight_layout()
        
        # Connect click event if not already connected
        if self._click_cid is None:
            self._click_cid = self.canvas.mpl_connect('button_press_event', self._on_click_plot)
        
        self.canvas.draw_idle()
    
    def _on_click_plot(self, event):
        """Handle click on performance plots to show value annotations."""
        if event.inaxes not in self._data_by_ax:
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        if x is None or self._alphas is None:
            return
        
        # Get data for this axis
        data = self._data_by_ax[ax]
        label = self._label_by_ax.get(ax, "Value")
        alphas = self._alphas
        
        # Interpolate to find values at clicked alpha
        cruise_val = np.interp(x, alphas, data["cruise"])
        takeoff_val = np.interp(x, alphas, data["takeoff"])
        
        # Remove old annotations for this axis
        if ax in self._click_artists_by_ax:
            for artist in self._click_artists_by_ax[ax]:
                try:
                    artist.remove()
                except:
                    pass
        self._click_artists_by_ax[ax] = []
        
        # Add marker at clicked position
        marker, = ax.plot([x], [y], 'yo', markeredgecolor='k', markersize=8, zorder=10)
        self._click_artists_by_ax[ax].append(marker)
        
        # Add annotation text
        txt = f"Alpha={x:.2f}°\nCruise {label}={cruise_val:.4f}\nTakeoff {label}={takeoff_val:.4f}"
        ann = ax.annotate(
            txt, (x, y), 
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="w", alpha=0.9, edgecolor='gray'),
            fontsize=8
        )
        self._click_artists_by_ax[ax].append(ann)
        
        self.canvas.draw_idle()
    
    def update_from_project(self):
        pass  # Nothing to update until calculated


class LiftingSurfacesTab(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._loading = False
        self._build_ui()
        self.update_from_project()

    def _build_ui(self):
        root = QVBoxLayout(self)

        h_layout = QHBoxLayout()
        root.addLayout(h_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        toolbar = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Flying Wing", "Conventional", "Canard", "Twin Fin"])
        toolbar.addWidget(QLabel("Preset"))
        toolbar.addWidget(self.preset_combo)
        preset_btn = QPushButton("Apply")
        preset_btn.clicked.connect(self._apply_preset)
        toolbar.addWidget(preset_btn)
        add_btn = QPushButton("Add Surface")
        add_btn.clicked.connect(self._add_surface)
        toolbar.addWidget(add_btn)
        add_vertical_btn = QPushButton("Add Vertical Surface")
        add_vertical_btn.clicked.connect(self._add_vertical_surface)
        toolbar.addWidget(add_vertical_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_surface)
        toolbar.addWidget(remove_btn)
        toolbar.addStretch()
        content_layout.addLayout(toolbar)

        selector = QHBoxLayout()
        selector.addWidget(QLabel("Active surface"))
        self.surface_combo = QComboBox()
        self.surface_combo.currentIndexChanged.connect(self._on_surface_selected)
        selector.addWidget(self.surface_combo, 1)
        self.main_wing_check = QCheckBox("Main wing")
        self.main_wing_check.toggled.connect(self._on_role_flags_changed)
        selector.addWidget(self.main_wing_check)
        self.trim_check = QCheckBox("For trim")
        self.trim_check.toggled.connect(self._on_role_flags_changed)
        selector.addWidget(self.trim_check)
        content_layout.addLayout(selector)

        identity_group = QGroupBox("Surface")
        identity_form = QFormLayout(identity_group)
        self.uid_edit = QLineEdit()
        self.name_edit = QLineEdit()
        self.role_combo = QComboBox()
        self.role_combo.addItems([
            SurfaceRole.MAIN_WING.value,
            SurfaceRole.HORIZONTAL_TAIL.value,
            SurfaceRole.CANARD.value,
            SurfaceRole.STABILATOR.value,
            SurfaceRole.VERTICAL_TAIL.value,
            SurfaceRole.FIN.value,
            SurfaceRole.WINGLET.value,
        ])
        self.role_combo.currentIndexChanged.connect(self._on_role_combo_changed)
        self.symmetry_combo = QComboBox()
        self.symmetry_combo.addItems([s.value for s in SymmetryMode])
        self.axis_combo = QComboBox()
        self.axis_combo.addItems([a.value for a in Axis])
        self.parent_combo = QComboBox()
        self.vertical_mode_combo = QComboBox()
        self.vertical_mode_combo.addItem("Full Geometry", "full_geometry")
        self.vertical_mode_combo.addItem("Flat Plate", "flat_plate")
        self.active_check = QCheckBox("Active")
        self.x_spin = _double_spin(-20.0, 20.0, 0.01, 3)
        self.y_spin = _double_spin(-20.0, 20.0, 0.01, 3)
        self.z_spin = _double_spin(-20.0, 20.0, 0.01, 3)
        self.incidence_spin = _double_spin(-20.0, 20.0, 0.1, 2)
        self.area_spin = _double_spin(0.001, 1000.0, 0.01, 4)
        self.ar_spin = _double_spin(0.1, 50.0, 0.1, 3)
        self.taper_spin = _double_spin(0.01, 2.0, 0.01, 3)
        self.chord_mode_combo = QComboBox()
        self.chord_mode_combo.addItem("Linear taper", "linear_taper")
        self.chord_mode_combo.addItem("Elliptical chord lift", "elliptical")
        self.chord_mode_combo.addItem("Bell chord lift", "bell")
        self.tip_floor_spin = _double_spin(0.0, 50.0, 1.0, 2)
        self.split_chord_offsets_check = QCheckBox("Split chord offsets LE/TE")
        self.sweep_spin = _double_spin(-80.0, 80.0, 0.5, 2)
        self.dihedral_spin = _double_spin(-45.0, 45.0, 0.5, 2)
        self.front_spar_root_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.front_spar_tip_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.rear_spar_root_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.rear_spar_tip_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.rear_spar_span_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.center_ext_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.center_span_spin = _double_spin(0.0, 100.0, 1.0, 2)
        self.bwb_blend_spin = _double_spin(0.0, 50.0, 1.0, 2)
        self.bwb_dihedral_spin = _double_spin(-20.0, 20.0, 0.5, 2)
        self.sections_spin = QSpinBox()
        self.sections_spin.setRange(2, 200)
        self.sections_spin.setSingleStep(1)
        self.center_linear_check = QCheckBox("Linear center extension")
        self.snap_check = QCheckBox("Snap sections")

        airfoil_group = QGroupBox("Airfoils")
        airfoil_form = QFormLayout(airfoil_group)
        self.bwb_airfoil_edit = QLineEdit()
        self.root_airfoil_edit = QLineEdit()
        self.tip_airfoil_edit = QLineEdit()
        airfoil_form.addRow("BWB airfoil", self.bwb_airfoil_edit)
        airfoil_form.addRow("Root airfoil", self.root_airfoil_edit)
        airfoil_form.addRow("Tip airfoil", self.tip_airfoil_edit)
        content_layout.addWidget(airfoil_group)

        for label, widget in (
            ("UID", self.uid_edit),
            ("Name", self.name_edit),
            ("Role", self.role_combo),
            ("Active", self.active_check),
            ("Symmetry", self.symmetry_combo),
            ("Local span axis", self.axis_combo),
            ("Attached to", self.parent_combo),
            ("Vertical geometry", self.vertical_mode_combo),
            ("Origin X [m]", self.x_spin),
            ("Origin Y [m]", self.y_spin),
            ("Origin Z [m]", self.z_spin),
            ("Incidence [deg]", self.incidence_spin),
        ):
            identity_form.addRow(label, widget)
        content_layout.addWidget(identity_group)

        planform_group = QGroupBox("Planform Parameters")
        planform_form = QFormLayout(planform_group)
        for label, widget in (
            ("Area [m^2]", self.area_spin),
            ("Aspect ratio", self.ar_spin),
            ("Taper ratio", self.taper_spin),
            ("Chord setup", self.chord_mode_combo),
            ("Tip chord floor [%]", self.tip_floor_spin),
            ("Chord offset split", self.split_chord_offsets_check),
            ("Sweep LE [deg]", self.sweep_spin),
            ("Dihedral [deg]", self.dihedral_spin),
            ("Front spar root [%]", self.front_spar_root_spin),
            ("Front spar tip [%]", self.front_spar_tip_spin),
            ("Rear spar root [%]", self.rear_spar_root_spin),
            ("Rear spar tip [%]", self.rear_spar_tip_spin),
            ("Rear spar span [%]", self.rear_spar_span_spin),
            ("Center chord ext [% root]", self.center_ext_spin),
            ("Center span [% half-span]", self.center_span_spin),
            ("Center extension", self.center_linear_check),
            ("BWB blend [% span]", self.bwb_blend_spin),
            ("BWB dihedral [deg]", self.bwb_dihedral_spin),
            ("Sections / ribs", self.sections_spin),
            ("Section snapping", self.snap_check),
        ):
            planform_form.addRow(label, widget)
        content_layout.addWidget(planform_group)

        bwb_group = QGroupBox("BWB Body Sections")
        bwb_layout = QVBoxLayout(bwb_group)
        self.bwb_table = QTableWidget()
        self.bwb_table.setColumnCount(5)
        self.bwb_table.setHorizontalHeaderLabels(["Y Pos [m]", "Chord [m]", "X Offset [m]", "Z Offset [m]", "Airfoil"])
        self.bwb_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bwb_table.itemChanged.connect(self._on_bwb_table_changed)
        bwb_layout.addWidget(self.bwb_table)
        bwb_btn_row = QHBoxLayout()
        bwb_add_btn = QPushButton("Add Section")
        bwb_add_btn.clicked.connect(self._add_bwb_section)
        bwb_btn_row.addWidget(bwb_add_btn)
        bwb_remove_btn = QPushButton("Remove Selected")
        bwb_remove_btn.clicked.connect(self._remove_bwb_section)
        bwb_btn_row.addWidget(bwb_remove_btn)
        bwb_btn_row.addStretch()
        bwb_layout.addLayout(bwb_btn_row)
        content_layout.addWidget(bwb_group)

        cs_group = QGroupBox("Control Surfaces")
        cs_layout = QVBoxLayout(cs_group)
        self.cs_table = QTableWidget()
        self.cs_table.setColumnCount(7)
        self.cs_table.setHorizontalHeaderLabels([
            "Name", "Type", "Span Start %", "Span End %",
            "Chord Inboard %", "Chord Outboard %", "Hinge Height"
        ])
        self.cs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cs_table.itemChanged.connect(self._on_cs_table_changed)
        cs_layout.addWidget(self.cs_table)
        cs_btn_row = QHBoxLayout()
        cs_add_btn = QPushButton("Add Control Surface")
        cs_add_btn.clicked.connect(self._add_control_surface)
        cs_btn_row.addWidget(cs_add_btn)
        cs_remove_btn = QPushButton("Remove Selected")
        cs_remove_btn.clicked.connect(self._remove_control_surface)
        cs_btn_row.addWidget(cs_remove_btn)
        cs_btn_row.addStretch()
        cs_layout.addLayout(cs_btn_row)
        content_layout.addWidget(cs_group)

        for widget in (
            self.uid_edit, self.name_edit, self.bwb_airfoil_edit, self.root_airfoil_edit, self.tip_airfoil_edit,
            self.symmetry_combo, self.axis_combo,
            self.parent_combo, self.vertical_mode_combo,
            self.active_check, self.x_spin, self.y_spin, self.z_spin, self.incidence_spin,
            self.area_spin, self.ar_spin, self.taper_spin, self.chord_mode_combo, self.tip_floor_spin,
            self.split_chord_offsets_check, self.sweep_spin, self.dihedral_spin,
            self.front_spar_root_spin, self.front_spar_tip_spin, self.rear_spar_root_spin,
            self.rear_spar_tip_spin, self.rear_spar_span_spin, self.center_ext_spin,
            self.center_span_spin, self.bwb_blend_spin, self.bwb_dihedral_spin,
            self.sections_spin, self.center_linear_check, self.snap_check,
        ):
            if hasattr(widget, "editingFinished"):
                widget.editingFinished.connect(self._write_current_surface)
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._write_current_surface)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._write_current_surface)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._write_current_surface)

        scroll.setWidget(content)
        h_layout.addWidget(scroll, 1)

        right_layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, 2)

        summary_group = QGroupBox("Derived Geometry")
        summary_form = QFormLayout(summary_group)
        self.summary_labels = {}
        for key, label in (
            ("span", "Span [m]"),
            ("half_span", "Half span [m]"),
            ("root_chord", "Root chord [m]"),
            ("tip_chord", "Tip chord [m]"),
            ("mac", "MAC [m]"),
            ("actual_area", "Actual area [m^2]"),
            ("actual_ar", "Actual AR"),
        ):
            self.summary_labels[key] = QLabel("-")
            summary_form.addRow(label, self.summary_labels[key])
        right_layout.addWidget(summary_group, 0)

        h_layout.addLayout(right_layout, 2)

    def update_from_project(self):
        self._loading = True
        current_uid = self._current_surface().uid if self._current_surface() else None
        self.surface_combo.clear()
        for surface in self._editable_surfaces():
            self.surface_combo.addItem(f"{surface.name} ({surface.uid})", surface.uid)
        if current_uid:
            idx = self.surface_combo.findData(current_uid)
            if idx >= 0:
                self.surface_combo.setCurrentIndex(idx)
        self._loading = False
        self._load_current_surface()

    def sync_to_project(self):
        self._write_current_surface()
        self._apply_all_winglet_constraints()
        self.project.sync_aircraft_main_wing_to_legacy()

    def _current_surface(self) -> Optional[LiftingSurface]:
        uid = self.surface_combo.currentData()
        for surface in self._editable_surfaces():
            if surface.uid == uid:
                return surface
        surfaces = self._editable_surfaces()
        return surfaces[0] if surfaces else None

    def _editable_surfaces(self) -> List[LiftingSurface]:
        return list(self.project.aircraft.surfaces)

    def _wing_parent_candidates(self, exclude_uid: str | None = None) -> List[LiftingSurface]:
        wing_roles = {
            SurfaceRole.MAIN_WING.value,
            SurfaceRole.HORIZONTAL_TAIL.value,
            SurfaceRole.CANARD.value,
            SurfaceRole.STABILATOR.value,
        }
        return [
            surface for surface in self.project.aircraft.surfaces
            if surface.uid != exclude_uid
            and _value(surface.role) in wing_roles
            and _value(surface.local_span_axis) not in (Axis.Z.value, Axis.NEG_Z.value)
        ]

    def _surface_by_uid(self, uid: str | None) -> Optional[LiftingSurface]:
        if not uid:
            return None
        for surface in self.project.aircraft.surfaces:
            if surface.uid == uid:
                return surface
        return None

    def _is_vertical_surface(self, surface: LiftingSurface) -> bool:
        return (
            _value(surface.role) in (SurfaceRole.VERTICAL_TAIL.value, SurfaceRole.FIN.value, SurfaceRole.WINGLET.value)
            or _value(surface.local_span_axis) in (Axis.Z.value, Axis.NEG_Z.value)
        )

    def _refresh_parent_combo(self, surface: LiftingSurface) -> None:
        current_parent = surface.transform.parent_uid
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem("None", "")
        for candidate in self._wing_parent_candidates(exclude_uid=surface.uid):
            self.parent_combo.addItem(f"{candidate.name} ({candidate.uid})", candidate.uid)
        idx = self.parent_combo.findData(current_parent or "")
        if idx < 0 and _value(surface.role) == SurfaceRole.WINGLET.value and self.parent_combo.count() > 1:
            idx = 1
            surface.transform = SurfaceTransform(
                origin_m=surface.transform.origin_m,
                orientation_euler_deg=surface.transform.orientation_euler_deg,
                parent_uid=self.parent_combo.itemData(idx),
            )
        self.parent_combo.setCurrentIndex(max(0, idx))
        self.parent_combo.blockSignals(False)

    def _update_vertical_controls(self, surface: LiftingSurface) -> None:
        is_vertical = self._is_vertical_surface(surface)
        is_winglet = _value(surface.role) == SurfaceRole.WINGLET.value
        self.parent_combo.setEnabled(is_winglet)
        self.vertical_mode_combo.setEnabled(is_vertical)
        self.area_spin.setEnabled(not is_winglet)
        chord_mode = self.chord_mode_combo.currentData() or "linear_taper"
        self.taper_spin.setEnabled(chord_mode == "linear_taper")
        self.tip_floor_spin.setEnabled(chord_mode != "linear_taper")
        self.split_chord_offsets_check.setEnabled(chord_mode != "linear_taper")

    def _apply_winglet_constraints(self, surface: LiftingSurface) -> None:
        if _value(surface.role) != SurfaceRole.WINGLET.value:
            return
        parent = self._surface_by_uid(surface.transform.parent_uid)
        if parent is None:
            return

        parent_plan = parent.planform
        parent_x, parent_y, parent_z = parent.transform.origin_m
        parent_half_span = parent_plan.half_span()
        parent_tip_chord = parent_plan.tip_chord()
        parent_tip_x = parent_x + math.tan(math.radians(parent_plan.sweep_le_deg)) * parent_half_span + parent_plan.leading_edge_offset_at_span_fraction(1.0)
        parent_tip_y = parent_y + parent_half_span
        parent_tip_z = parent_z + math.tan(math.radians(parent_plan.dihedral_deg)) * parent_half_span

        surface.transform = SurfaceTransform(
            origin_m=(parent_tip_x, parent_tip_y, parent_tip_z),
            orientation_euler_deg=surface.transform.orientation_euler_deg,
            parent_uid=parent.uid,
        )
        surface.local_span_axis = Axis.Z.value
        surface.symmetry = SymmetryMode.MIRRORED_ABOUT_XZ.value

        taper = max(0.01, surface.planform.taper_ratio)
        aspect_ratio = max(0.1, surface.planform.aspect_ratio)
        surface.planform.taper_ratio = taper
        surface.planform.aspect_ratio = aspect_ratio
        surface.planform.center_chord_extension_percent = 0.0
        surface.planform.center_section_span_percent = 0.0
        surface.planform.wing_area_m2 = surface.planform.area_for_root_chord(parent_tip_chord)
        surface.planform.reset_cache()

    def _apply_all_winglet_constraints(self) -> None:
        for surface in self.project.aircraft.surfaces:
            self._apply_winglet_constraints(surface)

    def _sync_spin_values_from_surface(self, surface: LiftingSurface) -> None:
        self._loading = True
        self.symmetry_combo.setCurrentText(_value(surface.symmetry))
        self.axis_combo.setCurrentText(_value(surface.local_span_axis))
        self.x_spin.setValue(surface.transform.origin_m[0])
        self.y_spin.setValue(surface.transform.origin_m[1])
        self.z_spin.setValue(surface.transform.origin_m[2])
        self.area_spin.setValue(surface.planform.wing_area_m2)
        self.ar_spin.setValue(surface.planform.aspect_ratio)
        self.taper_spin.setValue(surface.planform.taper_ratio)
        _set_combo_by_data(self.chord_mode_combo, surface.planform.chord_distribution_mode)
        self.tip_floor_spin.setValue(surface.planform.chord_distribution_tip_floor_percent)
        self.split_chord_offsets_check.setChecked(surface.planform.split_chord_distribution_offsets)
        self.center_ext_spin.setValue(surface.planform.center_chord_extension_percent)
        self.center_span_spin.setValue(surface.planform.center_section_span_percent)
        self.center_linear_check.setChecked(surface.planform.center_extension_linear)
        self.sections_spin.setValue(max(2, int(getattr(surface.airfoils, "num_sections", 13) or 13)))
        self._loading = False

    def _on_surface_selected(self, *_args):
        if not self._loading:
            self._load_current_surface()

    def _load_current_surface(self):
        surface = self._current_surface()
        if surface is None:
            return
        self._apply_winglet_constraints(surface)
        self._loading = True
        self.uid_edit.setText(surface.uid)
        self.name_edit.setText(surface.name)
        _set_combo(self.role_combo, _value(surface.role))
        _set_combo(self.symmetry_combo, _value(surface.symmetry))
        _set_combo(self.axis_combo, _value(surface.local_span_axis))
        self._refresh_parent_combo(surface)
        _set_combo_by_data(self.vertical_mode_combo, surface.external_refs.get("geometry_mode", "full_geometry"))
        self.active_check.setChecked(surface.active)
        self.main_wing_check.setChecked(_value(surface.role) == SurfaceRole.MAIN_WING.value)
        self.trim_check.setChecked(_value(surface.role) in (SurfaceRole.HORIZONTAL_TAIL.value, SurfaceRole.CANARD.value, SurfaceRole.STABILATOR.value))
        self.x_spin.setValue(surface.transform.origin_m[0])
        self.y_spin.setValue(surface.transform.origin_m[1])
        self.z_spin.setValue(surface.transform.origin_m[2])
        self.incidence_spin.setValue(surface.incidence_deg)
        self.bwb_airfoil_edit.setText(surface.airfoils.bwb_airfoil)
        self.root_airfoil_edit.setText(surface.airfoils.root_airfoil)
        self.tip_airfoil_edit.setText(surface.airfoils.tip_airfoil)
        self.sections_spin.setValue(max(2, int(getattr(surface.airfoils, "num_sections", 13) or 13)))
        self.area_spin.setValue(surface.planform.wing_area_m2)
        self.ar_spin.setValue(surface.planform.aspect_ratio)
        self.taper_spin.setValue(surface.planform.taper_ratio)
        _set_combo_by_data(self.chord_mode_combo, surface.planform.chord_distribution_mode)
        self.tip_floor_spin.setValue(surface.planform.chord_distribution_tip_floor_percent)
        self.split_chord_offsets_check.setChecked(surface.planform.split_chord_distribution_offsets)
        self.sweep_spin.setValue(surface.planform.sweep_le_deg)
        self.dihedral_spin.setValue(surface.planform.dihedral_deg)
        self.front_spar_root_spin.setValue(surface.planform.front_spar_root_percent)
        self.front_spar_tip_spin.setValue(surface.planform.front_spar_tip_percent)
        self.rear_spar_root_spin.setValue(surface.planform.rear_spar_root_percent)
        self.rear_spar_tip_spin.setValue(surface.planform.rear_spar_tip_percent)
        self.rear_spar_span_spin.setValue(surface.planform.rear_spar_span_percent)
        self.center_ext_spin.setValue(surface.planform.center_chord_extension_percent)
        self.center_span_spin.setValue(surface.planform.center_section_span_percent)
        self.center_linear_check.setChecked(surface.planform.center_extension_linear)
        self.bwb_blend_spin.setValue(surface.planform.bwb_blend_span_percent)
        self.bwb_dihedral_spin.setValue(surface.planform.bwb_dihedral_deg)
        self.snap_check.setChecked(surface.planform.snap_to_sections)
        self._update_vertical_controls(surface)
        self._loading = False
        self._update_bwb_table()
        self._update_cs_table()
        self._update_summary_and_plot(surface)

    def _on_role_combo_changed(self, *_args):
        if self._loading:
            return
        surface = self._current_surface()
        if surface is None:
            return
        role = self.role_combo.currentText()
        if role in (SurfaceRole.VERTICAL_TAIL.value, SurfaceRole.FIN.value, SurfaceRole.WINGLET.value):
            _set_combo(self.axis_combo, Axis.Z.value)
            if role in (SurfaceRole.VERTICAL_TAIL.value, SurfaceRole.FIN.value):
                _set_combo(self.symmetry_combo, SymmetryMode.SINGLE_CENTERLINE.value)
        elif self.axis_combo.currentText() in (Axis.Z.value, Axis.NEG_Z.value):
            _set_combo(self.axis_combo, Axis.Y.value)
            _set_combo(self.symmetry_combo, SymmetryMode.MIRRORED_ABOUT_XZ.value)
        self._write_current_surface()

    def _write_current_surface(self, *_args):
        if self._loading:
            return
        surface = self._current_surface()
        if surface is None:
            return
        old_uid = surface.uid
        surface.uid = self.uid_edit.text().strip() or old_uid
        surface.name = self.name_edit.text().strip() or surface.uid
        surface.role = self.role_combo.currentText()
        surface.active = self.active_check.isChecked()
        surface.symmetry = self.symmetry_combo.currentText()
        surface.local_span_axis = self.axis_combo.currentText()
        parent_uid = self.parent_combo.currentData() or None
        if surface.role == SurfaceRole.WINGLET.value and parent_uid is None:
            candidates = self._wing_parent_candidates(exclude_uid=surface.uid)
            if candidates:
                parent_uid = candidates[0].uid
            else:
                QMessageBox.warning(self, "Winglet Attachment", "A winglet must be attached to another wing surface. Add or keep a wing surface before marking this surface as a winglet.")
                surface.role = SurfaceRole.FIN.value
                _set_combo(self.role_combo, SurfaceRole.FIN.value)
        surface.transform = SurfaceTransform(
            origin_m=(self.x_spin.value(), self.y_spin.value(), self.z_spin.value()),
            orientation_euler_deg=surface.transform.orientation_euler_deg,
            parent_uid=parent_uid,
        )
        surface.incidence_deg = self.incidence_spin.value()
        surface.planform.wing_area_m2 = self.area_spin.value()
        surface.planform.aspect_ratio = self.ar_spin.value()
        surface.planform.taper_ratio = self.taper_spin.value()
        surface.planform.chord_distribution_mode = self.chord_mode_combo.currentData() or "linear_taper"
        surface.planform.chord_distribution_tip_floor_percent = self.tip_floor_spin.value()
        surface.planform.split_chord_distribution_offsets = self.split_chord_offsets_check.isChecked()
        surface.planform.sweep_le_deg = self.sweep_spin.value()
        surface.planform.dihedral_deg = self.dihedral_spin.value()
        surface.planform.front_spar_root_percent = self.front_spar_root_spin.value()
        surface.planform.front_spar_tip_percent = self.front_spar_tip_spin.value()
        surface.planform.rear_spar_root_percent = self.rear_spar_root_spin.value()
        surface.planform.rear_spar_tip_percent = self.rear_spar_tip_spin.value()
        surface.planform.rear_spar_span_percent = self.rear_spar_span_spin.value()
        surface.planform.center_chord_extension_percent = self.center_ext_spin.value()
        surface.planform.center_section_span_percent = self.center_span_spin.value()
        surface.planform.center_extension_linear = self.center_linear_check.isChecked()
        surface.planform.bwb_blend_span_percent = self.bwb_blend_spin.value()
        surface.planform.bwb_dihedral_deg = self.bwb_dihedral_spin.value()
        surface.planform.snap_to_sections = self.snap_check.isChecked()
        surface.airfoils.bwb_airfoil = self.bwb_airfoil_edit.text().strip() or surface.airfoils.bwb_airfoil
        surface.airfoils.root_airfoil = self.root_airfoil_edit.text().strip() or surface.airfoils.root_airfoil
        surface.airfoils.tip_airfoil = self.tip_airfoil_edit.text().strip() or surface.airfoils.tip_airfoil
        surface.airfoils.num_sections = int(self.sections_spin.value())
        self._sync_current_bwb_sections()
        surface.planform.reset_cache()
        if self._is_vertical_surface(surface):
            surface.external_refs["geometry_mode"] = self.vertical_mode_combo.currentData() or "full_geometry"
        else:
            surface.external_refs.pop("geometry_mode", None)
        self._sync_current_control_surfaces()
        if old_uid != surface.uid:
            self.update_from_project()
        else:
            self._apply_winglet_constraints(surface)
            self._refresh_parent_combo(surface)
            self._update_vertical_controls(surface)
            self._sync_spin_values_from_surface(surface)
            self._update_summary_and_plot(surface)

    def _on_role_flags_changed(self, *_args):
        if self._loading:
            return
        if self.main_wing_check.isChecked():
            _set_combo(self.role_combo, SurfaceRole.MAIN_WING.value)
        elif self.trim_check.isChecked() and self.role_combo.currentText() not in (
            SurfaceRole.HORIZONTAL_TAIL.value, SurfaceRole.CANARD.value, SurfaceRole.STABILATOR.value
        ):
            _set_combo(self.role_combo, SurfaceRole.HORIZONTAL_TAIL.value)
        self._write_current_surface()

    def _update_summary_and_plot(self, surface: LiftingSurface):
        self._apply_all_winglet_constraints()
        plan = surface.planform
        self.summary_labels["span"].setText(f"{plan.actual_span():.3f}")
        self.summary_labels["half_span"].setText(f"{0.5 * plan.actual_span():.3f}")
        self.summary_labels["root_chord"].setText(f"{plan.extended_root_chord():.3f}")
        self.summary_labels["tip_chord"].setText(f"{plan.tip_chord():.3f}")
        self.summary_labels["mac"].setText(f"{plan.mean_aerodynamic_chord():.3f}")
        self.summary_labels["actual_area"].setText(f"{plan.actual_area():.3f}")
        self.summary_labels["actual_ar"].setText(f"{plan.actual_aspect_ratio():.3f}")

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
        if self._is_vertical_surface(surface):
            for idx, candidate in enumerate(self.project.aircraft.surfaces):
                if not candidate.active:
                    continue
                color = colors[idx % len(colors)]
                selected = candidate.uid == surface.uid
                if self._is_vertical_surface(candidate):
                    self._plot_vertical_surface_side(ax, candidate, color=color, selected=selected)
                else:
                    self._plot_lifting_surface_side(ax, candidate, color=color, selected=selected)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Z [m]")
            ax.set_title("Aircraft Side View")
        else:
            for idx, candidate in enumerate(self.project.aircraft.surfaces):
                if not candidate.active:
                    continue
                color = colors[idx % len(colors)]
                selected = candidate.uid == surface.uid
                self._plot_surface_planform(ax, candidate, color=color, selected=selected)
            ax.set_xlabel("Y [m]")
            ax.set_ylabel("X [m]")
            ax.set_title("Aircraft Planform")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if self.project.aircraft.surfaces:
            ax.legend(fontsize="small", loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_surface_planform(self, ax, surface: LiftingSurface, color: str, selected: bool) -> None:
        plan = surface.planform
        origin_x, origin_y, _origin_z = surface.transform.origin_m
        symmetry = _value(surface.symmetry)
        axis = _value(surface.local_span_axis)
        linewidth = 2.6 if selected else 1.4
        alpha = 1.0 if selected else 0.65
        label = f"* {surface.name}" if selected else surface.name

        if axis in (Axis.Z.value, Axis.NEG_Z.value):
            mode = surface.external_refs.get("geometry_mode", "full_geometry")
            for station_index, y_station in enumerate(self._vertical_y_stations(surface)):
                station_label = label if station_index == 0 else f"{surface.name} mirror"
                if mode == "flat_plate":
                    x0 = origin_x
                    x1 = origin_x + plan.root_chord()
                    linestyle = "-" if station_index == 0 else "--"
                    ax.plot([y_station, y_station], [x0, x1], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=station_label)
                else:
                    self._plot_vertical_surface_top_sections(
                        ax,
                        surface,
                        y_station=y_station,
                        color=color,
                        selected=selected,
                        label=station_label,
                        mirrored=station_index > 0,
                    )
            return

        sections = self._surface_sections(surface)
        if not sections:
            return

        sides = [1.0]
        if symmetry == SymmetryMode.MIRRORED_ABOUT_XZ.value:
            sides = [1.0, -1.0]

        for side_index, y_sign in enumerate(sides):
            y_half = [origin_y + y_sign * s["y"] for s in sections]
            x_le = [origin_x + s["x_le"] for s in sections]
            x_te = [origin_x + s["x_le"] + s["chord"] for s in sections]
            ax.plot(y_half, x_le, color=color, linewidth=linewidth, alpha=alpha, label=label if side_index == 0 else None)
            ax.plot(y_half, x_te, color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([y_half[-1], y_half[-1]], [x_le[-1], x_te[-1]], color=color, linewidth=linewidth, alpha=alpha)

            # Section/rib placement lines.
            for y, xf, xb in zip(y_half, x_le, x_te):
                ax.plot([y, y], [xf, xb], color="#999999", linewidth=0.6, alpha=0.7)

            # Spars use the original Planform tab colors.
            front = [origin_x + s["x_le"] + s["chord"] * s["front_frac"] for s in sections]
            rear_sections = [s for s in sections if s["eta"] <= plan.rear_spar_span_percent / 100.0 + 1e-9]
            rear_y = [origin_y + y_sign * s["y"] for s in rear_sections]
            rear = [origin_x + s["x_le"] + s["chord"] * s["rear_frac"] for s in rear_sections]
            ax.plot(y_half, front, color="#2ca02c", linewidth=1.8, alpha=alpha, label="Front spar" if selected and side_index == 0 else None)
            if rear_y:
                ax.plot(rear_y, rear, color="#d62728", linewidth=1.6, alpha=alpha, label="Rear spar" if selected and side_index == 0 else None)

            # Control surfaces, filled like the original Planform tab.
            cs_colors = ["#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8"]
            total_half_span = sections[-1]["y"]
            for cs_idx, cs in enumerate(surface.control_surfaces):
                cs_color = cs_colors[cs_idx % len(cs_colors)]
                start_y = total_half_span * cs.span_start_percent / 100.0
                end_y = total_half_span * cs.span_end_percent / 100.0
                pts = [s for s in sections if start_y - 1e-9 <= s["y"] <= end_y + 1e-9]
                if not pts or pts[0]["y"] > start_y:
                    pts.insert(0, self._interp_surface_section(sections, start_y))
                if pts[-1]["y"] < end_y:
                    pts.append(self._interp_surface_section(sections, end_y))
                cs_y = []
                hinge_x = []
                te_x = []
                span_range = max(1e-9, end_y - start_y)
                for p in pts:
                    local = (p["y"] - start_y) / span_range
                    hinge_frac = (cs.chord_start_percent + (cs.chord_end_percent - cs.chord_start_percent) * local) / 100.0
                    cs_y.append(origin_y + y_sign * p["y"])
                    hinge_x.append(origin_x + p["x_le"] + p["chord"] * hinge_frac)
                    te_x.append(origin_x + p["x_le"] + p["chord"])
                ax.plot(cs_y, hinge_x, color=cs_color, linewidth=1.4, linestyle="--", alpha=alpha, label=cs.name if selected and side_index == 0 else None)
                ax.fill_between(cs_y, hinge_x, te_x, color=cs_color, alpha=0.3 if selected else 0.16)

        # MAC for selected surface.
        if selected and len(sections) >= 2:
            integral_yc = 0.0
            integral_c = 0.0
            for i in range(len(sections) - 1):
                s1, s2 = sections[i], sections[i + 1]
                dy = s2["y"] - s1["y"]
                avg_c = 0.5 * (s1["chord"] + s2["chord"])
                avg_y = 0.5 * (s1["y"] + s2["y"])
                integral_yc += avg_y * avg_c * dy
                integral_c += avg_c * dy
            if integral_c > 0.0:
                y_mac = integral_yc / integral_c
                sec = self._interp_surface_section(sections, y_mac)
                for y_sign in sides:
                    y = origin_y + y_sign * y_mac
                    x = origin_x + sec["x_le"]
                    ax.plot([y, y], [x, x + sec["chord"]], color="#000000", linewidth=2.0, label="MAC" if y_sign == sides[0] else None)

    def _vertical_y_stations(self, surface: LiftingSurface) -> List[float]:
        _x, y, _z = surface.transform.origin_m
        if _value(surface.symmetry) == SymmetryMode.MIRRORED_ABOUT_XZ.value and abs(y) > 1e-9:
            return [y, -y]
        return [y]

    def _plot_vertical_surface_top_sections(
        self,
        ax,
        surface: LiftingSurface,
        y_station: float,
        color: str,
        selected: bool,
        label: str,
        mirrored: bool,
    ) -> None:
        x0, _origin_y, _z0 = surface.transform.origin_m
        sections = self._surface_sections(surface)
        linewidth = 1.1 if selected else 0.75
        alpha = 0.85 if selected else 0.5
        linestyle = "--" if mirrored else "-"
        thickness_sign = -1.0 if mirrored else 1.0

        try:
            from core.naca_generator.naca456 import generate_naca_airfoil

            root_name = surface.airfoils.root_airfoil.lower().replace("naca", "").strip()
            tip_name = surface.airfoils.tip_airfoil.lower().replace("naca", "").strip()
            x_root, y_root = generate_naca_airfoil(root_name, n_points=80)
            x_tip, y_tip = generate_naca_airfoil(tip_name, n_points=80)
            step = max(1, len(sections) // 9)
            plotted_label = False
            incidence = math.radians(surface.incidence_deg)
            for section in sections[::step]:
                eta = section["eta"]
                chord = section["chord"]
                x_profile = x_root * (1.0 - eta) + x_tip * eta
                y_profile = y_root * (1.0 - eta) + y_tip * eta
                x_le = x0 + section["x_le"]
                plot_y = []
                plot_x = []
                for xi, yi in zip(x_profile, y_profile):
                    local_x = float(xi) * chord
                    local_y = float(yi) * chord
                    x_global = x_le + local_x * math.cos(incidence) - local_y * math.sin(incidence)
                    y_global = y_station + thickness_sign * (local_x * math.sin(incidence) + local_y * math.cos(incidence))
                    plot_y.append(y_global)
                    plot_x.append(x_global)
                ax.plot(
                    plot_y,
                    plot_x,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                    label=label if not plotted_label else None,
                )
                plotted_label = True
        except Exception:
            step = max(1, len(sections) // 9)
            plotted_label = False
            for section in sections[::step]:
                chord = section["chord"]
                thickness = 0.08 * chord
                x_le = x0 + section["x_le"]
                plot_y = [
                    y_station - thickness_sign * thickness / 2.0,
                    y_station + thickness_sign * thickness / 2.0,
                    y_station + thickness_sign * thickness / 2.0,
                    y_station - thickness_sign * thickness / 2.0,
                    y_station - thickness_sign * thickness / 2.0,
                ]
                plot_x = [x_le, x_le, x_le + chord, x_le + chord, x_le]
                ax.plot(plot_y, plot_x, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label if not plotted_label else None)
                plotted_label = True

    def _plot_vertical_surface_side(self, ax, surface: LiftingSurface, color: str, selected: bool) -> None:
        mode = surface.external_refs.get("geometry_mode", "full_geometry")
        x0, _y0, z0 = surface.transform.origin_m
        direction = 1.0 if _value(surface.local_span_axis) != Axis.NEG_Z.value else -1.0
        sections = self._surface_sections(surface)
        linewidth = 2.6 if selected else 1.4
        alpha = 1.0 if selected else 0.65
        label = f"* {surface.name}" if selected else surface.name

        z_vals = [z0 + direction * s["y"] for s in sections]
        x_le = [x0 + s["x_le"] for s in sections]
        x_te = [x0 + s["x_le"] + s["chord"] for s in sections]
        ax.plot(x_le, z_vals, color=color, linewidth=linewidth, alpha=alpha, label=label)
        ax.plot(x_te, z_vals, color=color, linewidth=linewidth, alpha=alpha)
        ax.plot([x_le[-1], x_te[-1]], [z_vals[-1], z_vals[-1]], color=color, linewidth=linewidth, alpha=alpha)
        if _value(surface.symmetry) == SymmetryMode.MIRRORED_ABOUT_XZ.value and abs(surface.transform.origin_m[1]) > 1e-9:
            ax.plot(x_le, z_vals, color=color, linewidth=max(1.0, linewidth - 0.7), alpha=0.95, linestyle="--", label=f"{surface.name} mirror")
            ax.plot(x_te, z_vals, color=color, linewidth=max(1.0, linewidth - 0.7), alpha=0.95, linestyle="--")
            ax.plot([x_le[-1], x_te[-1]], [z_vals[-1], z_vals[-1]], color=color, linewidth=max(1.0, linewidth - 0.7), alpha=0.95, linestyle="--")

        if mode == "flat_plate":
            ax.fill(x_le + list(reversed(x_te)), z_vals + list(reversed(z_vals)), color=color, alpha=0.12 if selected else 0.06)
            return

        for z, xf, xb in zip(z_vals, x_le, x_te):
            ax.plot([xf, xb], [z, z], color="#999999", linewidth=0.6, alpha=0.7)

        front = [x0 + s["x_le"] + s["chord"] * s["front_frac"] for s in sections]
        rear_sections = [s for s in sections if s["eta"] <= surface.planform.rear_spar_span_percent / 100.0 + 1e-9]
        rear_z = [z0 + direction * s["y"] for s in rear_sections]
        rear = [x0 + s["x_le"] + s["chord"] * s["rear_frac"] for s in rear_sections]
        ax.plot(front, z_vals, color="#2ca02c", linewidth=1.8, alpha=alpha, label="Front spar" if selected else None)
        if rear_z:
            ax.plot(rear, rear_z, color="#d62728", linewidth=1.6, alpha=alpha, label="Rear spar" if selected else None)

        cs_colors = ["#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8"]
        total_span = sections[-1]["y"]
        for cs_idx, cs in enumerate(surface.control_surfaces):
            start_z = total_span * cs.span_start_percent / 100.0
            end_z = total_span * cs.span_end_percent / 100.0
            pts = [s for s in sections if start_z - 1e-9 <= s["y"] <= end_z + 1e-9]
            if not pts or pts[0]["y"] > start_z:
                pts.insert(0, self._interp_surface_section(sections, start_z))
            if pts[-1]["y"] < end_z:
                pts.append(self._interp_surface_section(sections, end_z))
            hinge_x = []
            te_x = []
            cs_z = []
            span_range = max(1e-9, end_z - start_z)
            for p in pts:
                local = (p["y"] - start_z) / span_range
                hinge_frac = (cs.chord_start_percent + (cs.chord_end_percent - cs.chord_start_percent) * local) / 100.0
                hinge_x.append(x0 + p["x_le"] + p["chord"] * hinge_frac)
                te_x.append(x0 + p["x_le"] + p["chord"])
                cs_z.append(z0 + direction * p["y"])
            cs_color = cs_colors[cs_idx % len(cs_colors)]
            ax.plot(hinge_x, cs_z, color=cs_color, linewidth=1.4, linestyle="--", alpha=alpha, label=cs.name if selected else None)
            ax.fill_betweenx(cs_z, hinge_x, te_x, color=cs_color, alpha=0.3 if selected else 0.16)

    def _plot_lifting_surface_side(self, ax, surface: LiftingSurface, color: str, selected: bool) -> None:
        x0, _y0, z0 = surface.transform.origin_m
        sections = self._surface_sections(surface)
        linewidth = 1.2 if selected else 0.8
        alpha = 0.8 if selected else 0.45
        label = f"* {surface.name}" if selected else surface.name
        try:
            from core.naca_generator.naca456 import generate_naca_airfoil

            name = surface.airfoils.root_airfoil.lower().replace("naca", "").strip()
            x_airfoil, z_airfoil = generate_naca_airfoil(name, n_points=80)
            plotted_label = False
            step = max(1, len(sections) // 7)
            for section in sections[::step]:
                chord = section["chord"]
                x_le = x0 + section["x_le"]
                z_le = z0 + math.tan(math.radians(surface.planform.dihedral_deg)) * section["y"]
                twist = math.radians(surface.incidence_deg)
                xs = []
                zs = []
                for xi, zi in zip(x_airfoil, z_airfoil):
                    local_x = float(xi) * chord
                    local_z = float(zi) * chord
                    xs.append(x_le + local_x * math.cos(twist) - local_z * math.sin(twist))
                    zs.append(z_le + local_x * math.sin(twist) + local_z * math.cos(twist))
                ax.plot(xs, zs, linewidth=linewidth, color=color, alpha=alpha, label=label if not plotted_label else None)
                plotted_label = True
        except Exception:
            chord = surface.planform.root_chord()
            thickness = 0.08 * chord
            xs = [x0, x0 + chord, x0 + chord, x0, x0]
            zs = [z0 - thickness / 2.0, z0 - thickness / 2.0, z0 + thickness / 2.0, z0 + thickness / 2.0, z0 - thickness / 2.0]
            ax.plot(xs, zs, linewidth=linewidth, color=color, alpha=alpha, label=label)

    def _surface_sections(self, surface: LiftingSurface) -> List[dict]:
        plan = surface.planform
        n = max(3, int(getattr(surface.airfoils, "num_sections", 13) or 13))
        half_span = plan.half_span()
        root = plan.extended_root_chord()
        sections = []
        for i in range(n):
            eta = i / (n - 1)
            y = half_span * eta
            chord = plan.chord_at_span_fraction(eta)
            if eta == 0.0 and plan.center_chord_extension_percent > 0:
                chord = root
            x_le = math.tan(math.radians(plan.sweep_le_deg)) * y + plan.leading_edge_offset_at_span_fraction(eta)
            sections.append(
                {
                    "eta": eta,
                    "y": y,
                    "x_le": x_le,
                    "chord": chord,
                    "front_frac": (plan.front_spar_root_percent + (plan.front_spar_tip_percent - plan.front_spar_root_percent) * eta) / 100.0,
                    "rear_frac": (plan.rear_spar_root_percent + (plan.rear_spar_tip_percent - plan.rear_spar_root_percent) * eta) / 100.0,
                }
            )
        return sections

    def _interp_surface_section(self, sections: List[dict], y: float) -> dict:
        if y <= sections[0]["y"]:
            return dict(sections[0])
        if y >= sections[-1]["y"]:
            return dict(sections[-1])
        for i in range(len(sections) - 1):
            s1, s2 = sections[i], sections[i + 1]
            if s1["y"] <= y <= s2["y"]:
                t = (y - s1["y"]) / max(1e-9, s2["y"] - s1["y"])
                return {
                    key: (s1[key] + (s2[key] - s1[key]) * t)
                    for key in ("eta", "y", "x_le", "chord", "front_frac", "rear_frac")
                }
        return dict(sections[-1])

    def _update_bwb_table(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        self._loading = True
        sections = surface.planform.body_sections
        self.bwb_table.setRowCount(len(sections))
        for row, section in enumerate(sections):
            values = [section.y_pos, section.chord, section.x_offset, section.z_offset, section.airfoil]
            for col, value in enumerate(values):
                self.bwb_table.setItem(row, col, QTableWidgetItem(f"{value:.3f}" if isinstance(value, float) else str(value)))
        self._loading = False

    def _sync_current_bwb_sections(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        sections = []
        for row in range(self.bwb_table.rowCount()):
            try:
                sections.append(
                    BodySection(
                        y_pos=float(_table_text(self.bwb_table, row, 0, "0")),
                        chord=float(_table_text(self.bwb_table, row, 1, "1")),
                        x_offset=float(_table_text(self.bwb_table, row, 2, "0")),
                        z_offset=float(_table_text(self.bwb_table, row, 3, "0")),
                        airfoil=_table_text(self.bwb_table, row, 4, surface.airfoils.bwb_airfoil),
                    )
                )
            except ValueError:
                continue
        surface.planform.body_sections = sections

    def _on_bwb_table_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        self._sync_current_bwb_sections()
        surface = self._current_surface()
        if surface is not None:
            surface.planform.reset_cache()
            self._update_summary_and_plot(surface)

    def _add_bwb_section(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        next_y = max([section.y_pos for section in surface.planform.body_sections], default=-0.5) + 0.5
        surface.planform.body_sections.append(
            BodySection(
                y_pos=next_y,
                chord=max(0.05, surface.planform.root_chord()),
                x_offset=0.0,
                z_offset=0.0,
                airfoil=surface.airfoils.bwb_airfoil,
            )
        )
        surface.planform.reset_cache()
        self._update_bwb_table()
        self._update_summary_and_plot(surface)

    def _remove_bwb_section(self) -> None:
        surface = self._current_surface()
        row = self.bwb_table.currentRow()
        if surface is not None and 0 <= row < len(surface.planform.body_sections):
            surface.planform.body_sections.pop(row)
            surface.planform.reset_cache()
            self._update_bwb_table()
            self._update_summary_and_plot(surface)

    def _update_cs_table(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        self._loading = True
        self.cs_table.setRowCount(len(surface.control_surfaces))
        for row, cs in enumerate(surface.control_surfaces):
            values = [cs.name, cs.surface_type, cs.span_start_percent, cs.span_end_percent, cs.chord_start_percent, cs.chord_end_percent, cs.hinge_rel_height]
            for col, value in enumerate(values):
                self.cs_table.setItem(row, col, QTableWidgetItem(f"{value:.3f}" if isinstance(value, float) else str(value)))
        self._loading = False

    def _sync_current_control_surfaces(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        controls = []
        for row in range(self.cs_table.rowCount()):
            try:
                controls.append(
                    ControlSurface(
                        name=_table_text(self.cs_table, row, 0, f"Control {row + 1}"),
                        surface_type=_table_text(self.cs_table, row, 1, "Elevator"),
                        span_start_percent=float(_table_text(self.cs_table, row, 2, "0")),
                        span_end_percent=float(_table_text(self.cs_table, row, 3, "100")),
                        chord_start_percent=float(_table_text(self.cs_table, row, 4, "70")),
                        chord_end_percent=float(_table_text(self.cs_table, row, 5, "70")),
                        hinge_rel_height=float(_table_text(self.cs_table, row, 6, "0.5")),
                    )
                )
            except ValueError:
                continue
        surface.control_surfaces = controls
        surface.planform.control_surfaces = list(controls)

    def _on_cs_table_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        self._sync_current_control_surfaces()
        surface = self._current_surface()
        if surface is not None:
            self._update_summary_and_plot(surface)

    def _add_control_surface(self) -> None:
        surface = self._current_surface()
        if surface is None:
            return
        surface.control_surfaces.append(ControlSurface(name=f"Control{len(surface.control_surfaces) + 1}", surface_type="Elevator", span_start_percent=60.0, span_end_percent=100.0, chord_start_percent=70.0, chord_end_percent=70.0))
        surface.planform.control_surfaces = list(surface.control_surfaces)
        self._update_cs_table()
        self._update_summary_and_plot(surface)

    def _remove_control_surface(self) -> None:
        surface = self._current_surface()
        row = self.cs_table.currentRow()
        if surface is not None and 0 <= row < len(surface.control_surfaces):
            surface.control_surfaces.pop(row)
            surface.planform.control_surfaces = list(surface.control_surfaces)
            self._update_cs_table()
            self._update_summary_and_plot(surface)

    def _apply_preset(self):
        choice = self.preset_combo.currentText()
        if choice == "Conventional":
            self.project.aircraft = conventional_rc_aircraft_preset()
        elif choice == "Canard":
            self.project.aircraft = canard_rc_aircraft_preset()
        elif choice == "Twin Fin":
            self.project.aircraft = twin_fin_rc_aircraft_preset()
        else:
            self.project.sync_legacy_wing_to_aircraft()
        self.project.sync_aircraft_main_wing_to_legacy()
        self.update_from_project()

    def _add_surface(self):
        idx = len(self.project.aircraft.surfaces) + 1
        uid = f"surface_{idx}"
        self.project.aircraft.surfaces.append(
            LiftingSurface(
                uid=uid,
                name=f"Surface {idx}",
                role=SurfaceRole.HORIZONTAL_TAIL,
                symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
                planform=PlanformGeometry(wing_area_m2=0.1, aspect_ratio=4.0, taper_ratio=0.8),
            )
        )
        self.update_from_project()
        self._select_surface(uid)

    def _add_vertical_surface(self):
        idx = sum(1 for s in self.project.aircraft.surfaces if self._is_vertical_surface(s)) + 1
        uid = f"vertical_{idx}"
        existing = {s.uid for s in self.project.aircraft.surfaces}
        while uid in existing:
            idx += 1
            uid = f"vertical_{idx}"
        self.project.aircraft.surfaces.append(
            LiftingSurface(
                uid=uid,
                name=f"Vertical Surface {idx}",
                role=SurfaceRole.VERTICAL_TAIL,
                symmetry=SymmetryMode.SINGLE_CENTERLINE,
                local_span_axis=Axis.Z,
                transform=SurfaceTransform(
                    origin_m=(0.8, 0.0, 0.05),
                ),
                planform=PlanformGeometry(wing_area_m2=0.05, aspect_ratio=1.5, taper_ratio=0.7, sweep_le_deg=15.0),
                analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.055, zero_lift_aoa_deg=0.0, cm0=0.0, cl_max=0.9),
                external_refs={"geometry_mode": "full_geometry"},
            )
        )
        self.update_from_project()
        self._select_surface(uid)

    def _select_surface(self, uid: str) -> None:
        idx = self.surface_combo.findData(uid)
        if idx >= 0:
            self.surface_combo.setCurrentIndex(idx)
            self._load_current_surface()

    def _remove_surface(self):
        surface = self._current_surface()
        if surface is None:
            return
        if _value(surface.role) == SurfaceRole.MAIN_WING.value:
            QMessageBox.warning(self, "Surface", "Choose another main wing before removing this surface.")
            return
        self.project.aircraft.surfaces = [s for s in self.project.aircraft.surfaces if s is not surface]
        self.update_from_project()


class ControlSurfacesTab(QWidget):
    COLUMNS = ["Name", "Type", "Span start %", "Span end %", "Chord inboard %", "Chord outboard %", "Hinge height"]

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._loading = False
        self._build_ui()
        self.update_from_project()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Surface"))
        self.surface_combo = QComboBox()
        self.surface_combo.currentIndexChanged.connect(self._load_controls)
        top.addWidget(self.surface_combo, 1)
        add_btn = QPushButton("Add Control")
        add_btn.clicked.connect(self._add_control)
        top.addWidget(add_btn)
        remove_btn = QPushButton("Remove Control")
        remove_btn.clicked.connect(self._remove_control)
        top.addWidget(remove_btn)
        root.addLayout(top)
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table)

    def update_from_project(self):
        self._loading = True
        current = self.surface_combo.currentData()
        self.surface_combo.clear()
        for surface in self.project.aircraft.surfaces:
            self.surface_combo.addItem(f"{surface.name} ({surface.uid})", surface.uid)
        if current:
            idx = self.surface_combo.findData(current)
            if idx >= 0:
                self.surface_combo.setCurrentIndex(idx)
        self._loading = False
        self._load_controls()

    def sync_to_project(self):
        self._write_controls()

    def _surface(self) -> Optional[LiftingSurface]:
        uid = self.surface_combo.currentData()
        for surface in self.project.aircraft.surfaces:
            if surface.uid == uid:
                return surface
        return None

    def _load_controls(self, *_args):
        surface = self._surface()
        if surface is None:
            return
        self._loading = True
        self.table.setRowCount(len(surface.control_surfaces))
        for row, cs in enumerate(surface.control_surfaces):
            values = [cs.name, cs.surface_type, cs.span_start_percent, cs.span_end_percent, cs.chord_start_percent, cs.chord_end_percent, cs.hinge_rel_height]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.3f}" if isinstance(value, float) else str(value)))
        self.table.resizeColumnsToContents()
        self._loading = False

    def _write_controls(self):
        if self._loading:
            return
        surface = self._surface()
        if surface is None:
            return
        controls = []
        for row in range(self.table.rowCount()):
            try:
                controls.append(
                    ControlSurface(
                        name=_table_text(self.table, row, 0, f"Control {row + 1}"),
                        surface_type=_table_text(self.table, row, 1, "Elevator"),
                        span_start_percent=float(_table_text(self.table, row, 2, "0")),
                        span_end_percent=float(_table_text(self.table, row, 3, "100")),
                        chord_start_percent=float(_table_text(self.table, row, 4, "70")),
                        chord_end_percent=float(_table_text(self.table, row, 5, "70")),
                        hinge_rel_height=float(_table_text(self.table, row, 6, "0.5")),
                    )
                )
            except ValueError:
                continue
        surface.control_surfaces = controls
        surface.planform.control_surfaces = list(controls)

    def _on_item_changed(self, item):
        self._write_controls()

    def _add_control(self):
        surface = self._surface()
        if surface is None:
            return
        surface.control_surfaces.append(ControlSurface(name=f"Control {len(surface.control_surfaces) + 1}", surface_type="Elevator"))
        surface.planform.control_surfaces = list(surface.control_surfaces)
        self._load_controls()

    def _remove_control(self):
        surface = self._surface()
        row = self.table.currentRow()
        if surface is not None and 0 <= row < len(surface.control_surfaces):
            surface.control_surfaces.pop(row)
            surface.planform.control_surfaces = list(surface.control_surfaces)
            self._load_controls()


class VerticalSurfacesTab(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self._build_ui()
        self.update_from_project()

    def _build_ui(self):
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        add_btn = QPushButton("Add Vertical Surface")
        add_btn.clicked.connect(self._add_vertical)
        row.addWidget(add_btn)
        row.addStretch()
        root.addLayout(row)
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["UID", "Name", "Symmetry", "X", "Y", "Z", "Area"])
        self.table.itemChanged.connect(self._write_table)
        root.addWidget(self.table)
        root.addWidget(self.canvas)

    def update_from_project(self):
        surfaces = self._vertical_surfaces()
        self.table.blockSignals(True)
        self.table.setRowCount(len(surfaces))
        for row, surface in enumerate(surfaces):
            values = [surface.uid, surface.name, _value(surface.symmetry), *surface.transform.origin_m, surface.planform.wing_area_m2]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.3f}" if isinstance(value, float) else str(value)))
        self.table.blockSignals(False)
        self._plot()

    def sync_to_project(self):
        self._write_table()

    def _vertical_surfaces(self) -> List[LiftingSurface]:
        return [s for s in self.project.aircraft.surfaces if _value(s.role) in (SurfaceRole.VERTICAL_TAIL.value, SurfaceRole.FIN.value) or _value(s.local_span_axis) in (Axis.Z.value, Axis.NEG_Z.value)]

    def _write_table(self, *_args):
        surfaces = self._vertical_surfaces()
        for row, surface in enumerate(surfaces):
            try:
                surface.uid = _table_text(self.table, row, 0, surface.uid)
                surface.name = _table_text(self.table, row, 1, surface.name)
                surface.symmetry = _table_text(self.table, row, 2, _value(surface.symmetry))
                x = float(_table_text(self.table, row, 3, surface.transform.origin_m[0]))
                y = float(_table_text(self.table, row, 4, surface.transform.origin_m[1]))
                z = float(_table_text(self.table, row, 5, surface.transform.origin_m[2]))
                surface.transform = SurfaceTransform(origin_m=(x, y, z), orientation_euler_deg=surface.transform.orientation_euler_deg, parent_uid=surface.transform.parent_uid)
                surface.planform.wing_area_m2 = float(_table_text(self.table, row, 6, surface.planform.wing_area_m2))
                surface.planform.reset_cache()
            except ValueError:
                continue
        self._plot()

    def _add_vertical(self):
        self.project.aircraft.surfaces.append(
            LiftingSurface(
                uid=f"vertical_{len(self.project.aircraft.surfaces) + 1}",
                name="Vertical Surface",
                role=SurfaceRole.VERTICAL_TAIL,
                symmetry=SymmetryMode.SINGLE_CENTERLINE,
                local_span_axis=Axis.Z,
                transform=SurfaceTransform(origin_m=(0.8, 0.0, 0.05)),
                planform=PlanformGeometry(wing_area_m2=0.05, aspect_ratio=1.5, taper_ratio=0.7, sweep_le_deg=15.0),
                analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.055, zero_lift_aoa_deg=0.0, cm0=0.0, cl_max=0.9),
            )
        )
        self.update_from_project()

    def _plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for surface in self.project.aircraft.surfaces:
            if not surface.active:
                continue
            axis = _value(surface.local_span_axis)
            if axis in (Axis.Z.value, Axis.NEG_Z.value):
                self._plot_vertical_surface_side(ax, surface)
            else:
                self._plot_lifting_surface_side(ax, surface)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.3)
        ax.set_title("Aircraft Side View")
        if self.project.aircraft.surfaces:
            ax.legend(fontsize="small")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_vertical_surface_side(self, ax, surface: LiftingSurface) -> None:
        x0, _y, z0 = surface.transform.origin_m
        span = surface.planform.span()
        root = surface.planform.root_chord()
        tip = surface.planform.tip_chord()
        sweep = math.tan(math.radians(surface.planform.sweep_le_deg)) * span
        direction = 1.0 if _value(surface.local_span_axis) == Axis.Z.value else -1.0
        xs = [x0, x0 + root, x0 + sweep + tip, x0 + sweep, x0]
        zs = [z0, z0, z0 + direction * span, z0 + direction * span, z0]
        ax.plot(xs, zs, linewidth=2.4, color="#d62728", label=surface.name)

    def _plot_lifting_surface_side(self, ax, surface: LiftingSurface) -> None:
        x0, _y, z0 = surface.transform.origin_m
        sections = _surface_sections_for_plot(surface)
        try:
            from core.naca_generator.naca456 import generate_naca_airfoil

            name = surface.airfoils.root_airfoil.lower().replace("naca", "").strip()
            x, z = generate_naca_airfoil(name, n_points=80)
            plotted_label = False
            for idx, section in enumerate(sections[:: max(1, len(sections) // 7)] or [{"x_le": 0.0, "y": 0.0, "chord": surface.planform.root_chord()}]):
                chord = section["chord"]
                x_le = x0 + section["x_le"]
                z_le = z0 + math.tan(math.radians(surface.planform.dihedral_deg)) * section["y"]
                twist = math.radians(surface.incidence_deg)
                xs = []
                zs = []
                for xi, zi in zip(x, z):
                    local_x = float(xi) * chord
                    local_z = float(zi) * chord
                    xs.append(x_le + local_x * math.cos(twist) - local_z * math.sin(twist))
                    zs.append(z_le + local_x * math.sin(twist) + local_z * math.cos(twist))
                ax.plot(xs, zs, linewidth=1.0, color="#1f77b4", alpha=0.55, label=surface.name if not plotted_label else None)
                plotted_label = True
        except Exception:
            chord = surface.planform.root_chord()
            thickness = 0.08 * chord
            xs = [x0, x0 + chord, x0 + chord, x0, x0]
            zs = [z0 - thickness / 2.0, z0 - thickness / 2.0, z0 + thickness / 2.0, z0 + thickness / 2.0, z0 - thickness / 2.0]
            ax.plot(xs, zs, linewidth=1.5, color="#1f77b4", alpha=0.85, label=surface.name)


class FuselageTab(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._build_ui()
        self.update_from_project()

    def _build_ui(self):
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        add_btn = QPushButton("Add Fuselage / Body")
        add_btn.clicked.connect(self._add_body)
        row.addWidget(add_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_body)
        row.addWidget(remove_btn)
        row.addStretch()
        root.addLayout(row)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["UID", "Name", "Role", "Active", "X", "Length", "Width", "Height", "Drag area"])
        self.table.itemChanged.connect(self._write_table)
        root.addWidget(self.table)

    def update_from_project(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.project.aircraft.bodies))
        for row, body in enumerate(self.project.aircraft.bodies):
            env = body.envelope or BodyEnvelope()
            values = [body.uid, body.name, body.role, "yes" if body.active else "no", body.transform.origin_m[0], env.length_m, env.max_width_m, env.max_height_m, body.drag_area_estimate_m2 or 0.0]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.3f}" if isinstance(value, float) else str(value)))
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def sync_to_project(self):
        self._write_table()

    def _write_table(self, *_args):
        bodies = []
        for row in range(self.table.rowCount()):
            try:
                body = BodyObject(
                    uid=_table_text(self.table, row, 0, f"body_{row + 1}"),
                    name=_table_text(self.table, row, 1, f"Body {row + 1}"),
                    role=_table_text(self.table, row, 2, "fuselage"),
                    active=_table_text(self.table, row, 3, "yes").lower() in ("yes", "true", "1", "active"),
                    envelope=BodyEnvelope(
                        length_m=float(_table_text(self.table, row, 5, "0")),
                        max_width_m=float(_table_text(self.table, row, 6, "0")),
                        max_height_m=float(_table_text(self.table, row, 7, "0")),
                    ),
                    drag_area_estimate_m2=float(_table_text(self.table, row, 8, "0")),
                )
                x = float(_table_text(self.table, row, 4, "0"))
                body.transform.origin_m = (x, 0.0, 0.0)
                bodies.append(body)
            except ValueError:
                continue
        self.project.aircraft.bodies = bodies

    def _add_body(self):
        self.project.aircraft.bodies.append(
            BodyObject(uid=f"fuselage_{len(self.project.aircraft.bodies) + 1}", name="Fuselage", role="fuselage", envelope=BodyEnvelope(0.8, 0.12, 0.14))
        )
        self.update_from_project()

    def _remove_body(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.project.aircraft.bodies):
            self.project.aircraft.bodies.pop(row)
            self.update_from_project()


# --- Main Geometry Tab ---
class GeometryTab(QTabWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.lifting_surfaces_tab = LiftingSurfacesTab(project)
        self.fuselage_tab = FuselageTab(project)
        self.addTab(self.lifting_surfaces_tab, "Lifting Surfaces")
        self.addTab(self.fuselage_tab, "Fuselage / Bodies")

    def update_from_project(self):
        for tab in self._tabs():
            tab.project = self.project
            tab.update_from_project()

    def sync_to_project(self):
        for tab in self._tabs():
            tab.project = self.project
            if hasattr(tab, "sync_to_project"):
                tab.sync_to_project()
        self.project.sync_aircraft_main_wing_to_legacy()

    def _tabs(self):
        return (self.lifting_surfaces_tab, self.fuselage_tab)


def _double_spin(min_value: float, max_value: float, step: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(min_value, max_value)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    return spin


def _value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _set_combo(combo: QComboBox, value: str) -> None:
    idx = combo.findText(str(value))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _set_combo_by_data(combo: QComboBox, value: str) -> None:
    idx = combo.findData(str(value))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _table_text(table: QTableWidget, row: int, col: int, default) -> str:
    item = table.item(row, col)
    return item.text().strip() if item and item.text().strip() else str(default)


def _surface_sections_for_plot(surface: LiftingSurface) -> List[dict]:
    plan = surface.planform
    n = max(3, int(getattr(surface.airfoils, "num_sections", 13) or 13))
    half_span = plan.half_span()
    root = plan.extended_root_chord()
    sections = []
    for i in range(n):
        eta = i / (n - 1)
        y = half_span * eta
        chord = plan.chord_at_span_fraction(eta)
        if eta == 0.0 and plan.center_chord_extension_percent > 0:
            chord = root
        x_le = math.tan(math.radians(plan.sweep_le_deg)) * y + plan.leading_edge_offset_at_span_fraction(eta)
        sections.append(
            {
                "eta": eta,
                "y": y,
                "x_le": x_le,
                "chord": chord,
                "front_frac": (plan.front_spar_root_percent + (plan.front_spar_tip_percent - plan.front_spar_root_percent) * eta) / 100.0,
                "rear_frac": (plan.rear_spar_root_percent + (plan.rear_spar_tip_percent - plan.rear_spar_root_percent) * eta) / 100.0,
            }
        )
    return sections
