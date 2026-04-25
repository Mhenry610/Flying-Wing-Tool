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


# --- Main Geometry Tab ---
class GeometryTab(QTabWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        
        self.planform_tab = PlanformTab(project)
        self.twist_tab = TwistTrimTab(project)
        self.airfoil_tab = AirfoilTab(project)
        self.perf_tab = PerformanceTab(project)
        
        self.addTab(self.planform_tab, "Planform")
        self.addTab(self.twist_tab, "Twist & Trim")
        self.addTab(self.airfoil_tab, "Airfoil")
        self.addTab(self.perf_tab, "Performance")

        # Connect signals
        # TwistTrimTab emits dataChanged when analysis/optimization runs
        self.twist_tab.dataChanged.connect(self.airfoil_tab.update_from_project)
        self.twist_tab.dataChanged.connect(self.planform_tab.update_from_project)

    def update_from_project(self):
        self.project = self.project # Update reference if needed, though usually same object
        self.planform_tab.project = self.project
        self.planform_tab.update_from_project()
        
        self.twist_tab.project = self.project
        self.twist_tab.update_from_project()
        
        self.airfoil_tab.project = self.project
        self.airfoil_tab.update_from_project()
        
        self.perf_tab.project = self.project
        self.perf_tab.update_from_project()

    def sync_to_project(self):
        for tab in (self.planform_tab, self.twist_tab, self.airfoil_tab):
            tab.project = self.project
            if hasattr(tab, "sync_to_project"):
                tab.sync_to_project()
