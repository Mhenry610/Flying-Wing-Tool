from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.state import Project
from core.aircraft import MassItem
from services.aircraft import apply_mass_balance_to_reference, compute_mass_balance
from services.mission.apc_map import APCMap
from services.mission.motor import MotorProp
from services.mission.planner import AeroConfig, MissionSegment, evaluate_mission
from services.mission.sweep import compute_sweep
from services.mission.util import isa_density


class MissionTab(QTabWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.map = APCMap()
        self.m_prop_map = APCMap()
        self.APC_path = None
        self.m_prop_file_path = None

        # Create sub-tabs
        self.sweep_tab = self._create_sweep_tab()
        self.batch_tab = self._create_batch_tab()
        self.mission_tab = self._create_mission_tab()
        self.propulsion_tab = self._create_propulsion_tab()

        self.addTab(self.sweep_tab, "Sweep")
        self.addTab(self.batch_tab, "Batch")
        self.addTab(self.mission_tab, "Mission")
        self.addTab(self.propulsion_tab, "Propulsion Systems")

    def _create_sweep_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Controls
        left_layout = QVBoxLayout()
        form = QFormLayout()

        # File loading
        self.btn_file = QPushButton("Load APC .dat")
        self.lbl_file = QLabel("No file loaded")
        self.btn_file.clicked.connect(self.load_file)
        form.addRow(self.btn_file, self.lbl_file)

        # Motor parameters
        self.kv = QDoubleSpinBox()
        self.kv.setRange(1, 20000)
        self.kv.setValue(920)
        self.kv.setSuffix(" rpm/V")
        form.addRow("KV", self.kv)

        self.Ri = QDoubleSpinBox()
        self.Ri.setRange(0.1, 500)
        self.Ri.setValue(60)
        self.Ri.setSuffix(" mΩ")
        form.addRow("Resistance Ri", self.Ri)

        self.Io = QDoubleSpinBox()
        self.Io.setRange(0, 10)
        self.Io.setValue(0.7)
        self.Io.setSuffix(" A")
        form.addRow("Idle current Io", self.Io)

        self.Vio = QDoubleSpinBox()
        self.Vio.setRange(0, 60)
        self.Vio.setValue(12)
        self.Vio.setSuffix(" V")
        form.addRow("Voltage @ Io", self.Vio)

        self.vb = QDoubleSpinBox()
        self.vb.setRange(1, 120)
        self.vb.setValue(22.2)
        self.vb.setSuffix(" V")
        form.addRow("Battery voltage", self.vb)

        # Power marks
        self.power_marks = QLineEdit()
        self.power_marks.setText("0.15,0.3,0.5,1,3")
        self.marks_per_prop = QCheckBox("per prop")
        marks_row = QHBoxLayout()
        marks_row.addWidget(self.power_marks)
        marks_row.addWidget(self.marks_per_prop)
        marks_widget = QWidget()
        marks_widget.setLayout(marks_row)
        form.addRow("Power marks [kW]", marks_widget)

        self.nprops = QSpinBox()
        self.nprops.setRange(1, 16)
        self.nprops.setValue(1)
        form.addRow("# props", self.nprops)

        self.alt = QDoubleSpinBox()
        self.alt.setRange(0, 10000)
        self.alt.setValue(0)
        self.alt.setSuffix(" m")
        form.addRow("Altitude", self.alt)

        # Velocity range
        self.vmin = QDoubleSpinBox()
        self.vmin.setRange(0, 150)
        self.vmin.setValue(0)
        self.vmin.setSuffix(" m/s")
        form.addRow("Velocity min", self.vmin)

        self.vmax = QDoubleSpinBox()
        self.vmax.setRange(0.1, 200)
        self.vmax.setValue(40)
        self.vmax.setSuffix(" m/s")
        form.addRow("Velocity max", self.vmax)

        self.npts = QSpinBox()
        self.npts.setRange(5, 1000)
        self.npts.setValue(80)
        form.addRow("# points", self.npts)

        self.diam_in = QDoubleSpinBox()
        self.diam_in.setRange(1, 60)
        self.diam_in.setValue(12)
        self.diam_in.setSuffix(" in")
        form.addRow("Prop diameter", self.diam_in)

        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 1e6)
        self.smooth.setValue(0)
        self.smooth.setDecimals(3)
        form.addRow("Surface smoothing s", self.smooth)

        left_layout.addLayout(form)

        # Buttons
        self.btn_fit = QPushButton("Rebuild CT/CP Surface")
        self.btn_fit.clicked.connect(self.refit_surface)
        left_layout.addWidget(self.btn_fit)

        self.btn_run = QPushButton("Compute Sweep")
        self.btn_run.clicked.connect(self.run_sweep)
        left_layout.addWidget(self.btn_run)

        self.status = QLabel("")
        left_layout.addWidget(self.status)
        left_layout.addStretch()

        # Right: Plot
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))

        layout.addLayout(left_layout, 1)
        layout.addWidget(self.canvas, 2)

        return widget

    def _create_batch_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()

        self.batch_folder = QLineEdit()
        btn_sel_folder = QPushButton("Browse...")
        btn_sel_folder.clicked.connect(lambda: self._pick_folder(self.batch_folder))
        row1 = QHBoxLayout()
        row1.addWidget(self.batch_folder)
        row1.addWidget(btn_sel_folder)
        w1 = QWidget()
        w1.setLayout(row1)
        form.addRow("APC folder", w1)

        self.batch_outdir = QLineEdit()
        btn_sel_out = QPushButton("Browse...")
        btn_sel_out.clicked.connect(lambda: self._pick_folder(self.batch_outdir))
        row2 = QHBoxLayout()
        row2.addWidget(self.batch_outdir)
        row2.addWidget(btn_sel_out)
        w2 = QWidget()
        w2.setLayout(row2)
        form.addRow("Output dir", w2)

        layout.addLayout(form)

        self.btn_batch = QPushButton("Run Batch (CSV+PNG)")
        self.btn_batch.clicked.connect(self.run_batch)
        layout.addWidget(self.btn_batch)

        self.batch_log = QTextEdit()
        self.batch_log.setReadOnly(True)
        layout.addWidget(self.batch_log)

        return widget

    def _create_mission_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- Hidden Aero Fields (Kept for run_mission compatibility) ---
        self.m_Swing = QDoubleSpinBox()
        self.m_Swing.setVisible(False)
        self.m_Sfuse = QDoubleSpinBox()
        self.m_Sfuse.setVisible(False)
        self.m_Stail = QDoubleSpinBox()
        self.m_Stail.setVisible(False)
        self.m_cd_wing = QDoubleSpinBox()
        self.m_cd_wing.setValue(0.030)
        self.m_cd_wing.setVisible(False)
        self.m_cd_fuse = QDoubleSpinBox()
        self.m_cd_fuse.setValue(0.015)
        self.m_cd_fuse.setVisible(False)
        self.m_cd_tail = QDoubleSpinBox()
        self.m_cd_tail.setValue(0.010)
        self.m_cd_tail.setVisible(False)
        self.m_cl = QDoubleSpinBox()
        self.m_cl.setValue(0.6)
        self.m_cl.setVisible(False)
        self.m_clmax = QDoubleSpinBox()
        self.m_clmax.setValue(1.8)
        self.m_clmax.setVisible(False)
        self.m_mu = QDoubleSpinBox()
        self.m_mu.setValue(0.03)
        self.m_mu.setVisible(False)

        # Top Row
        top_layout = QHBoxLayout()
        self.btn_import = QPushButton("Import Geometry")
        self.btn_import.clicked.connect(self._import_from_geometry)
        top_layout.addWidget(self.btn_import)

        self.m_weight_kg = QDoubleSpinBox()
        self.m_weight_kg.setRange(0.1, 10000)
        self.m_weight_kg.setValue(12.5)
        self.m_weight_kg.setSuffix(" kg")
        top_layout.addWidget(QLabel("Weight:"))
        top_layout.addWidget(self.m_weight_kg)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Hardware Config
        hw_layout = QHBoxLayout()

        # 1. Battery
        gb_batt = QGroupBox("Battery")
        f_batt = QFormLayout(gb_batt)
        self.batt_chem = QComboBox()
        self.batt_chem.addItems(["LiPo (3.7V)", "LiIon (3.6V)", "LiFePO4 (3.2V)"])
        f_batt.addRow("Chemistry", self.batt_chem)
        self.batt_series = QSpinBox()
        self.batt_series.setRange(1, 100)
        self.batt_series.setValue(6)
        self.batt_series.setSuffix(" S")
        f_batt.addRow("Series", self.batt_series)
        self.batt_parallel = QSpinBox()
        self.batt_parallel.setRange(1, 100)
        self.batt_parallel.setValue(1)
        self.batt_parallel.setSuffix(" P")
        f_batt.addRow("Parallel", self.batt_parallel)
        self.batt_capacity_mAh = QDoubleSpinBox()
        self.batt_capacity_mAh.setRange(100, 100000)
        self.batt_capacity_mAh.setValue(5000)
        self.batt_capacity_mAh.setSuffix(" mAh")
        f_batt.addRow("Capacity", self.batt_capacity_mAh)
        self.batt_c_rating = QDoubleSpinBox()
        self.batt_c_rating.setRange(1, 200)
        self.batt_c_rating.setValue(30)
        self.batt_c_rating.setSuffix(" C")
        f_batt.addRow("C-Rating", self.batt_c_rating)
        hw_layout.addWidget(gb_batt)

        # 2. Motor
        gb_motor = QGroupBox("Motor")
        f_motor = QFormLayout(gb_motor)
        self.motor_kv = QDoubleSpinBox()
        self.motor_kv.setRange(1, 20000)
        self.motor_kv.setValue(920)
        self.motor_kv.setSuffix(" rpm/V")
        f_motor.addRow("KV", self.motor_kv)
        self.motor_ri_mohm = QDoubleSpinBox()
        self.motor_ri_mohm.setRange(0.1, 1000)
        self.motor_ri_mohm.setValue(60)
        self.motor_ri_mohm.setSuffix(" mΩ")
        f_motor.addRow("Ri", self.motor_ri_mohm)
        self.motor_io = QDoubleSpinBox()
        self.motor_io.setRange(0, 50)
        self.motor_io.setValue(0.7)
        self.motor_io.setSuffix(" A")
        f_motor.addRow("Io", self.motor_io)
        self.motor_vio = QDoubleSpinBox()
        self.motor_vio.setRange(0, 100)
        self.motor_vio.setValue(12)
        self.motor_vio.setSuffix(" V")
        f_motor.addRow("V @ Io", self.motor_vio)
        self.motor_count = QSpinBox()
        self.motor_count.setRange(1, 32)
        self.motor_count.setValue(1)
        f_motor.addRow("Count", self.motor_count)
        hw_layout.addWidget(gb_motor)

        # 3. Propeller
        gb_prop = QGroupBox("Propeller")
        f_prop = QFormLayout(gb_prop)
        self.prop_diam_in = QDoubleSpinBox()
        self.prop_diam_in.setRange(1, 100)
        self.prop_diam_in.setValue(12)
        self.prop_diam_in.setSuffix(" in")
        f_prop.addRow("Diameter", self.prop_diam_in)
        self.prop_pitch_in = QDoubleSpinBox()
        self.prop_pitch_in.setRange(1, 100)
        self.prop_pitch_in.setValue(6)
        self.prop_pitch_in.setSuffix(" in")
        f_prop.addRow("Pitch", self.prop_pitch_in)
        self.prop_family = QComboBox()
        self.prop_family.addItems(["APC Standard", "APC Electric", "APC SlowFly"])
        f_prop.addRow("Family", self.prop_family)

        prop_file_layout = QHBoxLayout()
        self.btn_prop_file = QPushButton("...")
        self.btn_prop_file.setFixedWidth(30)
        self.btn_prop_file.clicked.connect(self._load_mission_prop_file)
        self.prop_file_lbl = QLabel("None")
        prop_file_layout.addWidget(self.prop_file_lbl)
        prop_file_layout.addWidget(self.btn_prop_file)
        f_prop.addRow("APC .dat", prop_file_layout)
        hw_layout.addWidget(gb_prop)

        layout.addLayout(hw_layout)

        # Mission Parameters
        gb_params = QGroupBox("Mission Parameters")
        h_params = QHBoxLayout(gb_params)

        f_p1 = QFormLayout()
        self.surface_type = QComboBox()
        # Data: mu value
        self.surface_type.addItem("Concrete (0.02)", 0.02)
        self.surface_type.addItem("Asphalt (0.02)", 0.02)
        self.surface_type.addItem("Short Grass (0.05)", 0.05)
        self.surface_type.addItem("Long Grass (0.10)", 0.10)
        self.surface_type.addItem("Dirt (0.04)", 0.04)
        self.surface_type.addItem("Snow (0.03)", 0.03)
        self.surface_type.addItem("Ice (0.01)", 0.01)
        self.surface_type.currentIndexChanged.connect(self._update_mu)
        f_p1.addRow("Surface", self.surface_type)

        self.m_takeoff_dist = QDoubleSpinBox()
        self.m_takeoff_dist.setRange(1, 5000)
        self.m_takeoff_dist.setValue(120)
        self.m_takeoff_dist.setSuffix(" m")
        f_p1.addRow("TO Dist", self.m_takeoff_dist)

        self.dynamics_mode = QComboBox()
        self.dynamics_mode.addItems(
            ["Planner (steady-state)", "Simulator (3-DOF)", "Simulator (Simple)"]
        )
        f_p1.addRow("Mode", self.dynamics_mode)

        h_params.addLayout(f_p1)

        f_p2 = QFormLayout()
        self.m_cruise_speed = QDoubleSpinBox()
        self.m_cruise_speed.setRange(0, 200)
        self.m_cruise_speed.setValue(40)
        self.m_cruise_speed.setSuffix(" m/s")
        f_p2.addRow("Cruise Speed", self.m_cruise_speed)
        self.m_cruise_alt = QDoubleSpinBox()
        self.m_cruise_alt.setRange(0, 10000)
        self.m_cruise_alt.setValue(500)
        self.m_cruise_alt.setSuffix(" m")
        f_p2.addRow("Cruise Alt", self.m_cruise_alt)

        self.m_cruise_duration = QDoubleSpinBox()
        self.m_cruise_duration.setRange(0, 1e6)
        self.m_cruise_duration.setValue(30)
        self.m_cruise_time_unit = QComboBox()
        self.m_cruise_time_unit.addItems(["s", "min", "h"])
        dur_row = QHBoxLayout()
        dur_row.addWidget(self.m_cruise_duration)
        dur_row.addWidget(self.m_cruise_time_unit)
        dur_widget = QWidget()
        dur_widget.setLayout(dur_row)
        f_p2.addRow("Duration", dur_widget)

        h_params.addLayout(f_p2)

        layout.addWidget(gb_params)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_mission = QPushButton("Compute Mission Analysis")
        self.btn_mission.clicked.connect(self.run_mission)
        btn_row.addWidget(self.btn_mission)

        self.btn_jsbsim = QPushButton("Export JSBSim model")
        self.btn_jsbsim.clicked.connect(self.export_jsbsim_model)
        btn_row.addWidget(self.btn_jsbsim)

        self.chk_flightgear = QCheckBox("For FlightGear")
        self.chk_flightgear.setToolTip(
            "When checked, exports a complete FlightGear aircraft package "
            "including 3D model and configuration files"
        )
        btn_row.addWidget(self.chk_flightgear)

        layout.addLayout(btn_row)

        # Results Splitter/Layout
        res_layout = QHBoxLayout()

        # Table
        left_res = QVBoxLayout()
        self.m_table = QTableWidget(0, 10)
        self.m_table.setHorizontalHeaderLabels(
            [
                "name",
                "reachable",
                "duty",
                "rpm",
                "I_mot",
                "T_tot",
                "T_req",
                "Shortfall",
                "P_tot",
                "E_J",
            ]
        )
        left_res.addWidget(self.m_table)
        self.m_summary = QLabel("Ready")
        left_res.addWidget(self.m_summary)
        res_layout.addLayout(left_res, 1)

        # Plot
        self.mission_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        res_layout.addWidget(self.mission_canvas, 1)

        layout.addLayout(res_layout)

        return widget

    def _create_propulsion_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        controls = QHBoxLayout()
        sync_btn = QPushButton("Sync From Mission Controls")
        sync_btn.clicked.connect(self._sync_propulsion_systems_to_aircraft)
        controls.addWidget(sync_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_propulsion_system)
        controls.addWidget(remove_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.propulsion_table = QTableWidget(0, 8)
        self.propulsion_table.setHorizontalHeaderLabels([
            "UID", "Name", "Motors", "Motor KV", "Battery", "Prop", "APC file", "Matched"
        ])
        self.propulsion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.propulsion_table)
        return widget

    def _create_mass_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add Item")
        add_btn.clicked.connect(self._add_mass_item)
        controls.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_mass_item)
        controls.addWidget(remove_btn)
        apply_btn = QPushButton("Apply CG to Aircraft Reference")
        apply_btn.clicked.connect(self._apply_mass_cg)
        controls.addWidget(apply_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.mass_table = QTableWidget()
        self.mass_table.setColumnCount(7)
        self.mass_table.setHorizontalHeaderLabels(["Name", "Category", "Mass [kg]", "X [m]", "Y [m]", "Z [m]", "Notes"])
        self.mass_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mass_table.itemChanged.connect(self._on_mass_table_changed)
        layout.addWidget(self.mass_table)

        summary = QGroupBox("Mass Balance")
        summary_form = QFormLayout(summary)
        self.mass_total_label = QLabel("-")
        self.mass_cg_label = QLabel("-")
        self.mass_warning_label = QLabel("-")
        summary_form.addRow("Total mass", self.mass_total_label)
        summary_form.addRow("CG [m]", self.mass_cg_label)
        summary_form.addRow("Warnings", self.mass_warning_label)
        layout.addWidget(summary)
        return widget

    def _update_mu(self):
        mu = self.surface_type.currentData()
        if mu is not None:
            self.m_mu.setValue(float(mu))

    def _load_mission_prop_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open APC .dat", "", "APC data (*.dat);;All files (*)"
        )
        if not fn:
            return
        try:
            self.m_prop_map.load(fn, j_nodes=80, smooth=0.0)
            self.prop_file_lbl.setText(Path(fn).name)
            self.m_prop_file_path = fn
            if self.m_prop_map.d_in:
                self.prop_diam_in.setValue(float(self.m_prop_map.d_in))
            self.m_summary.setText(f"Loaded Prop: {Path(fn).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _plot_mission_profile(self, time_s, altitude_m, airspeed_mps, soc, phases=None):
        fig = self.mission_canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(111)

        time_arr = np.asarray(time_s)
        alt_arr = np.asarray(altitude_m)
        speed_arr = np.asarray(airspeed_mps)

        if phases:
            colors = ["#f2f2f2", "#e8f4ff"]
            for idx, phase in enumerate(phases):
                start = phase.get("start", 0.0)
                end = phase.get("end", 0.0)
                label = phase.get("label", "")
                if end <= start:
                    continue
                ax1.axvspan(start, end, color=colors[idx % len(colors)], alpha=0.4)
                if label:
                    ax1.text(
                        (start + end) / 2.0,
                        0.98,
                        label,
                        transform=ax1.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=8,
                    )

        (alt_line,) = ax1.plot(time_arr, alt_arr, "b-", label="Altitude (m)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Altitude (m)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(True)

        ax2 = ax1.twinx()
        (speed_line,) = ax2.plot(time_arr, speed_arr, "g-", label="Airspeed (m/s)")
        ax2.set_ylabel("Airspeed (m/s)", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

        lines = [alt_line, speed_line]
        labels = [line.get_label() for line in lines]

        if soc is not None and len(time_arr) == len(soc):
            soc_pct = np.asarray(soc) * 100.0
            (soc_line,) = ax2.plot(time_arr, soc_pct, "k--", label="SOC (%)")
            lines.append(soc_line)
            labels.append(soc_line.get_label())

        ax1.legend(lines, labels, loc="upper right", fontsize=8)
        fig.tight_layout()
        self.mission_canvas.draw_idle()

    def _get_aero_defaults(self) -> Dict[str, float]:
        cl_cruise = 0.6
        cd_wing = 0.03
        cd_fuse = 0.015
        cd_tail = 0.01
        cl_max = 1.8
        wing_area = 0.0

        if self.project is not None:
            try:
                wing_area = float(self.project.wing.planform.actual_area())
            except Exception:
                wing_area = 0.0

            perf = getattr(self.project.analysis, "performance_metrics", None)
            if perf:
                cl_cruise = float(perf.get("cruise_cl", cl_cruise))
                cd_wing = float(perf.get("cruise_cd", cd_wing))
            else:
                try:
                    cl_cruise = float(self.project.wing.twist_trim.design_cl)
                except Exception:
                    pass

                polars = getattr(self.project.analysis, "polars", None)
                if polars and "CL" in polars and "CD" in polars:
                    cls = np.array(polars["CL"])
                    cds = np.array(polars["CD"])
                    idx = np.argsort(cls)
                    cd_wing = float(np.interp(cl_cruise, cls[idx], cds[idx]))

            try:
                cl_max = float(self.project.wing.twist_trim.estimated_cl_max)
            except Exception:
                pass

        if wing_area > 0:
            self.m_Swing.setValue(wing_area)
        self.m_cd_wing.setValue(cd_wing)
        self.m_cd_fuse.setValue(cd_fuse)
        self.m_cd_tail.setValue(cd_tail)
        self.m_cl.setValue(cl_cruise)
        self.m_clmax.setValue(cl_max)

        return {
            "cl_cruise": cl_cruise,
            "cd_wing": cd_wing,
            "cd_fuse": cd_fuse,
            "cd_tail": cd_tail,
            "cl_max": cl_max,
            "wing_area": wing_area,
        }

    def _collect_profile(self, result):
        time_s: List[float] = []
        altitude_m: List[float] = []
        airspeed_mps: List[float] = []
        soc: List[float] = []
        phases: List[Dict[str, Any]] = []

        t_offset = 0.0
        for phase in result.phases:
            if phase.time is None or len(phase.time) == 0:
                continue
            phase_time = np.asarray(phase.time, dtype=float)
            alt = phase.states.get("altitude") if phase.states else None
            speed = phase.states.get("airspeed") if phase.states else None
            soc_hist = phase.states.get("SOC") if phase.states else None

            if alt is None:
                alt = np.full_like(phase_time, np.nan, dtype=float)
            if speed is None:
                speed = np.full_like(phase_time, np.nan, dtype=float)
            if soc_hist is None:
                soc_hist = np.full_like(phase_time, np.nan, dtype=float)

            phase_time = phase_time + t_offset
            if len(phase_time) > 0:
                phases.append(
                    {
                        "start": float(phase_time[0]),
                        "end": float(phase_time[-1]),
                        "label": phase.name,
                    }
                )
                t_offset = float(phase_time[-1])

            time_s.extend(phase_time.tolist())
            altitude_m.extend(np.asarray(alt).tolist())
            airspeed_mps.extend(np.asarray(speed).tolist())
            soc.extend(np.asarray(soc_hist).tolist())

        return time_s, altitude_m, airspeed_mps, soc, phases

    def _pick_folder(self, target: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder", "")
        if d:
            target.setText(d)

    def load_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open APC .dat", "", "APC data (*.dat);;All files (*)"
        )
        if not fn:
            return
        self.APC_path = fn
        self.lbl_file.setText(Path(fn).name)
        try:
            s = float(self.smooth.value())
            self.map.load(fn, j_nodes=80, smooth=s)
            if self.map.d_in is not None:
                self.diam_in.setValue(float(self.map.d_in))
            self.status.setText(
                f"Loaded. RPM bands: {len(self.map.rpms)} | J range: [{self.map.Jg[0]:.3f}, {self.map.Jg[-1]:.3f}]"
            )
        except Exception as e:
            self.status.setText(f"Load error: {e}")

    def refit_surface(self):
        if not self.APC_path:
            self.status.setText("Load a .dat first.")
            return
        try:
            s = float(self.smooth.value())
            self.map.load(self.APC_path, j_nodes=80, smooth=s)
            self.status.setText("Surface rebuilt.")
        except Exception as e:
            self.status.setText(f"Rebuild error: {e}")

    def run_sweep(self):
        if not self.map.loaded:
            self.status.setText("Load APC data first.")
            return

        try:
            KV = float(self.kv.value())
            Ri_mOhm = float(self.Ri.value())
            Io_A = float(self.Io.value())
            Vio = float(self.Vio.value()) if self.Vio.value() > 0 else None
            Vb = float(self.vb.value())
            nprops = int(self.nprops.value())
            alt_m = float(self.alt.value())
            rho = isa_density(alt_m)
            D_m = float(self.diam_in.value()) * 0.0254
            vmin = float(self.vmin.value())
            vmax = float(self.vmax.value())
            npts = int(self.npts.value())

            if vmax <= vmin:
                self.status.setText("Velocity max must be > min.")
                return

            V = np.linspace(vmin, vmax, npts)
            motor = MotorProp(KV, Ri_mOhm, Io_A, Vio)

            marks_W = self._parse_marks()
            marks_total = (
                [m * nprops for m in marks_W]
                if self.marks_per_prop.isChecked()
                else marks_W
            )

            out = compute_sweep(V, marks_total, motor, Vb, nprops, rho, D_m, self.map)
            series = out["series"]
            p_full_max = out["p_full_max"]

            self._plot_series(series)

            if self.marks_per_prop.isChecked():
                ptxt = ", ".join(f"{p / 1000.0:.2f}kW/prop" for p in marks_W)
            else:
                ptxt = ", ".join(f"{p / 1000.0:.2f}kW total" for p in marks_W)
            unattainable_all = [s["P"] for s in series if not s.get("reachable", True)]
            unatt = (
                ", ".join(f"{p / 1000.0:.2f}kW" for p in unattainable_all)
                if unattainable_all
                else "None"
            )
            self.status.setText(
                f"rho={rho:.3f} kg/m^3 | D={D_m:.3f} m | Power tracks: {ptxt} | Max full-power ≈ {p_full_max / 1000.0:.2f}kW | Unattainable: {unatt}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Sweep failed: {e}")

    def _parse_marks(self) -> List[float]:
        marks_txt = self.power_marks.text().strip()
        marks_W: List[float] = []
        if marks_txt:
            raw = re.sub(r"[\[\]]", "", marks_txt)
            for tok in re.split(r"[;,\s]+", raw):
                if not tok:
                    continue
                try:
                    num = float(re.sub(r"[^0-9eE+\-.]", "", tok))
                    t = tok.strip().lower()
                    if re.search(r"kw|\bk\b", t):
                        val = num * 1000.0
                    elif re.search(r"w", t):
                        val = num
                    else:
                        val = num * 1000.0
                    if val > 0:
                        marks_W.append(val)
                except Exception:
                    pass
        if not marks_W:
            marks_W = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        return marks_W

    def _plot_series(self, series: List[Dict[str, Any]]):
        fig = self.canvas.figure
        fig.clear()

        axes = fig.subplots(2, 2)
        ax_thrust, ax_rpm = axes[0]
        ax_eff, ax_torque = axes[1]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, s in enumerate(series):
            label = f"{s['P'] / 1000.0:.2f} kW"
            color = colors[i % len(colors)]
            if s.get("reachable", True):
                ax_thrust.plot(s["V"], s["thrust"], color=color, label=label)
                ax_rpm.plot(s["V"], s["rpm"], color=color)
                ax_eff.plot(s["V"], s["eff_overall"], color=color)
                ax_torque.plot(s["V"], s["torque"], color=color)
            else:
                ax_thrust.plot([], [], color=color, ls="--", label=label)

        ax_thrust.set_xlabel("Velocity [m/s]")
        ax_thrust.set_ylabel("Thrust [N]")
        ax_thrust.grid(True)
        ax_thrust.legend(fontsize=8)
        ax_rpm.set_xlabel("Velocity [m/s]")
        ax_rpm.set_ylabel("RPM")
        ax_rpm.grid(True)
        ax_eff.set_xlabel("Velocity [m/s]")
        ax_eff.set_ylabel("Overall eta")
        ax_eff.set_ylim(0, 1.05)
        ax_eff.grid(True)
        ax_torque.set_xlabel("Velocity [m/s]")
        ax_torque.set_ylabel("Torque [N·m]")
        ax_torque.grid(True)

        fig.tight_layout()
        self.canvas.draw_idle()

    def run_batch(self):
        folder = self.batch_folder.text().strip()
        outdir = self.batch_outdir.text().strip()
        if not folder:
            self.batch_log.append("Select an APC folder.")
            return
        if not outdir:
            self.batch_log.append("Select an output directory.")
            return
        try:
            Path(outdir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.batch_log.append(f"Output dir error: {e}")
            return

        marks_W = self._parse_marks()
        per_prop = self.marks_per_prop.isChecked()
        V = np.linspace(
            float(self.vmin.value()), float(self.vmax.value()), int(self.npts.value())
        )
        rho = isa_density(float(self.alt.value()))
        motor = MotorProp(
            float(self.kv.value()),
            float(self.Ri.value()),
            float(self.Io.value()),
            float(self.Vio.value()) if self.Vio.value() > 0 else None,
        )
        vb = float(self.vb.value())
        nprops = int(self.nprops.value())
        smooth = float(self.smooth.value())

        for path in sorted(Path(folder).glob("*.dat")):
            try:
                apc = APCMap()
                apc.load(str(path), j_nodes=80, smooth=smooth)
                D_m = (apc.d_in or 0.0) * 0.0254
                if D_m <= 0:
                    self.batch_log.append(f"[WARN] Skip {path.name}: unknown diameter")
                    continue
                marks_total = [m * nprops for m in marks_W] if per_prop else marks_W
                out = compute_sweep(V, marks_total, motor, vb, nprops, rho, D_m, apc)
                series = out["series"]

                # CSV
                csv_path = Path(outdir) / f"{path.stem}_sweep.csv"
                import csv

                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "prop",
                            "diam_in",
                            "mark_W",
                            "V_mps",
                            "thrust_total_N",
                            "rpm",
                            "eff_overall",
                            "torque_Nm_per_motor",
                        ]
                    )
                    for s in series:
                        mark = float(s["P"])
                        for V0, T, rpm, eff, tq in zip(
                            s["V"], s["thrust"], s["rpm"], s["eff_overall"], s["torque"]
                        ):
                            if not np.isfinite(rpm):
                                continue
                            w.writerow(
                                [
                                    path.stem,
                                    apc.d_in,
                                    mark,
                                    float(V0),
                                    float(T),
                                    float(rpm),
                                    float(eff),
                                    float(tq),
                                ]
                            )

                self.batch_log.append(f"[OK] {path.name} -> {csv_path.name}")
            except Exception as e:
                self.batch_log.append(f"[ERR] {path.name}: {e}")

    def run_mission(self):
        mode = self.dynamics_mode.currentText()
        self.sync_to_project()

        from services.dynamics.landing_gear import SURFACE_PRESETS, SurfaceType
        from services.propulsion.battery_model import BatteryPackConfig, get_cell_preset
        from services.propulsion.motor_model import MotorParameters
        from services.propulsion.propulsion_system import (
            ESC_PRESETS,
            IntegratedPropulsionSystem,
            PropellerSpec,
            PropulsionSystemConfig,
        )

        aero_defaults = self._get_aero_defaults()

        kv = float(self.motor_kv.value())
        ri_mohm = float(self.motor_ri_mohm.value())
        io = float(self.motor_io.value())
        vio = float(self.motor_vio.value()) if self.motor_vio.value() > 0 else None
        n_motors = int(self.motor_count.value())

        chem_label = self.batt_chem.currentText()
        if "LiIon" in chem_label:
            preset_name = "LiIon_18650_HighCap"
        elif "LiFe" in chem_label:
            preset_name = "LiFe_Standard"
        else:
            preset_name = "LiPo_Standard"

        cell = get_cell_preset(preset_name)
        cell.capacity_Ah = float(self.batt_capacity_mAh.value()) / 1000.0
        c_rate = float(self.batt_c_rating.value())
        cell.C_rate_max_continuous = c_rate
        cell.C_rate_max_burst = max(c_rate * 1.5, c_rate)

        battery = BatteryPackConfig(
            cell=cell,
            n_series=int(self.batt_series.value()),
            n_parallel=int(self.batt_parallel.value()),
        )
        Vb = battery.V_nominal
        batt_cap_Wh = battery.energy_Wh

        motor_params = MotorParameters(
            kv=kv,
            R_internal=ri_mohm / 1000.0,
            I_no_load=io,
        )
        if np.isfinite(battery.I_max_continuous) and battery.I_max_continuous > 0:
            motor_params.I_max = battery.I_max_continuous / max(1, n_motors)

        prop_diam_in = float(self.prop_diam_in.value())
        prop_pitch_in = float(self.prop_pitch_in.value())
        prop_family_label = self.prop_family.currentText()
        family_map = {
            "APC Standard": "Standard",
            "APC Electric": "Electric",
            "APC SlowFly": "SlowFly",
            "Standard": "Standard",
            "Electric": "Electric",
            "SlowFly": "SlowFly",
        }
        prop_family = family_map.get(prop_family_label, "Electric")
        prop_spec = PropellerSpec(
            diameter_in=prop_diam_in,
            pitch_in=prop_pitch_in,
            family=prop_family,
        )

        surface_label = self.surface_type.currentText().lower()
        if "concrete" in surface_label:
            surface_type = SurfaceType.CONCRETE
        elif "asphalt" in surface_label:
            surface_type = SurfaceType.ASPHALT
        elif "short grass" in surface_label:
            surface_type = SurfaceType.SHORT_GRASS
        elif "long grass" in surface_label:
            surface_type = SurfaceType.LONG_GRASS
        elif "dirt" in surface_label:
            surface_type = SurfaceType.DIRT
        elif "snow" in surface_label:
            surface_type = SurfaceType.SNOW
        elif "ice" in surface_label:
            surface_type = SurfaceType.ICE
        else:
            surface_type = SurfaceType.ASPHALT

        surface_props = SURFACE_PRESETS[surface_type]
        mu_roll = surface_props.mu_rolling
        mu_brake = surface_props.mu_braking
        self.m_mu.setValue(mu_roll)

        wing_area_m2 = (
            float(aero_defaults["wing_area"]) if aero_defaults["wing_area"] > 0 else 0.5
        )
        cd0 = float(
            aero_defaults["cd_wing"]
            + aero_defaults["cd_fuse"]
            + aero_defaults["cd_tail"]
        )
        cl_max = float(aero_defaults["cl_max"])

        prop_map = (
            self.m_prop_map if self.m_prop_map and self.m_prop_map.loaded else None
        )
        motor_for_apc = MotorProp(kv, ri_mohm, io, vio)
        rho0 = isa_density(0.0)
        diam_m = prop_diam_in * 0.0254
        apc_static_thrust = None
        if prop_map is not None:
            try:
                static = motor_for_apc.solve_rpm(Vb, 0.0, rho0, diam_m, prop_map)
                apc_static_thrust = static.get("T", 0.0) * n_motors
            except Exception:
                apc_static_thrust = None

        if mode == "Planner (steady-state)":
            if prop_map is None:
                self.m_summary.setText("Load APC .dat for planner mode.")
                return

            try:
                weight_N = float(self.m_weight_kg.value()) * 9.80665
                aero = AeroConfig(
                    cl_cruise=float(aero_defaults["cl_cruise"]),
                    cd_wing=float(aero_defaults["cd_wing"]),
                    cd_fuse=float(aero_defaults["cd_fuse"]),
                    cd_tail=float(aero_defaults["cd_tail"]),
                    s_wing_m2=wing_area_m2,
                    s_fuse_m2=0.0,
                    s_tail_m2=0.0,
                    cl_max=float(aero_defaults["cl_max"]),
                    mu_roll=mu_roll,
                )

                takeoff_seg = MissionSegment(
                    name="Takeoff",
                    duration_s=0.0,
                    speed_mps=0.0,
                    alt_m=0.0,
                    thrust_total_N=None,
                    mode="takeoff",
                    distance_m=float(self.m_takeoff_dist.value()),
                )

                unit = self.m_cruise_time_unit.currentText()
                dur = float(self.m_cruise_duration.value())
                if unit == "min":
                    dur_s = dur * 60.0
                elif unit == "h":
                    dur_s = dur * 3600.0
                else:
                    dur_s = dur

                cruise_seg = MissionSegment(
                    name="Cruise",
                    duration_s=dur_s,
                    speed_mps=float(self.m_cruise_speed.value()),
                    alt_m=float(self.m_cruise_alt.value()),
                    thrust_total_N=None,
                    mode="cruise",
                )
                segs = [takeoff_seg, cruise_seg]

                motor = MotorProp(kv, ri_mohm, io, vio)
                out = evaluate_mission(
                    segs,
                    prop_map,
                    motor,
                    Vb,
                    n_motors,
                    diam_m,
                    weight_N=weight_N,
                    aero=aero,
                    S_ref_m2=wing_area_m2,
                )
                rows = out["results"]

                self.m_table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    vals = [
                        row.get("name", ""),
                        str(row.get("reachable", False)),
                        f"{row.get('duty', 0):.4f}",
                        f"{row.get('rpm', 0):.1f}",
                        f"{row.get('I', 0):.3f}",
                        f"{row.get('T_total', 0):.2f}",
                        f"{row.get('thrust_required', 0):.2f}",
                        f"{row.get('shortfall_N', 0):.2f}",
                        f"{row.get('Pelec_total', 0):.1f}",
                        f"{row.get('energy_J', 0):.1f}",
                    ]
                    for c, v in enumerate(vals):
                        item = QTableWidgetItem(str(v))
                        self.m_table.setItem(r, c, item)

                used_Wh = out["total_energy_J"] / 3.6e6
                soc_used_pct = (
                    (used_Wh / batt_cap_Wh) * 100.0 if batt_cap_Wh > 0 else 0.0
                )

                summary = (
                    f"Total energy {used_Wh:.2f} Wh | Cap {batt_cap_Wh:.1f} Wh | "
                    f"Used {soc_used_pct:.1f}% | Time {out['total_time_s']:.1f} s"
                )
                if apc_static_thrust is not None:
                    summary += f" | APC static thrust {apc_static_thrust:.1f} N"
                self.m_summary.setText(summary)

                time_s = [0.0, 5.0, 5.0 + dur_s]
                alt_m = [
                    0.0,
                    float(self.m_cruise_alt.value()),
                    float(self.m_cruise_alt.value()),
                ]
                speed_mps = [
                    0.0,
                    float(self.m_cruise_speed.value()),
                    float(self.m_cruise_speed.value()),
                ]
                phases = [
                    {"start": 0.0, "end": 5.0, "label": "Takeoff"},
                    {"start": 5.0, "end": 5.0 + dur_s, "label": "Cruise"},
                ]
                self._plot_mission_profile(
                    time_s, alt_m, speed_mps, None, phases=phases
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Mission Error", f"Mission evaluation failed: {e}"
                )
            return

        try:
            import sys
            print("[DEBUG] Starting 3DOF mission simulation...", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            
            from services.aero_model import create_rigid_body_aero_model
            from services.mission.mission_definition import (
                MissionPhase,
                MissionPhaseType,
                MissionProfile,
            )
            from services.mission.mission_simulator import (
                MissionSimulator,
                SimulationConfig,
                set_mission_verbosity,
            )
            
            # Enable verbose logging to console
            set_mission_verbosity('INFO')

            mass_kg = float(self.m_weight_kg.value())
            cruise_speed = float(self.m_cruise_speed.value())
            cruise_alt = float(self.m_cruise_alt.value())

            unit = self.m_cruise_time_unit.currentText()
            dur = float(self.m_cruise_duration.value())
            if unit == "min":
                cruise_duration_s = dur * 60.0
            elif unit == "h":
                cruise_duration_s = dur * 3600.0
            else:
                cruise_duration_s = dur

            weight_N = mass_kg * 9.80665
            V_stall = np.sqrt(
                2.0 * weight_N / (rho0 * wing_area_m2 * max(cl_max, 1e-3))
            )
            V_rot = 1.1 * V_stall
            V_ref = 1.3 * V_stall

            max_thrust_N = 15.0
            propulsive_efficiency = 0.55
            if prop_map is not None:
                try:
                    cruise = motor_for_apc.solve_rpm(
                        Vb, max(1.0, cruise_speed), rho0, diam_m, prop_map
                    )
                    max_thrust_N = max(max_thrust_N, (apc_static_thrust or 0.0))
                    propulsive_efficiency = float(
                        np.clip(cruise.get("eta", 0.55), 0.1, 0.9)
                    )
                except Exception:
                    pass

            dynamics_mode = "3dof" if mode == "Simulator (3-DOF)" else "simple"
            config = SimulationConfig(
                verbose=True,
                verbose_calc_interval=5.0,
                dynamics_mode=dynamics_mode,
                mu_roll=mu_roll,
                mu_brake=mu_brake,
                max_bank_deg=45.0,
            )

            propulsion_system = None
            if dynamics_mode == "3dof":
                esc_presets = sorted(
                    ESC_PRESETS.values(), key=lambda p: p.I_max_continuous
                )
                esc_target = (
                    motor_params.I_max if np.isfinite(motor_params.I_max) else 60.0
                )
                esc = esc_presets[-1]
                for preset in esc_presets:
                    if preset.I_max_continuous >= esc_target:
                        esc = preset
                        break

                prop_config = PropulsionSystemConfig(
                    propeller=prop_spec,
                    motor=motor_params,
                    esc=esc,
                    battery=battery,
                    n_motors=n_motors,
                )
                propulsion_system = IntegratedPropulsionSystem(prop_config)
                try:
                    static_thrust = propulsion_system.get_static_thrust(
                        throttle=1.0,
                        SOC=1.0,
                        T_ambient=15.0,
                    )
                    if np.isfinite(static_thrust) and static_thrust > 0:
                        max_thrust_N = max(max_thrust_N, float(static_thrust))
                except Exception:
                    pass

            phases = [
                MissionPhase(
                    name="Ground Idle",
                    phase_type=MissionPhaseType.GROUND_IDLE,
                    duration=5.0,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Takeoff Roll",
                    phase_type=MissionPhaseType.TAKEOFF_ROLL,
                    end_speed=V_rot,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Rotation",
                    phase_type=MissionPhaseType.ROTATION,
                    duration=4.0,
                    end_altitude=5.0,
                    target_speed=V_rot,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Climb",
                    phase_type=MissionPhaseType.CLIMB,
                    end_altitude=cruise_alt,
                    target_climb_rate=2.0,
                    target_speed=cruise_speed,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Cruise",
                    phase_type=MissionPhaseType.CRUISE,
                    duration=cruise_duration_s,
                    target_altitude=cruise_alt,
                    target_speed=cruise_speed,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Descent",
                    phase_type=MissionPhaseType.DESCENT,
                    end_altitude=15.0,
                    target_descent_rate=2.0,
                    target_speed=max(V_ref, cruise_speed * 0.8),
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Approach",
                    phase_type=MissionPhaseType.APPROACH,
                    end_altitude=config.flare_height_m,
                    target_descent_rate=1.5,
                    target_speed=V_ref,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Flare",
                    phase_type=MissionPhaseType.LANDING_FLARE,
                    end_altitude=0.0,
                    target_descent_rate=0.5,
                    target_heading=0.0,
                ),
                MissionPhase(
                    name="Landing Roll",
                    phase_type=MissionPhaseType.LANDING_ROLL,
                    end_speed=2.0,
                    target_heading=0.0,
                ),
            ]

            mission = MissionProfile(
                name="GUI Mission",
                phases=phases,
                initial_altitude=0.0,
                initial_speed=0.0,
                initial_heading=0.0,
                initial_SOC=1.0,
                T_ambient=15.0,
                pressure_altitude=0.0,
                wind_speed=0.0,
                wind_direction=0.0,
                runway_heading=0.0,
                runway_length=float(self.m_takeoff_dist.value()),
                surface_friction=mu_roll,
            )

            # Build RigidBodyAeroModel with precomputed polars (matches takeoff_analysis3DOF.py)
            aero_model = None
            if self.project is not None:
                try:
                    print(f"Building RigidBodyAeroModel from project.wing...")
                    aero_model = create_rigid_body_aero_model(
                        wing_project=self.project.wing,
                        use_precomputed_polars=True,
                        verbose=True,
                    )
                    print("RigidBodyAeroModel ready for mission simulation")
                except Exception as e:
                    import traceback

                    print(f"Warning: Could not build RigidBodyAeroModel: {e}")
                    traceback.print_exc()
                    print("Falling back to linear aero model")
                    aero_model = None
            else:
                print(
                    f"DEBUG: Skipping RigidBodyAeroModel (project={self.project is not None})"
                )

            simulator = MissionSimulator(
                config=config,
                mass_kg=mass_kg,
                wing_area_m2=wing_area_m2,
                aero_model=aero_model,
                CL0=0.0,
                CLa=5.5,
                CD0=cd0,
                k_induced=0.04,
                CL_max=cl_max,
                max_thrust_N=max_thrust_N,
                propulsive_efficiency=propulsive_efficiency,
                battery_capacity_Wh=batt_cap_Wh,
                propulsion=propulsion_system,
            )

            print("[DEBUG] Starting simulate_mission()...", flush=True)
            result = simulator.simulate_mission(mission)
            print("[DEBUG] simulate_mission() completed", flush=True)
            def _stats(values: Any) -> Dict[str, float] | None:
                if values is None:
                    return None
                arr = np.asarray(values, dtype=float)
                if arr.size == 0:
                    return None
                if not np.any(np.isfinite(arr)):
                    return None
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return None
                return {
                    "min": float(np.min(arr)),
                    "avg": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                }

            phase_stats = []
            for phase_result in result.phases:
                phase_stats.append(
                    {
                        "name": phase_result.name,
                        "phase_type": phase_result.phase_type.name,
                        "start_time_s": float(phase_result.start_time),
                        "end_time_s": float(phase_result.end_time),
                        "duration_s": float(phase_result.duration),
                        "distance_m": float(phase_result.distance_traveled_m),
                        "energy_Wh": float(phase_result.energy_consumed_Wh),
                        "SOC_start": float(phase_result.SOC_start),
                        "SOC_end": float(phase_result.SOC_end),
                        "throttle": _stats(phase_result.controls.get("throttle")),
                        "thrust_N": _stats(phase_result.auxiliary.get("thrust")),
                    }
                )

            if self.project is not None:
                propulsion_snapshot = {
                    "battery": {
                        "chemistry": chem_label,
                        "series": int(battery.n_series),
                        "parallel": int(battery.n_parallel),
                        "capacity_mAh": float(battery.capacity_mAh),
                        "C_rate": float(c_rate),
                        "V_nominal": float(battery.V_nominal),
                        "energy_Wh": float(battery.energy_Wh),
                        "I_max": float(battery.I_max_continuous),
                    },
                    "motor": {
                        "kv": float(motor_params.kv),
                        "R_internal_ohm": float(motor_params.R_internal),
                        "I_no_load": float(motor_params.I_no_load),
                        "I_max": float(motor_params.I_max),
                    },
                    "propeller": {
                        "diameter_in": float(prop_spec.diameter_in),
                        "pitch_in": float(prop_spec.pitch_in),
                        "family": str(prop_spec.family),
                        "apc_file_path": self.m_prop_file_path,
                    },
                    "derived": {
                        "max_thrust_N": float(max_thrust_N),
                        "propulsive_efficiency": float(propulsive_efficiency),
                        "apc_static_thrust_N": float(apc_static_thrust)
                        if apc_static_thrust is not None
                        else None,
                        "prop_map_loaded": prop_map is not None,
                    },
                }
                self.project.analysis.mission_results = {
                    "summary": result.summary(),
                    "phases": phase_stats,
                    "propulsion": propulsion_snapshot,
                }
            self._populate_phase_stats_table(phase_stats)
            time_s, altitude_m, airspeed_mps, soc, phases = self._collect_profile(
                result
            )
            self._plot_mission_profile(
                time_s, altitude_m, airspeed_mps, soc, phases=phases
            )

            takeoff_distance = sum(
                p.distance_traveled_m
                for p in result.phases
                if p.phase_type
                in (MissionPhaseType.TAKEOFF_ROLL, MissionPhaseType.ROTATION)
            )
            landing_distance = sum(
                p.distance_traveled_m
                for p in result.phases
                if p.phase_type == MissionPhaseType.LANDING_ROLL
            )

            status = "SUCCESS" if result.success else f"FAILED: {result.failure_reason}"

            used_Wh = result.total_energy_Wh
            pct_used = (used_Wh / batt_cap_Wh) * 100.0 if batt_cap_Wh > 0 else 0.0

            summary = (
                f"{mode} | {status} | T {result.total_time_s:.1f}s | Dist {result.total_distance_m:.1f}m | "
                f"E {used_Wh:.2f}Wh ({pct_used:.1f}% of {batt_cap_Wh:.1f}Wh) | "
                f"TO {takeoff_distance:.1f}m | Lnd {landing_distance:.1f}m"
            )
            if apc_static_thrust is not None:
                summary += f" | APC static thrust {apc_static_thrust:.1f} N"
            self.m_summary.setText(summary)
            print("[DEBUG] Mission simulation completed successfully", flush=True)
        except Exception as e:
            import traceback
            print(f"[DEBUG] EXCEPTION CAUGHT: {e}", flush=True)
            traceback.print_exc()
            QMessageBox.critical(
                self, "Mission Error", f"Mission simulation failed: {e}"
            )

    def export_jsbsim_model(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "JSBSim Export", "No project loaded.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Export JSBSim Model", "")
        if not output_dir:
            return

        from core.jsbsim_gen import (
            JSBSimExportConfig,
            JSBSimPropellerTable,
            JSBSimPropulsionConfig,
            export_jsbsim_project,
        )
        from services.propulsion.battery_model import BatteryPackConfig, get_cell_preset
        from services.propulsion.propulsion_system import PropellerSpec

        self.sync_to_project()

        mission_results = getattr(self.project.analysis, "mission_results", None) or {}
        summary = mission_results.get("summary", {}) if isinstance(mission_results, dict) else {}
        max_power_w = summary.get("max_power_W") if summary else None

        kv = float(self.motor_kv.value())
        ri_mohm = float(self.motor_ri_mohm.value())
        io = float(self.motor_io.value())
        vio = float(self.motor_vio.value()) if self.motor_vio.value() > 0 else None
        n_motors = int(self.motor_count.value())

        chem_label = self.batt_chem.currentText()
        if "LiIon" in chem_label:
            preset_name = "LiIon_18650_HighCap"
        elif "LiFe" in chem_label:
            preset_name = "LiFe_Standard"
        else:
            preset_name = "LiPo_Standard"

        cell = get_cell_preset(preset_name)
        cell.capacity_Ah = float(self.batt_capacity_mAh.value()) / 1000.0
        c_rate = float(self.batt_c_rating.value())
        cell.C_rate_max_continuous = c_rate
        cell.C_rate_max_burst = max(c_rate * 1.5, c_rate)

        battery = BatteryPackConfig(
            cell=cell,
            n_series=int(self.batt_series.value()),
            n_parallel=int(self.batt_parallel.value()),
        )
        Vb = float(battery.V_nominal)

        if max_power_w is None or not np.isfinite(max_power_w):
            max_power_w = battery.P_max_continuous * max(1, n_motors)
        max_power_w = float(max(max_power_w, 0.0))
        engine_power_w = max_power_w / max(1, n_motors)

        prop_diam_in = float(self.prop_diam_in.value())
        prop_pitch_in = float(self.prop_pitch_in.value())
        prop_family_label = self.prop_family.currentText()
        family_map = {
            "APC Standard": "Standard",
            "APC Electric": "Electric",
            "APC SlowFly": "SlowFly",
            "Standard": "Standard",
            "Electric": "Electric",
            "SlowFly": "SlowFly",
        }
        prop_family = family_map.get(prop_family_label, "Electric")

        prop_table = None
        prop_map = self.m_prop_map if self.m_prop_map and self.m_prop_map.loaded else None
        if prop_map is None and self.m_prop_file_path:
            try:
                prop_map = APCMap()
                prop_map.load(self.m_prop_file_path, j_nodes=60, smooth=0.0)
            except Exception:
                prop_map = None

        if prop_map is not None:
            try:
                motor_for_apc = MotorProp(kv, ri_mohm, io, vio)
                rho0 = isa_density(float(self.m_cruise_alt.value()))
                diam_m = (prop_map.d_in or prop_diam_in) * 0.0254
                cruise_speed = float(self.m_cruise_speed.value())
                rpm_result = motor_for_apc.solve_rpm(
                    Vb,
                    max(1.0, cruise_speed),
                    rho0,
                    diam_m,
                    prop_map,
                )
                rpm = float(rpm_result.get("rpm", 0.0))
                if np.isfinite(rpm) and rpm > 0.0:
                    j_vals = np.asarray(prop_map.Jg, dtype=float)
                    ct_vals = []
                    cp_vals = []
                    for j_val in j_vals:
                        ct_val, cp_val = prop_map.ct_cp(rpm, float(j_val))
                        ct_vals.append(ct_val)
                        cp_vals.append(cp_val)
                    if j_vals.size > 0 and j_vals[0] > 0.0:
                        j_vals = np.concatenate([[0.0], j_vals])
                        ct_vals = [ct_vals[0]] + ct_vals
                        cp_vals = [cp_vals[0]] + cp_vals
                    prop_table = JSBSimPropellerTable(
                        advance_ratio=np.asarray(j_vals, dtype=float),
                        ct=np.asarray(ct_vals, dtype=float),
                        cp=np.asarray(cp_vals, dtype=float),
                    )
            except Exception:
                prop_table = None

        if prop_table is None:
            try:
                prop_spec = PropellerSpec(
                    diameter_in=prop_diam_in,
                    pitch_in=prop_pitch_in,
                    family=prop_family,
                )
                j_vals = np.linspace(0.0, 2.4, 25)
                ct_vals, cp_vals = prop_spec.get_coefficients(j_vals)
                prop_table = JSBSimPropellerTable(
                    advance_ratio=np.asarray(j_vals, dtype=float),
                    ct=np.asarray(ct_vals, dtype=float),
                    cp=np.asarray(cp_vals, dtype=float),
                )
            except Exception:
                prop_table = None

        max_thrust_N = summary.get("apc_static_thrust_N")
        if max_thrust_N is None:
            max_thrust_N = summary.get("max_thrust_N")

        propulsion_config = JSBSimPropulsionConfig(
            engine_power_w=engine_power_w,
            engine_count=max(1, n_motors),
            propeller_diameter_in=prop_diam_in,
            propeller_blades=2,
            propeller_gearratio=1.0,
            propeller_table=prop_table,
            max_thrust_N=max_thrust_N,
        )

        export_config = JSBSimExportConfig(
            model_name=self.project.wing.name,
            propulsion=propulsion_config,
        )

        if not summary:
            QMessageBox.information(
                self,
                "JSBSim Export",
                "No mission results found. Using battery continuous power for propulsion.",
            )

        # Check if FlightGear package export is requested
        if self.chk_flightgear.isChecked():
            from core.flightgear_gen import export_flightgear_package

            try:
                fg_result = export_flightgear_package(
                    self.project, output_dir, export_config
                )
                details = [
                    f"FlightGear package created at:",
                    f"  {fg_result.package_dir}",
                    "",
                    "Files generated:",
                    f"  {fg_result.set_xml_path}",
                    f"  {fg_result.jsbsim_result.aircraft_path}",
                ]
                if fg_result.jsbsim_result.engine_path:
                    details.append(f"  {fg_result.jsbsim_result.engine_path}")
                if fg_result.jsbsim_result.propeller_path:
                    details.append(f"  {fg_result.jsbsim_result.propeller_path}")
                details.extend([
                    f"  {fg_result.model_xml_path}",
                    f"  {fg_result.model_ac_path}",
                    "",
                    "To use in FlightGear:",
                    f"  fgfs --aircraft={fg_result.aircraft_name} "
                    f"--aircraft-dir={fg_result.package_dir}/..",
                ])
                QMessageBox.information(
                    self,
                    "FlightGear Export",
                    "\n".join(details),
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "FlightGear Export Error",
                    f"Failed to export FlightGear package:\n{e}",
                )
        else:
            result = export_jsbsim_project(self.project, output_dir, export_config)
            details = [f"Aircraft: {result.aircraft_path}"]
            if result.engine_path:
                details.append(f"Engine: {result.engine_path}")
            if result.propeller_path:
                details.append(f"Propeller: {result.propeller_path}")
            QMessageBox.information(
                self,
                "JSBSim Export",
                "Export complete:\n" + "\n".join(details),
            )

    def _mass_categories(self) -> List[str]:
        return [
            "battery", "motor", "esc", "servo", "payload", "receiver",
            "autopilot", "structure", "fuselage", "landing_gear", "other",
        ]

    def _refresh_mass_table(self):
        if not hasattr(self, "mass_table") or self.project is None:
            return
        self.mass_table.blockSignals(True)
        self.mass_table.setRowCount(len(self.project.aircraft.mass_items))
        for row, item in enumerate(self.project.aircraft.mass_items):
            values = [
                item.name,
                item.category,
                item.mass_kg,
                item.cg_m[0],
                item.cg_m[1],
                item.cg_m[2],
                item.notes,
            ]
            for col, value in enumerate(values):
                text = f"{value:.4f}" if isinstance(value, float) else str(value)
                self.mass_table.setItem(row, col, QTableWidgetItem(text))
        self.mass_table.blockSignals(False)
        self._update_mass_summary()

    def _sync_mass_table(self):
        if not hasattr(self, "mass_table") or self.project is None:
            return
        items = []
        categories = set(self._mass_categories())
        for row in range(self.mass_table.rowCount()):
            try:
                name = _table_text(self.mass_table, row, 0, f"Mass {row + 1}")
                category = _table_text(self.mass_table, row, 1, "other").lower()
                if category not in categories:
                    category = "other"
                mass_kg = max(0.0, float(_table_text(self.mass_table, row, 2, "0")))
                cg = (
                    float(_table_text(self.mass_table, row, 3, "0")),
                    float(_table_text(self.mass_table, row, 4, "0")),
                    float(_table_text(self.mass_table, row, 5, "0")),
                )
                uid = _uid_from_name(name, row)
                items.append(MassItem(uid=uid, name=name, mass_kg=mass_kg, cg_m=cg, category=category, notes=_table_text(self.mass_table, row, 6, "")))
            except ValueError:
                continue
        self.project.aircraft.mass_items = items
        self._update_mass_summary()

    def _on_mass_table_changed(self, _item):
        self._sync_mass_table()

    def _add_mass_item(self):
        if self.project is None:
            return
        idx = len(self.project.aircraft.mass_items) + 1
        self.project.aircraft.mass_items.append(
            MassItem(uid=f"mass_{idx}", name=f"Mass {idx}", mass_kg=0.1, cg_m=self.project.aircraft.reference.cg_m, category="other")
        )
        self._refresh_mass_table()

    def _remove_mass_item(self):
        if self.project is None:
            return
        row = self.mass_table.currentRow()
        if 0 <= row < len(self.project.aircraft.mass_items):
            self.project.aircraft.mass_items.pop(row)
            self._refresh_mass_table()

    def _apply_mass_cg(self):
        if self.project is None:
            return
        self._sync_mass_table()
        balance = apply_mass_balance_to_reference(self.project.aircraft)
        self._update_mass_summary(balance)

    def _update_mass_summary(self, balance=None):
        if not hasattr(self, "mass_total_label") or self.project is None:
            return
        balance = balance or compute_mass_balance(self.project.aircraft)
        self.mass_total_label.setText(f"{balance.total_mass_kg:.4f} kg")
        self.mass_cg_label.setText(f"({balance.cg_m[0]:.4f}, {balance.cg_m[1]:.4f}, {balance.cg_m[2]:.4f})")
        self.mass_warning_label.setText("; ".join(balance.warnings) if balance.warnings else "-")

    def _battery_nominal_voltage(self) -> float:
        chem = self.batt_chem.currentText()
        cell_v = 3.7
        if "LiIon" in chem:
            cell_v = 3.6
        elif "LiFePO4" in chem:
            cell_v = 3.2
        return float(self.batt_series.value()) * cell_v

    def _mission_propulsion_snapshot(self) -> Dict[str, Any]:
        return {
            "uid": "mission_propulsion",
            "name": "Mission propulsion",
            "source": "mission_gui",
            "motor_count": int(self.motor_count.value()),
            "motor": {
                "kv_rpm_per_v": float(self.motor_kv.value()),
                "ri_mohm": float(self.motor_ri_mohm.value()),
                "io_a": float(self.motor_io.value()),
                "v_at_io": float(self.motor_vio.value()),
            },
            "battery": {
                "chemistry": self.batt_chem.currentText(),
                "series": int(self.batt_series.value()),
                "parallel": int(self.batt_parallel.value()),
                "capacity_mAh": float(self.batt_capacity_mAh.value()),
                "c_rating": float(self.batt_c_rating.value()),
                "nominal_voltage_v": self._battery_nominal_voltage(),
            },
            "propeller": {
                "diameter_in": float(self.prop_diam_in.value()),
                "pitch_in": float(self.prop_pitch_in.value()),
                "family": self.prop_family.currentText(),
                "apc_file_path": self.m_prop_file_path,
            },
            "matching": {
                "source": "APC map" if self.m_prop_file_path else "parametric",
                "matched": bool(self.m_prop_file_path or self.prop_family.currentText()),
            },
        }

    def _sync_propulsion_systems_to_aircraft(self):
        if self.project is None:
            return
        snapshot = self._mission_propulsion_snapshot()
        systems = [s for s in self.project.aircraft.propulsion_systems if s.get("uid") != snapshot["uid"]]
        systems.insert(0, snapshot)
        self.project.aircraft.propulsion_systems = systems
        self._refresh_propulsion_table()

    def _refresh_propulsion_table(self):
        if not hasattr(self, "propulsion_table") or self.project is None:
            return
        systems = self.project.aircraft.propulsion_systems
        self.propulsion_table.setRowCount(len(systems))
        for row, system in enumerate(systems):
            motor = system.get("motor", {})
            battery = system.get("battery", {})
            prop = system.get("propeller", {})
            matching = system.get("matching", {})
            values = [
                system.get("uid", ""),
                system.get("name", ""),
                system.get("motor_count", 1),
                motor.get("kv_rpm_per_v", ""),
                f"{battery.get('series', '-') }S{battery.get('parallel', '-') }P {battery.get('capacity_mAh', '-') } mAh",
                f"{prop.get('diameter_in', '-') }x{prop.get('pitch_in', '-') } {prop.get('family', '')}",
                Path(prop.get("apc_file_path", "")).name if prop.get("apc_file_path") else "",
                "yes" if matching.get("matched") else "no",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.propulsion_table.setItem(row, col, item)

    def _remove_propulsion_system(self):
        if self.project is None or not hasattr(self, "propulsion_table"):
            return
        row = self.propulsion_table.currentRow()
        if 0 <= row < len(self.project.aircraft.propulsion_systems):
            self.project.aircraft.propulsion_systems.pop(row)
            self._refresh_propulsion_table()

    def update_from_project(self):
        if self.project is None:
            return
        self._refresh_mass_table()
        gui_settings = getattr(self.project.mission, "gui_settings", None)
        if gui_settings:
            self._apply_gui_settings(gui_settings)

        mission_results = getattr(self.project.analysis, "mission_results", None)
        if mission_results:
            summary = mission_results.get("summary", {})
            phases = mission_results.get("phases", [])

            if summary:
                status = "SUCCESS" if summary.get("success") else "FAILED"
                if summary.get("failure_reason"):
                    status = f"FAILED: {summary.get('failure_reason')}"
                summary_text = (
                    f"Saved Mission | {status} | T {summary.get('total_time_s', 0.0):.1f}s | "
                    f"Dist {summary.get('total_distance_m', 0.0):.1f}m | "
                    f"E {summary.get('total_energy_Wh', 0.0):.2f}Wh | "
                    f"Final SOC {summary.get('final_SOC_percent', 0.0):.1f}%"
                )
                self.m_summary.setText(summary_text)

            self._populate_phase_stats_table(phases)
        self._refresh_propulsion_table()

    def sync_to_project(self):
        if self.project is None:
            return
        self._sync_mass_table()
        self._sync_propulsion_systems_to_aircraft()
        self.project.mission.gui_settings = self._collect_gui_settings()

    def _collect_gui_settings(self) -> Dict[str, Any]:
        return {
            "weight_kg": float(self.m_weight_kg.value()),
            "batt_chem": self.batt_chem.currentText(),
            "batt_series": int(self.batt_series.value()),
            "batt_parallel": int(self.batt_parallel.value()),
            "batt_capacity_mAh": float(self.batt_capacity_mAh.value()),
            "batt_c_rating": float(self.batt_c_rating.value()),
            "motor_kv": float(self.motor_kv.value()),
            "motor_ri_mohm": float(self.motor_ri_mohm.value()),
            "motor_io": float(self.motor_io.value()),
            "motor_vio": float(self.motor_vio.value()),
            "motor_count": int(self.motor_count.value()),
            "prop_diam_in": float(self.prop_diam_in.value()),
            "prop_pitch_in": float(self.prop_pitch_in.value()),
            "prop_family": self.prop_family.currentText(),
            "prop_file_path": self.m_prop_file_path,
            "surface_type": self.surface_type.currentText(),
            "takeoff_dist_m": float(self.m_takeoff_dist.value()),
            "dynamics_mode": self.dynamics_mode.currentText(),
            "cruise_speed_mps": float(self.m_cruise_speed.value()),
            "cruise_alt_m": float(self.m_cruise_alt.value()),
            "cruise_duration": float(self.m_cruise_duration.value()),
            "cruise_time_unit": self.m_cruise_time_unit.currentText(),
        }

    def _apply_gui_settings(self, settings: Dict[str, Any]) -> None:
        def _set_combo(combo: QComboBox, value: Any) -> None:
            if value is None:
                return
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def _set_spin(spin: Any, value: Any) -> None:
            if value is None:
                return
            try:
                # QSpinBox expects int, QDoubleSpinBox expects float
                if hasattr(spin, 'decimals'):
                    spin.setValue(float(value))
                else:
                    spin.setValue(int(float(value)))
            except Exception:
                return

        _set_spin(self.m_weight_kg, settings.get("weight_kg"))
        _set_combo(self.batt_chem, settings.get("batt_chem"))
        _set_spin(self.batt_series, settings.get("batt_series"))
        _set_spin(self.batt_parallel, settings.get("batt_parallel"))
        _set_spin(self.batt_capacity_mAh, settings.get("batt_capacity_mAh"))
        _set_spin(self.batt_c_rating, settings.get("batt_c_rating"))
        _set_spin(self.motor_kv, settings.get("motor_kv"))
        _set_spin(self.motor_ri_mohm, settings.get("motor_ri_mohm"))
        _set_spin(self.motor_io, settings.get("motor_io"))
        _set_spin(self.motor_vio, settings.get("motor_vio"))
        _set_spin(self.motor_count, settings.get("motor_count"))
        _set_spin(self.prop_diam_in, settings.get("prop_diam_in"))
        _set_spin(self.prop_pitch_in, settings.get("prop_pitch_in"))
        _set_combo(self.prop_family, settings.get("prop_family"))
        _set_combo(self.surface_type, settings.get("surface_type"))
        _set_spin(self.m_takeoff_dist, settings.get("takeoff_dist_m"))
        _set_combo(self.dynamics_mode, settings.get("dynamics_mode"))
        _set_spin(self.m_cruise_speed, settings.get("cruise_speed_mps"))
        _set_spin(self.m_cruise_alt, settings.get("cruise_alt_m"))
        _set_spin(self.m_cruise_duration, settings.get("cruise_duration"))
        _set_combo(self.m_cruise_time_unit, settings.get("cruise_time_unit"))
        self._update_mu()

        prop_file = settings.get("prop_file_path")
        if prop_file:
            try:
                path = Path(prop_file)
                if path.exists():
                    self.m_prop_map.load(str(path), j_nodes=80, smooth=0.0)
                    self.prop_file_lbl.setText(path.name)
                    self.m_prop_file_path = str(path)
                    if self.m_prop_map.d_in:
                        self.prop_diam_in.setValue(float(self.m_prop_map.d_in))
            except Exception:
                pass

    def _populate_phase_stats_table(self, phases: List[Dict[str, Any]]) -> None:
        if not phases:
            self.m_table.setRowCount(0)
            return

        self.m_table.setRowCount(len(phases))
        self.m_table.setHorizontalHeaderLabels(
            [
                "phase",
                "thr_min",
                "thr_avg",
                "thr_max",
                "T_min",
                "T_avg",
                "T_max",
                "dur_s",
                "dist_m",
                "E_Wh",
            ]
        )

        def _fmt(val: Any, fmt: str) -> str:
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return "-"
            try:
                return format(float(val), fmt)
            except Exception:
                return "-"

        for r, phase in enumerate(phases):
            throttle = phase.get("throttle") or {}
            thrust = phase.get("thrust_N") or {}
            row_vals = [
                phase.get("name", ""),
                _fmt(throttle.get("min"), ".2f"),
                _fmt(throttle.get("avg"), ".2f"),
                _fmt(throttle.get("max"), ".2f"),
                _fmt(thrust.get("min"), ".1f"),
                _fmt(thrust.get("avg"), ".1f"),
                _fmt(thrust.get("max"), ".1f"),
                _fmt(phase.get("duration_s"), ".1f"),
                _fmt(phase.get("distance_m"), ".1f"),
                _fmt(phase.get("energy_Wh"), ".2f"),
            ]
            for c, v in enumerate(row_vals):
                self.m_table.setItem(r, c, QTableWidgetItem(str(v)))

    def _import_from_geometry(self):
        """Import parameters from the geometry and analysis sections."""
        try:
            # 1. Weight
            gtw = self.project.wing.twist_trim.gross_takeoff_weight_kg
            if gtw > 0:
                self.m_weight_kg.setValue(gtw)

            # 2. Wing Area (Actual)
            area = self.project.wing.planform.actual_area()
            if area > 0:
                self.m_Swing.setValue(area)

            # 3. Cruise Altitude
            alt = self.project.wing.twist_trim.cruise_altitude_m
            self.m_cruise_alt.setValue(alt)

            # 4. Design CL & CD & Speed from Performance Metrics (Preferred)
            perf = self.project.analysis.performance_metrics
            if perf:
                # Cruise
                if "cruise_cl" in perf:
                    self.m_cl.setValue(perf["cruise_cl"])
                if "cruise_cd" in perf:
                    self.m_cd_wing.setValue(perf["cruise_cd"])
                if "cruise_velocity" in perf:
                    self.m_cruise_speed.setValue(perf["cruise_velocity"])

                # Takeoff (Optional mapping? Mission tab doesn't have explicit takeoff CL/CD inputs for the mission profile,
                # but uses CLmax. We can stick to estimated CLmax from twist trim or use takeoff_cl from perf if desired.
                # For now, let's keep CLmax from twist trim as it's the limit, not the operating point.)

                self.m_summary.setText(
                    f"Imported performance metrics. V={perf.get('cruise_velocity', 0):.1f} m/s, CL={perf.get('cruise_cl', 0):.3f}, CD={perf.get('cruise_cd', 0):.4f}"
                )

            else:
                # Fallback to Twist Trim settings + Interpolation
                cl = self.project.wing.twist_trim.design_cl
                self.m_cl.setValue(cl)

                # Drag (CD) from Polars
                polars = self.project.analysis.polars
                if polars and "CL" in polars and "CD" in polars:
                    cls = np.array(polars["CL"])
                    cds = np.array(polars["CD"])
                    idx = np.argsort(cls)
                    cd_interp = np.interp(cl, cls[idx], cds[idx])
                    self.m_cd_wing.setValue(float(cd_interp))

                # Calculate Cruise Speed
                if area > 0 and cl > 0 and gtw > 0:
                    rho = isa_density(alt)
                    weight_N = gtw * 9.80665
                    v_cruise = np.sqrt(2 * weight_N / (rho * area * cl))
                    self.m_cruise_speed.setValue(float(v_cruise))

                self.m_summary.setText(
                    "Imported geometry data (Fallback). Calculated V and interpolated CD."
                )

            # 5. CL Max (Always from Twist Trim estimation)
            cl_max = self.project.wing.twist_trim.estimated_cl_max
            self.m_clmax.setValue(cl_max)

        except Exception as e:
            QMessageBox.warning(
                self, "Import Error", f"Failed to import geometry data: {e}"
            )


def _table_text(table: QTableWidget, row: int, col: int, default: Any) -> str:
    item = table.item(row, col)
    return item.text().strip() if item and item.text().strip() else str(default)


def _uid_from_name(name: str, row: int) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return text or f"mass_{row + 1}"
