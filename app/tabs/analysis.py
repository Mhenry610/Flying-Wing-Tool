from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from scipy.interpolate import RegularGridInterpolator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QMessageBox,
    QComboBox, QCheckBox, QScrollArea, QSplitter
)

from core.state import Project
from services.aero_analysis import AeroAnalysisService, OptimizationObjectives, OptimizationConstraints

class AnalysisTab(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.service = AeroAnalysisService()
        
        # State for interactive plots
        self._grid_Res = None
        self._grid_alphas = None
        self._Z_by_ax = {}
        self._label_by_ax = {}
        self._click_artists_by_ax = {}
        self._click_figtexts_by_ax = {}
        self._click_cid = None
        self._cbs = []
        self._opt_markers = []
        self._opt_design_point = None
        self._kulfan_opt = None # Store optimized airfoil

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        
        # Splitter: Left (Controls) | Right (Plots)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- Left Panel: Controls ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # 1. Inputs Group
        input_group = QGroupBox("Analysis Inputs")
        il = QGridLayout(input_group)
        
        il.addWidget(QLabel("Airfoil Source"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Root Airfoil", "Tip Airfoil", "Custom"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        il.addWidget(self.source_combo, 0, 1, 1, 3)
        
        il.addWidget(QLabel("Airfoil Name/Path"), 1, 0)
        self.airfoil_edit = QLineEdit()
        self.airfoil_edit.setPlaceholderText("e.g. naca2412 or path/to/file.dat")
        il.addWidget(self.airfoil_edit, 1, 1, 1, 3)
        
        # Alpha Range
        il.addWidget(QLabel("Alpha Min [deg]"), 2, 0)
        self.a_min = QDoubleSpinBox(); self.a_min.setRange(-90, 90); self.a_min.setValue(-5.0)
        il.addWidget(self.a_min, 2, 1)
        
        il.addWidget(QLabel("Alpha Max [deg]"), 2, 2)
        self.a_max = QDoubleSpinBox(); self.a_max.setRange(-90, 90); self.a_max.setValue(15.0)
        il.addWidget(self.a_max, 2, 3)
        
        il.addWidget(QLabel("Points"), 3, 0)
        self.a_pts = QSpinBox(); self.a_pts.setRange(5, 200); self.a_pts.setValue(40)
        il.addWidget(self.a_pts, 3, 1)
        
        # Re Range
        il.addWidget(QLabel("Re Min"), 4, 0)
        self.re_min = QDoubleSpinBox(); self.re_min.setRange(1e3, 1e8); self.re_min.setValue(1e4); self.re_min.setDecimals(0)
        il.addWidget(self.re_min, 4, 1)
        
        il.addWidget(QLabel("Re Max"), 4, 2)
        self.re_max = QDoubleSpinBox(); self.re_max.setRange(1e3, 1e9); self.re_max.setValue(5e6); self.re_max.setDecimals(0)
        il.addWidget(self.re_max, 4, 3)
        
        il.addWidget(QLabel("Points"), 5, 0)
        self.re_pts = QSpinBox(); self.re_pts.setRange(5, 200); self.re_pts.setValue(40)
        il.addWidget(self.re_pts, 5, 1)
        
        il.addWidget(QLabel("Mach"), 6, 0)
        self.mach = QDoubleSpinBox(); self.mach.setRange(0, 0.95); self.mach.setValue(0.0); self.mach.setSingleStep(0.05)
        il.addWidget(self.mach, 6, 1)
        
        run_btn = QPushButton("Run NeuralFoil Sweep")
        run_btn.clicked.connect(self.run_sweep)
        il.addWidget(run_btn, 7, 0, 1, 4)
        
        controls_layout.addWidget(input_group)
        
        # 2. Optimization Group
        opt_group = QGroupBox("Kulfan Optimization")
        ol = QGridLayout(opt_group)
        
        # Objectives
        ol.addWidget(QLabel("Objectives:"), 0, 0, 1, 4)
        
        self.chk_max_ld = QCheckBox("Max L/D"); self.chk_max_ld.setChecked(True)
        ol.addWidget(self.chk_max_ld, 1, 0, 1, 2)
        self.w_max_ld = QDoubleSpinBox(); self.w_max_ld.setValue(1.0)
        ol.addWidget(QLabel("w:"), 1, 2); ol.addWidget(self.w_max_ld, 1, 3)
        
        self.chk_min_cd = QCheckBox("Min Cd"); self.chk_min_cd.setChecked(False)
        ol.addWidget(self.chk_min_cd, 2, 0, 1, 2)
        self.w_min_cd = QDoubleSpinBox(); self.w_min_cd.setValue(1.0)
        ol.addWidget(QLabel("w:"), 2, 2); ol.addWidget(self.w_min_cd, 2, 3)
        
        self.chk_target_cl = QCheckBox("Target Cl"); self.chk_target_cl.setChecked(False)
        ol.addWidget(self.chk_target_cl, 3, 0, 1, 2)
        self.w_target_cl = QDoubleSpinBox(); self.w_target_cl.setValue(1.0)
        ol.addWidget(QLabel("w:"), 3, 2); ol.addWidget(self.w_target_cl, 3, 3)
        
        self.chk_target_cm = QCheckBox("Target Cm"); self.chk_target_cm.setChecked(False)
        ol.addWidget(self.chk_target_cm, 4, 0, 1, 2)
        self.w_target_cm = QDoubleSpinBox(); self.w_target_cm.setValue(1.0)
        ol.addWidget(QLabel("w:"), 4, 2); ol.addWidget(self.w_target_cm, 4, 3)
        
        # Targets
        ol.addWidget(QLabel("Target Cl val:"), 5, 0)
        self.target_cl_val = QDoubleSpinBox(); self.target_cl_val.setValue(0.8)
        ol.addWidget(self.target_cl_val, 5, 1)
        
        ol.addWidget(QLabel("Target Cm val:"), 5, 2)
        self.target_cm_val = QDoubleSpinBox(); self.target_cm_val.setRange(-1, 1); self.target_cm_val.setValue(-0.05)
        ol.addWidget(self.target_cm_val, 5, 3)
        
        # Operating Point
        ol.addWidget(QLabel("Opt Alpha:"), 6, 0)
        self.opt_alpha = QDoubleSpinBox(); self.opt_alpha.setRange(-20, 20); self.opt_alpha.setValue(4.0)
        ol.addWidget(self.opt_alpha, 6, 1)
        
        ol.addWidget(QLabel("Opt Re:"), 6, 2)
        self.opt_re = QDoubleSpinBox(); self.opt_re.setRange(1e3, 1e8); self.opt_re.setValue(3e5); self.opt_re.setDecimals(0)
        ol.addWidget(self.opt_re, 6, 3)
        
        # Constraints
        ol.addWidget(QLabel("Min t/c:"), 7, 0)
        self.min_tc = QDoubleSpinBox(); self.min_tc.setValue(0.012); self.min_tc.setSingleStep(0.001); self.min_tc.setDecimals(3)
        ol.addWidget(self.min_tc, 7, 1)
        
        self.opt_alpha_var = QCheckBox("Optimize Alpha"); self.opt_alpha_var.setChecked(True)
        ol.addWidget(self.opt_alpha_var, 7, 2, 1, 2)
        
        # Actions
        self.use_opt_check = QCheckBox("Use optimized airfoil for sweep")
        self.use_opt_check.setChecked(True)
        ol.addWidget(self.use_opt_check, 8, 0, 1, 4)
        
        opt_btn = QPushButton("Run Optimization")
        opt_btn.clicked.connect(self.run_optimization)
        ol.addWidget(opt_btn, 9, 0, 1, 4)
        
        controls_layout.addWidget(opt_group)
        controls_layout.addStretch()
        
        scroll.setWidget(controls_widget)
        splitter.addWidget(scroll)
        
        # --- Right Panel: Plots ---
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)
        
        # Set splitter ratio
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        # Initial Plot Setup
        self._create_plots()
        self._update_inputs_from_project()

    def _create_plots(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.35, wspace=0.25)
        self.ax_geom = self.figure.add_subplot(gs[0, :])
        self.ax_cl = self.figure.add_subplot(gs[1, 0])
        self.ax_cd = self.figure.add_subplot(gs[1, 1])
        self.ax_ld = self.figure.add_subplot(gs[2, 0])
        self.ax_cm = self.figure.add_subplot(gs[2, 1])
        
        self.ax_geom.set_title("Airfoil Geometry")
        self.ax_geom.set_aspect("equal")
        self.ax_geom.grid(True, alpha=0.3)
        
        for ax, title in zip([self.ax_cl, self.ax_cd, self.ax_ld, self.ax_cm], 
                             ["CL", "CD", "L/D", "Cm"]):
            ax.set_title(title)
            ax.set_xlabel("Re")
            ax.set_ylabel("Alpha [deg]")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            
        self.canvas.draw_idle()
        
        # Connect click event
        if self._click_cid:
            self.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.canvas.mpl_connect('button_press_event', self._on_click_contour)

    def _update_inputs_from_project(self):
        self.update_from_project()

    def _on_source_changed(self, text: str):
        if text == "Root Airfoil":
            self.airfoil_edit.setText(self.project.wing.airfoil.root_airfoil)
            self.airfoil_edit.setEnabled(False)
        elif text == "Tip Airfoil":
            self.airfoil_edit.setText(self.project.wing.airfoil.tip_airfoil)
            self.airfoil_edit.setEnabled(False)
        else:
            self.airfoil_edit.setEnabled(True)
            self.airfoil_edit.setFocus()

    def run_sweep(self):
        try:
            # 1. Get Airfoil
            if self.use_opt_check.isChecked() and self._kulfan_opt is not None:
                af = self._kulfan_opt
            else:
                af_name = self.airfoil_edit.text()
                af = self.service.make_airfoil(af_name)
            
            # 2. Build Grid
            alphas = np.linspace(self.a_min.value(), self.a_max.value(), self.a_pts.value())
            Res = np.logspace(np.log10(self.re_min.value()), np.log10(self.re_max.value()), self.re_pts.value())
            Re_grid, alpha_grid = np.meshgrid(Res, alphas)
            
            # 3. Run Analysis
            mach = self.mach.value()
            aero = self.service.eval_neuralfoil(af, alpha_grid, Re_grid, Mach=mach)
            
            # 4. Plot
            self._plot_results(af, Re_grid, alpha_grid, aero)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Sweep failed: {e}")

    def _plot_results(self, af, Re_grid, alpha_grid, aero):
        # Clear previous colorbars
        for cb in self._cbs:
            try: cb.remove()
            except: pass
        self._cbs = []
        
        # Geometry
        self.ax_geom.clear()
        try:
            coords = af.coordinates
            if coords is not None:
                self.ax_geom.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=1.5)
            self.ax_geom.set_aspect("equal")
            self.ax_geom.grid(True, alpha=0.3)
            self.ax_geom.set_title(f"Airfoil: {getattr(af, 'name', 'Custom')}")
            self.ax_geom.set_xlabel("x/c")
            self.ax_geom.set_ylabel("y/c")
            
            # Plot Camber Line
            try:
                x_camber = np.linspace(0, 1, 100)
                y_camber = af.local_camber(x_camber)
                self.ax_geom.plot(x_camber, y_camber, 'g--', linewidth=1.0, alpha=0.5, label='Camber')
                self.ax_geom.legend(fontsize='small')
            except Exception:
                pass # Ignore if camber calc fails
        except Exception as e:
            self.ax_geom.text(0.5, 0.5, f"Error plotting airfoil:\n{str(e)}", 
                            transform=self.ax_geom.transAxes, ha='center', va='center')
            
        # Contours
        CL = aero["CL"]
        CD = np.maximum(aero["CD"], 1e-6)
        CM = aero["CM"]
        LD = CL / CD
        
        self._grid_Res = Re_grid[0, :] # Assuming meshgrid structure
        self._grid_alphas = alpha_grid[:, 0]
        self._Z_by_ax = {self.ax_cl: CL, self.ax_cd: CD, self.ax_ld: LD, self.ax_cm: CM}
        self._label_by_ax = {self.ax_cl: 'CL', self.ax_cd: 'CD', self.ax_ld: 'L/D', self.ax_cm: 'Cm'}
        
        def plot_contour(ax, Z, label):
            ax.clear()
            cs = ax.contourf(Re_grid, alpha_grid, Z, levels=20)
            ax.contour(Re_grid, alpha_grid, Z, levels=10, colors='k', linewidths=0.5, alpha=0.3)
            cb = self.figure.colorbar(cs, ax=ax)
            cb.set_label(label)
            self._cbs.append(cb)
            ax.set_xscale("log")
            ax.set_xlabel("Re")
            ax.set_ylabel("Alpha [deg]")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            
        plot_contour(self.ax_cl, CL, "CL")
        plot_contour(self.ax_cd, CD, "CD")
        plot_contour(self.ax_ld, LD, "L/D")
        plot_contour(self.ax_cm, CM, "Cm")
        
        # Plot design point if exists
        if self._opt_design_point:
            alpha_opt, Re_opt = self._opt_design_point
            for ax in [self.ax_cl, self.ax_cd, self.ax_ld, self.ax_cm]:
                ax.plot([Re_opt], [alpha_opt], 'ro', mfc='none', mew=2, ms=8)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def run_optimization(self):
        try:
            # 1. Setup Objectives
            objs = OptimizationObjectives(
                max_ld=self.chk_max_ld.isChecked(),
                min_cd=self.chk_min_cd.isChecked(),
                target_cl=self.target_cl_val.value() if self.chk_target_cl.isChecked() else None,
                target_cm=self.target_cm_val.value() if self.chk_target_cm.isChecked() else None,
                w_max_ld=self.w_max_ld.value(),
                w_min_cd=self.w_min_cd.value(),
                w_target_cl=self.w_target_cl.value(),
                w_target_cm=self.w_target_cm.value()
            )
            
            # 2. Setup Constraints
            constrs = OptimizationConstraints(
                min_thickness=self.min_tc.value(),
                optimize_alpha=self.opt_alpha_var.isChecked()
            )
            
            # 3. Initial Airfoil
            af_name = self.airfoil_edit.text()
            af_init = self.service.make_airfoil(af_name)
            
            # 4. Run
            alpha = self.opt_alpha.value()
            Re = self.opt_re.value()
            mach = self.mach.value()
            
            self._opt_design_point = (alpha, Re)
            
            af_opt, res = self.service.optimize_kulfan(
                af_init, alpha, Re, mach, objs, constrs
            )
            
            self._kulfan_opt = af_opt
            self.use_opt_check.setChecked(True)
            
            # Show result summary
            msg = "Optimization Complete!\n\n"
            for k, v in res.items():
                msg += f"{k}: {v:.4f}\n"
            QMessageBox.information(self, "Success", msg)
            
            # Auto-run sweep with new airfoil
            self.run_sweep()
            
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error", str(e))

    def _on_click_contour(self, event):
        if event.inaxes not in self._Z_by_ax: return
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        # Interpolate
        Z = self._Z_by_ax[ax]
        
        val_str = "N/A"
        try:
            # Grid is (alphas, Res) corresponding to Z's shape
            # x is Re, y is Alpha
            # RegularGridInterpolator requires strictly increasing grid points.
            # We'll sort them if necessary, but for now assume standard usage or handle simple inversion.
            
            grid_alphas = self._grid_alphas
            grid_Res = self._grid_Res
            
            # Check for decreasing arrays and flip if needed (simple check)
            # Note: This is a quick fix; robust handling would involve sorting Z too.
            # But since Z corresponds to the grid, if we flip the grid we must flip Z.
            # Let's just try direct interpolation. If it fails due to unsorted, we catch it.
            
            interp = RegularGridInterpolator((grid_alphas, grid_Res), Z, bounds_error=False, fill_value=None)
            val = interp((y, x))
            val_str = f"{val.item():.4f}"
        except Exception as e:
            # Fallback or error logging
            print(f"Interpolation failed: {e}")
        
        # Remove old annotations
        if ax in self._click_artists_by_ax:
            for a in self._click_artists_by_ax[ax]: a.remove()
        self._click_artists_by_ax[ax] = []
        
        # Add marker
        m, = ax.plot([x], [y], 'yo', markeredgecolor='k')
        self._click_artists_by_ax[ax].append(m)
        
        # Add text
        lbl = self._label_by_ax.get(ax, "Val")
        txt = f"Re={x:.2e}\nAlpha={y:.2f}\n{lbl}={val_str}"
        ann = ax.annotate(txt, (x, y), xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        self._click_artists_by_ax[ax].append(ann)
        self.canvas.draw_idle()

    def update_from_project(self):
        settings = getattr(self.project.analysis, "gui_settings", {}).get("analysis_tab", {})
        if settings:
            self._apply_gui_settings(settings)

        # Refresh airfoil names if they changed
        if self.source_combo.currentText() == "Root Airfoil":
            self.airfoil_edit.setText(self.project.wing.airfoil.root_airfoil)
        elif self.source_combo.currentText() == "Tip Airfoil":
            self.airfoil_edit.setText(self.project.wing.airfoil.tip_airfoil)

    def sync_to_project(self):
        if self.project is None:
            return
        settings = getattr(self.project.analysis, "gui_settings", None)
        if settings is None:
            self.project.analysis.gui_settings = {}
            settings = self.project.analysis.gui_settings
        settings["analysis_tab"] = self._collect_gui_settings()

    def _collect_gui_settings(self) -> Dict[str, Any]:
        return {
            "airfoil_source": self.source_combo.currentText(),
            "airfoil_text": self.airfoil_edit.text(),
            "alpha_min_deg": float(self.a_min.value()),
            "alpha_max_deg": float(self.a_max.value()),
            "alpha_points": int(self.a_pts.value()),
            "re_min": float(self.re_min.value()),
            "re_max": float(self.re_max.value()),
            "re_points": int(self.re_pts.value()),
            "mach": float(self.mach.value()),
            "max_ld_enabled": bool(self.chk_max_ld.isChecked()),
            "min_cd_enabled": bool(self.chk_min_cd.isChecked()),
            "target_cl_enabled": bool(self.chk_target_cl.isChecked()),
            "target_cm_enabled": bool(self.chk_target_cm.isChecked()),
            "w_max_ld": float(self.w_max_ld.value()),
            "w_min_cd": float(self.w_min_cd.value()),
            "w_target_cl": float(self.w_target_cl.value()),
            "w_target_cm": float(self.w_target_cm.value()),
            "target_cl_value": float(self.target_cl_val.value()),
            "target_cm_value": float(self.target_cm_val.value()),
            "opt_alpha": float(self.opt_alpha.value()),
            "opt_re": float(self.opt_re.value()),
            "min_tc": float(self.min_tc.value()),
            "optimize_alpha": bool(self.opt_alpha_var.isChecked()),
            "use_optimized_airfoil": bool(self.use_opt_check.isChecked()),
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
                if hasattr(spin, "decimals"):
                    spin.setValue(float(value))
                else:
                    spin.setValue(int(float(value)))
            except Exception:
                return

        def _set_check(check: QCheckBox, value: Any) -> None:
            if value is not None:
                check.setChecked(bool(value))

        _set_combo(self.source_combo, settings.get("airfoil_source"))
        if self.source_combo.currentText() == "Custom":
            self.airfoil_edit.setText(str(settings.get("airfoil_text", "")))
        _set_spin(self.a_min, settings.get("alpha_min_deg"))
        _set_spin(self.a_max, settings.get("alpha_max_deg"))
        _set_spin(self.a_pts, settings.get("alpha_points"))
        _set_spin(self.re_min, settings.get("re_min"))
        _set_spin(self.re_max, settings.get("re_max"))
        _set_spin(self.re_pts, settings.get("re_points"))
        _set_spin(self.mach, settings.get("mach"))
        _set_check(self.chk_max_ld, settings.get("max_ld_enabled"))
        _set_check(self.chk_min_cd, settings.get("min_cd_enabled"))
        _set_check(self.chk_target_cl, settings.get("target_cl_enabled"))
        _set_check(self.chk_target_cm, settings.get("target_cm_enabled"))
        _set_spin(self.w_max_ld, settings.get("w_max_ld"))
        _set_spin(self.w_min_cd, settings.get("w_min_cd"))
        _set_spin(self.w_target_cl, settings.get("w_target_cl"))
        _set_spin(self.w_target_cm, settings.get("w_target_cm"))
        _set_spin(self.target_cl_val, settings.get("target_cl_value"))
        _set_spin(self.target_cm_val, settings.get("target_cm_value"))
        _set_spin(self.opt_alpha, settings.get("opt_alpha"))
        _set_spin(self.opt_re, settings.get("opt_re"))
        _set_spin(self.min_tc, settings.get("min_tc"))
        _set_check(self.opt_alpha_var, settings.get("optimize_alpha"))
        _set_check(self.use_opt_check, settings.get("use_optimized_airfoil"))
