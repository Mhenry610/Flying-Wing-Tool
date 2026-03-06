import os
import sys
import datetime
import traceback
from typing import Dict, Any, Optional

# Ensure project root on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Qt and PyVistaQt
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYVISTA_QT_API", "PySide6")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# App services and core imports
from refactor.core.parsing import parse_wing_data  # [refactor.core.parsing.parse_wing_data()](refactor/core/parsing.py:1)
from refactor.core.parsing import prettify_xml     # [refactor.core.parsing.prettify_xml()](refactor/core/parsing.py:1)
from refactor.core.cpacs_gen import generate_cpacs_xml  # [refactor.core.cpacs_gen.generate_cpacs_xml()](refactor/core/cpacs_gen.py:1)
from refactor.core.xflr5_gen import (  # [refactor.core.xflr5_gen.create_xflr5_xml_structure()](refactor/core/xflr5_gen.py:1)
    create_xflr5_xml_structure,
    create_airfoil_dat_files,
)
from refactor.services.viewer import process_cpacs_data  # [refactor.services.viewer.process_cpacs_data()](refactor/services/viewer.py:1)
from refactor.services.step_export import (              # [refactor.services.step_export.build_full_step_from_processed()](refactor/services/step_export.py:1)
    build_full_step_from_processed,
    write_step,
)
from refactor.services.fixture_export import (           # [refactor.services.fixture_export.FixtureParams](refactor/services/fixture_export.py:1)
    FixtureParams,
    build_fixture_layout,
    write_fixture_step,
)

# PyVistaQt for inline 3D
_pyvistaqt_import_ok = True
try:
    from pyvistaqt import BackgroundPlotter
    import pyvista as pv
    # Safer defaults for performance and predictability
    try:
        pv.set_jupyter_backend(None)
    except Exception:
        pass
    try:
        pv.global_theme.smooth_shading = True
        pv.global_theme.show_edges = False
        # Set a subtle gray gradient background by default
        # Using two-color background (top/bottom) for a gradient feel
        pv.global_theme.background = "dimgray"
        pv.global_theme.background_color = "dimgray"
        pv.global_theme.background2 = (0.22, 0.22, 0.22)  # slightly darker at bottom
        pv.global_theme.floor_enable = False
    except Exception:
        pass
except Exception:
    _pyvistaqt_import_ok = False


class LogPane(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setFont(QtGui.QFont("Consolas", 10))

    def log(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        self.appendPlainText(line)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class InlinePyVistaWidget(QtWidgets.QWidget):
    """
    Native Qt host for PyVista BackgroundPlotter without separate top-level window.
    This wraps BackgroundPlotter and extracts the underlying QWidget (app_window/window/_main_window/_window)
    then sets it as a child of this widget with proper layout to keep it contained.
    """
    def __init__(self, parent=None, logger: Optional[LogPane] = None):
        super().__init__(parent)
        self._logger = logger.log if logger else (lambda m: None)
        self._plotter = None
        self._content = None
        self._ready = False

        # Persist per-actor properties across redraws (as chosen in the Editor)
        # Keys are actor names added via name=... in add_mesh
        # Values: {'visible': bool, 'opacity': float}
        self._actor_prefs: Dict[str, Dict[str, Any]] = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        if not _pyvistaqt_import_ok:
            info = QtWidgets.QLabel("PyVista/PyVistaQt/Vtk not available.\nInstall: pip install pyside6 pyvista pyvistaqt vtk")
            info.setAlignment(Qt.AlignCenter)
            lay.addWidget(info)
            return

        try:
            # Construct BackgroundPlotter without popping external window
            ctor_errors = []
            for ctor in (lambda: BackgroundPlotter(show=False, off_screen=False),
                         lambda: BackgroundPlotter(show=False),
                         lambda: BackgroundPlotter()):
                try:
                    self._plotter = ctor()
                    break
                except Exception as e:
                    ctor_errors.append(str(e))
                    self._plotter = None
            if self._plotter is None:
                raise RuntimeError("BackgroundPlotter construction failed: " + " | ".join(ctor_errors))
            # Apply gradient gray background explicitly in case theme is overridden later
            try:
                # set_background accepts color and optionally top/bottom for gradient
                # Newer PyVista: set_background(color, top=None)
                # We call twice to ensure gradient on versions supporting background2
                self._plotter.set_background(color=(0.33, 0.33, 0.33), top=(0.22, 0.22, 0.22))
            except Exception:
                try:
                    # Fallback: two-step for older versions
                    self._plotter.set_background((0.33, 0.33, 0.33))
                    if hasattr(self._plotter, "set_background2"):
                        self._plotter.set_background2((0.22, 0.22, 0.22))
                except Exception:
                    pass

            # Access embedded Qt window
            app_window = getattr(self._plotter, "app_window", None) or \
                         getattr(self._plotter, "window", None) or \
                         getattr(self._plotter, "_main_window", None) or \
                         getattr(self._plotter, "_window", None)
            if app_window is None:
                raise RuntimeError("Could not access PyVistaQt main window QWidget.")

            # Ensure it is a child and laid out here
            app_window.setParent(self, Qt.WindowType.Widget)
            app_window.setWindowFlags(Qt.Widget)
            app_window.show()
            self._content = app_window
            lay.addWidget(self._content)
            self._ready = True
        except Exception as e:
            self._logger(f"InlinePyVistaWidget init error: {e}")
            msg = QtWidgets.QLabel(f"Inline PyVista init error:\n{e}")
            msg.setAlignment(Qt.AlignCenter)
            lay.addWidget(msg)

    def is_ready(self) -> bool:
        return bool(self._ready and self._plotter is not None and self._content is not None)

    def clear(self):
        try:
            if self._plotter:
                self._plotter.clear()
        except Exception:
            pass

    def _normalize_actor_name(self, raw: str) -> str:
        """
        Normalize actor names to stable component keys so Editor choices persist even if
        PyVista renames internals (e.g., adds suffixes).
        Maps variants like 'spars', 'spars_m', 'spars (0)' -> canonical keys:
          wing, spars, ribs, elevon, cutter, wing_m, spars_m, ribs_m, elevon_m, cutter_m
        """
        try:
            name = (raw or "").strip().lower()
            # remove any parenthetical suffixes pyvista may append
            if "(" in name:
                name = name.split("(", 1)[0].strip()
            # keep _m mirrored suffix if present
            if name.endswith("_m"):
                base = name[:-2]
                if base in ("wing", "spars", "ribs", "elevon", "cutter"):
                    return f"{base}_m"
            # standard components
            for base in ("wing", "spars", "ribs", "elevon", "cutter"):
                if name.startswith(base):
                    return base if "_m" not in name else f"{base}_m"
            # fallback to raw
            return (raw or "").strip()
        except Exception:
            return (raw or "").strip()

    def _snapshot_actor_prefs(self):
        """Capture current actor visibility/opacity from the plotter actors with normalized keys."""
        try:
            if not self._plotter:
                return
            actors = getattr(self._plotter, "actors", {}) or {}
            for raw_name, actor in actors.items():
                try:
                    key = self._normalize_actor_name(raw_name)
                    vis = bool(actor.GetVisibility())
                    prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
                    opacity = float(prop.GetOpacity()) if prop else 1.0
                    self._actor_prefs[key] = {"visible": vis, "opacity": opacity}
                except Exception:
                    continue
        except Exception:
            pass

    def _apply_actor_prefs(self):
        """Re-apply cached visibility/opacity to actors after re-adding meshes using normalized keys."""
        try:
            if not self._plotter or not self._actor_prefs:
                return
            actors = getattr(self._plotter, "actors", {}) or {}
            for raw_name, actor in actors.items():
                key = self._normalize_actor_name(raw_name)
                if key in self._actor_prefs and actor is not None:
                    prefs = self._actor_prefs[key]
                    try:
                        actor.SetVisibility(1 if prefs.get("visible", True) else 0)
                    except Exception:
                        pass
                    try:
                        prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
                        if prop is not None and "opacity" in prefs:
                            prop.SetOpacity(float(prefs["opacity"]))
                    except Exception:
                        pass
        except Exception:
            pass

    def _snapshot_actor_prefs(self):
        """Capture current actor visibility/opacity from the plotter actors."""
        try:
            if not self._plotter:
                return
            # actors is a dict: name -> vtkActor
            actors = getattr(self._plotter, "actors", {}) or {}
            for name, actor in actors.items():
                try:
                    vis = bool(actor.GetVisibility())
                    # Some PyVista versions store opacity on actor.GetProperty().GetOpacity()
                    prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
                    opacity = float(prop.GetOpacity()) if prop else 1.0
                    self._actor_prefs[name] = {"visible": vis, "opacity": opacity}
                except Exception:
                    continue
        except Exception:
            pass

    def _apply_actor_prefs(self):
        """Re-apply cached visibility/opacity to actors after re-adding meshes."""
        try:
            if not self._plotter or not self._actor_prefs:
                return
            actors = getattr(self._plotter, "actors", {}) or {}
            for name, actor in actors.items():
                if name in self._actor_prefs and actor is not None:
                    prefs = self._actor_prefs[name]
                    try:
                        actor.SetVisibility(1 if prefs.get("visible", True) else 0)
                    except Exception:
                        pass
                    try:
                        prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
                        if prop is not None and "opacity" in prefs:
                            prop.SetOpacity(float(prefs["opacity"]))
                    except Exception:
                        pass
        except Exception:
            pass

    def update_scene(self, processed, include_elevon=True):
        if not self.is_ready() or processed is None or not getattr(processed, "success", False):
            return
        try:
            from refactor.services.viewer_pv import (  # [refactor.services.viewer_pv._build_wing_mesh()](refactor/services/viewer_pv.py:400)
                _build_wing_mesh,
                _build_group_mesh,
                _build_cutter_wedge,
            )
            import pyvista as _pv

            self.clear()
            
            # Build base meshes
            # Prefer STEP parity: loft wing from dihedraled rib profiles when available
            ribs_profiles = getattr(processed, "dihedraled_rib_profiles", None)
            if ribs_profiles and len(ribs_profiles) >= 2:
                wing_mesh = _build_wing_mesh(None, dihedraled_rib_profiles=ribs_profiles)
            else:
                wing_mesh = _build_wing_mesh(processed.wing_vertices)
            spars_mesh = _build_group_mesh(processed.spar_surfaces)
            ribs_mesh = _build_group_mesh(processed.rib_surfaces)

            elevon_mesh = None
            if include_elevon and getattr(processed, "elevon_surfaces", None):
                elevon_mesh = _build_group_mesh(processed.elevon_surfaces)

            elevon_angle_deg = float(getattr(processed, "elevon_angle_deg", 0.0) or 0.0)
            cutter_mesh = _build_cutter_wedge(processed, elevon_angle_deg)

            # Helper: create a mirrored copy across the Y=0 plane by negating Y
            def mirror_copy(mesh: Optional[_pv.PolyData]) -> Optional[_pv.PolyData]:
                if mesh is None:
                    return None
                try:
                    m = mesh.copy(deep=True)
                    pts = m.points.copy()
                    pts[:, 1] *= -1.0  # mirror about Y=0
                    m.points = pts
                    return m
                except Exception:
                    return None

            # Compute mirrored counterparts
            wing_m = mirror_copy(wing_mesh)
            spars_m = mirror_copy(spars_mesh)
            ribs_m = mirror_copy(ribs_mesh)
            elevon_m = mirror_copy(elevon_mesh) if elevon_mesh is not None else None
            cutter_m = mirror_copy(cutter_mesh) if cutter_mesh is not None else None

            # Add original side
            if wing_mesh is not None:
                self._plotter.add_mesh(wing_mesh, color="cyan", smooth_shading=True, name="wing", opacity=0.35)
            if spars_mesh is not None:
                self._plotter.add_mesh(spars_mesh, color="red", smooth_shading=True, name="spars", opacity=0.95)
            if ribs_mesh is not None:
                self._plotter.add_mesh(ribs_mesh, color="royalblue", smooth_shading=True, name="ribs", opacity=0.98)
            if include_elevon and elevon_mesh is not None:
                self._plotter.add_mesh(elevon_mesh, color="orange", smooth_shading=True, name="elevon", opacity=0.95)
            if cutter_mesh is not None:
                self._plotter.add_mesh(cutter_mesh, color="magenta", smooth_shading=True, name="cutter", opacity=0.7)

            # Add mirrored side
            if wing_m is not None:
                # default mirrored wing opacity handled by external prefs; use 0.35 visual base regardless
                self._plotter.add_mesh(wing_m, color="cyan", smooth_shading=True, name="wing_m", opacity=0.35)
            if spars_m is not None:
                self._plotter.add_mesh(spars_m, color="red", smooth_shading=True, name="spars_m", opacity=0.95)
            if ribs_m is not None:
                self._plotter.add_mesh(ribs_m, color="royalblue", smooth_shading=True, name="ribs_m", opacity=0.98)
            if include_elevon and elevon_m is not None:
                # default mirrored elevon opacity handled by external prefs; base opacity here
                self._plotter.add_mesh(elevon_m, color="orange", smooth_shading=True, name="elevon_m", opacity=0.95)
            if cutter_m is not None:
                self._plotter.add_mesh(cutter_m, color="magenta", smooth_shading=True, name="cutter_m", opacity=0.7)

            try:
                self._plotter.camera_position = "iso"
            except Exception:
                pass

            try:
                self._plotter.app.processEvents()
            except Exception:
                pass
        except Exception as e:
            self._logger(f"InlinePyVistaWidget update error: {e}")


class CpacsGeneratorTab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        props_group = QtWidgets.QGroupBox("Structural & Geometric Properties")
        props_layout = QtWidgets.QGridLayout(props_group)
        props_layout.setColumnStretch(1, 1)

        self.thickness_edit = QtWidgets.QLineEdit("0.00635")
        self.dihedral_edit = QtWidgets.QLineEdit("2.0")

        props_layout.addWidget(QtWidgets.QLabel("Structural Thickness (m):"), 0, 0)
        props_layout.addWidget(self.thickness_edit, 0, 1)
        props_layout.addWidget(QtWidgets.QLabel("Dihedral Angle (deg):"), 1, 0)
        props_layout.addWidget(self.dihedral_edit, 1, 1)

        layout.addWidget(props_group)

        self.gen_btn = QtWidgets.QPushButton("Generate CPACS XML & Send to Viewer")
        self.gen_btn.clicked.connect(self.generate)
        layout.addWidget(self.gen_btn)

        self.xml_view = QtWidgets.QPlainTextEdit()
        self.xml_view.setReadOnly(True)
        self.xml_view.setFont(QtGui.QFont("Courier New", 10))
        self.xml_view.setFixedHeight(220)
        self.xml_view.setPlainText("Generated CPACS XML will appear here...")
        layout.addWidget(self.xml_view, 1)

    def generate(self):
        if not self.app.parsed_data:
            QtWidgets.QMessageBox.critical(self, "Error", "Please open and load a WingData.txt file first.")
            return

        self.app.log("Generating CPACS XML...")
        model_name = os.path.splitext(os.path.basename(self.app.filepath or ""))[0] if self.app.filepath else "wing"
        try:
            thickness = float(self.thickness_edit.text())
            dihedral = float(self.dihedral_edit.text())
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric inputs for thickness/dihedral.")
            return

        cpacs_root = generate_cpacs_xml(self.app.parsed_data, model_name, thickness, dihedral)
        pretty = prettify_xml(cpacs_root)
        self.xml_view.setPlainText(pretty)

        self.app.log("CPACS XML generated. You can now save the file.")
        self.app.add_save_task("CPACS XML", ".cpacs.xml", pretty)

        self.app.transfer_cpacs_to_viewer(pretty.encode("utf-8"), model_name)


class Xflr5Tab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        config = QtWidgets.QGroupBox("XFLR5 Wing Configuration")
        grid = QtWidgets.QGridLayout(config)
        grid.setColumnStretch(1, 1)

        self.wing_name = QtWidgets.QLineEdit("Main Wing")
        self.wing_type = QtWidgets.QComboBox()
        self.wing_type.addItems(["MAINWING", "ELEVATOR", "FIN"])

        grid.addWidget(QtWidgets.QLabel("Wing Name:"), 0, 0)
        grid.addWidget(self.wing_name, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Wing Type:"), 1, 0)
        grid.addWidget(self.wing_type, 1, 1)

        pos_row = QtWidgets.QHBoxLayout()
        self.pos_x = QtWidgets.QLineEdit("0")
        self.pos_y = QtWidgets.QLineEdit("0")
        self.pos_z = QtWidgets.QLineEdit("-0.5")
        for w in (self.pos_x, self.pos_y, self.pos_z):
            w.setFixedWidth(80)
            pos_row.addWidget(w)
        pos_row.addStretch(1)

        grid.addWidget(QtWidgets.QLabel("Position (x, y, z):"), 2, 0)
        grid.addLayout(pos_row, 2, 1)

        self.symmetric_chk = QtWidgets.QCheckBox("Symmetric")
        self.symmetric_chk.setChecked(True)
        grid.addWidget(self.symmetric_chk, 3, 0, 1, 2)

        self.dat_chk = QtWidgets.QCheckBox("Also create .dat files for airfoils")
        self.dat_chk.setChecked(True)
        grid.addWidget(self.dat_chk, 4, 0, 1, 2)

        layout.addWidget(config)

        self.gen_btn = QtWidgets.QPushButton("Generate XFLR5 XML & DAT Files")
        self.gen_btn.clicked.connect(self.generate)
        layout.addWidget(self.gen_btn)

    def generate(self):
        if not self.app.parsed_data:
            QtWidgets.QMessageBox.critical(self, "Error", "Please open and load a WingData.txt file first.")
            return

        self.app.log("Generating XFLR5 XML (ignoring Structures/Controls)...")
        try:
            pos = (float(self.pos_x.text()), float(self.pos_y.text()), float(self.pos_z.text()))
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric inputs for position.")
            return

        root = create_xflr5_xml_structure(
            self.app.parsed_data,
            self.wing_name.text(),
            self.wing_type.currentText(),
            pos,
            bool(self.symmetric_chk.isChecked()),
        )

        pretty = prettify_xml(root)
        if not pretty.startswith("<?xml"):
            pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty
        final_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE explane>\n' + pretty.split('\n', 1)[1]

        self.app.log("XFLR5 XML generated. You can now save the file.")
        self.app.add_save_task("XFLR5 XML", ".xml", final_xml)

        if self.dat_chk.isChecked():
            self.app.log("Creating .dat files...")
            out_dir = os.path.dirname(self.app.filepath) if self.app.filepath else "."
            create_airfoil_dat_files(self.app.parsed_data, out_dir)


class StepperTab(QtWidgets.QWidget):
    """
    Rebuilds the CPACS 3D Viewer / STEP Exporter tab using Qt widgets and inline PyVista scene.
    Layout mirrors the Tk version:
      - Top toolbar (load, visibility toggles, export buttons)
      - Controls row (left: rib/elevon, center: per-spar overrides, right: interlocking params)
      - Inline 3D viewport under the controls row
    """
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.cpacs_root = None
        self.processed = None
        self.file_name_hint = "default_model"

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)

        # Top toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(8)

        self.load_btn = QtWidgets.QPushButton("Load CPACS File...")
        self.load_btn.clicked.connect(self.load_from_file)
        toolbar.addWidget(self.load_btn)

        toolbar.addStretch(1)

        self.export_fixture_btn = QtWidgets.QPushButton("Generate & Export Fixture Layout (STEP)")
        self.export_fixture_btn.clicked.connect(self.generate_and_export_layout)
        toolbar.addWidget(self.export_fixture_btn)

        # Toggle: spline-based wing surface for STEP export
        self.use_spline_wing_chk = QtWidgets.QCheckBox("Use spline wing for STEP")
        self.use_spline_wing_chk.setToolTip("Loft wing from spline-stitched airfoil curves. Other geometry still uses point loft.")
        self.use_spline_wing_chk.setChecked(False)
        toolbar.addWidget(self.use_spline_wing_chk)

        self.export_step_btn = QtWidgets.QPushButton("Export 3D Model (STEP)")
        self.export_step_btn.clicked.connect(self.export_step_file)
        toolbar.addWidget(self.export_step_btn)

        root.addLayout(toolbar)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(16)

        # Track external visibility/opacity per component (incl. mirrored)
        # keys: wing, spars, ribs, elevon, cutter, wing_m, spars_m, ribs_m, elevon_m, cutter_m
        self.viewer_actor_prefs: Dict[str, Dict[str, Any]] = {}
        def _init_pref(key, visible=True, opacity=1.0):
            self.viewer_actor_prefs.setdefault(key, {"visible": visible, "opacity": opacity})
        for base in ("wing","spars","ribs","elevon","cutter"):
            _init_pref(base, True, 1.0)
            # Set mirrored defaults; wing_m and elevon_m at 0.5, others remain 1.0
            default_opacity_m = 0.5 if base in ("wing", "elevon") else 1.0
            _init_pref(f"{base}_m", True, default_opacity_m)

        # Left controls
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Rib Thickness:"))
        self.rib_thickness = QtWidgets.QLineEdit("0.00635")
        self.rib_thickness.setFixedWidth(100)
        left.addWidget(self.rib_thickness)

        left.addSpacing(6)
        left.addWidget(QtWidgets.QLabel("Elevon Down Angle (°):"))
        self.elevon_angle = QtWidgets.QLineEdit("30.0")
        self.elevon_angle.setFixedWidth(100)
        self.elevon_angle.editingFinished.connect(self.draw_geometry)
        left.addWidget(self.elevon_angle)

        left.addSpacing(6)
        left.addWidget(QtWidgets.QLabel("Deflection Height Scale:"))
        self.deflection_height_scale = QtWidgets.QLineEdit("1.2")
        self.deflection_height_scale.setFixedWidth(100)
        self.deflection_height_scale.setToolTip("Scales the deflection cutter height while keeping hinge line and angle.")
        self.deflection_height_scale.editingFinished.connect(self.draw_geometry)
        left.addWidget(self.deflection_height_scale)

        left.addSpacing(6)
        left.addWidget(QtWidgets.QLabel("Cutter Hinge Reference:"))
        self.cutter_hinge_ref = QtWidgets.QComboBox()
        self.cutter_hinge_ref.addItems(["Top of rear spar", "Bottom of rear spar", "Centerline of rear spar"])
        self.cutter_hinge_ref.currentIndexChanged.connect(self.draw_geometry)
        left.addWidget(self.cutter_hinge_ref)

        controls.addLayout(left, 0)

        # Center controls: per-spar overrides in a scroll area
        center_group = QtWidgets.QGroupBox("Individual Spar Controls")
        center_v = QtWidgets.QVBoxLayout(center_group)

        self.spar_scroll = QtWidgets.QScrollArea()
        self.spar_scroll.setWidgetResizable(True)
        self.spar_inner = QtWidgets.QWidget()
        self.spar_form = QtWidgets.QFormLayout(self.spar_inner)
        self.spar_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.spar_scroll.setWidget(self.spar_inner)
        center_v.addWidget(self.spar_scroll)

        controls.addWidget(center_group, 1)

        # Right controls: interlocking structure
        right_group = QtWidgets.QGroupBox("Interlocking Structure Generation")
        right_l = QtWidgets.QFormLayout(right_group)

        self.material_thickness_mm = QtWidgets.QLineEdit("6.35")
        self.slot_clearance_mm = QtWidgets.QLineEdit("0.15")
        self.use_dogbones = QtWidgets.QCheckBox("Add CNC Dog-bones")
        self.use_dogbones.stateChanged.connect(self._toggle_dogbones)
        self.tool_diameter_mm = QtWidgets.QLineEdit("3.175")
        self.tool_diameter_mm.setEnabled(False)
        self.add_cradle = QtWidgets.QCheckBox("Add under-spar cradle"); self.add_cradle.setChecked(True)
        self.tab_width_mm = QtWidgets.QLineEdit("15.0")
        self.tab_spacing_mm = QtWidgets.QLineEdit("80.0")
        self.tab_edge_margin_mm = QtWidgets.QLineEdit("12.0")

        right_l.addRow("Material Thickness (mm):", self.material_thickness_mm)
        right_l.addRow("Slot Clearance (mm):", self.slot_clearance_mm)
        right_l.addRow(self.use_dogbones)
        right_l.addRow("Tool Diameter (mm):", self.tool_diameter_mm)
        right_l.addRow(self.add_cradle)
        right_l.addRow("Tab Width (mm):", self.tab_width_mm)
        right_l.addRow("Tab Spacing (mm):", self.tab_spacing_mm)
        right_l.addRow("Tab Edge Margin (mm):", self.tab_edge_margin_mm)

        controls.addWidget(right_group, 0)

        # External visibility/opacity controls
        vis_group = QtWidgets.QGroupBox("Viewer Layers")
        vis_layout = QtWidgets.QGridLayout(vis_group)
        headers = ["Component", "Visible", "Opacity"]
        for c, h in enumerate(headers):
            lbl = QtWidgets.QLabel(h)
            f = lbl.font(); f.setBold(True); lbl.setFont(f)
            vis_layout.addWidget(lbl, 0, c)
        self._layer_check: Dict[str, QtWidgets.QCheckBox] = {}
        self._layer_opacity: Dict[str, QtWidgets.QDoubleSpinBox] = {}

        def add_layer_row(row, key, label):
            chk = QtWidgets.QCheckBox()
            chk.setChecked(self.viewer_actor_prefs[key]["visible"])
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 1.0); spin.setSingleStep(0.05); spin.setDecimals(2)
            spin.setValue(self.viewer_actor_prefs[key]["opacity"])
            self._layer_check[key] = chk
            self._layer_opacity[key] = spin
            vis_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            vis_layout.addWidget(chk, row, 1)
            vis_layout.addWidget(spin, row, 2)
            chk.stateChanged.connect(lambda _=None, k=key: self._on_layer_changed(k))
            spin.valueChanged.connect(lambda _=None, k=key: self._on_layer_changed(k))

        rowi = 1
        for base,label in (("wing","Wing"),("spars","Spars"),("ribs","Ribs"),("elevon","Elevon"),("cutter","Cutter")):
            add_layer_row(rowi, base, label); rowi += 1

        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        vis_layout.addWidget(sep, rowi, 0, 1, 3); rowi += 1

        for base,label in (("wing_m","Wing (Mirrored)"),
                           ("spars_m","Spars (Mirrored)"),
                           ("ribs_m","Ribs (Mirrored)"),
                           ("elevon_m","Elevon (Mirrored)"),
                           ("cutter_m","Cutter (Mirrored)")):
            add_layer_row(rowi, base, label); rowi += 1

        controls.addWidget(vis_group, 0)

        # Apply button
        self.apply_btn = QtWidgets.QPushButton("Apply Overrides & Redraw")
        self.apply_btn.clicked.connect(self.apply_overrides)
        controls.addWidget(self.apply_btn, 0, Qt.AlignBottom)

        root.addLayout(controls)

        # Inline PyVista viewport
        self.inline_pv = InlinePyVistaWidget(self.app, self.app.log_pane)
        self.inline_pv.setMinimumHeight(400)
        root.addWidget(self.inline_pv, 1)

    def _toggle_dogbones(self):
        self.tool_diameter_mm.setEnabled(self.use_dogbones.isChecked())

    def populate_spar_controls(self, spar_uids, initial_thicknesses):
        # Clear existing
        while self.spar_form.count():
            item = self.spar_form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.spar_thick_edits: Dict[str, QtWidgets.QLineEdit] = {}
        self.spar_offset_edits: Dict[str, QtWidgets.QLineEdit] = {}

        uids = list(spar_uids or [])
        init_map = dict(initial_thicknesses or {})
        if not uids and init_map:
            uids = list(init_map.keys())

        for uid in uids:
            if not uid:
                continue
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            label = QtWidgets.QLabel(f"{uid}:")
            label.setMinimumWidth(140)
            row_layout.addWidget(label)

            row_layout.addWidget(QtWidgets.QLabel("Thick:"))
            thick = QtWidgets.QLineEdit(str(init_map.get(uid, 0.00635)))
            thick.setFixedWidth(80)
            row_layout.addWidget(thick)
            self.spar_thick_edits[uid] = thick

            row_layout.addSpacing(6)
            row_layout.addWidget(QtWidgets.QLabel("Offset:"))
            offset = QtWidgets.QLineEdit("0.0")
            offset.setFixedWidth(80)
            row_layout.addWidget(offset)
            self.spar_offset_edits[uid] = offset

            self.spar_form.addRow(row_widget)

    def draw_geometry(self):
        if not self.processed or not getattr(self.processed, "success", False):
            return
        try:
            # propagate UI into processed for viewer services
            try:
                self.processed.cutter_hinge_mode = self.cutter_hinge_ref.currentText()
                self.processed.elevon_angle_deg = float(self.elevon_angle.text())
                # Optional: deflection height scale for cutter sizing
                try:
                    self.processed.deflection_height_scale = float(self.deflection_height_scale.text())
                except Exception:
                    self.processed.deflection_height_scale = 1.0
            except Exception:
                pass

            if self.inline_pv.is_ready():
                # Merge external prefs into plotter-side prefs before redraw
                if hasattr(self.inline_pv, "_actor_prefs"):
                    for k, v in self.viewer_actor_prefs.items():
                        self.inline_pv._actor_prefs[k] = {"visible": bool(v["visible"]),
                                                          "opacity": float(v["opacity"])}
                self.inline_pv.update_scene(self.processed, include_elevon=True)
                # After redraw, re-apply again in case new actors were created
                try:
                    self.inline_pv._apply_actor_prefs()
                except Exception:
                    pass
        except Exception as e:
            self.app.log(f"draw_geometry inline error: {e}")

    def apply_overrides(self):
        if self.cpacs_root is None:
            QtWidgets.QMessageBox.warning(self, "Overrides", "No CPACS loaded to apply overrides.")
            return
        try:
            # Gather UI values
            try:
                rib_thick = float(self.rib_thickness.text())
            except Exception:
                rib_thick = None

            spar_thicks: Dict[str, float] = {}
            for uid, edit in self.spar_thick_edits.items():
                try:
                    spar_thicks[uid] = float(edit.text())
                except Exception:
                    continue

            spar_offsets: Dict[str, float] = {}
            for uid, edit in self.spar_offset_edits.items():
                try:
                    spar_offsets[uid] = float(edit.text())
                except Exception:
                    continue

            # Mutate XML thickness values
            import xml.etree.ElementTree as ET
            # Update ribs
            if rib_thick is not None:
                for rib_def in self.cpacs_root.findall('.//ribsDefinition'):
                    t_elem = rib_def.find('.//ribCrossSection/material/thickness')
                    if t_elem is not None:
                        t_elem.text = str(rib_thick)
            # Update spars
            for spar_seg in self.cpacs_root.findall('.//sparSegment'):
                uid = spar_seg.get('uID')
                if uid in spar_thicks:
                    t_elem = spar_seg.find('.//sparCrossSection/web1/material/thickness')
                    if t_elem is not None:
                        t_elem.text = str(spar_thicks[uid])

            # Reprocess
            xml_bytes = QtCore.QByteArray()
            try:
                # Using ElementTree tostring on the existing cpacs_root
                import xml.etree.ElementTree as ET
                xml_bytes = ET.tostring(self.cpacs_root, encoding="utf-8")
            except Exception:
                pass

            self.processed = process_cpacs_data(xml_bytes, spar_offsets=spar_offsets)

            if getattr(self.processed, "success", False):
                # Refresh controls and preserve user offsets
                preserved = {uid: self.spar_offset_edits.get(uid, QtWidgets.QLineEdit("0.0")).text()
                             for uid in (self.processed.spar_uids or [])}
                try:
                    self.rib_thickness.setText(str(float(self.processed.initial_rib_thickness)))
                except Exception:
                    pass
                self.populate_spar_controls(self.processed.spar_uids, self.processed.initial_spar_thicknesses)
                for uid, val in preserved.items():
                    if uid in self.spar_offset_edits:
                        self.spar_offset_edits[uid].setText(str(val))

                try:
                    self.processed.cutter_hinge_mode = self.cutter_hinge_ref.currentText()
                    self.processed.elevon_angle_deg = float(self.elevon_angle.text())
                    try:
                        self.processed.deflection_height_scale = float(self.deflection_height_scale.text())
                    except Exception:
                        self.processed.deflection_height_scale = 1.0
                except Exception:
                    pass
                self.draw_geometry()
                self.app.log("Overrides applied and viewer redrawn.")
            else:
                self.app.log("Processing failed; inline preview not updated.")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Apply Overrides Error", str(e))

    def load_from_string(self, xml_bytes: bytes, filename_hint: str):
        import xml.etree.ElementTree as ET
        try:
            self.cpacs_root = ET.fromstring(xml_bytes)
            self.file_name_hint = filename_hint
            self.app.log(f"Viewer: Received CPACS for '{filename_hint}'. Processing...")

            self.processed = process_cpacs_data(xml_bytes, spar_offsets={})

            if getattr(self.processed, "success", False):
                try:
                    self.rib_thickness.setText(str(float(self.processed.initial_rib_thickness)))
                except Exception:
                    pass
                self.populate_spar_controls(self.processed.spar_uids, self.processed.initial_spar_thicknesses)
            else:
                self.populate_spar_controls([], {})

            try:
                self.processed.cutter_hinge_mode = self.cutter_hinge_ref.currentText()
                self.processed.elevon_angle_deg = float(self.elevon_angle.text())
                try:
                    self.processed.deflection_height_scale = float(self.deflection_height_scale.text())
                except Exception:
                    self.processed.deflection_height_scale = 1.0
            except Exception:
                pass
            # Respect toggles after load
            self.draw_geometry()
            self.app.log("Viewer: Processing complete.")
        except Exception as e:
            self.app.log(f"Viewer load_from_string error: {e}")

    def load_from_file(self):
        dlg = QtWidgets.QFileDialog(self, "Select a CPACS file")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters(["XML files (*.xml)", "CPACS files (*.cpacs)", "All files (*.*)"])
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        path = paths[0]
        try:
            with open(path, "rb") as f:
                xml_bytes = f.read()
            self.load_from_string(xml_bytes, os.path.splitext(os.path.basename(path))[0])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    def export_step_file(self):
        if not self.processed:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load or generate CPACS first.")
            return
        dlg = QtWidgets.QFileDialog(self, "Save 3D Model STEP File")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setDefaultSuffix("step")
        dlg.selectFile(f"{self.file_name_hint}_3D.step")
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]
        try:
            # Propagate current GUI state to processed for export
            try:
                self.processed.cutter_hinge_mode = self.cutter_hinge_ref.currentText()
                self.processed.elevon_angle_deg = float(self.elevon_angle.text())
                self.processed.deflection_height_scale = float(self.deflection_height_scale.text())
            except Exception:
                pass
            angle = None
            try:
                angle = float(self.elevon_angle.text())
            except Exception:
                angle = None
            shape = build_full_step_from_processed(
                self.processed,
                scale_to_mm=True,
                cut_wing_with_elevon_opening=True,
                hollow_skin_scale=0.98,
                elevon_angle_deg=angle,
                use_spline_wing=bool(self.use_spline_wing_chk.isChecked())
            )
            ok = write_step(shape, path)
            if ok:
                QtWidgets.QMessageBox.information(self, "Export Successful", f"Exported STEP to:\n{path}")
                self.app.log(f"STEP exported: {path}")
            else:
                raise RuntimeError("Writer returned failure.")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "STEP Export Error", str(e))

    def generate_and_export_layout(self):
        if not self.processed:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load or generate CPACS first.")
            return
        dlg = QtWidgets.QFileDialog(self, "Save Fixture Layout STEP File")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setDefaultSuffix("step")
        dlg.selectFile(f"{self.file_name_hint}_fixture_layout.step")
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]
        try:
            params = FixtureParams(
                material_thickness_mm=float(self.material_thickness_mm.text()),
                slot_clearance_mm=float(self.slot_clearance_mm.text()),
                add_cradle=bool(self.add_cradle.isChecked()),
                tab_width_mm=float(self.tab_width_mm.text()),
                tab_spacing_mm=float(self.tab_spacing_mm.text()),
                tab_edge_margin_mm=float(self.tab_edge_margin_mm.text()),
                base_plate_margin_mm=20.0,
                add_tabs=True,
                slot_base_plate=True,
            )
            layout = build_fixture_layout(self.processed, params)
            ok = write_fixture_step(layout.final_layout, path)
            if ok:
                QtWidgets.QMessageBox.information(self, "Export Successful", f"Exported fixture layout to:\n{path}")
                self.app.log(f"Fixture STEP exported: {path}")
            else:
                raise RuntimeError("Writer returned failure.")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Fixture Export Error", str(e))


    # External layer change handler
    def _on_layer_changed(self, key: str):
        # Update local state
        try:
            self.viewer_actor_prefs[key]["visible"] = bool(self._layer_check[key].isChecked())
            self.viewer_actor_prefs[key]["opacity"] = float(self._layer_opacity[key].value())
        except Exception:
            return
        # Push to InlinePyVistaWidget immediately if actors exist
        try:
            pvw = self.inline_pv
            if pvw and getattr(pvw, "_plotter", None) is not None:
                # update actor prefs store
                if hasattr(pvw, "_actor_prefs"):
                    pvw._actor_prefs[key] = {"visible": self.viewer_actor_prefs[key]["visible"],
                                             "opacity": self.viewer_actor_prefs[key]["opacity"]}
                # apply to live actors without full redraw
                actors = getattr(pvw._plotter, "actors", {}) or {}
                norm = getattr(pvw, "_normalize_actor_name", None)
                for raw_name, actor in actors.items():
                    candidate = norm(raw_name) if norm else raw_name
                    if candidate == key and actor is not None:
                        try:
                            actor.SetVisibility(1 if self.viewer_actor_prefs[key]["visible"] else 0)
                        except Exception:
                            pass
                        try:
                            prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
                            if prop is not None:
                                prop.SetOpacity(self.viewer_actor_prefs[key]["opacity"])
                        except Exception:
                            pass
                try:
                    pvw._plotter.render()
                    if hasattr(pvw._plotter, "app"):
                        pvw._plotter.app.processEvents()
                except Exception:
                    pass
        except Exception:
            pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Wing Data Converter & Viewer (Qt)")
        self.resize(1400, 950)

        self.filepath: Optional[str] = None
        self.parsed_data: Optional[Dict[str, Any]] = None
        self.save_tasks: Dict[str, Dict[str, str]] = {}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QGridLayout(central)
        root.setRowStretch(1, 1)
        root.setColumnStretch(0, 1)

        # Top file input
        top_group = QtWidgets.QGroupBox("Input File (WingData.txt)")
        top_layout = QtWidgets.QGridLayout(top_group)
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.open_file)
        self.path_edit = QtWidgets.QLineEdit("")
        self.path_edit.setReadOnly(True)
        # New: spline refinement control (number of points)
        self.refine_spin = QtWidgets.QSpinBox()
        self.refine_spin.setRange(0, 2000)
        self.refine_spin.setValue(0)
        self.refine_spin.setToolTip("If >0, resample airfoil profiles to this many points using a spline")
        refine_label = QtWidgets.QLabel("Airfoil refine to points:")
        refine_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_layout.addWidget(self.browse_btn, 0, 0)
        top_layout.addWidget(self.path_edit, 0, 1)
        top_layout.addWidget(refine_label, 0, 2)
        top_layout.addWidget(self.refine_spin, 0, 3)
        top_layout.setColumnStretch(1, 1)
        root.addWidget(top_group, 0, 0)

        # Notebook (tabs)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1, 0)

        # Bottom logging and save
        bottom = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(10, 10, 10, 10)

        self.log_pane = LogPane()
        bottom_layout.addWidget(self.log_pane, 1)
        self.save_btn = QtWidgets.QPushButton("Save Generated Files...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_files)
        bottom_layout.addWidget(self.save_btn, 0, Qt.AlignTop)

        root.addWidget(bottom, 2, 0)

        # Tabs
        self.cpacs_tab = CpacsGeneratorTab(self)
        self.xflr5_tab = Xflr5Tab(self)
        self.stepper_tab = StepperTab(self)
        self.tabs.addTab(self.cpacs_tab, "CPACS Generator")
        self.tabs.addTab(self.xflr5_tab, "XFLR5 Generator")
        self.tabs.addTab(self.stepper_tab, "CPACS 3D Viewer / STEP Exporter")

    # Logging
    def log(self, message: str):
        self.log_pane.log(message)

    # File opening
    def open_file(self):
        dlg = QtWidgets.QFileDialog(self, "Select WingData.txt")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters(["Text files (*.txt)", "All files (*.*)"])
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        filepath = paths[0]
        self.filepath = filepath
        self.path_edit.setText(filepath)
        self.log(f"Loading data from: {os.path.basename(filepath)}")
        # Read refine-to value (0 means no refinement)
        refine_val = None
        try:
            rv = int(self.refine_spin.value()) if hasattr(self, "refine_spin") else 0
            if rv > 1:
                refine_val = rv
                self.log(f"Airfoil refinement requested: resample to {rv} points.")
        except Exception:
            refine_val = None
        data, msg = parse_wing_data(filepath, refine_to=refine_val)
        if data:
            self.parsed_data = data
            self.log(msg)
            if self.parsed_data.get('structures'):
                self.log("  Found 'Structures' data.")
            if self.parsed_data.get('controls'):
                self.log("  Found 'Controls' data.")
            self.save_tasks = {}
            self.save_btn.setEnabled(False)
        else:
            self.log(f"ERROR: {msg}")
            QtWidgets.QMessageBox.critical(self, "Parsing Error", msg)

    # Save generated files
    def add_save_task(self, name: str, extension: str, content: str):
        self.save_tasks[name] = {'ext': extension, 'content': content}
        self.save_btn.setEnabled(True)

    def save_files(self):
        if not self.save_tasks:
            QtWidgets.QMessageBox.warning(self, "No Files", "No files have been generated to save.")
            return
        base_filename = os.path.splitext(os.path.basename(self.filepath or ""))[0] if self.filepath else "model"
        for name, task in self.save_tasks.items():
            dlg = QtWidgets.QFileDialog(self, f"Save {name}")
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            dlg.setDefaultSuffix(task['ext'].lstrip('.'))
            dlg.selectFile(base_filename + task['ext'])
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                continue
            save_path = dlg.selectedFiles()[0]
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(task['content'])
                self.log(f"Successfully saved: {os.path.basename(save_path)}")
            except Exception as e:
                self.log(f"ERROR saving {name}: {e}")
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save {name}:\n{e}")
        self.log("Save process complete.")

    # Cross-tab transfer
    def transfer_cpacs_to_viewer(self, xml_bytes: bytes, filename_hint: str):
        self.log("Transferring generated CPACS data to 3D Viewer tab...")
        self.stepper_tab.load_from_string(xml_bytes, filename_hint)
        self.tabs.setCurrentWidget(self.stepper_tab)
        self.log("Transfer complete. Viewer updated.")


def launch():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
