import datetime
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# PyVista imports
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter

    _pyvistaqt_import_ok = True
except ImportError:
    _pyvistaqt_import_ok = False
    pv = None
    BackgroundPlotter = None

from core.state import Project
from core.flow5_gen import export_flow5_project

# CPACS imports - still needed for 3D preview functionality
from core.export.cpacs_gen import generate_cpacs_xml
from services.export.adapter import project_to_cpacs_data

# DXF export imports
from services.export.dxf_export import (
    GridNestingParams,
    SplitUserChoice,
    is_ezdxf_available,
)
from services.export.dxf_export import (
    RibExportParams as DxfRibParams,
)
from services.export.dxf_export import (
    SparExportParams as DxfSparParams,
)
from services.export.viewer import ProcessedCpacs, process_cpacs_data
from services.geometry import AeroSandboxService

# Import the migrated PyVista viewer service
# We use a lazy import or direct import if available
try:
    from services.export.viewer_pv import (
        _build_cutter_wedge,
        _build_group_mesh,
        _build_wing_mesh,
    )
except ImportError:
    pass


class LogPane(QPlainTextEdit):
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


class InlinePyVistaWidget(QWidget):
    """
    Native Qt host for PyVista BackgroundPlotter without separate top-level window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plotter = None
        self._content = None
        self._ready = False

        # Persist per-actor properties across redraws
        self._actor_prefs: Dict[str, Dict[str, Any]] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        if not _pyvistaqt_import_ok:
            info = QLabel(
                "PyVista/PyVistaQt/Vtk not available.\nInstall: pip install pyside6 pyvista pyvistaqt vtk"
            )
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(info)
            return

        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            info = QLabel("3D preview disabled in offscreen Qt mode.")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(info)
            return

        try:
            # Construct BackgroundPlotter without popping external window
            self._plotter = BackgroundPlotter(show=False)

            # Set background
            try:
                self._plotter.set_background(
                    color=(0.33, 0.33, 0.33), top=(0.22, 0.22, 0.22)
                )
            except Exception:
                self._plotter.set_background("dimgray")

            # Access embedded Qt window
            app_window = (
                getattr(self._plotter, "app_window", None)
                or getattr(self._plotter, "window", None)
                or getattr(self._plotter, "_main_window", None)
                or getattr(self._plotter, "_window", None)
            )

            if app_window is None:
                raise RuntimeError("Could not access PyVistaQt main window QWidget.")

            # Ensure it is a child and laid out here
            app_window.setParent(self)  # Qt.WindowType.Widget is default
            # app_window.setWindowFlags(Qt.WindowType.Widget) # Might not be needed in PyQt6
            app_window.show()
            self._content = app_window
            lay.addWidget(self._content)
            self._ready = True
        except Exception as e:
            msg = QLabel(f"Inline PyVista init error:\n{e}")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(msg)

    def is_ready(self) -> bool:
        return bool(
            self._ready and self._plotter is not None and self._content is not None
        )

    def clear(self):
        try:
            if self._plotter:
                self._plotter.clear()
        except Exception:
            pass

    def _normalize_actor_name(self, raw: str) -> str:
        try:
            name = (raw or "").strip().lower()
            if "(" in name:
                name = name.split("(", 1)[0].strip()
            if name.endswith("_m"):
                base = name[:-2]
                if base in ("wing", "spars", "ribs", "elevon", "cutter"):
                    return f"{base}_m"
            for base in ("wing", "spars", "ribs", "elevon", "cutter"):
                if name.startswith(base):
                    return base if "_m" not in name else f"{base}_m"
            return (raw or "").strip()
        except Exception:
            return (raw or "").strip()

    def _apply_actor_prefs(self):
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
                        prop = (
                            actor.GetProperty()
                            if hasattr(actor, "GetProperty")
                            else None
                        )
                        if prop is not None and "opacity" in prefs:
                            prop.SetOpacity(float(prefs["opacity"]))
                    except Exception:
                        pass
        except Exception:
            pass

    def update_scene(self, processed: ProcessedCpacs, include_elevon=True):
        if (
            not self.is_ready()
            or processed is None
            or not getattr(processed, "success", False)
        ):
            return
        try:
            from services.export.viewer_pv import (
                _build_cutter_wedge,
                _build_group_mesh,
                _build_wing_mesh,
            )

            self.clear()

            # Build meshes
            # Use loft_profiles (defining sections) for the wing skin if available,
            # otherwise fallback to rib profiles or vertices.
            loft_profiles = getattr(processed, "loft_profiles", None)
            ribs_profiles = getattr(processed, "dihedraled_rib_profiles", None)

            if loft_profiles is not None and len(loft_profiles) >= 2:
                wing_mesh = _build_wing_mesh(
                    None, dihedraled_rib_profiles=loft_profiles
                )
            elif ribs_profiles is not None and len(ribs_profiles) >= 2:
                wing_mesh = _build_wing_mesh(
                    None, dihedraled_rib_profiles=ribs_profiles
                )
            else:
                wing_mesh = _build_wing_mesh(processed.wing_vertices)

            spars_mesh = _build_group_mesh(processed.spar_surfaces)
            ribs_mesh = _build_group_mesh(processed.rib_surfaces)

            elevon_mesh = None
            if include_elevon and getattr(processed, "elevon_surfaces", None):
                elevon_mesh = _build_group_mesh(processed.elevon_surfaces)

            elevon_angle_deg = float(getattr(processed, "elevon_angle_deg", 0.0) or 0.0)
            cutter_mesh = _build_cutter_wedge(processed, elevon_angle_deg)

            # Helper for mirroring
            def mirror_copy(mesh):
                if mesh is None:
                    return None
                try:
                    m = mesh.copy(deep=True)
                    pts = m.points.copy()
                    pts[:, 1] *= -1.0
                    m.points = pts
                    return m
                except Exception:
                    return None

            wing_m = mirror_copy(wing_mesh)
            spars_m = mirror_copy(spars_mesh)
            ribs_m = mirror_copy(ribs_mesh)
            elevon_m = mirror_copy(elevon_mesh)
            cutter_m = mirror_copy(cutter_mesh)

            # Add to plotter
            if wing_mesh:
                self._plotter.add_mesh(
                    wing_mesh,
                    color="cyan",
                    smooth_shading=True,
                    name="wing",
                    opacity=0.35,
                )
            if spars_mesh:
                self._plotter.add_mesh(
                    spars_mesh,
                    color="red",
                    smooth_shading=True,
                    name="spars",
                    opacity=0.95,
                )
            if ribs_mesh:
                self._plotter.add_mesh(
                    ribs_mesh,
                    color="royalblue",
                    smooth_shading=True,
                    name="ribs",
                    opacity=0.98,
                )
            if elevon_mesh:
                self._plotter.add_mesh(
                    elevon_mesh,
                    color="orange",
                    smooth_shading=True,
                    name="elevon",
                    opacity=0.95,
                )
            if cutter_mesh:
                self._plotter.add_mesh(
                    cutter_mesh,
                    color="magenta",
                    smooth_shading=True,
                    name="cutter",
                    opacity=0.7,
                )

            if wing_m:
                self._plotter.add_mesh(
                    wing_m,
                    color="cyan",
                    smooth_shading=True,
                    name="wing_m",
                    opacity=0.35,
                )
            if spars_m:
                self._plotter.add_mesh(
                    spars_m,
                    color="red",
                    smooth_shading=True,
                    name="spars_m",
                    opacity=0.95,
                )
            if ribs_m:
                self._plotter.add_mesh(
                    ribs_m,
                    color="royalblue",
                    smooth_shading=True,
                    name="ribs_m",
                    opacity=0.98,
                )
            if elevon_m:
                self._plotter.add_mesh(
                    elevon_m,
                    color="orange",
                    smooth_shading=True,
                    name="elevon_m",
                    opacity=0.95,
                )
            if cutter_m:
                self._plotter.add_mesh(
                    cutter_m,
                    color="magenta",
                    smooth_shading=True,
                    name="cutter_m",
                    opacity=0.7,
                )

            try:
                self._plotter.camera_position = "iso"
            except Exception:
                pass

        except Exception as e:
            print(f"InlinePyVistaWidget update error: {e}")


class ExportTab(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.processed: Optional[ProcessedCpacs] = None
        self.cpacs_root = None

        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Add Generate button (Unified specific)
        self.gen_btn = QPushButton("Generate from Project")
        self.gen_btn.clicked.connect(self.generate_from_project)
        toolbar.addWidget(self.gen_btn)

        toolbar.addStretch(1)

        self.export_fixture_btn = QPushButton("Generate & Export Fixture Layout (STEP)")
        self.export_fixture_btn.clicked.connect(self.generate_and_export_layout)
        toolbar.addWidget(self.export_fixture_btn)

        self.use_spline_wing_chk = QCheckBox("Use spline wing for STEP")
        self.use_spline_wing_chk.setToolTip(
            "Loft wing from spline-stitched airfoil curves."
        )
        self.use_spline_wing_chk.setChecked(True)  # Default to True as per user request
        toolbar.addWidget(self.use_spline_wing_chk)

        self.cut_wing_chk = QCheckBox("Cut wing for elevon")
        self.cut_wing_chk.setChecked(True)
        self.cut_wing_chk.setToolTip("Cut the elevon opening from the wing solid.")
        toolbar.addWidget(self.cut_wing_chk)

        self.export_flow5_btn = QPushButton("Export Flow5")
        self.export_flow5_btn.setToolTip("Export plane geometry as Flow5-compatible XML + airfoil .dat files")
        self.export_flow5_btn.clicked.connect(self.export_flow5_file)
        toolbar.addWidget(self.export_flow5_btn)

        self.export_step_btn = QPushButton("Export 3D Model (STEP)")
        self.export_step_btn.clicked.connect(self.export_step_file)
        toolbar.addWidget(self.export_step_btn)

        self.export_cfd_wing_btn = QPushButton("Export CFD Wing (STEP)")
        self.export_cfd_wing_btn.setToolTip("Export solid half-wing OML for CFD analysis (no internal structure)")
        self.export_cfd_wing_btn.clicked.connect(self.export_cfd_wing_file)
        toolbar.addWidget(self.export_cfd_wing_btn)

        root.addLayout(toolbar)

        # Elevon angle setting (single line)
        elevon_row = QHBoxLayout()
        elevon_row.setSpacing(8)
        elevon_row.addWidget(QLabel("Control Surface Angle (°):"))
        self.elevon_angle = QLineEdit("30.0")
        self.elevon_angle.setFixedWidth(60)
        self.elevon_angle.editingFinished.connect(self.draw_geometry)
        elevon_row.addWidget(self.elevon_angle)
        elevon_row.addStretch()
        root.addLayout(elevon_row)

        # Initialize viewer actor prefs with defaults (used by InlinePyVistaWidget)
        self.viewer_actor_prefs = {}
        self._layer_check = {}
        self._layer_opacity = {}
        for base in ("wing", "spars", "ribs", "elevon", "cutter"):
            self.viewer_actor_prefs[base] = {"visible": True, "opacity": 1.0}
            default_opacity_m = 0.5 if base in ("wing", "elevon") else 1.0
            self.viewer_actor_prefs[f"{base}_m"] = {
                "visible": True,
                "opacity": default_opacity_m,
            }

        # === Fixture Export Section ===
        fixture_group = self._build_fixture_group()
        root.addWidget(fixture_group, 0)

        # === DXF Export Section (below controls, above 3D viewer) ===
        dxf_group = self._build_dxf_group()
        root.addWidget(dxf_group, 0)

        # Inline PyVista viewport
        self.inline_pv = InlinePyVistaWidget(self)
        self.inline_pv.setMinimumHeight(400)
        root.addWidget(self.inline_pv, 1)

    def _build_fixture_group(self) -> QGroupBox:
        """Build Fixture Export Parameters UI group."""
        group = QGroupBox("Fixture Export Parameters")
        form = QFormLayout(group)

        # Material thickness (for fixture jig material - plywood/MDF)
        self.fixture_material_thickness = QDoubleSpinBox()
        self.fixture_material_thickness.setRange(1.0, 25.0)
        self.fixture_material_thickness.setValue(6.35)  # 1/4" default
        self.fixture_material_thickness.setSuffix(" mm")
        self.fixture_material_thickness.setToolTip(
            "Thickness of fixture material (plywood/MDF)"
        )
        form.addRow("Material Thickness:", self.fixture_material_thickness)

        # Slot clearance (for spar/rib slots in fixtures)
        self.fixture_slot_clearance = QDoubleSpinBox()
        self.fixture_slot_clearance.setRange(0.0, 1.0)
        self.fixture_slot_clearance.setValue(0.15)
        self.fixture_slot_clearance.setSingleStep(0.05)
        self.fixture_slot_clearance.setSuffix(" mm")
        self.fixture_slot_clearance.setToolTip(
            "Clearance for spar/rib slots in fixtures"
        )
        form.addRow("Slot Clearance:", self.fixture_slot_clearance)

        # Add cradle toggle
        self.fixture_add_cradle = QCheckBox("Add Under-Cradle")
        self.fixture_add_cradle.setChecked(True)
        self.fixture_add_cradle.setToolTip(
            "Add cradle pieces that follow lower wing surface"
        )
        form.addRow(self.fixture_add_cradle)

        # Tab width (for base plate interlocking)
        self.fixture_tab_width = QDoubleSpinBox()
        self.fixture_tab_width.setRange(5.0, 50.0)
        self.fixture_tab_width.setValue(15.0)
        self.fixture_tab_width.setSuffix(" mm")
        self.fixture_tab_width.setToolTip("Width of tabs for base plate slots")
        form.addRow("Tab Width:", self.fixture_tab_width)

        # Tab spacing
        self.fixture_tab_spacing = QDoubleSpinBox()
        self.fixture_tab_spacing.setRange(20.0, 200.0)
        self.fixture_tab_spacing.setValue(80.0)
        self.fixture_tab_spacing.setSuffix(" mm")
        self.fixture_tab_spacing.setToolTip("Spacing between tabs along fixture")
        form.addRow("Tab Spacing:", self.fixture_tab_spacing)

        return group

    def _build_dxf_group(self) -> QGroupBox:
        """Build 2D Manufacturing Export (DXF) UI group."""
        group = QGroupBox("2D Manufacturing Export (DXF)")
        main_layout = QVBoxLayout(group)

        # Check ezdxf availability
        ezdxf_ok = is_ezdxf_available()
        if not ezdxf_ok:
            warning = QLabel("⚠️ ezdxf not installed. Run: pip install ezdxf")
            warning.setStyleSheet("color: orange; font-weight: bold;")
            main_layout.addWidget(warning)

        # Horizontal layout for settings groups
        settings_row = QHBoxLayout()

        # === Structural Parameters (from Structure Tab - read-only) ===
        struct_group = QGroupBox("Structural Parameters (from Structure Tab)")
        struct_form = QFormLayout(struct_group)

        # Read-only displays of values from project.wing.planform
        plan = self.project.wing.planform

        self.dxf_rib_thickness_label = QLabel(f"{plan.rib_thickness_mm:.1f} mm")
        struct_form.addRow("Rib Thickness:", self.dxf_rib_thickness_label)

        self.dxf_spar_thickness_label = QLabel(f"{plan.spar_thickness_mm:.1f} mm")
        struct_form.addRow("Spar Thickness:", self.dxf_spar_thickness_label)

        stringer_info = f"{plan.stringer_count} × {plan.stringer_height_mm:.1f}×{plan.stringer_thickness_mm:.1f} mm"
        self.dxf_stringer_label = QLabel(
            stringer_info if plan.stringer_count > 0 else "None"
        )
        struct_form.addRow("Stringers:", self.dxf_stringer_label)

        self.dxf_lightening_label = QLabel(f"{plan.rib_lightening_fraction * 100:.0f}%")
        struct_form.addRow("Lightening Fraction:", self.dxf_lightening_label)

        # Refresh button to update from project
        refresh_btn = QPushButton("↻ Refresh from Project")
        refresh_btn.clicked.connect(self._refresh_dxf_struct_params)
        struct_form.addRow(refresh_btn)

        settings_row.addWidget(struct_group)

        # === Export Options ===
        options_group = QGroupBox("Export Options")
        options_form = QFormLayout(options_group)

        self.dxf_add_spar_notches = QCheckBox("Add Spar Notches")
        self.dxf_add_spar_notches.setChecked(True)
        options_form.addRow(self.dxf_add_spar_notches)

        self.dxf_add_stringer_cutouts = QCheckBox("Add Stringer Cutouts")
        self.dxf_add_stringer_cutouts.setChecked(True)
        options_form.addRow(self.dxf_add_stringer_cutouts)

        self.dxf_add_elevon_cutouts = QCheckBox("Add Elevon Cutouts")
        self.dxf_add_elevon_cutouts.setChecked(True)
        self.dxf_add_elevon_cutouts.setToolTip(
            "Truncate ribs at hinge line in elevon region"
        )
        options_form.addRow(self.dxf_add_elevon_cutouts)

        self.dxf_add_lightening_holes = QCheckBox("Add Lightening Holes")
        self.dxf_add_lightening_holes.setChecked(False)
        options_form.addRow(self.dxf_add_lightening_holes)

        # Grain direction indicator (new)
        self.dxf_show_grain = QCheckBox("Add Grain Direction Arrows")
        self.dxf_show_grain.setChecked(False)
        self.dxf_show_grain.setToolTip(
            "Add grain direction arrows and labels on ENGRAVE layer:\n"
            "- Ribs: Chordwise (LE to TE)\n"
            "- Spars: Spanwise (Root to Tip)"
        )
        options_form.addRow(self.dxf_show_grain)

        self.dxf_label_ribs = QCheckBox("Label Ribs")
        self.dxf_label_ribs.setChecked(True)
        options_form.addRow(self.dxf_label_ribs)

        self.dxf_label_spars = QCheckBox("Label Spars")
        self.dxf_label_spars.setChecked(True)
        options_form.addRow(self.dxf_label_spars)

        settings_row.addWidget(options_group)

        # === Manufacturing Clearances ===
        clearance_group = QGroupBox("Manufacturing Clearances")
        clearance_form = QFormLayout(clearance_group)

        self.dxf_spar_notch_clearance = QDoubleSpinBox()
        self.dxf_spar_notch_clearance.setRange(0.0, 1.0)
        self.dxf_spar_notch_clearance.setValue(0.1)
        self.dxf_spar_notch_clearance.setSingleStep(0.05)
        self.dxf_spar_notch_clearance.setSuffix(" mm")
        self.dxf_spar_notch_clearance.setToolTip(
            "Clearance added to spar notch width for slip fit"
        )
        clearance_form.addRow("Spar Notch Clearance:", self.dxf_spar_notch_clearance)

        self.dxf_rib_notch_clearance = QDoubleSpinBox()
        self.dxf_rib_notch_clearance.setRange(0.0, 1.0)
        self.dxf_rib_notch_clearance.setValue(0.1)
        self.dxf_rib_notch_clearance.setSingleStep(0.05)
        self.dxf_rib_notch_clearance.setSuffix(" mm")
        self.dxf_rib_notch_clearance.setToolTip(
            "Clearance added to rib notch width in spars"
        )
        clearance_form.addRow("Rib Notch Clearance:", self.dxf_rib_notch_clearance)

        self.dxf_stringer_slot_clearance = QDoubleSpinBox()
        self.dxf_stringer_slot_clearance.setRange(0.0, 1.0)
        self.dxf_stringer_slot_clearance.setValue(0.2)
        self.dxf_stringer_slot_clearance.setSingleStep(0.05)
        self.dxf_stringer_slot_clearance.setSuffix(" mm")
        self.dxf_stringer_slot_clearance.setToolTip(
            "Clearance for stringer slots in ribs"
        )
        clearance_form.addRow(
            "Stringer Slot Clearance:", self.dxf_stringer_slot_clearance
        )

        self.dxf_notch_depth_percent = QDoubleSpinBox()
        self.dxf_notch_depth_percent.setRange(20.0, 60.0)
        self.dxf_notch_depth_percent.setValue(50.0)
        self.dxf_notch_depth_percent.setSuffix(" %")
        self.dxf_notch_depth_percent.setToolTip(
            "How deep notches cut into each other (% of local thickness)\n"
            "Typically 50% for half-lap joints."
        )
        clearance_form.addRow("Interlock Depth:", self.dxf_notch_depth_percent)

        settings_row.addWidget(clearance_group)

        # === Sheet Nesting ===
        nest_group = QGroupBox("Sheet Nesting")
        nest_form = QFormLayout(nest_group)

        self.dxf_sheet_width = QDoubleSpinBox()
        self.dxf_sheet_width.setRange(100, 99999)
        self.dxf_sheet_width.setValue(600)
        self.dxf_sheet_width.setSuffix(" mm")
        nest_form.addRow("Sheet Width:", self.dxf_sheet_width)

        self.dxf_sheet_height = QDoubleSpinBox()
        self.dxf_sheet_height.setRange(100, 99999)
        self.dxf_sheet_height.setValue(300)
        self.dxf_sheet_height.setSuffix(" mm")
        nest_form.addRow("Sheet Height:", self.dxf_sheet_height)

        self.dxf_part_spacing = QDoubleSpinBox()
        self.dxf_part_spacing.setRange(1, 100)
        self.dxf_part_spacing.setValue(5)
        self.dxf_part_spacing.setSuffix(" mm")
        nest_form.addRow("Part Spacing:", self.dxf_part_spacing)

        self.dxf_edge_margin = QDoubleSpinBox()
        self.dxf_edge_margin.setRange(0, 200)
        self.dxf_edge_margin.setValue(10)
        self.dxf_edge_margin.setSuffix(" mm")
        nest_form.addRow("Edge Margin:", self.dxf_edge_margin)

        # Rotation fitting info label (fixed ±5°, not user-adjustable)
        rotation_info = QLabel("Auto-rotation: ±5° (1° steps)")
        rotation_info.setStyleSheet("color: gray; font-style: italic;")
        rotation_info.setToolTip(
            "Long parts will be automatically rotated up to ±5° to fit on the sheet.\n"
            "This is a fixed setting for optimal nesting."
        )
        nest_form.addRow(rotation_info)

        # Finger joint splitting option
        self.dxf_allow_splitting = QCheckBox("Allow Finger Joint Splitting")
        self.dxf_allow_splitting.setChecked(True)
        self.dxf_allow_splitting.setToolTip(
            "If a spar is too long to fit even with rotation,\n"
            "offer to split it with finger joints.\n\n"
            "Split position avoids rib notches by 20mm."
        )
        nest_form.addRow(self.dxf_allow_splitting)

        settings_row.addWidget(nest_group)

        main_layout.addLayout(settings_row)

        # === Export Buttons ===
        btn_row = QHBoxLayout()

        self.dxf_export_ribs_btn = QPushButton("Export Ribs (DXF)")
        self.dxf_export_ribs_btn.clicked.connect(self._export_ribs_dxf)
        self.dxf_export_ribs_btn.setEnabled(ezdxf_ok)
        btn_row.addWidget(self.dxf_export_ribs_btn)

        self.dxf_export_spars_btn = QPushButton("Export Spars (DXF)")
        self.dxf_export_spars_btn.clicked.connect(self._export_spars_dxf)
        self.dxf_export_spars_btn.setEnabled(ezdxf_ok)
        btn_row.addWidget(self.dxf_export_spars_btn)

        self.dxf_export_nested_btn = QPushButton("Export Nested Layout")
        self.dxf_export_nested_btn.clicked.connect(self._export_nested_dxf)
        self.dxf_export_nested_btn.setEnabled(ezdxf_ok)
        btn_row.addWidget(self.dxf_export_nested_btn)

        main_layout.addLayout(btn_row)

        # Status label
        status = "Ready (ezdxf available)" if ezdxf_ok else "ezdxf not available"
        self.dxf_status = QLabel(f"Status: {status}")
        main_layout.addWidget(self.dxf_status)

        return group

    def _export_ribs_dxf(self):
        """Export rib profiles to DXF files using profiles.py.

        Also exports elevon rib pieces (aft portions) for ribs in control surface regions.
        """
        try:
            import ezdxf
            from ezdxf import units

            from services.export.profiles import (
                generate_elevon_rib_profile,
                generate_grain_indicator,
                generate_rib_profile,
                get_lightening_hole_geometries,
                get_stringer_slot_polylines,
            )
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Missing dependency: {e}")
            return

        sections = self._spanwise_sections_for_export()
        if sections is None:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Ribs"
        )
        if not output_dir:
            return

        try:
            params = self._dxf_rib_params()
            # Note: Rib thickness, spar thickness, stringer dimensions are read
            # from project.wing.planform inside generate_rib_profile()

            generated_files = []
            elevon_rib_count = 0
            show_grain = self.dxf_show_grain.isChecked()

            for section in sections:
                try:
                    profile = generate_rib_profile(section, self.project, params)
                except Exception as e:
                    print(f"Skipping section {section.index}: {e}")
                    continue

                doc, msp = self._new_dxf_doc(ezdxf, units)

                # Draw main outline
                if len(profile.outline) >= 3:
                    points_2d = [(float(p[0]), float(p[1])) for p in profile.outline]
                    msp.add_lwpolyline(
                        points_2d, close=True, dxfattribs={"layer": "CUT"}
                    )

                # Draw stringer slots as separate polylines
                if profile.stringer_slots:
                    slot_polylines = get_stringer_slot_polylines(profile.stringer_slots)
                    for poly in slot_polylines:
                        points_2d = [(float(p[0]), float(p[1])) for p in poly]
                        msp.add_lwpolyline(
                            points_2d, close=True, dxfattribs={"layer": "CUT"}
                        )

                # Draw lightening holes
                if profile.lightening_holes:
                    hole_geoms = get_lightening_hole_geometries(
                        profile.lightening_holes
                    )
                    for geom_type, geom_data in hole_geoms:
                        if geom_type == "circle":
                            cx, cz, radius = geom_data
                            msp.add_circle(
                                center=(cx, cz),
                                radius=radius,
                                dxfattribs={"layer": "CUT"},
                            )
                        elif geom_type == "polyline":
                            points_2d = [(float(p[0]), float(p[1])) for p in geom_data]
                            msp.add_lwpolyline(
                                points_2d, close=True, dxfattribs={"layer": "CUT"}
                            )

                # Draw grain direction indicator
                if show_grain:
                    try:
                        # Calculate rib bounds for grain indicator
                        z_values = [p[1] for p in profile.outline]
                        rib_height = max(z_values) - min(z_values)
                        grain = generate_grain_indicator(
                            "rib", profile.chord_mm, rib_height
                        )

                        # Draw arrow shaft
                        msp.add_line(
                            start=grain.arrow_start,
                            end=grain.arrow_end,
                            dxfattribs={"layer": "ENGRAVE"},
                        )
                        # Draw arrowhead
                        msp.add_line(
                            start=grain.arrow_end,
                            end=grain.arrowhead_left,
                            dxfattribs={"layer": "ENGRAVE"},
                        )
                        msp.add_line(
                            start=grain.arrow_end,
                            end=grain.arrowhead_right,
                            dxfattribs={"layer": "ENGRAVE"},
                        )
                        # Draw label
                        text = msp.add_text(
                            grain.label_text,
                            height=2.0,
                            dxfattribs={"layer": "ENGRAVE"},
                        )
                        text.set_placement(grain.label_position)
                    except Exception as e:
                        print(f"Could not add grain indicator: {e}")

                # Add rib label
                if self.dxf_label_ribs.isChecked():
                    label = f"R{section.index + 1}"
                    cx = profile.chord_mm * 0.35
                    cz = 0  # Approximate center
                    try:
                        text = msp.add_text(
                            label, height=3.0, dxfattribs={"layer": "ENGRAVE"}
                        )
                        text.set_placement((cx, cz))
                    except Exception:
                        pass

                # Save main rib file
                filename = os.path.join(output_dir, f"rib_{section.index + 1:02d}.dxf")
                doc.saveas(filename)
                generated_files.append(filename)

                # Generate elevon rib piece if this rib has an elevon cutout
                if profile.elevon_cutout and profile.elevon_cutout.has_cutout:
                    try:
                        elevon_profile = generate_elevon_rib_profile(
                            section,
                            self.project,
                            profile.elevon_cutout,
                            max_deflection_deg=self._elevon_deflection_deg(),
                            hinge_gap_mm=params.spar_notch_clearance_mm,  # Use same clearance
                        )
                        if elevon_profile and len(elevon_profile.outline) >= 3:
                            elev_doc, elev_msp = self._new_dxf_doc(ezdxf, units)

                            # Draw elevon rib outline
                            points_2d = [
                                (float(p[0]), float(p[1]))
                                for p in elevon_profile.outline
                            ]
                            elev_msp.add_lwpolyline(
                                points_2d, close=True, dxfattribs={"layer": "CUT"}
                            )

                            # Add elevon rib label
                            if self.dxf_label_ribs.isChecked():
                                label = f"E{section.index + 1}"  # E for Elevon
                                cx = (
                                    elevon_profile.hinge_x_mm
                                    + elevon_profile.chord_mm * 0.3
                                )
                                cz = 0
                                try:
                                    text = elev_msp.add_text(
                                        label,
                                        height=3.0,
                                        dxfattribs={"layer": "ENGRAVE"},
                                    )
                                    text.set_placement((cx, cz))
                                except Exception:
                                    pass

                            # Save elevon rib file
                            elev_filename = os.path.join(
                                output_dir, f"elevon_rib_{section.index + 1:02d}.dxf"
                            )
                            elev_doc.saveas(elev_filename)
                            generated_files.append(elev_filename)
                            elevon_rib_count += 1
                    except Exception as e:
                        print(
                            f"Could not generate elevon rib for section {section.index}: {e}"
                        )

            # Build summary message
            summary = f"Exported {len(generated_files)} DXF files to:\n{output_dir}\n\n"
            summary += f"Main ribs: {len(generated_files) - elevon_rib_count}\n"
            if elevon_rib_count > 0:
                summary += f"Elevon ribs: {elevon_rib_count}\n"
            
            summary += "\nInterlocking: Notches (edge cutouts)"


            QMessageBox.information(self, "Success", summary)
            self.dxf_status.setText(f"Status: Exported {len(generated_files)} ribs")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"DXF export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_spars_dxf(self):
        """Export spar plates to DXF files using profiles.py.

        For BWB aircraft with body_sections, generates separate files for
        body and wing regions.
        """
        try:
            import ezdxf
            from ezdxf import units

            from services.export.profiles import (
                generate_grain_indicator,
                generate_separated_spar_profiles,
            )
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Missing dependency: {e}")
            return

        sections = self._spanwise_sections_for_export(min_count=2)
        if sections is None:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Spars"
        )
        if not output_dir:
            return

        try:
            # Check interlocking type
            show_grain = self.dxf_show_grain.isChecked()

            params = self._dxf_spar_params()

            generated_files = []
            cut_line_count = 0

            for spar_type in ["front", "rear"]:
                try:
                    # Use separated profiles for BWB support (now returns single spar with cut lines)
                    profiles = generate_separated_spar_profiles(
                        self.project, sections, spar_type, params
                    )
                except Exception as e:
                    print(f"Skipping {spar_type} spar: {e}")
                    continue

                for profile in profiles:
                    doc, msp = self._new_dxf_doc(ezdxf, units, include_cutline=True)

                    # Draw spar outline (tapered profile following airfoil)
                    # In Tabs & Slots mode, this includes tab protrusions
                    if len(profile.outline) >= 3:
                        points_2d = [
                            (float(p[0]), float(p[1])) for p in profile.outline
                        ]
                        msp.add_lwpolyline(
                            points_2d, close=True, dxfattribs={"layer": "CUT"}
                        )

                    # Draw cut lines for BWB separation
                    if profile.cut_lines:
                        for cut_line in profile.cut_lines:
                            # Draw vertical line from lower to upper edge
                            msp.add_line(
                                start=(cut_line.x_along_span_mm, cut_line.z_lower_mm),
                                end=(cut_line.x_along_span_mm, cut_line.z_upper_mm),
                                dxfattribs={"layer": "CUTLINE"},
                            )
                            # Add label for the cut line
                            if cut_line.label and self.dxf_label_spars.isChecked():
                                try:
                                    text = msp.add_text(
                                        cut_line.label,
                                        height=2.0,
                                        dxfattribs={"layer": "ENGRAVE"},
                                    )
                                    text.set_placement(
                                        (
                                            cut_line.x_along_span_mm + 2,
                                            cut_line.z_upper_mm + 2,
                                        )
                                    )
                                except Exception:
                                    pass
                            cut_line_count += 1

                    # Draw grain direction indicator
                    if show_grain:
                        try:
                            grain = generate_grain_indicator(
                                f"{spar_type}_spar",
                                profile.length_mm,
                                profile.max_height_mm,
                            )

                            # Draw arrow shaft
                            msp.add_line(
                                start=grain.arrow_start,
                                end=grain.arrow_end,
                                dxfattribs={"layer": "ENGRAVE"},
                            )
                            # Draw arrowhead
                            msp.add_line(
                                start=grain.arrow_end,
                                end=grain.arrowhead_left,
                                dxfattribs={"layer": "ENGRAVE"},
                            )
                            msp.add_line(
                                start=grain.arrow_end,
                                end=grain.arrowhead_right,
                                dxfattribs={"layer": "ENGRAVE"},
                            )
                            # Draw label
                            text = msp.add_text(
                                grain.label_text,
                                height=2.0,
                                dxfattribs={"layer": "ENGRAVE"},
                            )
                            text.set_placement(grain.label_position)
                        except Exception as e:
                            print(f"Could not add grain indicator: {e}")

                    # Add label with height info
                    if self.dxf_label_spars.isChecked():
                        label = f"{spar_type.upper()} SPAR"
                        cx = profile.length_mm / 2
                        cz = profile.max_height_mm - 5
                        try:
                            text = msp.add_text(
                                label, height=3.0, dxfattribs={"layer": "ENGRAVE"}
                            )
                            text.set_placement((cx, cz))
                        except Exception:
                            pass

                    # Save file (simple naming - one file per spar type, with region suffix for BWB)
                    filename = os.path.join(output_dir, f"{spar_type}_spar_{profile.spar_region}.dxf")
                    doc.saveas(filename)
                    generated_files.append(filename)


                    # Log height info
                    print(
                        f"{spar_type.upper()} spar: root={profile.height_at_root_mm:.1f}mm, tip={profile.height_at_tip_mm:.1f}mm"
                    )

            # Build message
            msg = (
                f"Exported {len(generated_files)} spar DXF files to:\n{output_dir}\n\n"
            )
            msg += "Spar heights are computed from airfoil geometry."
            if cut_line_count > 0:
                msg += f"\n\n{cut_line_count} cut lines added (green) for BWB body/wing separation."

            QMessageBox.information(self, "Success", msg)
            self.dxf_status.setText(f"Status: Exported {len(generated_files)} spars")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"DXF export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_nested_dxf(self):
        """Export nested layout of all parts on sheets, grouped by material thickness."""
        try:
            import shutil
            import tempfile

            import ezdxf
            from ezdxf import units

            from services.export.dxf_export import (
                PartInfo,
                generate_nested_layout_by_thickness,
                generate_nested_layout_with_fitting,
            )
            from services.export.profiles import (
                generate_elevon_rib_profile,
                generate_rib_profile,
                generate_separated_spar_profiles,
                get_lightening_hole_geometries,
                get_stringer_slot_polylines,
            )
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Missing dependency: {e}")
            return

        sections = self._spanwise_sections_for_export()
        if sections is None:
            return

        # Ask for output file (will become base name for thickness-grouped files)
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Nested Layout", "nested_layout.dxf", "DXF Files (*.dxf)"
        )
        if not output_path:
            return

        # Create temp directory for individual parts
        temp_dir = tempfile.mkdtemp(prefix="dxf_nest_")

        try:
            plan = self.project.wing.planform

            # Get material thicknesses from planform
            rib_thickness = plan.rib_thickness_mm
            spar_thickness = plan.spar_thickness_mm

            rib_params = self._dxf_rib_params()

            # List of PartInfo with thickness tracking
            parts: list[PartInfo] = []

            # Generate ribs to temp files
            for section in sections:
                try:
                    profile = generate_rib_profile(section, self.project, rib_params)
                except Exception as e:
                    print(f"Skipping section {section.index}: {e}")
                    continue

                doc, msp = self._new_dxf_doc(ezdxf, units)

                # Draw main outline
                if len(profile.outline) >= 3:
                    points_2d = [(float(p[0]), float(p[1])) for p in profile.outline]
                    msp.add_lwpolyline(
                        points_2d, close=True, dxfattribs={"layer": "CUT"}
                    )

                # Draw stringer slots
                if profile.stringer_slots:
                    slot_polylines = get_stringer_slot_polylines(profile.stringer_slots)
                    for poly in slot_polylines:
                        points_2d = [(float(p[0]), float(p[1])) for p in poly]
                        msp.add_lwpolyline(
                            points_2d, close=True, dxfattribs={"layer": "CUT"}
                        )

                # Draw lightening holes
                if profile.lightening_holes:
                    hole_geoms = get_lightening_hole_geometries(
                        profile.lightening_holes
                    )
                    for geom_type, geom_data in hole_geoms:
                        if geom_type == "circle":
                            cx, cz, radius = geom_data
                            msp.add_circle(
                                center=(cx, cz),
                                radius=radius,
                                dxfattribs={"layer": "CUT"},
                            )
                        elif geom_type == "polyline":
                            points_2d = [(float(p[0]), float(p[1])) for p in geom_data]
                            msp.add_lwpolyline(
                                points_2d, close=True, dxfattribs={"layer": "CUT"}
                            )

                # Add rib label
                label = f"R{section.index + 1}"
                cx = profile.chord_mm * 0.35
                cz = 0
                try:
                    text = msp.add_text(
                        label, height=3.0, dxfattribs={"layer": "ENGRAVE"}
                    )
                    text.set_placement((cx, cz))
                except Exception:
                    pass

                # Save to temp
                filename = os.path.join(temp_dir, f"rib_{section.index + 1:02d}.dxf")
                doc.saveas(filename)
                parts.append(PartInfo(file_path=filename, thickness_mm=rib_thickness))

                # Generate elevon rib if applicable
                if profile.elevon_cutout and profile.elevon_cutout.has_cutout:
                    try:
                        elevon_profile = generate_elevon_rib_profile(
                            section,
                            self.project,
                            profile.elevon_cutout,
                            max_deflection_deg=self._elevon_deflection_deg(),
                            hinge_gap_mm=rib_params.spar_notch_clearance_mm,
                        )
                        if elevon_profile and len(elevon_profile.outline) >= 3:
                            elev_doc, elev_msp = self._new_dxf_doc(ezdxf, units)

                            points_2d = [
                                (float(p[0]), float(p[1]))
                                for p in elevon_profile.outline
                            ]
                            elev_msp.add_lwpolyline(
                                points_2d, close=True, dxfattribs={"layer": "CUT"}
                            )

                            label = f"E{section.index + 1}"
                            cx = (
                                elevon_profile.hinge_x_mm
                                + elevon_profile.chord_mm * 0.3
                            )
                            try:
                                text = elev_msp.add_text(
                                    label, height=3.0, dxfattribs={"layer": "ENGRAVE"}
                                )
                                text.set_placement((cx, 0))
                            except Exception:
                                pass

                            elev_filename = os.path.join(
                                temp_dir, f"elevon_rib_{section.index + 1:02d}.dxf"
                            )
                            elev_doc.saveas(elev_filename)
                            # Elevon ribs use same thickness as main ribs
                            parts.append(
                                PartInfo(
                                    file_path=elev_filename, thickness_mm=rib_thickness
                                )
                            )
                    except Exception as e:
                        print(f"Could not generate elevon rib: {e}")

            # Generate spars to temp files
            spar_params = self._dxf_spar_params()

            # Track spar profiles for splitting (only spars can be split)
            spar_profiles: dict[str, object] = {}  # filepath -> SparProfile

            for spar_type in ["front", "rear"]:
                try:
                    profiles = generate_separated_spar_profiles(
                        self.project, sections, spar_type, spar_params
                    )
                except Exception as e:
                    print(f"Skipping {spar_type} spar: {e}")
                    continue

                for idx, profile in enumerate(profiles):
                    doc, msp = self._new_dxf_doc(ezdxf, units)

                    if len(profile.outline) >= 3:
                        points_2d = [
                            (float(p[0]), float(p[1])) for p in profile.outline
                        ]
                        msp.add_lwpolyline(
                            points_2d, close=True, dxfattribs={"layer": "CUT"}
                        )

                    label = f"{spar_type.upper()} SPAR"
                    cx = profile.length_mm / 2
                    cz = profile.max_height_mm - 5
                    try:
                        text = msp.add_text(
                            label, height=3.0, dxfattribs={"layer": "ENGRAVE"}
                        )
                        text.set_placement((cx, cz))
                    except Exception:
                        pass

                    # Use index if multiple segments (BWB body/wing separation)
                    suffix = f"_{idx}" if len(profiles) > 1 else ""
                    filename = os.path.join(temp_dir, f"{spar_type}_spar{suffix}.dxf")
                    doc.saveas(filename)
                    # Spars use spar thickness
                    parts.append(
                        PartInfo(file_path=filename, thickness_mm=spar_thickness)
                    )
                    # Track spar profile for potential splitting
                    spar_profiles[filename] = profile

            # Set up nesting parameters
            nesting_params = GridNestingParams(
                sheet_width_mm=self.dxf_sheet_width.value(),
                sheet_height_mm=self.dxf_sheet_height.value(),
                part_spacing_mm=self.dxf_part_spacing.value(),
                margin_mm=self.dxf_edge_margin.value(),
                allow_rotation=True,  # Fixed ±5° rotation
                allow_splitting=self.dxf_allow_splitting.isChecked(),
            )

            # Use new fitting algorithm if splitting is allowed
            if self.dxf_allow_splitting.isChecked():
                # Use advanced nesting with rotation/splitting
                result = generate_nested_layout_with_fitting(
                    parts,
                    output_path,
                    nesting_params,
                    spar_profiles=spar_profiles,
                    split_callback=self._handle_split_prompt,
                )
                
                if not result.success:
                    if result.error_message:
                        QMessageBox.warning(self, "Nesting Incomplete", result.error_message)
                    return
                
                generated_files = result.output_files
                
                # Build result message
                unique_thicknesses = sorted(set(p.thickness_mm for p in parts))
                thickness_str = ", ".join(f"{t:.1f}mm" for t in unique_thicknesses)
                
                msg_parts = [
                    f"Placed {result.placed_count} of {result.total_parts} parts"
                ]
                if result.split_count > 0:
                    msg_parts.append(f"{result.split_count} parts split with finger joints")
                if result.skipped_parts:
                    msg_parts.append(f"{len(result.skipped_parts)} parts skipped")
                
                if len(generated_files) == 1:
                    msg = (
                        f"Exported nested layout to:\n{generated_files[0]}\n\n"
                        + "\n".join(msg_parts) + "\n\n"
                        f"Sheet size: {nesting_params.sheet_width_mm} × {nesting_params.sheet_height_mm} mm\n"
                        f"Material thickness: {thickness_str}"
                    )
                else:
                    file_list = "\n".join(
                        f"  • {os.path.basename(f)}" for f in generated_files
                    )
                    msg = (
                        f"{len(generated_files)} sheets generated:\n{file_list}\n\n"
                        + "\n".join(msg_parts) + "\n\n"
                        f"Sheet size: {nesting_params.sheet_width_mm} × {nesting_params.sheet_height_mm} mm\n"
                        f"Material thicknesses: {thickness_str}"
                    )
            else:
                # Use simple thickness-grouped nesting (original behavior)
                generated_files = generate_nested_layout_by_thickness(
                    parts, output_path, nesting_params
                )

                # Determine unique thicknesses for message
                unique_thicknesses = sorted(set(p.thickness_mm for p in parts))
                thickness_str = ", ".join(f"{t:.1f}mm" for t in unique_thicknesses)

                if len(generated_files) == 1:
                    msg = (
                        f"Exported nested layout with {len(parts)} parts to:\n{generated_files[0]}\n\n"
                        f"Sheet size: {nesting_params.sheet_width_mm} × {nesting_params.sheet_height_mm} mm\n"
                        f"Material thickness: {thickness_str}"
                    )
                else:
                    file_list = "\n".join(
                        f"  • {os.path.basename(f)}" for f in generated_files
                    )
                    msg = (
                        f"Exported {len(parts)} parts to {len(generated_files)} sheets "
                        f"(grouped by thickness):\n{file_list}\n\n"
                        f"Sheet size: {nesting_params.sheet_width_mm} × {nesting_params.sheet_height_mm} mm\n"
                        f"Material thicknesses: {thickness_str}"
                    )

            QMessageBox.information(self, "Success", msg)
            self.dxf_status.setText(
                f"Status: Exported nested layout ({len(parts)} parts, {len(generated_files)} sheets)"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Nested export failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def _handle_split_prompt(
        self, part_name: str, part_length: float, sheet_width: float
    ) -> SplitUserChoice:
        """
        Show dialog asking user if they want to split an oversized part.
        
        Args:
            part_name: Name of the part that doesn't fit
            part_length: Actual length of the part
            sheet_width: Available sheet width
        
        Returns:
            SplitUserChoice enum value
        """
        msg = (
            f"Part '{part_name}' is too long to fit on the sheet:\n\n"
            f"  Part length: {part_length:.1f} mm\n"
            f"  Sheet width: {sheet_width:.1f} mm\n\n"
            "Would you like to split this part with finger joints?\n"
            "(Split position will avoid rib notch locations)"
        )
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Oversized Part")
        dialog.setText(msg)
        dialog.setIcon(QMessageBox.Icon.Question)
        
        # Add custom buttons
        yes_btn = dialog.addButton("Yes, Split This", QMessageBox.ButtonRole.YesRole)
        yes_all_btn = dialog.addButton("Yes to All", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = dialog.addButton("Skip This Part", QMessageBox.ButtonRole.NoRole)
        cancel_btn = dialog.addButton("Cancel Export", QMessageBox.ButtonRole.RejectRole)
        
        dialog.exec()
        
        clicked = dialog.clickedButton()
        if clicked == yes_btn:
            return SplitUserChoice.YES_THIS_ONLY
        elif clicked == yes_all_btn:
            return SplitUserChoice.YES_TO_ALL
        elif clicked == skip_btn:
            return SplitUserChoice.NO_SKIP
        else:
            return SplitUserChoice.CANCEL

    def _refresh_dxf_struct_params(self):
        """Refresh the DXF structural parameters display from project."""
        plan = self.project.wing.planform

        self.dxf_rib_thickness_label.setText(f"{plan.rib_thickness_mm:.1f} mm")
        self.dxf_spar_thickness_label.setText(f"{plan.spar_thickness_mm:.1f} mm")

        if plan.stringer_count > 0:
            stringer_info = f"{plan.stringer_count} × {plan.stringer_height_mm:.1f}×{plan.stringer_thickness_mm:.1f} mm"
        else:
            stringer_info = "None"
        self.dxf_stringer_label.setText(stringer_info)

        self.dxf_lightening_label.setText(f"{plan.rib_lightening_fraction * 100:.0f}%")
        self.dxf_lightening_label.setToolTip(
            f"Fraction: {plan.rib_lightening_fraction*100:.0f}%\n"
            f"Margin: {plan.lightening_hole_margin_mm:.1f} mm\n"
            f"Shape: {plan.lightening_hole_shape}"
        )

    def sync_to_project(self):
        if self.project is None:
            return
        settings = getattr(self.project.analysis, "gui_settings", None)
        if settings is None:
            self.project.analysis.gui_settings = {}
            settings = self.project.analysis.gui_settings
        settings["export_tab"] = self._collect_gui_settings()

    def update_from_project(self):
        if self.project is None:
            return
        settings = getattr(self.project.analysis, "gui_settings", {}).get("export_tab", {})
        if settings:
            self._apply_gui_settings(settings)
        self._refresh_dxf_struct_params()

    def _collect_gui_settings(self) -> Dict[str, Any]:
        return {
            "use_spline_wing_step": bool(self.use_spline_wing_chk.isChecked()),
            "cut_wing_for_elevon": bool(self.cut_wing_chk.isChecked()),
            "elevon_angle_deg": self.elevon_angle.text(),
            "fixture_material_thickness_mm": float(self.fixture_material_thickness.value()),
            "fixture_slot_clearance_mm": float(self.fixture_slot_clearance.value()),
            "fixture_add_cradle": bool(self.fixture_add_cradle.isChecked()),
            "fixture_tab_width_mm": float(self.fixture_tab_width.value()),
            "fixture_tab_spacing_mm": float(self.fixture_tab_spacing.value()),
            "dxf_add_spar_notches": bool(self.dxf_add_spar_notches.isChecked()),
            "dxf_add_stringer_cutouts": bool(self.dxf_add_stringer_cutouts.isChecked()),
            "dxf_add_elevon_cutouts": bool(self.dxf_add_elevon_cutouts.isChecked()),
            "dxf_add_lightening_holes": bool(self.dxf_add_lightening_holes.isChecked()),
            "dxf_show_grain": bool(self.dxf_show_grain.isChecked()),
            "dxf_label_ribs": bool(self.dxf_label_ribs.isChecked()),
            "dxf_label_spars": bool(self.dxf_label_spars.isChecked()),
            "dxf_spar_notch_clearance_mm": float(self.dxf_spar_notch_clearance.value()),
            "dxf_rib_notch_clearance_mm": float(self.dxf_rib_notch_clearance.value()),
            "dxf_stringer_slot_clearance_mm": float(self.dxf_stringer_slot_clearance.value()),
            "dxf_notch_depth_percent": float(self.dxf_notch_depth_percent.value()),
            "dxf_sheet_width_mm": float(self.dxf_sheet_width.value()),
            "dxf_sheet_height_mm": float(self.dxf_sheet_height.value()),
            "dxf_part_spacing_mm": float(self.dxf_part_spacing.value()),
            "dxf_edge_margin_mm": float(self.dxf_edge_margin.value()),
            "dxf_allow_splitting": bool(self.dxf_allow_splitting.isChecked()),
            "viewer_actor_prefs": dict(getattr(self, "viewer_actor_prefs", {}) or {}),
        }

    def _apply_gui_settings(self, settings: Dict[str, Any]) -> None:
        def _set_spin(spin: Any, value: Any) -> None:
            if value is None:
                return
            try:
                spin.setValue(float(value))
            except Exception:
                return

        def _set_check(check: QCheckBox, value: Any) -> None:
            if value is not None:
                check.setChecked(bool(value))

        _set_check(self.use_spline_wing_chk, settings.get("use_spline_wing_step"))
        _set_check(self.cut_wing_chk, settings.get("cut_wing_for_elevon"))
        if settings.get("elevon_angle_deg") is not None:
            self.elevon_angle.setText(str(settings.get("elevon_angle_deg")))
        _set_spin(
            self.fixture_material_thickness,
            settings.get("fixture_material_thickness_mm"),
        )
        _set_spin(self.fixture_slot_clearance, settings.get("fixture_slot_clearance_mm"))
        _set_check(self.fixture_add_cradle, settings.get("fixture_add_cradle"))
        _set_spin(self.fixture_tab_width, settings.get("fixture_tab_width_mm"))
        _set_spin(self.fixture_tab_spacing, settings.get("fixture_tab_spacing_mm"))
        _set_check(self.dxf_add_spar_notches, settings.get("dxf_add_spar_notches"))
        _set_check(self.dxf_add_stringer_cutouts, settings.get("dxf_add_stringer_cutouts"))
        _set_check(self.dxf_add_elevon_cutouts, settings.get("dxf_add_elevon_cutouts"))
        _set_check(self.dxf_add_lightening_holes, settings.get("dxf_add_lightening_holes"))
        _set_check(self.dxf_show_grain, settings.get("dxf_show_grain"))
        _set_check(self.dxf_label_ribs, settings.get("dxf_label_ribs"))
        _set_check(self.dxf_label_spars, settings.get("dxf_label_spars"))
        _set_spin(self.dxf_spar_notch_clearance, settings.get("dxf_spar_notch_clearance_mm"))
        _set_spin(self.dxf_rib_notch_clearance, settings.get("dxf_rib_notch_clearance_mm"))
        _set_spin(
            self.dxf_stringer_slot_clearance,
            settings.get("dxf_stringer_slot_clearance_mm"),
        )
        _set_spin(self.dxf_notch_depth_percent, settings.get("dxf_notch_depth_percent"))
        _set_spin(self.dxf_sheet_width, settings.get("dxf_sheet_width_mm"))
        _set_spin(self.dxf_sheet_height, settings.get("dxf_sheet_height_mm"))
        _set_spin(self.dxf_part_spacing, settings.get("dxf_part_spacing_mm"))
        _set_spin(self.dxf_edge_margin, settings.get("dxf_edge_margin_mm"))
        _set_check(self.dxf_allow_splitting, settings.get("dxf_allow_splitting"))
        actor_prefs = settings.get("viewer_actor_prefs")
        if isinstance(actor_prefs, dict):
            self.viewer_actor_prefs.update(actor_prefs)

    def _spanwise_sections_for_export(self, min_count: int = 1) -> Optional[List[Any]]:
        try:
            sections = AeroSandboxService(self.project).spanwise_sections()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate sections: {e}")
            return None

        if len(sections) < min_count:
            msg = (
                "Need at least 2 sections for spar generation."
                if min_count > 1
                else "No sections available. Check project geometry."
            )
            QMessageBox.warning(self, "Error", msg)
            return None
        return sections

    def _dxf_rib_params(self):
        from services.export.profiles import RibProfileParams

        plan = self.project.wing.planform
        return RibProfileParams(
            include_spar_notches=self.dxf_add_spar_notches.isChecked(),
            include_stringer_slots=self.dxf_add_stringer_cutouts.isChecked(),
            include_lightening_holes=self.dxf_add_lightening_holes.isChecked(),
            include_elevon_cutout=self.dxf_add_elevon_cutouts.isChecked(),
            spar_notch_clearance_mm=self.dxf_spar_notch_clearance.value(),
            spar_notch_depth_percent=self.dxf_notch_depth_percent.value(),
            stringer_slot_clearance_mm=self.dxf_stringer_slot_clearance.value(),
            lightening_hole_margin_mm=plan.lightening_hole_margin_mm,
            lightening_hole_shape=plan.lightening_hole_shape,
        )

    def _dxf_spar_params(self):
        from services.export.profiles import SparProfileParams

        return SparProfileParams(
            include_rib_notches=True,
            rib_notch_clearance_mm=self.dxf_rib_notch_clearance.value(),
            rib_notch_depth_percent=self.dxf_notch_depth_percent.value(),
        )

    def _new_dxf_doc(self, ezdxf: Any, units: Any, include_cutline: bool = False):
        doc = ezdxf.new("R2010")
        doc.header["$INSUNITS"] = units.MM
        doc.layers.add(name="CUT", color=1)
        doc.layers.add(name="ENGRAVE", color=5)
        if include_cutline:
            doc.layers.add(name="CUTLINE", color=3)
        return doc, doc.modelspace()

    def _elevon_deflection_deg(self) -> float:
        try:
            return abs(float(self.elevon_angle.text()))
        except ValueError:
            return 30.0

    def load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CPACS", "", "XML Files (*.xml)"
        )
        if path:
            try:
                with open(path, "r") as f:
                    xml_str = f.read()
                self.cpacs_root = ET.fromstring(xml_str)
                self.process_and_display(xml_str)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CPACS: {e}")

    def generate_from_project(self):
        try:
            data = project_to_cpacs_data(self.project)
            model_name = self.project.wing.name or "FlyingWing"
            # Use rib thickness from project planform
            thick = str(
                self.project.wing.planform.rib_thickness_mm / 1000.0
            )  # Convert mm to m

            self.cpacs_root = generate_cpacs_xml(
                data,
                model_name=model_name,
                thickness_value=thick,
                dihedral_deg=str(self.project.wing.planform.dihedral_deg),
            )
            xml_str = ET.tostring(self.cpacs_root, encoding="unicode")
            self.process_and_display(xml_str)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate CPACS: {e}")
            import traceback

            traceback.print_exc()

    def process_and_display(self, xml_str: str):
        self.processed = process_cpacs_data(xml_str)
        if self.processed.success:
            self.populate_spar_controls()
            self.draw_geometry()
        else:
            QMessageBox.warning(
                self, "Warning", "CPACS processing failed or returned no geometry."
            )

    def populate_spar_controls(self):
        # Individual spar controls removed - now using project planform values
        pass

    def draw_geometry(self):
        if not self.processed or not self.processed.success:
            return
        try:
            # Propagate UI values to processed object
            try:
                self.processed.elevon_angle_deg = float(self.elevon_angle.text())
            except:
                self.processed.elevon_angle_deg = 0.0

            # Hardcode deflection height scale to 1.0
            self.processed.deflection_height_scale = 1.0

            # Hardcode cutter hinge mode to centerline
            self.processed.cutter_hinge_mode = "Centerline of rear spar"

            if self.inline_pv.is_ready():
                # Sync prefs
                self.inline_pv._actor_prefs = self.viewer_actor_prefs.copy()
                self.inline_pv.update_scene(self.processed, include_elevon=True)
                self.inline_pv._apply_actor_prefs()
        except Exception as e:
            print(f"draw_geometry error: {e}")

    def export_flow5_file(self):
        """
        Export Flow5 XML and airfoil .dat files from the current project state.
        
        Flow5 is an open-source VLM/Panel code. The export creates:
        - A single .xml file with the xflplane format
        - Multiple .dat airfoil files (one per section, ≤240 points each)
        """
        if self.project is None:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Flow5 Export"
        )
        if not output_dir:
            return

        try:
            model_name = self.project.wing.name or "FlyingWingProject"
            
            xml_path, dat_files = export_flow5_project(
                self.project,
                output_dir,
                model_name=model_name,
                max_airfoil_points=240,
            )

            msg = (
                f"Flow5 export successful!\n\n"
                f"XML file: {os.path.basename(xml_path)}\n"
                f"Airfoil files: {len(dat_files)} .dat files\n\n"
                f"Location: {output_dir}"
            )
            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Flow5 export failed: {e}")
            import traceback
            traceback.print_exc()

    def export_step_file(self):
        """Export 3D CAD model using new direct Project -> STEP pipeline."""
        if self.project is None:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        try:
            from services.export.geometry_builder import WingGeometryConfig
            from services.export.step_export import write_step
            from services.step_export import build_step_from_project

            # Build config from UI options
            stringer_count = getattr(self.project.wing.planform, "stringer_count", 0) or 0
            lightening_fraction = getattr(self.project.wing.planform, "rib_lightening_fraction", 0.0) or 0.0
            config = WingGeometryConfig(
                include_skin=True,
                include_wingbox=True,
                include_ribs=True,
                include_stringers=stringer_count > 0,
                include_control_surfaces=self.cut_wing_chk.isChecked(),
                scale_factor=1000.0,  # m to mm
                mirror_to_full_aircraft=True,
                include_spar_notches=True,
                include_stringer_slots=stringer_count > 0,
                include_lightening_holes=lightening_fraction > 0.0,
                lightening_hole_shape=self.project.wing.planform.lightening_hole_shape,
                apply_sweep_correction=True,
            )

            # Build geometry directly from Project (no CPACS intermediate)
            shape = build_step_from_project(self.project, config)

            if shape.IsNull():
                QMessageBox.warning(self, "Error", "No geometry generated.")
                return

            path, _ = QFileDialog.getSaveFileName(
                self, "Save STEP", "", "STEP Files (*.stp *.step)"
            )
            if path:
                if write_step(shape, path):
                    QMessageBox.information(self, "Success", f"Saved STEP to {path}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to write STEP file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export STEP: {e}")
            import traceback

            traceback.print_exc()

    def export_cfd_wing_file(self):
        """Export solid half-wing for CFD analysis (OML only, no internal structure)."""
        if self.project is None:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        try:
            from services.export.step_export import write_step
            from services.step_export import build_cfd_wing_solid

            # Generate default filename: {project_name}CFD.step
            model_name = self.project.wing.name or "FlyingWing"
            # Sanitize for filesystem
            safe_name = "".join(c for c in model_name if c.isalnum() or c in "._- ")
            default_filename = f"{safe_name}CFD.step"

            path, _ = QFileDialog.getSaveFileName(
                self, "Save CFD Wing", default_filename, "STEP Files (*.stp *.step)"
            )
            if not path:
                return

            # Build CFD wing solid (half-wing, spline profiles, no internal structure)
            shape = build_cfd_wing_solid(
                self.project,
                scale_to_mm=True,
            )

            if shape.IsNull():
                QMessageBox.warning(self, "Error", "Failed to generate CFD wing geometry.")
                return

            if write_step(shape, path):
                QMessageBox.information(
                    self, "Success",
                    f"CFD wing exported successfully!\n\n"
                    f"File: {os.path.basename(path)}\n"
                    f"Type: Solid half-wing (OML only)\n"
                    f"Units: mm\n\n"
                    f"Note: No internal structure included.\n"
                    f"Import into CFD software and mirror as needed."
                )
            else:
                QMessageBox.critical(self, "Error", "Failed to write STEP file.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"CFD wing export failed: {e}")
            import traceback
            traceback.print_exc()

    def generate_and_export_layout(self):
        """Generate fixture assembly using the new geometry pipeline."""
        if self.project is None:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        try:
            from services.export.geometry_builder import (
                build_geometry_from_project,
                WingGeometryConfig,
            )
            from services.export.step_export import write_step

            # Build geometry configuration for fixtures
            config = WingGeometryConfig(
                # Enable fixtures
                include_fixtures=True,
                fixture_material_thickness_mm=self.fixture_material_thickness.value(),
                fixture_slot_clearance_mm=self.fixture_slot_clearance.value(),
                fixture_add_cradle=self.fixture_add_cradle.isChecked(),
                fixture_tab_width_mm=self.fixture_tab_width.value(),
                fixture_tab_spacing_mm=self.fixture_tab_spacing.value(),

                # Need wing structure for cutting
                include_skin=False,
                include_wingbox=True,   # Need spar geometry
                include_ribs=True,      # Need rib geometry for cutting
                include_stringers=False,
                include_control_surfaces=False,

                # Use planform values
                include_spar_notches=True,
                include_stringer_slots=False,
                include_lightening_holes=False,
            )

            # Build geometry
            geometry = build_geometry_from_project(self.project, config)

            if not geometry.fixture_assembly:
                QMessageBox.warning(self, "Error", "Failed to generate fixture assembly.")
                return

            # Apply scaling (m -> mm) to fixture assembly
            from core.occ_utils.shapes import scale_shape
            scaled_assembly = scale_shape(geometry.fixture_assembly, config.scale_factor)

            # Export
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Fixture Layout", "", "STEP Files (*.stp *.step)"
            )
            if path:
                if write_step(scaled_assembly, path):
                    fixture_count = len(geometry.fixtures) + len(geometry.cradles)
                    QMessageBox.information(
                        self, "Success",
                        f"Saved fixture layout to {path}\n"
                        f"Fixtures: {len(geometry.fixtures)}, "
                        f"Cradles: {len(geometry.cradles)}, "
                        f"Base plate: {'Yes' if geometry.base_plate else 'No'}"
                    )
                else:
                    QMessageBox.critical(self, "Error", "Failed to write STEP file.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fixture export failed: {e}")
            import traceback
            traceback.print_exc()
