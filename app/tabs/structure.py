# app/tabs/structure.py
"""
Structure tab UI for material selection and structural analysis.
"""
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QScrollArea, QSplitter, QFrame,
    QTextEdit, QTabWidget
)
from PyQt6.QtCore import Qt

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    Figure = None

from core.state import Project
from core.models.materials import (
    MATERIAL_PRESETS, 
    get_preset_names_by_category,
    get_material_by_name
)
from services.geometry import AeroSandboxService


class StructureTab(QWidget):
    """Structure tab for material selection and structural analysis."""
    
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.analysis_result: Optional[Dict[str, Any]] = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top section: Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # Materials Group
        materials_group = QGroupBox("Materials")
        materials_layout = QFormLayout(materials_group)
        
        # Spar material selector
        self.spar_material_combo = QComboBox()
        self._populate_material_combo(self.spar_material_combo)
        self.spar_material_combo.currentTextChanged.connect(self._on_spar_material_changed)
        materials_layout.addRow("Spar Material:", self.spar_material_combo)
        
        # Skin material selector
        self.skin_material_combo = QComboBox()
        self._populate_material_combo(self.skin_material_combo)
        self.skin_material_combo.currentTextChanged.connect(self._on_skin_material_changed)
        materials_layout.addRow("Skin Material:", self.skin_material_combo)
        
        # Rib material selector
        self.rib_material_combo = QComboBox()
        self._populate_material_combo(self.rib_material_combo)
        self.rib_material_combo.currentTextChanged.connect(self._on_rib_material_changed)
        materials_layout.addRow("Rib Material:", self.rib_material_combo)
        
        # Stringer material selector
        self.stringer_material_combo = QComboBox()
        self._populate_material_combo(self.stringer_material_combo)
        self.stringer_material_combo.currentTextChanged.connect(self._on_stringer_material_changed)
        materials_layout.addRow("Stringer Material:", self.stringer_material_combo)
        
        # Advanced checkbox (for showing custom material editing)
        self.advanced_chk = QCheckBox("Show Advanced Properties")
        self.advanced_chk.stateChanged.connect(self._on_advanced_toggled)
        materials_layout.addRow(self.advanced_chk)
        
        controls_layout.addWidget(materials_group)
        
        # Geometry Group
        geometry_group = QGroupBox("Geometry")
        geometry_layout = QFormLayout(geometry_group)
        
        self.spar_thickness_spin = QDoubleSpinBox()
        self.spar_thickness_spin.setRange(0.5, 50.0)
        self.spar_thickness_spin.setSingleStep(0.5)
        self.spar_thickness_spin.setValue(3.0)
        self.spar_thickness_spin.setSuffix(" mm")
        self.spar_thickness_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Spar Thickness:", self.spar_thickness_spin)
        
        self.skin_thickness_spin = QDoubleSpinBox()
        self.skin_thickness_spin.setRange(0.1, 20.0)
        self.skin_thickness_spin.setSingleStep(0.1)
        self.skin_thickness_spin.setValue(1.5)
        self.skin_thickness_spin.setSuffix(" mm")
        self.skin_thickness_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Skin Thickness:", self.skin_thickness_spin)
        
        self.rib_thickness_spin = QDoubleSpinBox()
        self.rib_thickness_spin.setRange(0.5, 20.0)
        self.rib_thickness_spin.setSingleStep(0.5)
        self.rib_thickness_spin.setValue(3.0)
        self.rib_thickness_spin.setSuffix(" mm")
        self.rib_thickness_spin.setToolTip("Rib plate thickness. Ribs are placed at each wing section location.")
        self.rib_thickness_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Rib Thickness:", self.rib_thickness_spin)
        
        # Stringer configuration (longitudinal stiffeners between spars)
        self.stringer_count_spin = QSpinBox()
        self.stringer_count_spin.setRange(0, 20)
        self.stringer_count_spin.setValue(0)
        self.stringer_count_spin.setToolTip("Number of longitudinal stiffeners per skin panel. "
                                             "Stringers divide skin into smaller panels, increasing buckling resistance.")
        self.stringer_count_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Stringer Count:", self.stringer_count_spin)
        
        self.stringer_height_spin = QDoubleSpinBox()
        self.stringer_height_spin.setRange(1.0, 50.0)
        self.stringer_height_spin.setSingleStep(1.0)
        self.stringer_height_spin.setValue(10.0)
        self.stringer_height_spin.setSuffix(" mm")
        self.stringer_height_spin.setToolTip("Height/depth of stringer cross-section")
        self.stringer_height_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Stringer Width:", self.stringer_height_spin)
        
        self.stringer_thickness_spin = QDoubleSpinBox()
        self.stringer_thickness_spin.setRange(0.5, 10.0)
        self.stringer_thickness_spin.setSingleStep(0.5)
        self.stringer_thickness_spin.setValue(1.5)
        self.stringer_thickness_spin.setSuffix(" mm")
        self.stringer_thickness_spin.setToolTip("Thickness of stringer web")
        self.stringer_thickness_spin.valueChanged.connect(self._on_geometry_changed)
        geometry_layout.addRow("Stringer Thickness:", self.stringer_thickness_spin)
        
        controls_layout.addWidget(geometry_group)
        
        # Constraints Group
        constraints_group = QGroupBox("Design Constraints")
        constraints_layout = QFormLayout(constraints_group)
        
        self.fos_spin = QDoubleSpinBox()
        self.fos_spin.setRange(1.0, 5.0)
        self.fos_spin.setSingleStep(0.1)
        self.fos_spin.setValue(1.5)
        self.fos_spin.valueChanged.connect(self._on_geometry_changed)
        constraints_layout.addRow("Factor of Safety:", self.fos_spin)
        
        self.max_deflection_spin = QDoubleSpinBox()
        self.max_deflection_spin.setRange(1.0, 50.0)
        self.max_deflection_spin.setSingleStep(1.0)
        self.max_deflection_spin.setValue(15.0)
        self.max_deflection_spin.setSuffix(" %")
        self.max_deflection_spin.valueChanged.connect(self._on_geometry_changed)
        constraints_layout.addRow("Max Tip Deflection:", self.max_deflection_spin)
        
        self.max_twist_spin = QDoubleSpinBox()
        self.max_twist_spin.setRange(0.5, 10.0)
        self.max_twist_spin.setSingleStep(0.5)
        self.max_twist_spin.setValue(3.0)
        self.max_twist_spin.setSuffix(" °")
        self.max_twist_spin.setToolTip("Maximum allowable tip twist (combined bending + torsion).\n"
                                        "Excessive twist reduces control surface effectiveness\n"
                                        "and can cause aileron reversal at high speeds.")
        self.max_twist_spin.valueChanged.connect(self._on_geometry_changed)
        constraints_layout.addRow("Max Tip Twist:", self.max_twist_spin)
        
        self.load_factor_spin = QDoubleSpinBox()
        self.load_factor_spin.setRange(1.0, 10.0)
        self.load_factor_spin.setSingleStep(0.5)
        self.load_factor_spin.setValue(2.5)
        constraints_layout.addRow("Design Load Factor:", self.load_factor_spin)

        self.flight_condition_combo = QComboBox()
        self.flight_condition_combo.addItem("Cruise", "cruise")
        self.flight_condition_combo.addItem("Takeoff", "takeoff")
        self.flight_condition_combo.setToolTip(
            "Named operating point used to generate aerodynamic loads for structural analysis."
        )
        constraints_layout.addRow("Analysis Point:", self.flight_condition_combo)
        
        controls_layout.addWidget(constraints_group)
        
        # Visualization Group
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        
        self.exaggerate_chk = QCheckBox("Exaggerate Deformation")
        self.exaggerate_chk.setChecked(False)
        self.exaggerate_chk.stateChanged.connect(self._on_viz_changed)
        viz_layout.addRow(self.exaggerate_chk)
        
        self.exaggeration_factor_spin = QDoubleSpinBox()
        self.exaggeration_factor_spin.setRange(1.0, 100.0)
        self.exaggeration_factor_spin.setSingleStep(1.0)
        self.exaggeration_factor_spin.setValue(10.0)
        self.exaggeration_factor_spin.setSuffix("x")
        self.exaggeration_factor_spin.setEnabled(False)  # Disabled until checkbox is checked
        self.exaggeration_factor_spin.valueChanged.connect(self._on_viz_changed)
        viz_layout.addRow("Exaggeration Factor:", self.exaggeration_factor_spin)
        
        controls_layout.addWidget(viz_group)
        
        # Analysis Buttons (part of controls row)
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        
        self.analyze_btn = QPushButton("Run Structural\nAnalysis")
        self.analyze_btn.setMinimumSize(120, 60)
        self.analyze_btn.clicked.connect(self._run_analysis)
        button_layout.addWidget(self.analyze_btn)
        
        self.optimize_btn = QPushButton("Optimize\nThickness")
        self.optimize_btn.setMinimumSize(120, 60)
        self.optimize_btn.setToolTip("Find minimum-mass spar and skin thicknesses\n"
                                      "that satisfy stress, buckling, and deflection constraints")
        self.optimize_btn.clicked.connect(self._run_optimization)
        button_layout.addWidget(self.optimize_btn)
        
        self.export_report_btn = QPushButton("Export\nPDF Report")
        self.export_report_btn.setMinimumSize(120, 60)
        self.export_report_btn.setEnabled(False)  # Enabled after analysis
        self.export_report_btn.clicked.connect(self._export_pdf_report)
        button_layout.addWidget(self.export_report_btn)
        
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)
        
        main_layout.addLayout(controls_layout)
        
        # Advanced Options Group (hidden by default, shown when "Show Advanced Properties" is checked)
        self.advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout(self.advanced_group)
        
        # Boundary condition dropdown
        self.boundary_condition_combo = QComboBox()
        self.boundary_condition_combo.addItems(["Simply Supported", "Semi-Restrained", "Clamped"])
        self.boundary_condition_combo.setCurrentText("Semi-Restrained")
        self.boundary_condition_combo.setToolTip("Edge support condition for skin panels. "
                                                   "Simply Supported: k=4.0, Semi-Restrained: k=5.2, Clamped: k=6.35")
        self.boundary_condition_combo.currentTextChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Boundary Condition:", self.boundary_condition_combo)
        
        # Curvature effect checkbox
        self.include_curvature_chk = QCheckBox("Include Curvature Effect")
        self.include_curvature_chk.setChecked(True)
        self.include_curvature_chk.setToolTip("Account for airfoil camber curvature between spars. "
                                               "Curved panels have higher buckling resistance (Batdorf correction).")
        self.include_curvature_chk.stateChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow(self.include_curvature_chk)
        
        # Post-buckling analysis toggle
        self.post_buckling_chk = QCheckBox("Enable Post-Buckling Analysis")
        self.post_buckling_chk.setChecked(False)
        self.post_buckling_chk.setToolTip("Allow skin panels to buckle and redistribute load to stringers.\n"
                                          "Requires stringers to be defined. Uses von Kármán effective width\n"
                                          "and tension field theory for post-buckled shear.")
        self.post_buckling_chk.stateChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow(self.post_buckling_chk)
        
        # --- Analysis Parameters Section ---
        advanced_layout.addRow(QLabel(""))  # Spacer
        advanced_layout.addRow(QLabel("<b>Analysis Parameters</b>"))
        
        # Stringer section type
        self.stringer_section_combo = QComboBox()
        self.stringer_section_combo.addItems(["rectangular", "L_section", "T_section", "hat"])
        self.stringer_section_combo.setCurrentText("rectangular")
        self.stringer_section_combo.setToolTip("Stringer cross-section type for crippling analysis.\n"
                                                "Rectangular: simple balsa strip on edge.\n"
                                                "L/T section: formed sections with flanges.")
        self.stringer_section_combo.currentTextChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Stringer Section Type:", self.stringer_section_combo)
        
        # Rib lightening hole fraction
        self.rib_lightening_spin = QDoubleSpinBox()
        self.rib_lightening_spin.setRange(0.0, 0.7)
        self.rib_lightening_spin.setSingleStep(0.05)
        self.rib_lightening_spin.setValue(0.4)
        self.rib_lightening_spin.setToolTip("Fraction of rib material removed for lightening holes (0.0-0.7).\n"
                                             "0.4 = 40% material removed, typical for balsa ribs.")
        self.rib_lightening_spin.valueChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Rib Lightening Fraction:", self.rib_lightening_spin)
        
        self.lightening_margin_spin = QDoubleSpinBox()
        self.lightening_margin_spin.setRange(5.0, 50.0)
        self.lightening_margin_spin.setSingleStep(1.0)
        self.lightening_margin_spin.setValue(10.0)
        self.lightening_margin_spin.setSuffix(" mm")
        self.lightening_margin_spin.setToolTip("Minimum distance from hole edges to rib boundaries and spar notches.")
        self.lightening_margin_spin.valueChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Hole Margin:", self.lightening_margin_spin)
        
        self.hole_shape_combo = QComboBox()
        self.hole_shape_combo.addItems(["circular", "oval"])
        self.hole_shape_combo.setCurrentText("circular")
        self.hole_shape_combo.setToolTip("Geometry of lightening holes.")
        self.hole_shape_combo.currentTextChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Hole Shape:", self.hole_shape_combo)
        
        # Spar cap width
        self.spar_cap_width_spin = QDoubleSpinBox()
        self.spar_cap_width_spin.setRange(1.0, 50.0)
        self.spar_cap_width_spin.setSingleStep(1.0)
        self.spar_cap_width_spin.setValue(10.0)
        self.spar_cap_width_spin.setSuffix(" mm")
        self.spar_cap_width_spin.setToolTip("Width of spar cap bearing on ribs (for crushing check).")
        self.spar_cap_width_spin.valueChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Spar Cap Width:", self.spar_cap_width_spin)
        
        # Fastener/adhesive mass fraction
        self.fastener_fraction_spin = QDoubleSpinBox()
        self.fastener_fraction_spin.setRange(0.0, 0.30)
        self.fastener_fraction_spin.setSingleStep(0.02)
        self.fastener_fraction_spin.setValue(0.10)
        self.fastener_fraction_spin.setToolTip("Mass adder for fasteners, adhesive, and assembly hardware.\n"
                                                "0.10 = 10% added to structural mass.")
        self.fastener_fraction_spin.valueChanged.connect(self._on_geometry_changed)
        advanced_layout.addRow("Fastener/Adhesive Adder:", self.fastener_fraction_spin)
        
        # --- User-Defined Material Section ---
        advanced_layout.addRow(QLabel(""))  # Spacer
        advanced_layout.addRow(QLabel("<b>User-Defined Material</b>"))
        
        self.user_mat_E1 = QDoubleSpinBox()
        self.user_mat_E1.setRange(0.1, 500)
        self.user_mat_E1.setSingleStep(0.5)
        self.user_mat_E1.setValue(3.5)
        self.user_mat_E1.setSuffix(" GPa")
        self.user_mat_E1.setToolTip("Longitudinal elastic modulus (along grain/fiber)")
        advanced_layout.addRow("E₁ (longitudinal):", self.user_mat_E1)
        
        self.user_mat_E2 = QDoubleSpinBox()
        self.user_mat_E2.setRange(0.01, 100)
        self.user_mat_E2.setSingleStep(0.1)
        self.user_mat_E2.setValue(0.2)
        self.user_mat_E2.setSuffix(" GPa")
        self.user_mat_E2.setToolTip("Transverse elastic modulus (across grain/fiber)")
        advanced_layout.addRow("E₂ (transverse):", self.user_mat_E2)
        
        self.user_mat_G12 = QDoubleSpinBox()
        self.user_mat_G12.setRange(0.01, 100)
        self.user_mat_G12.setSingleStep(0.1)
        self.user_mat_G12.setValue(0.25)
        self.user_mat_G12.setSuffix(" GPa")
        self.user_mat_G12.setToolTip("In-plane shear modulus")
        advanced_layout.addRow("G₁₂ (shear):", self.user_mat_G12)
        
        self.user_mat_density = QDoubleSpinBox()
        self.user_mat_density.setRange(10, 10000)
        self.user_mat_density.setSingleStep(10)
        self.user_mat_density.setValue(160)
        self.user_mat_density.setSuffix(" kg/m³")
        self.user_mat_density.setToolTip("Material density")
        advanced_layout.addRow("Density:", self.user_mat_density)
        
        self.user_mat_sigma_c = QDoubleSpinBox()
        self.user_mat_sigma_c.setRange(0.1, 2000)
        self.user_mat_sigma_c.setSingleStep(1)
        self.user_mat_sigma_c.setValue(10)
        self.user_mat_sigma_c.setSuffix(" MPa")
        self.user_mat_sigma_c.setToolTip("Compressive strength along grain/fiber")
        advanced_layout.addRow("σ compression:", self.user_mat_sigma_c)
        
        self.user_mat_sigma_t = QDoubleSpinBox()
        self.user_mat_sigma_t.setRange(0.1, 2000)
        self.user_mat_sigma_t.setSingleStep(1)
        self.user_mat_sigma_t.setValue(14)
        self.user_mat_sigma_t.setSuffix(" MPa")
        self.user_mat_sigma_t.setToolTip("Tensile strength along grain/fiber")
        advanced_layout.addRow("σ tension:", self.user_mat_sigma_t)
        
        self.apply_user_material_btn = QPushButton("Apply as 'User Defined' Material")
        self.apply_user_material_btn.setToolTip("Apply these values to the 'User Defined' preset\n"
                                                 "and select it for all components.")
        self.apply_user_material_btn.clicked.connect(self._apply_user_defined_material)
        advanced_layout.addRow(self.apply_user_material_btn)
        
        # Hide by default
        self.advanced_group.setVisible(False)
        main_layout.addWidget(self.advanced_group)
        
        # Bottom section: Results
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Summary
        summary_group = QGroupBox("Results Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText("Run analysis to see results...")
        summary_layout.addWidget(self.summary_text)
        
        results_splitter.addWidget(summary_group)
        
        # Right: Plots
        plots_group = QGroupBox("Structural Response")
        plots_layout = QVBoxLayout(plots_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            plots_layout.addWidget(self.canvas)
        else:
            plots_layout.addWidget(QLabel("Matplotlib not available for plotting"))
        
        results_splitter.addWidget(plots_group)
        results_splitter.setSizes([300, 500])
        
        main_layout.addWidget(results_splitter, stretch=1)
        
    def _populate_material_combo(self, combo: QComboBox):
        """Populate a material combo box with presets organized by category."""
        combo.clear()
        categories = get_preset_names_by_category()
        
        for category, names in categories.items():
            for name in names:
                combo.addItem(name)
    
    def _on_spar_material_changed(self, name: str):
        """Handle spar material selection change."""
        if hasattr(self.project.wing, 'planform'):
            self.project.wing.planform.spar_material_name = name
    
    def _on_skin_material_changed(self, name: str):
        """Handle skin material selection change."""
        if hasattr(self.project.wing, 'planform'):
            self.project.wing.planform.skin_material_name = name
    
    def _on_rib_material_changed(self, name: str):
        """Handle rib material selection change."""
        if hasattr(self.project.wing, 'planform'):
            self.project.wing.planform.rib_material_name = name
    
    def _on_stringer_material_changed(self, name: str):
        """Handle stringer material selection change."""
        if hasattr(self.project.wing, 'planform'):
            self.project.wing.planform.stringer_material_name = name
    
    def _on_advanced_toggled(self, state: int):
        """Handle advanced mode toggle - show/hide advanced buckling options."""
        self.advanced_group.setVisible(state == Qt.CheckState.Checked.value)
    
    def _on_geometry_changed(self):
        """Handle geometry parameter changes including advanced options."""
        if hasattr(self.project.wing, 'planform'):
            planform = self.project.wing.planform
            # Basic geometry
            planform.spar_thickness_mm = self.spar_thickness_spin.value()
            planform.skin_thickness_mm = self.skin_thickness_spin.value()
            planform.rib_thickness_mm = self.rib_thickness_spin.value()
            planform.factor_of_safety = self.fos_spin.value()
            planform.max_tip_deflection_percent = self.max_deflection_spin.value()
            planform.max_tip_twist_deg = self.max_twist_spin.value()
            
            # Advanced buckling options (stringer configuration)
            planform.stringer_count = self.stringer_count_spin.value()
            planform.stringer_height_mm = self.stringer_height_spin.value()
            planform.stringer_thickness_mm = self.stringer_thickness_spin.value()
            
            # Boundary condition (convert display text to internal key)
            bc_text = self.boundary_condition_combo.currentText()
            bc_map = {
                "Simply Supported": "simply_supported",
                "Semi-Restrained": "semi_restrained",
                "Clamped": "clamped",
            }
            planform.skin_boundary_condition = bc_map.get(bc_text, "semi_restrained")
            
            # Curvature effect
            planform.include_curvature_effect = self.include_curvature_chk.isChecked()
            
            # Post-buckling toggle
            planform.post_buckling_enabled = self.post_buckling_chk.isChecked()
            
            # Advanced analysis parameters
            if hasattr(self, 'stringer_section_combo'):
                planform.stringer_section_type = self.stringer_section_combo.currentText()
            if hasattr(self, 'rib_lightening_spin'):
                planform.rib_lightening_fraction = self.rib_lightening_spin.value()
            if hasattr(self, 'lightening_margin_spin'):
                planform.lightening_hole_margin_mm = self.lightening_margin_spin.value()
            if hasattr(self, 'hole_shape_combo'):
                planform.lightening_hole_shape = self.hole_shape_combo.currentText()
            if hasattr(self, 'spar_cap_width_spin'):
                planform.spar_cap_width_mm = self.spar_cap_width_spin.value()
            if hasattr(self, 'fastener_fraction_spin'):
                planform.fastener_adhesive_fraction = self.fastener_fraction_spin.value()
    
    def _apply_user_defined_material(self):
        """Apply user-defined material values to the 'User Defined' preset."""
        from core.models.materials import MATERIAL_PRESETS, StructuralMaterial, MaterialType
        
        # Create custom material from UI values
        custom = StructuralMaterial(
            name="User Defined",
            material_type=MaterialType.ORTHOTROPIC,
            E_1=self.user_mat_E1.value() * 1e9,  # GPa to Pa
            E_2=self.user_mat_E2.value() * 1e9,
            E_3=self.user_mat_E2.value() * 1e9,  # Assume E3 ≈ E2
            G_12=self.user_mat_G12.value() * 1e9,
            G_13=self.user_mat_G12.value() * 1e9,
            G_23=self.user_mat_G12.value() * 0.2 * 1e9,  # Estimate G23 ≈ 0.2 * G12
            nu_12=0.3,
            nu_13=0.3,
            nu_23=0.4,
            density=self.user_mat_density.value(),
            sigma_1_tension=self.user_mat_sigma_t.value() * 1e6,  # MPa to Pa
            sigma_1_compression=self.user_mat_sigma_c.value() * 1e6,
            sigma_2_tension=self.user_mat_sigma_t.value() * 0.1 * 1e6,  # Estimate transverse
            sigma_2_compression=self.user_mat_sigma_c.value() * 0.2 * 1e6,
            tau_12=self.user_mat_sigma_c.value() * 0.25 * 1e6,  # Estimate shear strength
        )
        
        # Update the global preset
        MATERIAL_PRESETS["User Defined"] = custom
        
        # Select "User Defined" in all material dropdowns
        for combo in [self.spar_material_combo, self.skin_material_combo, 
                      self.rib_material_combo, self.stringer_material_combo]:
            idx = combo.findText("User Defined")
            if idx >= 0:
                combo.setCurrentIndex(idx)
        
        # Update planform material names
        if hasattr(self.project.wing, 'planform'):
            self.project.wing.planform.spar_material_name = "User Defined"
            self.project.wing.planform.skin_material_name = "User Defined"
            self.project.wing.planform.rib_material_name = "User Defined"
            self.project.wing.planform.stringer_material_name = "User Defined"
        
        QMessageBox.information(self, "Material Applied", 
                               f"Custom material applied as 'User Defined'.\n"
                               f"E₁={self.user_mat_E1.value()} GPa, "
                               f"σc={self.user_mat_sigma_c.value()} MPa, "
                               f"ρ={self.user_mat_density.value()} kg/m³")
    
    def _on_viz_changed(self):
        """Handle visualization setting changes."""
        # Enable/disable exaggeration factor based on checkbox
        self.exaggeration_factor_spin.setEnabled(self.exaggerate_chk.isChecked())
        
        # If we have analysis results, redraw the plots
        if self.analysis_result is not None:
            self._update_plots(self.analysis_result)
    
    def _get_exaggeration_factor(self) -> float:
        """Get the current exaggeration factor for visualization."""
        if self.exaggerate_chk.isChecked():
            return self.exaggeration_factor_spin.value()
        return 1.0

    def _selected_structural_condition_name(self) -> str:
        """Get the currently selected named structural-analysis operating point."""
        selected = self.flight_condition_combo.currentData()
        if selected in {"cruise", "takeoff"}:
            return str(selected)
        return "cruise"

    @staticmethod
    def _format_condition_name(condition_name: Optional[str]) -> str:
        """Format a stored flight-condition key for display."""
        key = str(condition_name or "custom").strip().lower()
        label_map = {
            "cruise": "Cruise",
            "takeoff": "Takeoff",
            "custom": "Custom",
        }
        return label_map.get(key, key.replace("_", " ").title())

    def _build_structural_flight_condition_request(self) -> Dict[str, Any]:
        """Build the structural flight-condition request from the current UI state."""
        return {
            "condition_name": self._selected_structural_condition_name(),
            "load_factor": self.load_factor_spin.value(),
        }

    def _run_analysis(self):
        """Run structural analysis."""
        try:
            # Create service
            service = AeroSandboxService(self.project)
            
            # Run analysis
            flight_condition = self._build_structural_flight_condition_request()
            
            result = service.run_aerostructural_analysis(flight_condition=flight_condition)
            self.analysis_result = result
            
            # Store in project state for persistence
            self.project.analysis.structural_analysis = result
            
            # Check for errors
            if "error" in result:
                QMessageBox.warning(
                    self, 
                    "Analysis Error", 
                    f"Structural analysis failed:\n{result['error']}"
                )
                self.export_report_btn.setEnabled(False)
                return
            
            # Update UI with results
            self._update_summary(result)
            self._update_plots(result)
            
            # Enable export button
            self.export_report_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            self.export_report_btn.setEnabled(False)
    
    def _update_summary(self, result: Dict[str, Any]):
        """Update the summary text with analysis results."""
        struct = result.get("structure", {})
        feasible = result.get("feasible", False)
        
        # Get factor of safety for reference
        fos = struct.get('factor_of_safety', 1.5)
        
        # Get stringer count and curvature status from planform
        stringer_count = 0
        include_curvature = True
        post_buckling_enabled = False
        max_deflection_percent = 15.0
        if hasattr(self.project.wing, 'planform'):
            planform = self.project.wing.planform
            stringer_count = getattr(planform, 'stringer_count', 0)
            include_curvature = getattr(planform, 'include_curvature_effect', True)
            post_buckling_enabled = getattr(planform, 'post_buckling_enabled', False)
            max_deflection_percent = getattr(planform, 'max_tip_deflection_percent', 15.0)
        
        # Helper to format margin with status
        def margin_status(value, threshold, name):
            if value is None or value == float('inf'):
                return f"{name}: N/A"
            status = "OK" if value >= threshold else "FAIL"
            return f"{name}: {value:.2f} (>{threshold:.1f}) [{status}]"
        
        summary_lines = [
            "=" * 40,
            "STRUCTURAL ANALYSIS RESULTS",
            "=" * 40,
            "",
            f"Status: {'FEASIBLE ✓' if feasible else 'NOT FEASIBLE ✗'}",
            "",
            "--- Key Metrics ---",
            f"Structural Mass: {struct.get('mass_kg', 0):.3f} kg",
            f"Tip Deflection: {struct.get('tip_deflection_m', 0) * 1000:.1f} mm",
            f"Max Stress: {struct.get('max_stress_MPa', 0):.1f} MPa",
            "",
            "--- ALL Safety Margins (must exceed FOS) ---",
        ]
        
        # Stress margin
        stress_margin = struct.get('stress_margin', 0)
        summary_lines.append(margin_status(stress_margin, fos, "Stress Margin"))
        
        # Skin buckling margin (or post-buckling if enabled)
        min_buckling = struct.get('min_buckling_margin', 0)
        if post_buckling_enabled:
            is_pb_feasible = struct.get('is_post_buckling_feasible')
            min_pb_margin = struct.get('min_post_buckling_margin')
            pb_status = "OK" if is_pb_feasible else "FAIL"
            if min_pb_margin is not None:
                summary_lines.append(f"Post-Buckling Margin: {min_pb_margin:.2f} (>1.0) [{pb_status}]")
            else:
                summary_lines.append(f"Post-Buckling: {'Enabled' if post_buckling_enabled else 'Disabled'} [{pb_status}]")
            summary_lines.append(f"  (Skin allowed to buckle: {min_buckling:.2f})")
        else:
            summary_lines.append(margin_status(min_buckling, fos, "Skin Buckling"))
        
        # Spar buckling margin
        min_spar = struct.get('min_spar_buckling_margin')
        if min_spar is not None:
            summary_lines.append(margin_status(min_spar, fos, "Spar Web Buckling"))
        
        # Skin shear buckling (Priority 3 - from torsion)
        min_skin_shear = struct.get('min_skin_shear_buckling_margin')
        if min_skin_shear is not None and min_skin_shear != float('inf'):
            summary_lines.append(margin_status(min_skin_shear, fos, "Skin Shear Buckling"))
        
        # Combined biaxial buckling (Priority 5)
        min_combined = struct.get('min_combined_buckling_margin')
        if min_combined is not None and min_combined != float('inf'):
            summary_lines.append(margin_status(min_combined, fos, "Combined σ+τ Buckling"))
        
        # Stringer crippling (Priority 6)
        min_stringer = struct.get('min_stringer_crippling_margin')
        stringer_mode = struct.get('stringer_failure_mode')
        if min_stringer is not None and min_stringer != float('inf') and stringer_count > 0:
            mode_str = f" ({stringer_mode})" if stringer_mode else ""
            status = "OK" if min_stringer >= fos else "FAIL"
            summary_lines.append(f"Stringer Crippling{mode_str}: {min_stringer:.2f} (>{fos:.1f}) [{status}]")
        
        # Rib failure margins
        min_rib_buckling = struct.get('min_rib_buckling_margin')
        min_rib_crushing = struct.get('min_rib_crushing_margin')
        if min_rib_buckling is not None or min_rib_crushing is not None:
            if min_rib_buckling is not None and min_rib_buckling != float('inf'):
                summary_lines.append(margin_status(min_rib_buckling, fos, "Rib Shear Buckling"))
            if min_rib_crushing is not None and min_rib_crushing != float('inf'):
                summary_lines.append(margin_status(min_rib_crushing, fos, "Rib Crushing"))
        
        # Deflection check
        tip_deflection_m = struct.get('tip_deflection_m', 0)
        half_span = self.project.wing.planform.half_span() if hasattr(self.project.wing, 'planform') else 1.0
        deflection_percent = abs(tip_deflection_m) / half_span * 100 if half_span > 0 else 0
        deflection_status = "OK" if deflection_percent <= max_deflection_percent else "FAIL"
        summary_lines.append(f"Tip Deflection: {deflection_percent:.1f}% (<{max_deflection_percent:.0f}%) [{deflection_status}]")
        
        # Twist constraint check
        tip_twist_deg = struct.get('tip_twist_deg')
        max_twist_deg = struct.get('max_tip_twist_deg', 3.0)
        twist_margin = struct.get('twist_margin')
        if tip_twist_deg is not None:
            twist_status = "OK" if twist_margin is not None and twist_margin >= 1.0 else "FAIL"
            summary_lines.append(f"Tip Twist: {tip_twist_deg:.2f}° (<{max_twist_deg:.1f}°) [{twist_status}]")
        
        # Add torsion information if available
        torque = struct.get('torque', [])
        tau_torsion_spar = struct.get('tau_torsion_spar', [])
        has_torsion = (torque and len(torque) > 0 and any(t != 0 for t in torque)) or \
                      (tau_torsion_spar and len(tau_torsion_spar) > 0 and any(t != 0 for t in tau_torsion_spar))
        
        if has_torsion:
            summary_lines.append("")
            summary_lines.append("--- Torsion Analysis ---")
            if torque and len(torque) > 0:
                max_torque = max(abs(t) for t in torque)
                summary_lines.append(f"Max Torque: {max_torque:.2f} N·m")
            if tau_torsion_spar and len(tau_torsion_spar) > 0:
                max_tau_tors = max(abs(t) for t in tau_torsion_spar) / 1e6  # Convert to MPa
                summary_lines.append(f"Max Torsion Shear: {max_tau_tors:.2f} MPa")
            summary_lines.append("(Spar buckling margin includes combined shear)")
        
        summary_lines.extend([
            "",
            "--- Flight Condition ---",
            f"Analysis Point: {self._format_condition_name(result.get('flight_condition', {}).get('condition_name'))}",
            f"Load Factor: {result.get('flight_condition', {}).get('load_factor', 2.5):.1f} g",
        ])
        flight_condition = result.get("flight_condition", {})
        velocity_mps = flight_condition.get("velocity_mps")
        alpha_deg = flight_condition.get("alpha_deg")
        altitude_m = flight_condition.get("altitude_m")
        if velocity_mps is not None:
            summary_lines.append(f"Velocity: {float(velocity_mps):.2f} m/s")
        if alpha_deg is not None:
            summary_lines.append(f"Alpha: {float(alpha_deg):.2f} deg")
        if altitude_m is not None:
            summary_lines.append(f"Altitude: {float(altitude_m):.0f} m")
        
        # Add mass breakdown if available
        mass_breakdown = struct.get('mass_breakdown')
        if mass_breakdown:
            summary_lines.extend([
                "",
                "--- Mass Breakdown ---",
            ])
            component_labels = {
                'wingbox_skins': 'Wingbox Skins',
                'spar_webs': 'Spar Webs',
                'leading_edge': 'Leading Edge',
                'trailing_edge': 'Trailing Edge',
                'ribs': 'Ribs',
                'stringers': 'Stringers',
                'control_surfaces': 'Control Surfaces',
                'fasteners_adhesive': 'Fasteners/Adhesive',
            }
            for key, label in component_labels.items():
                if key in mass_breakdown:
                    mass_g = mass_breakdown[key] * 1000
                    if mass_g > 0.1:  # Only show if non-negligible
                        summary_lines.append(f"  {label}: {mass_g:.1f} g")
            
            # Show counts
            n_ribs = mass_breakdown.get('n_ribs', 0)
            n_stringers = mass_breakdown.get('n_stringers', 0)
            n_control_surfaces = mass_breakdown.get('n_control_surfaces', 0)
            if n_ribs > 0:
                summary_lines.append(f"  (Ribs: {n_ribs} total)")
            if n_stringers > 0:
                summary_lines.append(f"  (Stringers: {n_stringers} total)")
            if n_control_surfaces > 0:
                summary_lines.append(f"  (Control Surfaces: {n_control_surfaces} total)")
        
        # Add advanced options status if any are active
        if stringer_count > 0 or not include_curvature or post_buckling_enabled:
            summary_lines.extend([
                "",
                "--- Configuration ---",
            ])
            if stringer_count > 0:
                summary_lines.append(f"Stringers: {stringer_count} per panel")
            if include_curvature:
                summary_lines.append("Curvature Effect: Enabled (Batdorf)")
            else:
                summary_lines.append("Curvature Effect: Disabled (flat plate)")
            if post_buckling_enabled:
                summary_lines.append("Post-Buckling: Enabled")
        
        # Add assumptions
        assumptions = struct.get('assumptions')
        if assumptions:
            summary_lines.extend([
                "",
                "--- Modeling Assumptions ---",
            ])
            for assumption in assumptions:
                summary_lines.append(f"  • {assumption}")
        
        self.summary_text.setPlainText("\n".join(summary_lines))
    
    def _update_plots(self, result: Dict[str, Any]):
        """Update the matplotlib plots with analysis results."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        struct = result.get("structure", {})
        if not struct:
            return
        
        self.figure.clear()
        
        y = struct.get("y", [])
        if not y:
            return
        
        # Convert y to mm for display
        y_mm = [yi * 1000 for yi in y]
        
        # Create 2x3 subplot grid: 2D plots + 3D view + buckling margins + twist
        ax1 = self.figure.add_subplot(2, 3, 1)
        ax2 = self.figure.add_subplot(2, 3, 2)
        ax3 = self.figure.add_subplot(2, 3, 3)
        ax4 = self.figure.add_subplot(2, 3, 4, projection='3d')
        ax5 = self.figure.add_subplot(2, 3, 5)
        ax6 = self.figure.add_subplot(2, 3, 6)  # Twist plot
        
        # Plot 1: Displacement
        displacement = struct.get("displacement", [])
        if displacement:
            disp_mm = [d * 1000 for d in displacement]
            ax1.plot(y_mm, disp_mm, 'b-', linewidth=1.5)
            ax1.set_xlabel("Spanwise Position [mm]", fontsize=8)
            ax1.set_ylabel("Displacement [mm]", fontsize=8)
            ax1.set_title("Vertical Deflection", fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=7)
        
        # Plot 2: Bending Moment
        moment = struct.get("bending_moment", [])
        if moment:
            ax2.plot(y_mm, moment, 'r-', linewidth=1.5)
            ax2.set_xlabel("Spanwise Position [mm]", fontsize=8)
            ax2.set_ylabel("Bending Moment [N·m]", fontsize=8)
            ax2.set_title("Bending Moment", fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', labelsize=7)
        
        # Plot 3: Load Distribution (Aerodynamic, Inertial, Net)
        aero_load = struct.get("aero_load", [])
        inertial_load = struct.get("inertial_load", [])
        net_load = struct.get("net_load", [])
        
        if aero_load and inertial_load and net_load:
            ax3.plot(y_mm, aero_load, 'b-', linewidth=1.5, label='Aero (lift)')
            ax3.plot(y_mm, inertial_load, 'r-', linewidth=1.5, label='Inertial (weight)')
            ax3.plot(y_mm, net_load, 'k-', linewidth=2.0, label='Net load')
            ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax3.set_xlabel("Spanwise Position [mm]", fontsize=8)
            ax3.set_ylabel("Load [N/m]", fontsize=8)
            ax3.set_title("Spanwise Load Distribution", fontsize=9)
            ax3.legend(loc='upper right', fontsize=6)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', labelsize=7)
            ax3.fill_between(y_mm, 0, aero_load, alpha=0.2, color='blue')
            ax3.fill_between(y_mm, 0, inertial_load, alpha=0.2, color='red')
        
        # Plot 4: Deformed Wing 3D View
        self._draw_deformed_wing(ax4, struct)
        
        # Plot 5: Buckling Margins
        self._draw_buckling_margins(ax5, struct, y_mm)
        
        # Plot 6: Wing Twist Distribution
        self._draw_twist_plot(ax6, struct, y_mm)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _draw_buckling_margins(self, ax, struct: Dict[str, Any], y_mm: list):
        """Draw buckling margins plot showing skin and spar web margins vs span."""
        fos = struct.get('factor_of_safety', 1.5)
        
        # Get buckling margin data
        skin_margin = struct.get("buckling_margin", [])
        spar_margin = struct.get("spar_buckling_margin", [])
        
        has_data = False
        
        if skin_margin:
            # Cap very large values for visualization
            skin_margin_capped = [min(m, 20.0) for m in skin_margin]
            ax.plot(y_mm, skin_margin_capped, 'b-', linewidth=1.5, label='Skin buckling')
            has_data = True
        
        if spar_margin:
            # Cap very large values for visualization
            spar_margin_capped = [min(m, 20.0) for m in spar_margin]
            ax.plot(y_mm, spar_margin_capped, 'g-', linewidth=1.5, label='Spar web shear')
            has_data = True
        
        if has_data:
            # Draw FOS reference line
            ax.axhline(y=fos, color='red', linestyle='--', linewidth=1.5, label=f'FOS = {fos:.1f}')
            
            # Shade the unsafe region
            if y_mm:
                ax.fill_between(y_mm, 0, fos, alpha=0.1, color='red')
            
            ax.set_xlabel("Spanwise Position [mm]", fontsize=8)
            ax.set_ylabel("Buckling Margin", fontsize=8)
            ax.set_title("Buckling Safety Margins", fontsize=9)
            ax.legend(loc='upper right', fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=7)
            
            # Set y-axis limits
            ax.set_ylim(bottom=0)
            if skin_margin or spar_margin:
                all_margins = list(skin_margin or []) + list(spar_margin or [])
                max_margin = min(max(all_margins) if all_margins else 10, 20)
                ax.set_ylim(0, max(max_margin * 1.1, fos * 1.5))
        else:
            ax.text(0.5, 0.5, "No buckling data", ha='center', va='center', transform=ax.transAxes)
    
    def _draw_twist_plot(self, ax, struct: Dict[str, Any], y_mm: list):
        """
        Draw wing twist distribution plot.
        
        Shows:
        - Bending twist (slope from beam deflection) in degrees
        - Torsional twist (from torque/GJ integration) if available
        - Total twist (sum of both)
        """
        import numpy as np
        import math
        
        # Get slope data (beam rotation due to bending)
        slope = struct.get("slope", [])
        torque = struct.get("torque", [])
        GJ = struct.get("GJ", [])
        y = struct.get("y", [])
        
        has_data = False
        
        if slope and len(slope) > 0:
            # Convert slope from radians to degrees
            slope_deg = [math.degrees(s) for s in slope]
            ax.plot(y_mm, slope_deg, 'purple', linewidth=1.5, label='Bending twist')
            has_data = True
            
            # Calculate torsional twist if torque and GJ are available
            if torque and GJ and len(torque) > 0 and len(GJ) > 0 and len(y) > 1:
                try:
                    y_arr = np.array(y)
                    torque_arr = np.array(torque)
                    GJ_arr = np.array(GJ)
                    
                    # Twist rate = T / GJ [rad/m]
                    twist_rate = torque_arr / (GJ_arr + 1e-10)
                    
                    # Integrate from root (y=0) to get total torsional twist
                    # θ(y) = ∫₀ʸ (T/GJ) dy
                    dy = np.gradient(y_arr)
                    torsion_twist_rad = np.cumsum(twist_rate * dy)
                    torsion_twist_deg = np.degrees(torsion_twist_rad)
                    
                    # Only plot if there's meaningful torsional twist
                    if np.max(np.abs(torsion_twist_deg)) > 0.01:
                        ax.plot(y_mm, torsion_twist_deg, 'orange', linewidth=1.5, 
                                linestyle='--', label='Torsion twist')
                        
                        # Total twist = bending + torsion
                        total_twist = np.array(slope_deg) + torsion_twist_deg
                        ax.plot(y_mm, total_twist, 'red', linewidth=2.0, label='Total twist')
                except Exception:
                    pass  # Skip torsion calculation if it fails
        
        if has_data:
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel("Spanwise Position [mm]", fontsize=8)
            ax.set_ylabel("Twist [deg]", fontsize=8)
            ax.set_title("Wing Twist Distribution", fontsize=9)
            ax.legend(loc='best', fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=7)
        else:
            ax.text(0.5, 0.5, "No twist data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Wing Twist Distribution", fontsize=9)
    
    def _draw_deformed_wing(self, ax, struct: Dict[str, Any]):
        """
        Draw a 3D view showing undeformed and deformed wing using AeroSandbox's
        full lofted body mesh (with airfoil thickness).
        
        Includes both heave (vertical displacement) and twist (from beam slope).
        Uses mesh_body() for proper 3D visualization matching the three-view style.
        """
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        y = struct.get("y", [])
        displacement = struct.get("displacement", [])
        
        if not y or not displacement:
            ax.text(0.5, 0.5, 0.5, "No data", ha='center', va='center')
            return
        
        y = np.array(y)
        displacement = np.array(displacement)
        
        # Get exaggeration factor from UI
        exaggeration = self._get_exaggeration_factor()
        
        # Try to use the AeroSandbox lofted body approach
        try:
            service = AeroSandboxService(self.project)
            
            # Get undeformed and deformed wings (with twist from slope)
            wing_undeformed, wing_deformed = service.get_wing_pair_for_visualization(
                structural_result=self.analysis_result,
                exaggeration_factor=exaggeration,
            )
            
            # Get full 3D body meshes (with airfoil thickness) for both wings
            # Using mesh_body() instead of mesh_thin_surface() for lofted geometry
            points_undef, faces_undef = wing_undeformed.mesh_body(
                method="quad",
                chordwise_resolution=24,  # Higher resolution for smooth surfaces
                mesh_tips=True,
                mesh_trailing_edge=True,
            )
            
            points_def, faces_def = wing_deformed.mesh_body(
                method="quad",
                chordwise_resolution=24,
                mesh_tips=True,
                mesh_trailing_edge=True,
            )
            
            # --- Draw undeformed wing (wireframe) ---
            undeformed_polys = points_undef[faces_undef]
            poly_undef = Poly3DCollection(
                undeformed_polys,
                alpha=0.15,
                facecolor='gray',
                edgecolor='dimgray',
                linewidths=0.2,
            )
            ax.add_collection3d(poly_undef)
            
            # --- Draw deformed wing (solid shaded) ---
            deformed_polys = points_def[faces_def]
            poly_def = Poly3DCollection(
                deformed_polys,
                alpha=0.7,
                facecolor='teal',
                edgecolor='darkslategray',
                linewidths=0.3,
            )
            ax.add_collection3d(poly_def)
            
            # Add legend-like labels
            deformed_label = 'Deformed'
            if exaggeration > 1.0:
                deformed_label = f'Deformed ({exaggeration:.0f}x)'
            ax.plot([], [], 's', color='gray', alpha=0.3, markersize=10, label='Undeformed')
            ax.plot([], [], 's', color='teal', alpha=0.7, markersize=10, label=deformed_label)
            ax.legend(loc='upper left', fontsize=7)
            
            # Set axis properties
            ax.set_xlabel("X [m]", fontsize=8)
            ax.set_ylabel("Y [m]", fontsize=8)
            ax.set_zlabel("Z [m]", fontsize=8)
            title = "Deformed Wing Shape"
            if exaggeration > 1.0:
                title += f" (exaggerated {exaggeration:.0f}x)"
            ax.set_title(title, fontsize=9)
            
            # Compute axis limits from actual geometry with EQUAL ASPECT RATIO
            all_points = np.vstack([points_undef, points_def])
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
            
            # Calculate ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            # Find the maximum range to create equal aspect ratio
            max_range = max(x_range, y_range, z_range)
            
            # Add margin (10% of max range)
            margin = max_range * 0.1
            max_range_with_margin = max_range + 2 * margin
            
            # Calculate centers
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            
            # Set equal limits centered on data
            ax.set_xlim(x_center - max_range_with_margin / 2, x_center + max_range_with_margin / 2)
            ax.set_ylim(y_center - max_range_with_margin / 2, y_center + max_range_with_margin / 2)
            ax.set_zlim(z_center - max_range_with_margin / 2, z_center + max_range_with_margin / 2)
            
            # Set equal aspect ratio (matplotlib 3.3+)
            try:
                ax.set_box_aspect([1, 1, 1])
            except AttributeError:
                # Fallback for older matplotlib versions
                pass
            
            # Set view angle for good visibility
            ax.view_init(elev=25, azim=-60)
            
        except Exception as e:
            # Fallback to simple planform visualization if ASB approach fails
            self._draw_deformed_wing_fallback(ax, struct, y, displacement, str(e))
    
    def _draw_deformed_wing_fallback(self, ax, struct: Dict[str, Any], 
                                      y: 'np.ndarray', displacement: 'np.ndarray',
                                      error_msg: str = ""):
        """
        Fallback visualization using simplified planform when ASB mesh is unavailable.
        """
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Get wing geometry from project
        try:
            planform = self.project.wing.planform
            half_span = planform.half_span()
            root_chord = planform.root_chord()
            tip_chord = planform.tip_chord()
            sweep_le_deg = planform.sweep_le_deg
        except Exception:
            # Fallback to simple geometry
            half_span = y[-1] if len(y) > 0 else 1.0
            root_chord = 0.3
            tip_chord = 0.15
            sweep_le_deg = 20.0
        
        # Create simplified wing planform vertices
        n_pts = 20
        eta = np.linspace(0, 1, n_pts)
        y_span = eta * half_span
        
        # Chord distribution (linear taper)
        chords = root_chord * (1 - eta) + tip_chord * eta
        
        # Leading edge x position (with sweep)
        sweep_le_rad = np.radians(sweep_le_deg)
        x_le = y_span * np.tan(sweep_le_rad)
        x_te = x_le + chords
        
        # Interpolate displacement
        z_deformed = np.interp(y_span, y, displacement)
        z_undeformed = np.zeros_like(y_span)
        
        # --- Draw undeformed wing (wireframe) ---
        ax.plot(x_le, y_span, z_undeformed, 'k--', linewidth=0.8, alpha=0.5)
        ax.plot(x_te, y_span, z_undeformed, 'k--', linewidth=0.8, alpha=0.5)
        ax.plot([x_le[0], x_te[0]], [0, 0], [0, 0], 'k--', linewidth=0.8, alpha=0.5)
        ax.plot([x_le[-1], x_te[-1]], [y_span[-1], y_span[-1]], [0, 0], 'k--', linewidth=0.8, alpha=0.5)
        
        # Left wing (mirrored)
        ax.plot(x_le, -y_span, z_undeformed, 'k--', linewidth=0.8, alpha=0.5)
        ax.plot(x_te, -y_span, z_undeformed, 'k--', linewidth=0.8, alpha=0.5)
        ax.plot([x_le[-1], x_te[-1]], [-y_span[-1], -y_span[-1]], [0, 0], 'k--', linewidth=0.8, alpha=0.5)
        
        # --- Draw deformed wing (solid) ---
        ax.plot(x_le, y_span, z_deformed, 'b-', linewidth=1.5, label='Deformed')
        ax.plot(x_te, y_span, z_deformed, 'b-', linewidth=1.5)
        ax.plot([x_le[0], x_te[0]], [0, 0], [z_deformed[0], z_deformed[0]], 'b-', linewidth=1.5)
        ax.plot([x_le[-1], x_te[-1]], [y_span[-1], y_span[-1]], [z_deformed[-1], z_deformed[-1]], 'b-', linewidth=1.5)
        
        # Left wing (mirrored)
        ax.plot(x_le, -y_span, z_deformed, 'b-', linewidth=1.5)
        ax.plot(x_te, -y_span, z_deformed, 'b-', linewidth=1.5)
        ax.plot([x_le[-1], x_te[-1]], [-y_span[-1], -y_span[-1]], [z_deformed[-1], z_deformed[-1]], 'b-', linewidth=1.5)
        
        # Surface patches
        verts = []
        for i in range(len(y_span) - 1):
            quad = [
                [x_le[i], y_span[i], z_deformed[i]],
                [x_te[i], y_span[i], z_deformed[i]],
                [x_te[i+1], y_span[i+1], z_deformed[i+1]],
                [x_le[i+1], y_span[i+1], z_deformed[i+1]],
            ]
            verts.append(quad)
            quad_left = [
                [x_le[i], -y_span[i], z_deformed[i]],
                [x_te[i], -y_span[i], z_deformed[i]],
                [x_te[i+1], -y_span[i+1], z_deformed[i+1]],
                [x_le[i+1], -y_span[i+1], z_deformed[i+1]],
            ]
            verts.append(quad_left)
        
        poly = Poly3DCollection(verts, alpha=0.3, facecolor='C0', edgecolor='none')
        ax.add_collection3d(poly)
        
        # Set axis properties
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        title = "Deformed Wing (Fallback)"
        if error_msg:
            title += f"\n[{error_msg[:40]}...]"
        ax.set_title(title, fontsize=8)
        
        # Set equal aspect ratio limits
        max_disp = max(abs(displacement.min()), abs(displacement.max())) if len(displacement) > 0 else 0.1
        max_range = max(half_span, root_chord, max_disp * 2) * 1.2
        
        # Center the view
        x_center = root_chord / 2
        y_center = 0  # Symmetric wing centered at y=0
        z_center = max_disp / 2 if len(displacement) > 0 else 0
        
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)
        
        # Set equal aspect ratio (matplotlib 3.3+)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
        
        ax.view_init(elev=25, azim=-60)
    
    def sync_to_project(self):
        """Push all editable controls into project state before saving."""
        if self.project is None or not hasattr(self.project.wing, "planform"):
            return
        planform = self.project.wing.planform
        planform.spar_material_name = self.spar_material_combo.currentText()
        planform.skin_material_name = self.skin_material_combo.currentText()
        planform.rib_material_name = self.rib_material_combo.currentText()
        planform.stringer_material_name = self.stringer_material_combo.currentText()
        self._on_geometry_changed()
        settings = getattr(self.project.analysis, "gui_settings", None)
        if settings is None:
            self.project.analysis.gui_settings = {}
            settings = self.project.analysis.gui_settings
        settings["structure_tab"] = self._collect_gui_settings()

    def _collect_gui_settings(self) -> Dict[str, Any]:
        return {
            "advanced_visible": bool(self.advanced_chk.isChecked()),
            "flight_condition": self.flight_condition_combo.currentData(),
            "load_factor": float(self.load_factor_spin.value()),
            "exaggerate_deformation": bool(self.exaggerate_chk.isChecked()),
            "exaggeration_factor": float(self.exaggeration_factor_spin.value()),
            "user_mat_E1_GPa": float(self.user_mat_E1.value()),
            "user_mat_E2_GPa": float(self.user_mat_E2.value()),
            "user_mat_G12_GPa": float(self.user_mat_G12.value()),
            "user_mat_density_kg_m3": float(self.user_mat_density.value()),
            "user_mat_sigma_c_MPa": float(self.user_mat_sigma_c.value()),
            "user_mat_sigma_t_MPa": float(self.user_mat_sigma_t.value()),
        }

    def _apply_gui_settings(self, settings: Dict[str, Any]) -> None:
        def _set_spin(spin: Any, value: Any) -> None:
            if value is None:
                return
            try:
                spin.setValue(float(value))
            except Exception:
                return

        if settings.get("advanced_visible") is not None:
            self.advanced_chk.setChecked(bool(settings.get("advanced_visible")))

        condition = settings.get("flight_condition")
        if condition is not None:
            idx = self.flight_condition_combo.findData(condition)
            if idx >= 0:
                self.flight_condition_combo.setCurrentIndex(idx)

        _set_spin(self.load_factor_spin, settings.get("load_factor"))
        if settings.get("exaggerate_deformation") is not None:
            self.exaggerate_chk.setChecked(bool(settings.get("exaggerate_deformation")))
            self.exaggeration_factor_spin.setEnabled(self.exaggerate_chk.isChecked())
        _set_spin(self.exaggeration_factor_spin, settings.get("exaggeration_factor"))
        _set_spin(self.user_mat_E1, settings.get("user_mat_E1_GPa"))
        _set_spin(self.user_mat_E2, settings.get("user_mat_E2_GPa"))
        _set_spin(self.user_mat_G12, settings.get("user_mat_G12_GPa"))
        _set_spin(self.user_mat_density, settings.get("user_mat_density_kg_m3"))
        _set_spin(self.user_mat_sigma_c, settings.get("user_mat_sigma_c_MPa"))
        _set_spin(self.user_mat_sigma_t, settings.get("user_mat_sigma_t_MPa"))

    def update_from_project(self):
        """Update UI from project state."""
        if not hasattr(self.project.wing, 'planform'):
            return
        
        planform = self.project.wing.planform
        
        # Block signals to avoid triggering change handlers
        self.spar_material_combo.blockSignals(True)
        self.skin_material_combo.blockSignals(True)
        self.rib_material_combo.blockSignals(True)
        self.stringer_material_combo.blockSignals(True)
        self.spar_thickness_spin.blockSignals(True)
        self.skin_thickness_spin.blockSignals(True)
        self.rib_thickness_spin.blockSignals(True)
        self.rib_lightening_spin.blockSignals(True)
        self.lightening_margin_spin.blockSignals(True)
        self.hole_shape_combo.blockSignals(True)
        self.fos_spin.blockSignals(True)
        self.max_deflection_spin.blockSignals(True)
        self.max_twist_spin.blockSignals(True)
        # Stringer geometry (now in Geometry group)
        self.stringer_count_spin.blockSignals(True)
        self.stringer_height_spin.blockSignals(True)
        self.stringer_thickness_spin.blockSignals(True)
        # Advanced options
        self.boundary_condition_combo.blockSignals(True)
        self.include_curvature_chk.blockSignals(True)
        self.post_buckling_chk.blockSignals(True)
        self.stringer_section_combo.blockSignals(True)
        self.spar_cap_width_spin.blockSignals(True)
        self.fastener_fraction_spin.blockSignals(True)
        
        try:
            # Update material combos
            idx = self.spar_material_combo.findText(planform.spar_material_name)
            if idx >= 0:
                self.spar_material_combo.setCurrentIndex(idx)
            
            idx = self.skin_material_combo.findText(planform.skin_material_name)
            if idx >= 0:
                self.skin_material_combo.setCurrentIndex(idx)
            
            idx = self.rib_material_combo.findText(getattr(planform, 'rib_material_name', planform.skin_material_name))
            if idx >= 0:
                self.rib_material_combo.setCurrentIndex(idx)
            
            # Stringer material
            idx = self.stringer_material_combo.findText(getattr(planform, 'stringer_material_name', planform.skin_material_name))
            if idx >= 0:
                self.stringer_material_combo.setCurrentIndex(idx)
            
            # Update geometry
            self.spar_thickness_spin.setValue(planform.spar_thickness_mm)
            self.skin_thickness_spin.setValue(planform.skin_thickness_mm)
            self.rib_thickness_spin.setValue(getattr(planform, 'rib_thickness_mm', 3.0))
            self.rib_lightening_spin.setValue(getattr(planform, 'rib_lightening_fraction', 0.4))
            self.lightening_margin_spin.setValue(getattr(planform, 'lightening_hole_margin_mm', 10.0))
            
            idx = self.hole_shape_combo.findText(getattr(planform, 'lightening_hole_shape', 'circular'))
            if idx >= 0:
                self.hole_shape_combo.setCurrentIndex(idx)
                
            self.fos_spin.setValue(planform.factor_of_safety)
            self.max_deflection_spin.setValue(planform.max_tip_deflection_percent)
            self.max_twist_spin.setValue(getattr(planform, 'max_tip_twist_deg', 3.0))
            
            # Update advanced buckling options
            self.stringer_count_spin.setValue(getattr(planform, 'stringer_count', 0))
            self.stringer_height_spin.setValue(getattr(planform, 'stringer_height_mm', 10.0))
            self.stringer_thickness_spin.setValue(getattr(planform, 'stringer_thickness_mm', 1.5))
            idx = self.stringer_section_combo.findText(
                getattr(planform, 'stringer_section_type', 'rectangular')
            )
            if idx >= 0:
                self.stringer_section_combo.setCurrentIndex(idx)
            self.spar_cap_width_spin.setValue(getattr(planform, 'spar_cap_width_mm', 10.0))
            self.fastener_fraction_spin.setValue(
                getattr(planform, 'fastener_adhesive_fraction', 0.10)
            )
            
            # Map internal key to display text for boundary condition
            bc_key = getattr(planform, 'skin_boundary_condition', 'semi_restrained')
            bc_display_map = {
                "simply_supported": "Simply Supported",
                "semi_restrained": "Semi-Restrained",
                "clamped": "Clamped",
            }
            bc_display = bc_display_map.get(bc_key, "Semi-Restrained")
            idx = self.boundary_condition_combo.findText(bc_display)
            if idx >= 0:
                self.boundary_condition_combo.setCurrentIndex(idx)
            
            self.include_curvature_chk.setChecked(
                getattr(planform, 'include_curvature_effect', True)
            )
            
            # Post-buckling toggle
            self.post_buckling_chk.setChecked(
                getattr(planform, 'post_buckling_enabled', False)
            )

            saved_condition = (
                self.project.analysis.structural_analysis.get("flight_condition", {}).get("condition_name")
                if self.project.analysis.structural_analysis
                else None
            )
            if saved_condition:
                combo_index = self.flight_condition_combo.findData(
                    str(saved_condition).strip().lower()
                )
                if combo_index >= 0:
                    self.flight_condition_combo.setCurrentIndex(combo_index)

            saved_load_factor = (
                self.project.analysis.structural_analysis.get("flight_condition", {}).get("load_factor")
                if self.project.analysis.structural_analysis
                else None
            )
            if saved_load_factor is not None:
                self.load_factor_spin.setValue(float(saved_load_factor))

            settings = getattr(self.project.analysis, "gui_settings", {}).get("structure_tab", {})
            if settings:
                self._apply_gui_settings(settings)
            
        finally:
            # Re-enable signals
            self.spar_material_combo.blockSignals(False)
            self.skin_material_combo.blockSignals(False)
            self.rib_material_combo.blockSignals(False)
            self.stringer_material_combo.blockSignals(False)
            self.spar_thickness_spin.blockSignals(False)
            self.skin_thickness_spin.blockSignals(False)
            self.rib_thickness_spin.blockSignals(False)
            self.rib_lightening_spin.blockSignals(False)
            self.lightening_margin_spin.blockSignals(False)
            self.hole_shape_combo.blockSignals(False)
            self.fos_spin.blockSignals(False)
            self.max_deflection_spin.blockSignals(False)
            self.max_twist_spin.blockSignals(False)
            # Stringer geometry (now in Geometry group)
            self.stringer_count_spin.blockSignals(False)
            self.stringer_height_spin.blockSignals(False)
            self.stringer_thickness_spin.blockSignals(False)
            # Advanced options
            self.boundary_condition_combo.blockSignals(False)
            self.include_curvature_chk.blockSignals(False)
            self.post_buckling_chk.blockSignals(False)
            self.stringer_section_combo.blockSignals(False)
            self.spar_cap_width_spin.blockSignals(False)
            self.fastener_fraction_spin.blockSignals(False)

        if self.project.analysis.structural_analysis:
            self.analysis_result = self.project.analysis.structural_analysis
            if 'structure' in self.analysis_result:
                self._update_summary(self.analysis_result)
                self._update_plots(self.analysis_result)
                self.export_report_btn.setEnabled(True)
    
    def _run_optimization(self):
        """Run thickness optimization to minimize mass."""
        try:
            from services.structure import optimize_wingbox_thickness
            
            # Create service
            service = AeroSandboxService(self.project)
            
            # Get current settings
            planform = self.project.wing.planform
            
            # Show progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Optimizing thickness...", None, 0, 0, self)
            progress.setWindowTitle("Optimization")
            progress.setMinimumDuration(0)
            progress.show()
            
            # Get sections and lift distribution (similar to run_aerostructural_analysis)
            from services.structure import WingBoxSection, StringerProperties, RibProperties
            from core.models.materials import get_material_by_name
            
            sections = service.spanwise_sections()
            half_span = planform.half_span()
            
            # Build WingBoxSection list
            wing_box_sections = []
            for sec in sections:
                y_pos = sec.y_m
                chord = sec.chord_m
                t_over_c = 0.12
                if hasattr(sec, 'airfoil') and sec.airfoil is not None:
                    try:
                        t_over_c = sec.airfoil.max_thickness()
                    except Exception:
                        pass
                
                eta = abs(y_pos) / half_span if half_span > 0 else 0
                front_spar = (
                    planform.front_spar_root_percent / 100 * (1 - eta) +
                    planform.front_spar_tip_percent / 100 * eta
                )
                rear_spar = planform.rear_spar_root_percent / 100
                
                wing_box_sections.append(WingBoxSection(
                    y=abs(y_pos),
                    chord=chord,
                    thickness_ratio=t_over_c,
                    front_spar_xsi=front_spar,
                    rear_spar_xsi=rear_spar,
                ))
            
            wing_box_sections.sort(key=lambda s: s.y)
            
            # Get lift distribution
            flight_condition = service.resolve_structural_flight_condition(
                self._build_structural_flight_condition_request()
            )
            load_factor = float(flight_condition.get("load_factor", self.load_factor_spin.value()))
            y_aero, lift_per_span = service.get_spanwise_lift_distribution(
                velocity=flight_condition.get("velocity_mps"),
                alpha=flight_condition.get("alpha_deg"),
                altitude_m=flight_condition.get("altitude_m"),
                load_factor=load_factor,
                n_spanwise_points=100,
                analysis_method=getattr(self.project.wing.twist_trim, "structural_spanload_model", "vlm"),
            )
            
            import numpy as _np_std
            from scipy.interpolate import interp1d
            lift_interp = interp1d(
                y_aero, lift_per_span, kind='linear',
                bounds_error=False, fill_value=(float(lift_per_span[0]), 0.0)
            )
            
            def lift_distribution(y):
                return lift_interp(_np_std.atleast_1d(y))
            
            # Get moment distribution for torsion (if available)
            moment_distribution = None
            try:
                y_moment, moment_per_span = service.get_spanwise_moment_distribution(
                    velocity=flight_condition.get("velocity_mps"),
                    alpha=flight_condition.get("alpha_deg"),
                    load_factor=load_factor,
                    n_spanwise_points=100,
                    altitude_m=flight_condition.get("altitude_m"),
                )
                moment_interp = interp1d(
                    y_moment, moment_per_span, kind='linear',
                    bounds_error=False, fill_value=(float(moment_per_span[0]), 0.0)
                )
                
                def moment_distribution(y):
                    return moment_interp(_np_std.atleast_1d(y))
                
                print("[Optimizer] Moment distribution computed for torsion constraints")
            except Exception as e:
                print(f"[Optimizer] Could not compute moment distribution: {e}")
                moment_distribution = None
            
            # Get materials
            spar_material = get_material_by_name(planform.spar_material_name)
            skin_material = get_material_by_name(planform.skin_material_name)
            
            # Rib positions
            rib_positions = sorted([abs(sec.y_m) for sec in sections])
            
            # Get stringer parameters from UI
            current_stringer_count = self.stringer_count_spin.value()
            stringer_height_m = self.stringer_height_spin.value() / 1000  # mm to m
            stringer_thickness_m = self.stringer_thickness_spin.value() / 1000  # mm to m
            
            # Get rib properties
            rib_material = get_material_by_name(getattr(planform, 'rib_material_name', planform.skin_material_name))
            rib_thickness_mm = self.rib_thickness_spin.value()
            
            from services.structure import RibProperties
            rib_props = RibProperties(
                thickness_m=rib_thickness_mm / 1000,
                material=rib_material,
                lightening_hole_fraction=0.4,  # Default
                spar_cap_width_m=0.010,  # Default
            )
            
            # Run optimization with stringer count search
            result = optimize_wingbox_thickness(
                sections=wing_box_sections,
                spar_material=spar_material,
                skin_material=skin_material,
                lift_distribution=lift_distribution,
                moment_distribution=moment_distribution,
                rib_positions=rib_positions,
                spar_thickness_init=planform.spar_thickness_mm,
                skin_thickness_init=planform.skin_thickness_mm,
                rib_thickness_init=rib_thickness_mm,
                rib_thickness_bounds=(1.0, 15.0),  # 1-15mm range
                rib_props=rib_props,
                stringer_count_range=(0, max(8, current_stringer_count + 4)),  # Search range
                stringer_height_m=stringer_height_m,
                stringer_thickness_m=stringer_thickness_m,
                factor_of_safety=planform.factor_of_safety,
                max_deflection_fraction=planform.max_tip_deflection_percent / 100,
                max_twist_deg=getattr(planform, 'max_tip_twist_deg', 3.0),
                verbose=True,
            )
            
            progress.close()
            
            if result.success:
                # Show results dialog
                msg = (
                    f"Optimization Successful!\n\n"
                    f"Optimized Design:\n"
                    f"  Spar Thickness: {result.spar_thickness_mm:.2f} mm\n"
                    f"  Skin Thickness: {result.skin_thickness_mm:.2f} mm\n"
                    f"  Rib Thickness: {result.rib_thickness_mm:.2f} mm\n"
                    f"  Stringer Count: {result.stringer_count} per panel\n\n"
                    f"Mass Reduction:\n"
                    f"  Before: {result.initial_mass_kg*1000:.1f} g\n"
                    f"  After:  {result.optimized_mass_kg*1000:.1f} g\n"
                    f"  Saved:  {result.mass_reduction_percent:.1f}%\n\n"
                    f"Safety Margins at Optimum:\n"
                    f"  Stress: {result.stress_margin:.2f}\n"
                    f"  Skin Buckling: {result.skin_buckling_margin:.2f}\n"
                    f"  Spar Buckling: {result.spar_buckling_margin:.2f}\n"
                    f"  Rib Crushing: {result.rib_crushing_margin:.2f}\n"
                    f"  Tip Deflection: {result.tip_deflection_percent:.1f}%\n"
                    f"  Tip Twist: {result.tip_twist_deg:.2f}° (margin: {result.twist_margin:.2f})\n\n"
                    f"Apply optimized design?"
                )
                
                reply = QMessageBox.question(
                    self, "Optimization Result", msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Update UI with optimized values
                    self.spar_thickness_spin.setValue(result.spar_thickness_mm)
                    self.skin_thickness_spin.setValue(result.skin_thickness_mm)
                    self.rib_thickness_spin.setValue(result.rib_thickness_mm)
                    self.stringer_count_spin.setValue(result.stringer_count)
                    
                    # Update planform
                    planform.spar_thickness_mm = result.spar_thickness_mm
                    planform.skin_thickness_mm = result.skin_thickness_mm
                    planform.rib_thickness_mm = result.rib_thickness_mm
                    if hasattr(planform, 'stringer_count'):
                        planform.stringer_count = result.stringer_count
                    
                    # Re-run analysis with new values
                    self._run_analysis()
            else:
                QMessageBox.warning(
                    self, "Optimization Failed",
                    f"Optimization did not converge:\n{result.message}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_pdf_report(self):
        """Export structural analysis report to PDF."""
        if self.analysis_result is None:
            QMessageBox.warning(self, "No Data", "Run analysis first before exporting report.")
            return
        
        try:
            from PyQt6.QtWidgets import QFileDialog
            from matplotlib.backends.backend_pdf import PdfPages
            from datetime import datetime
            
            # Get save path
            default_name = f"structural_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save PDF Report", default_name, "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Create PDF with multiple pages
            with PdfPages(file_path) as pdf:
                # Page 1: Summary and key metrics with Go/No-Go indicator
                fig1 = self._create_summary_page()
                pdf.savefig(fig1, bbox_inches='tight')
                fig1.clear()
                
                # Page 2: NEW - Margin Summary Bar Chart
                fig1b = self._create_margin_summary_page()
                pdf.savefig(fig1b, bbox_inches='tight')
                fig1b.clear()
                
                # Page 3: Structural response plots
                fig2 = self._create_response_page()
                pdf.savefig(fig2, bbox_inches='tight')
                fig2.clear()
                
                # Page 4: Buckling analysis
                fig3 = self._create_buckling_page()
                pdf.savefig(fig3, bbox_inches='tight')
                fig3.clear()
                
                # Page 5: Shear and Torsion analysis
                fig4 = self._create_shear_torsion_page()
                pdf.savefig(fig4, bbox_inches='tight')
                fig4.clear()
                
                # Page 6: Mass breakdown
                fig5 = self._create_mass_page()
                pdf.savefig(fig5, bbox_inches='tight')
                fig5.clear()
            
            QMessageBox.information(
                self, "Export Complete",
                f"Report saved to:\n{file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export PDF: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_summary_page(self):
        """Create summary page for PDF report with Go/No-Go indicator."""
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig = Figure(figsize=(8.5, 11))
        
        # Title
        fig.suptitle("Structural Analysis Report", fontsize=16, fontweight='bold', y=0.98)
        
        struct = self.analysis_result.get("structure", {})
        planform = self.project.wing.planform
        is_feasible = self.analysis_result.get('feasible', False)
        
        # Create two subplots: Go/No-Go banner (top) and summary text (bottom)
        ax_banner = fig.add_axes([0.1, 0.88, 0.8, 0.07])  # [left, bottom, width, height]
        ax_text = fig.add_axes([0.05, 0.05, 0.9, 0.80])
        
        # --- Go/No-Go Banner ---
        ax_banner.axis('off')
        if is_feasible:
            banner_color = '#44BB44'  # Green
            banner_text = "✓ DESIGN FEASIBLE"
        else:
            banner_color = '#FF4444'  # Red
            banner_text = "✗ DESIGN INFEASIBLE"
        
        rect = mpatches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=banner_color, edgecolor='black', linewidth=2,
            transform=ax_banner.transAxes
        )
        ax_banner.add_patch(rect)
        ax_banner.text(0.5, 0.5, banner_text, transform=ax_banner.transAxes,
                       ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        
        # --- Summary Text ---
        ax_text.axis('off')
        
        # Build summary text
        summary = [
            f"Project: {self.project.wing.name}",
            f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "=" * 50,
            "CONFIGURATION",
            "=" * 50,
            f"Wing Area: {planform.wing_area_m2:.2f} m²",
            f"Aspect Ratio: {planform.aspect_ratio:.1f}",
            f"Half-Span: {planform.half_span():.3f} m",
            "",
            f"Spar Material: {planform.spar_material_name}",
            f"Skin Material: {planform.skin_material_name}",
            f"Spar Thickness: {planform.spar_thickness_mm:.1f} mm",
            f"Skin Thickness: {planform.skin_thickness_mm:.1f} mm",
            "",
            "=" * 50,
            "RESULTS",
            "=" * 50,
            f"Structural Mass: {struct.get('mass_kg', 0)*1000:.1f} g",
            f"Tip Deflection: {struct.get('tip_deflection_m', 0)*1000:.1f} mm",
            f"Max Stress: {struct.get('max_stress_MPa', 0):.1f} MPa",
            "",
            f"Stress Margin: {struct.get('stress_margin', 0):.2f}",
            f"Skin Buckling Margin: {struct.get('min_buckling_margin', 0):.2f}",
            f"Spar Buckling Margin: {struct.get('min_spar_buckling_margin', 0):.2f}",
        ]
        
        # Add rib failure margins if available
        min_rib_buckling = struct.get('min_rib_buckling_margin')
        min_rib_crushing = struct.get('min_rib_crushing_margin')
        if min_rib_buckling is not None:
            summary.append(f"Rib Shear Buckling Margin: {min_rib_buckling:.2f}")
        if min_rib_crushing is not None:
            summary.append(f"Rib Crushing Margin: {min_rib_crushing:.2f}")
        
        # Add stringer crippling if available
        min_stringer = struct.get('min_stringer_crippling_margin')
        if min_stringer is not None and min_stringer != float('inf'):
            summary.append(f"Stringer Crippling Margin: {min_stringer:.2f}")
        
        # Add torsion info if available
        torque = struct.get('torque', [])
        min_torsion_margin = struct.get('min_torsion_margin')
        if torque and len(torque) > 0 and any(t != 0 for t in torque):
            max_torque = max(abs(t) for t in torque)
            summary.append("")
            summary.append("--- Torsion Analysis ---")
            summary.append(f"Max Torque: {max_torque:.2f} N·m")
            if min_torsion_margin is not None:
                summary.append(f"Combined Shear Margin: {min_torsion_margin:.2f}")
        
        ax_text.text(0.05, 0.98, '\n'.join(summary), transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        return fig
    
    def _create_margin_summary_page(self):
        """Create margin summary bar chart page for PDF report."""
        from matplotlib.figure import Figure
        import numpy as np
        
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle("Safety Margin Summary", fontsize=16, fontweight='bold', y=0.96)
        
        struct = self.analysis_result.get("structure", {})
        fos = struct.get('factor_of_safety', 1.5)
        
        # Collect all margins
        margins = {}
        
        stress_margin = struct.get('stress_margin')
        if stress_margin is not None and stress_margin < 100:
            margins['Stress'] = stress_margin
        
        min_buckling = struct.get('min_buckling_margin')
        if min_buckling is not None and min_buckling < 100:
            margins['Skin Buckling'] = min_buckling
        
        min_spar = struct.get('min_spar_buckling_margin')
        if min_spar is not None and min_spar < 100:
            margins['Spar Shear Buckling'] = min_spar
        
        min_rib_buckling = struct.get('min_rib_buckling_margin')
        if min_rib_buckling is not None and min_rib_buckling < 100:
            margins['Rib Buckling'] = min_rib_buckling
        
        min_rib_crushing = struct.get('min_rib_crushing_margin')
        if min_rib_crushing is not None and min_rib_crushing < 100:
            margins['Rib Crushing'] = min_rib_crushing
        
        min_stringer = struct.get('min_stringer_crippling_margin')
        if min_stringer is not None and min_stringer < 100:
            margins['Stringer Crippling'] = min_stringer
        
        min_skin_shear = struct.get('min_skin_shear_buckling_margin')
        if min_skin_shear is not None and min_skin_shear < 100:
            margins['Skin Shear Buckling'] = min_skin_shear
        
        min_combined = struct.get('min_combined_buckling_margin')
        if min_combined is not None and min_combined < 100:
            margins['Combined σ+τ'] = min_combined
        
        min_torsion = struct.get('min_torsion_margin')
        if min_torsion is not None and min_torsion < 100:
            margins['Combined Shear'] = min_torsion
        
        # Twist margin (note: this is max_allowable/actual, so >1 is OK)
        twist_margin = struct.get('twist_margin')
        if twist_margin is not None and twist_margin < 100:
            margins['Tip Twist'] = twist_margin
        
        if not margins:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No margin data available", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create bar chart
        ax = fig.add_subplot(111)
        
        names = list(margins.keys())
        values = list(margins.values())
        
        # Cap values for visualization
        values_capped = [min(v, 10.0) for v in values]
        
        # Color code: red < 1.0, yellow 1.0-FOS, green > FOS
        colors = []
        for v in values:
            if v < 1.0:
                colors.append('#FF4444')  # Red - FAIL
            elif v < fos:
                colors.append('#FFAA00')  # Yellow - Below FOS
            else:
                colors.append('#44BB44')  # Green - OK
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values_capped, color=colors, edgecolor='black', height=0.6)
        
        # Add reference lines
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Minimum (1.0)')
        ax.axvline(x=fos, color='orange', linestyle=':', linewidth=2, label=f'FOS ({fos:.1f})')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            # Show actual value even if capped
            label = f'{val:.2f}'
            if val > 10.0:
                label = f'{val:.1f}'
            x_pos = min(val, 10.0) + 0.1
            if x_pos > 9.5:
                x_pos = min(val, 10.0) - 0.5
                ax.text(x_pos, i, label, va='center', ha='right', fontsize=10, color='white', fontweight='bold')
            else:
                ax.text(x_pos, i, label, va='center', ha='left', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Margin of Safety', fontsize=12)
        ax.set_xlim(0, max(values_capped) * 1.2)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add interpretation note
        note_text = (
            "Color coding:\n"
            "  RED = Margin < 1.0 (FAILS)\n"
            f"  YELLOW = 1.0 ≤ Margin < {fos:.1f} (Below target FOS)\n"
            f"  GREEN = Margin ≥ {fos:.1f} (Meets FOS requirement)"
        )
        fig.text(0.1, 0.02, note_text, fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        return fig
    
    def _create_response_page(self):
        """Create structural response plots page."""
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle("Structural Response", fontsize=14, fontweight='bold')
        
        struct = self.analysis_result.get("structure", {})
        y = struct.get("y", [])
        
        if not y:
            return fig
        
        y_mm = [yi * 1000 for yi in y]
        
        # Plot 1: Displacement
        ax1 = fig.add_subplot(2, 2, 1)
        displacement = struct.get("displacement", [])
        if displacement:
            ax1.plot(y_mm, [d * 1000 for d in displacement], 'b-', linewidth=1.5)
            ax1.set_xlabel("Spanwise Position [mm]")
            ax1.set_ylabel("Displacement [mm]")
            ax1.set_title("Vertical Deflection")
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bending Moment
        ax2 = fig.add_subplot(2, 2, 2)
        moment = struct.get("bending_moment", [])
        if moment:
            ax2.plot(y_mm, moment, 'r-', linewidth=1.5)
            ax2.set_xlabel("Spanwise Position [mm]")
            ax2.set_ylabel("Bending Moment [N·m]")
            ax2.set_title("Bending Moment")
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Shear Force
        ax3 = fig.add_subplot(2, 2, 3)
        shear = struct.get("shear_force", [])
        if shear:
            ax3.plot(y_mm, shear, 'g-', linewidth=1.5)
            ax3.set_xlabel("Spanwise Position [mm]")
            ax3.set_ylabel("Shear Force [N]")
            ax3.set_title("Shear Force")
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Load Distribution
        ax4 = fig.add_subplot(2, 2, 4)
        aero_load = struct.get("aero_load", [])
        inertial_load = struct.get("inertial_load", [])
        net_load = struct.get("net_load", [])
        if aero_load:
            ax4.plot(y_mm, aero_load, 'b-', label='Aero')
            ax4.plot(y_mm, inertial_load, 'r-', label='Inertial')
            ax4.plot(y_mm, net_load, 'k-', linewidth=2, label='Net')
            ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax4.set_xlabel("Spanwise Position [mm]")
            ax4.set_ylabel("Load [N/m]")
            ax4.set_title("Load Distribution")
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def _create_buckling_page(self):
        """Create buckling analysis page."""
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle("Buckling Analysis", fontsize=14, fontweight='bold')
        
        struct = self.analysis_result.get("structure", {})
        y = struct.get("y", [])
        fos = struct.get('factor_of_safety', 1.5)
        
        if not y:
            return fig
        
        y_mm = [yi * 1000 for yi in y]
        
        # Plot 1: Skin/Spar Buckling Margins (top)
        ax1 = fig.add_subplot(3, 1, 1)
        skin_margin = struct.get("buckling_margin", [])
        spar_margin = struct.get("spar_buckling_margin", [])
        
        if skin_margin:
            skin_margin_capped = [min(m, 20.0) for m in skin_margin]
            ax1.plot(y_mm, skin_margin_capped, 'b-', linewidth=1.5, label='Skin buckling')
        if spar_margin:
            spar_margin_capped = [min(m, 20.0) for m in spar_margin]
            ax1.plot(y_mm, spar_margin_capped, 'g-', linewidth=1.5, label='Spar web shear')
        
        ax1.axhline(y=fos, color='red', linestyle='--', linewidth=1.5, label=f'FOS = {fos:.1f}')
        ax1.fill_between(y_mm, 0, fos, alpha=0.1, color='red')
        ax1.set_xlabel("Spanwise Position [mm]")
        ax1.set_ylabel("Buckling Margin")
        ax1.set_title("Skin & Spar Buckling Safety Margins")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Rib Failure Margins (middle)
        ax2 = fig.add_subplot(3, 1, 2)
        rib_buckling = struct.get("rib_shear_buckling_margin", [])
        rib_crushing = struct.get("rib_crushing_margin", [])
        min_rib_buckling = struct.get("min_rib_buckling_margin")
        min_rib_crushing = struct.get("min_rib_crushing_margin")
        
        has_rib_data = (rib_buckling and len(rib_buckling) > 0) or (rib_crushing and len(rib_crushing) > 0)
        
        if has_rib_data:
            # Get rib positions from the analysis - approximate from section positions
            # Rib margins are at discrete rib locations, not continuous
            n_ribs = max(len(rib_buckling) if rib_buckling else 0, len(rib_crushing) if rib_crushing else 0)
            if n_ribs > 0 and y_mm:
                # Approximate rib positions as evenly spaced along span
                rib_y_mm = [y_mm[0] + i * (y_mm[-1] - y_mm[0]) / max(1, n_ribs - 1) for i in range(n_ribs)]
                
                if rib_buckling and len(rib_buckling) == n_ribs:
                    rib_buckling_capped = [min(m, 50.0) for m in rib_buckling]
                    ax2.bar([y - 5 for y in rib_y_mm], rib_buckling_capped, width=10, 
                            color='purple', alpha=0.7, label='Rib shear buckling')
                
                if rib_crushing and len(rib_crushing) == n_ribs:
                    rib_crushing_capped = [min(m, 50.0) for m in rib_crushing]
                    ax2.bar([y + 5 for y in rib_y_mm], rib_crushing_capped, width=10,
                            color='orange', alpha=0.7, label='Rib crushing/bearing')
            
            ax2.axhline(y=fos, color='red', linestyle='--', linewidth=1.5, label=f'FOS = {fos:.1f}')
            ax2.fill_between([0, y_mm[-1]], 0, fos, alpha=0.1, color='red')
            ax2.set_xlabel("Spanwise Position [mm]")
            ax2.set_ylabel("Margin")
            ax2.set_title("Rib Failure Safety Margins")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(bottom=0)
            
            # Add annotation with minimum values
            annotation_text = []
            if min_rib_buckling is not None:
                status = "OK" if min_rib_buckling >= fos else "FAIL"
                annotation_text.append(f"Min Rib Buckling: {min_rib_buckling:.2f} [{status}]")
            if min_rib_crushing is not None:
                status = "OK" if min_rib_crushing >= fos else "FAIL"
                annotation_text.append(f"Min Rib Crushing: {min_rib_crushing:.2f} [{status}]")
            if annotation_text:
                ax2.text(0.02, 0.98, '\n'.join(annotation_text), transform=ax2.transAxes,
                         fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, "No rib failure data available\n(Rib properties may not be specified)",
                     ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title("Rib Failure Safety Margins")
        
        # Plot 3: Stresses (bottom)
        ax3 = fig.add_subplot(3, 1, 3)
        sigma_spar = struct.get("sigma_spar", [])
        sigma_skin = struct.get("sigma_skin", [])
        
        if sigma_spar:
            sigma_spar_mpa = [s / 1e6 for s in sigma_spar]
            ax3.plot(y_mm, sigma_spar_mpa, 'b-', linewidth=1.5, label='Spar stress')
        if sigma_skin:
            sigma_skin_mpa = [s / 1e6 for s in sigma_skin]
            ax3.plot(y_mm, sigma_skin_mpa, 'g-', linewidth=1.5, label='Skin stress')
        
        ax3.set_xlabel("Spanwise Position [mm]")
        ax3.set_ylabel("Stress [MPa]")
        ax3.set_title("Bending Stress Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def _create_shear_torsion_page(self):
        """Create shear and torsion analysis page."""
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle("Shear & Torsion Analysis", fontsize=14, fontweight='bold')
        
        struct = self.analysis_result.get("structure", {})
        y = struct.get("y", [])
        fos = struct.get('factor_of_safety', 1.5)
        
        if not y:
            return fig
        
        y_mm = [yi * 1000 for yi in y]
        
        # Plot 1: Torque Distribution
        ax1 = fig.add_subplot(3, 1, 1)
        torque = struct.get("torque", [])
        
        if torque and len(torque) == len(y):
            torque_nm = [t for t in torque]  # Already in N*m
            ax1.plot(y_mm, torque_nm, 'b-', linewidth=1.5, label='Torque')
            ax1.fill_between(y_mm, 0, torque_nm, alpha=0.2, color='blue')
            ax1.set_ylabel("Torque [N·m]")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No torque data available\n(moment_distribution not provided)",
                     ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        
        ax1.set_xlabel("Spanwise Position [mm]")
        ax1.set_title("Torque Distribution (from Pitching Moment)")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Shear Stress Breakdown
        ax2 = fig.add_subplot(3, 1, 2)
        tau_spar = struct.get("tau_spar", [])  # Bending shear
        tau_torsion_spar = struct.get("tau_torsion_spar", [])
        tau_total_spar = struct.get("tau_total_spar", [])
        
        has_shear_data = tau_spar and len(tau_spar) == len(y)
        
        if has_shear_data:
            tau_spar_mpa = [t / 1e6 for t in tau_spar]
            ax2.plot(y_mm, tau_spar_mpa, 'g-', linewidth=1.5, label='Bending shear τ_bend')
            
            if tau_torsion_spar and len(tau_torsion_spar) == len(y):
                tau_torsion_mpa = [t / 1e6 for t in tau_torsion_spar]
                ax2.plot(y_mm, tau_torsion_mpa, 'b-', linewidth=1.5, label='Torsion shear τ_tors')
            
            if tau_total_spar and len(tau_total_spar) == len(y):
                tau_total_mpa = [t / 1e6 for t in tau_total_spar]
                ax2.plot(y_mm, tau_total_mpa, 'r-', linewidth=2.0, label='Combined τ_total')
            
            ax2.set_ylabel("Shear Stress [MPa]")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No shear stress data available",
                     ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        
        ax2.set_xlabel("Spanwise Position [mm]")
        ax2.set_title("Spar Web Shear Stress Breakdown")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Torsional Shear in Skin vs Spar
        ax3 = fig.add_subplot(3, 1, 3)
        tau_torsion_skin = struct.get("tau_torsion_skin", [])
        
        has_torsion_data = (tau_torsion_skin and len(tau_torsion_skin) == len(y)) or \
                           (tau_torsion_spar and len(tau_torsion_spar) == len(y))
        
        if has_torsion_data:
            if tau_torsion_skin and len(tau_torsion_skin) == len(y):
                tau_skin_mpa = [t / 1e6 for t in tau_torsion_skin]
                ax3.plot(y_mm, tau_skin_mpa, 'b-', linewidth=1.5, label='Skin τ_torsion')
            
            if tau_torsion_spar and len(tau_torsion_spar) == len(y):
                tau_spar_tors_mpa = [t / 1e6 for t in tau_torsion_spar]
                ax3.plot(y_mm, tau_spar_tors_mpa, 'g-', linewidth=1.5, label='Spar τ_torsion')
            
            ax3.set_ylabel("Torsional Shear Stress [MPa]")
            ax3.legend()
            
            # Add note about shear flow distribution
            max_tau = max(max(tau_skin_mpa) if tau_torsion_skin else 0,
                          max(tau_spar_tors_mpa) if tau_torsion_spar else 0)
            if max_tau > 0:
                ax3.text(0.02, 0.98, f"Max torsion shear: {max_tau:.2f} MPa",
                         transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, "No torsion data available\n(moment_distribution not provided)",
                     ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        
        ax3.set_xlabel("Spanwise Position [mm]")
        ax3.set_title("Torsional Shear Stress Distribution (Bredt's Formula)")
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def _create_mass_page(self):
        """Create mass breakdown page."""
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle("Mass Breakdown", fontsize=14, fontweight='bold')
        
        struct = self.analysis_result.get("structure", {})
        mass_breakdown = struct.get("mass_breakdown", {})
        
        if not mass_breakdown:
            return fig
        
        # Pie chart of mass components
        ax1 = fig.add_subplot(1, 1, 1)
        
        labels = []
        sizes = []
        component_labels = {
            'wingbox_skins': 'Wingbox Skins',
            'spar_webs': 'Spar Webs',
            'leading_edge': 'Leading Edge',
            'trailing_edge': 'Trailing Edge',
            'ribs': 'Ribs',
            'stringers': 'Stringers',
            'control_surfaces': 'Control Surfaces',
            'fasteners_adhesive': 'Fasteners/Adhesive',
        }
        
        for key, label in component_labels.items():
            if key in mass_breakdown and mass_breakdown[key] > 0.001:
                labels.append(f"{label}\n({mass_breakdown[key]*1000:.1f} g)")
                sizes.append(mass_breakdown[key])
        
        if sizes:
            colors = plt.cm.Set3.colors[:len(sizes)]
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90
            )
            ax1.set_title(f"Total Mass: {mass_breakdown.get('total', 0)*1000:.1f} g")
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
