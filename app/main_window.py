import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, 
    QMenuBar, QMenu, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QAction

# Ensure core/services imports work if running from here
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import Project
from services.optimization import OptimizationService
from app.tabs.geometry import GeometryTab
from app.tabs.analysis import AnalysisTab
from app.tabs.mission import MissionTab
from app.tabs.export import ExportTab
from app.tabs.structure import StructureTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flying Wing Unified Tool")
        self.resize(1200, 800)
        
        # State
        self.project = Project()
        self.opt_service = OptimizationService()
        
        # UI Components
        self._init_ui()
        
    def _init_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project...", self)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("Tools")
        
        opt_action = QAction("Run Twist Optimization", self)
        opt_action.triggered.connect(self.run_optimization)
        tools_menu.addAction(opt_action)
        
        # Central Widget (Tabs)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab_geometry = GeometryTab(self.project)
        self.tab_analysis = AnalysisTab(self.project)
        self.tab_mission = MissionTab(self.project)
        self.tab_structure = StructureTab(self.project)
        self.tab_export = ExportTab(self.project)
        
        self.tabs.addTab(self.tab_geometry, "Geometry")
        self.tabs.addTab(self.tab_analysis, "Analysis")
        self.tabs.addTab(self.tab_mission, "Mission")
        self.tabs.addTab(self.tab_structure, "Structure")
        self.tabs.addTab(self.tab_export, "Export")
        
    def refresh_tabs(self):
        """Notify tabs to update their UI from the project state."""
        self.tab_geometry.project = self.project
        self.tab_geometry.update_from_project()
        
        self.tab_analysis.project = self.project
        self.tab_analysis.update_from_project()
        
        self.tab_mission.project = self.project
        self.tab_mission.update_from_project()
        
        self.tab_structure.project = self.project
        self.tab_structure.update_from_project()
        
        self.tab_export.project = self.project
        if hasattr(self.tab_export, "update_from_project"):
            self.tab_export.update_from_project()

    def sync_tabs_to_project(self):
        """Push current widget values into project state before saving/exporting."""
        errors = []
        for tab in (
            self.tab_geometry,
            self.tab_analysis,
            self.tab_mission,
            self.tab_structure,
            self.tab_export,
        ):
            if hasattr(tab, "sync_to_project"):
                try:
                    tab.sync_to_project()
                except Exception as exc:
                    errors.append(f"{tab.__class__.__name__}: {exc}")
        if errors:
            raise RuntimeError("Failed to sync project state: " + "; ".join(errors))

    def new_project(self):
        self.project = Project()
        self.refresh_tabs()
        
    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if path:
            try:
                self.project = Project.load(path)
                self.refresh_tabs()
                QMessageBox.information(self, "Success", "Project loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project: {e}")
                
    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if path:
            try:
                self.sync_tabs_to_project()
                self.project.save(path)
                QMessageBox.information(self, "Success", "Project saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {e}")

    def run_optimization(self):
        try:
            self.project = self.opt_service.run_twist_optimization(self.project)
            self.refresh_tabs()
            QMessageBox.information(self, "Success", "Twist optimization complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Optimization failed: {e}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
