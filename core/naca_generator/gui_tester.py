import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSpinBox, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Ensure we can import the core package
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach 'Unified Directory'
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.naca_generator.naca456 import generate_naca_airfoil

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class NACAViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NACA Generator Tester")
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Designation input
        controls_layout.addWidget(QLabel("Designation:"))
        self.designation_input = QLineEdit("63-412")
        self.designation_input.setToolTip("e.g., 2412, 0012, 63-412, 65-210")
        controls_layout.addWidget(self.designation_input)
        
        # Points input
        controls_layout.addWidget(QLabel("Points:"))
        self.points_input = QSpinBox()
        self.points_input.setRange(10, 1000)
        self.points_input.setValue(100)
        self.points_input.setSingleStep(10)
        controls_layout.addWidget(self.points_input)
        
        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate_airfoil)
        controls_layout.addWidget(self.generate_btn)
        
        # Add stretch to keep controls to the left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Plot area
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Initial generation
        self.generate_airfoil()
        
    def generate_airfoil(self):
        designation = self.designation_input.text().strip()
        n_points = self.points_input.value()
        
        if not designation:
            return
            
        try:
            x, y = generate_naca_airfoil(designation, n_points=n_points)
            
            self.plot_airfoil(x, y, designation)
            self.statusBar().showMessage(f"Generated {designation} with {len(x)} points.")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))
            self.statusBar().showMessage(f"Error: {str(e)}")
            
    def plot_airfoil(self, x, y, title):
        self.canvas.axes.cla()
        
        # Plot the airfoil
        self.canvas.axes.plot(x, y, 'b-', label=title, linewidth=1.5)
        self.canvas.axes.plot(x, y, 'b.', markersize=3) # Show points
        
        # Plot mean line approximation (midpoints)
        # Since x goes TE->LE->TE, we need to split
        le_idx = np.argmin(x)
        
        # Setup plot
        self.canvas.axes.set_title(f"NACA {title}")
        self.canvas.axes.set_xlabel("x/c")
        self.canvas.axes.set_ylabel("y/c")
        self.canvas.axes.grid(True, linestyle=':', alpha=0.6)
        self.canvas.axes.axis('equal')
        
        # Add a margin to the view
        self.canvas.axes.margins(0.1)
        
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = NACAViewer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
