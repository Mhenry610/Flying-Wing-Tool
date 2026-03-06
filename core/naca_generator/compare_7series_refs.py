import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.naca_generator.naca456 import generate_naca_airfoil

def load_dat_file(filepath):
    """Load NACA coordinates from a .dat file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Skip header
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                data.append([x, y])
            except ValueError:
                continue
                
    return np.array(data)

def compare_single_airfoil(designation, ref_filename):
    print(f"\n--- Comparing {designation} ---")
    
    # Load reference
    ref_file = os.path.join(os.path.dirname(__file__), ref_filename)
    if not os.path.exists(ref_file):
        print(f"Error: Reference file {ref_file} not found.")
        return None, None, None, None
        
    ref_data = load_dat_file(ref_file)
    ref_x = ref_data[:, 0]
    ref_y = ref_data[:, 1]
    
    # Generate airfoil
    gen_x, gen_y = generate_naca_airfoil(designation, n_points=200)
    
    # Find LE index for reference
    ref_le_idx = np.argmin(ref_x)
    
    # Find LE index for generated
    gen_le_idx = np.argmin(gen_x)
    
    # Upper surface (TE to LE)
    ref_x_upper = ref_x[:ref_le_idx+1]
    ref_y_upper = ref_y[:ref_le_idx+1]
    
    gen_x_upper = gen_x[:gen_le_idx+1]
    gen_y_upper = gen_y[:gen_le_idx+1]
    
    # Lower surface (LE to TE)
    ref_x_lower = ref_x[ref_le_idx:]
    ref_y_lower = ref_y[ref_le_idx:]
    
    gen_x_lower = gen_x[gen_le_idx:]
    gen_y_lower = gen_y[gen_le_idx:]
    
    # Interpolate
    # Upper: x is decreasing, so flip for interpolation
    gen_y_interp_upper = np.interp(ref_x_upper[::-1], gen_x_upper[::-1], gen_y_upper[::-1])
    gen_y_interp_upper = gen_y_interp_upper[::-1] # Flip back
    
    # Lower: x is increasing
    gen_y_interp_lower = np.interp(ref_x_lower, gen_x_lower, gen_y_lower)
    
    # Calculate errors
    err_upper = gen_y_interp_upper - ref_y_upper
    err_lower = gen_y_interp_lower - ref_y_lower
    
    rmse_upper = np.sqrt(np.mean(err_upper**2))
    rmse_lower = np.sqrt(np.mean(err_lower**2))
    max_err_upper = np.max(np.abs(err_upper))
    max_err_lower = np.max(np.abs(err_lower))
    
    print(f"Upper Surface RMSE: {rmse_upper:.6f}")
    print(f"Upper Surface Max Error: {max_err_upper:.6f}")
    print(f"Lower Surface RMSE: {rmse_lower:.6f}")
    print(f"Lower Surface Max Error: {max_err_lower:.6f}")
    
    return ref_x, ref_y, gen_x, gen_y

def compare_all():
    comparisons = [
        ("747A315", "NACA_747A315.dat"),
        ("747A415", "NACA_747A415.dat")
    ]
    
    for des, ref in comparisons:
        ref_x, ref_y, gen_x, gen_y = compare_single_airfoil(des, ref)
        
        if ref_x is None:
            continue
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(ref_x, ref_y, 'ko', label='Reference (File)', markersize=3, fillstyle='none')
        plt.plot(gen_x, gen_y, 'r-', label='Generated (Code)', linewidth=1)
        plt.title(f"Comparison: {des}")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        # Show plot
        plt.show()

if __name__ == "__main__":
    compare_all()
