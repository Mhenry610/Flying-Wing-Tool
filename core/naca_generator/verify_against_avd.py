import numpy as np
import sys
import os

# Add the Unified Directory to path
sys.path.append(r'd:\AI code\flying wing tool\Full Set\Unified Directory')

from core.naca_generator.naca456 import generate_naca_airfoil

def verify_naca_63_012():
    print("Verifying NACA 63-012 against avd.tex...")
    
    # Data from avd.tex (NACA Profile 63-012)
    # x in percent, y in percent
    avd_data = np.array([
        [0.0000, 0.0000],
        [0.5000, 0.9837],
        [0.7500, 1.1938],
        [1.2500, 1.5181],
        [2.5000, 2.0980],
        [5.0000, 2.9256],
        [7.5000, 3.5432],
        [10.0000, 4.0386],
        [15.0000, 4.8003],
        [20.0000, 5.3431],
        [25.0000, 5.7115],
        [30.0000, 5.9304],
        [35.0000, 6.0005],
        [40.0000, 5.9220],
        [45.0000, 5.7051],
        [50.0000, 5.3698],
        [55.0000, 4.9354],
        [60.0000, 4.4205],
        [65.0000, 3.8398],
        [70.0000, 3.2113],
        [75.0000, 2.5567],
        [80.0000, 1.9002],
        [85.0000, 1.2724],
        [90.0000, 0.7076],
        [95.0000, 0.2521],
        [100.0000, 0.0000]
    ])
    
    x_ref = avd_data[:, 0] / 100.0
    y_ref = avd_data[:, 1] / 100.0
    
    # Generate airfoil
    # 63-012: Family 1 (63), cl=0, toc=0.12
    # Designation "63-012"
    x_gen, y_gen = generate_naca_airfoil("63-012", n_points=200)
    
    # Extract upper surface (y >= 0)
    # x_gen goes 1 -> 0 -> 1
    # Upper surface is the first half (1 -> 0)
    le_idx = np.argmin(x_gen)
    x_upper = x_gen[:le_idx+1]
    y_upper = y_gen[:le_idx+1]
    
    # Interpolate generated y to reference x
    # x_upper is decreasing, so flip for interp
    y_interp = np.interp(x_ref, x_upper[::-1], y_upper[::-1])
    
    # Compare
    diff = y_interp - y_ref
    max_diff = np.max(np.abs(diff))
    
    print(f"Max difference: {max_diff:.6f}")
    
    # Print table
    print(f"{'x':>8} {'y_ref':>8} {'y_gen':>8} {'diff':>8}")
    for i in range(len(x_ref)):
        print(f"{x_ref[i]:8.4f} {y_ref[i]:8.4f} {y_interp[i]:8.4f} {diff[i]:8.4f}")
        
    if max_diff < 1e-4:
        print("\nSUCCESS: Verification passed (tolerance 1e-4)")
    else:
        print("\nWARNING: Verification exceeded tolerance")

if __name__ == "__main__":
    verify_naca_63_012()
