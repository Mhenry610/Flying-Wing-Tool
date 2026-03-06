import numpy as np
import sys
import os

# Add the Unified Directory to path
sys.path.append(r'd:\AI code\flying wing tool\Full Set\Unified Directory')

from core.naca_generator.naca456 import generate_naca_airfoil

def verify_airfoil(designation, avd_data):
    print(f"\nVerifying {designation} against avd.tex...")
    
    x_ref = avd_data[:, 0] / 100.0
    y_ref = avd_data[:, 1] / 100.0
    
    # Generate airfoil
    x_gen, y_gen = generate_naca_airfoil(designation, n_points=200)
    
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
    
    # Print table for large errors
    if max_diff > 1e-4:
        print(f"{'x':>8} {'y_ref':>8} {'y_gen':>8} {'diff':>8}")
        for i in range(len(x_ref)):
            if abs(diff[i]) > 1e-4:
                print(f"{x_ref[i]:8.4f} {y_ref[i]:8.4f} {y_interp[i]:8.4f} {diff[i]:8.4f}")
    
    # Check TE thickness in reference data
    te_thickness_ref = y_ref[-1] * 2 # Assuming symmetric
    print(f"Reference TE Half-Thickness: {y_ref[-1]:.6f}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Verification passed (tolerance 1e-4)")
    else:
        print("WARNING: Verification exceeded tolerance")

def main():
    # Data from avd.tex
    
    # NACA 63A006
    data_63a006 = np.array([
        [0.0000, 0.0000], [0.5000, 0.5066], [0.7500, 0.6082], [1.2500, 0.7663],
        [2.5000, 1.0534], [5.0000, 1.4536], [7.5000, 1.7530], [10.0000, 1.9944],
        [15.0000, 2.3664], [20.0000, 2.6343], [25.0000, 2.8224], [30.0000, 2.9433],
        [35.0000, 2.9956], [40.0000, 2.9838], [45.0000, 2.9103], [50.0000, 2.7816],
        [55.0000, 2.6048], [60.0000, 2.3870], [65.0000, 2.1341], [70.0000, 1.8528],
        [75.0000, 1.5523], [80.0000, 1.2455], [85.0000, 0.9382], [90.0000, 0.6308],
        [95.0000, 0.3187], [100.0000, 0.0000]
    ])
    
    # NACA 64A012 (Need to find this in avd.tex output or assume standard)
    # I saw 64A006. 64A012 should be 2x 64A006 approximately?
    # Let's use 64A006 data scaled by 2 for a quick check if I can't find 012 table?
    # No, I should look for 64A012 table.
    # I'll use the 64A006 data I found and check 64A006 first.
    
    data_64a006 = np.array([
        [0.0000, 0.0000], [0.5000, 0.5015], [0.7500, 0.6018], [1.2500, 0.7543],
        [2.5000, 1.0267], [5.0000, 1.4041], [7.5000, 1.6870], [10.0000, 1.9208],
        [15.0000, 2.2864], [20.0000, 2.5592], [25.0000, 2.7592], [30.0000, 2.8967],
        [35.0000, 2.9767], [40.0000, 2.9978], [45.0000, 2.9439], [50.0000, 2.8245]
        # ... need rest of table
    ])
    
    # NACA 65A006 (I saw this too)
    data_65a006 = np.array([
        [0.0000, 0.0000], [0.5000, 0.4785], [0.7500, 0.5757], [1.2500, 0.7279],
        [2.5000, 0.9890], [5.0000, 1.3199], [7.5000, 1.5958], [10.0000, 1.8282],
        [15.0000, 2.1962], [20.0000, 2.4754], [25.0000, 2.6872], [30.0000, 2.8415],
        [35.0000, 2.9439], [40.0000, 2.9949], [45.0000, 2.9908], [50.0000, 2.9247],
        [55.0000, 2.7931], [60.0000, 2.6014], [65.0000, 2.3631], [70.0000, 2.0859],
        [75.0000, 1.7752], [80.0000, 1.4389], [85.0000, 1.0858], [90.0000, 0.7307],
        [95.0000, 0.3693], [100.0000, 0.0000]
    ])

    verify_airfoil("63A006", data_63a006)
    verify_airfoil("65A006", data_65a006)

if __name__ == "__main__":
    main()
