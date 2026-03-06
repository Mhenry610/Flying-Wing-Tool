import numpy as np
import sys
import os

# Add the Unified Directory to path
sys.path.append(r'd:\AI code\flying wing tool\Full Set\Unified Directory')

from core.naca_generator.naca456 import generate_naca_airfoil

def check_airfoil(designation):
    print(f"\nChecking {designation}...")
    x, y = generate_naca_airfoil(designation, n_points=200)
    
    # Split into upper and lower
    le_idx = np.argmin(x)
    x_upper = x[:le_idx+1]
    y_upper = y[:le_idx+1]
    x_lower = x[le_idx:]
    y_lower = y[le_idx:]
    
    # Check TE (first and last points)
    print(f"TE Upper: ({x_upper[0]:.6f}, {y_upper[0]:.6f})")
    print(f"TE Lower: ({x_lower[-1]:.6f}, {y_lower[-1]:.6f})")
    
    # Check for overlap near TE (last 5% chord)
    # Interpolate to common x near TE
    x_check = np.linspace(0.95, 1.0, 20)
    
    # x_upper is 1 -> 0, x_lower is 0 -> 1
    y_u_int = np.interp(x_check, x_upper[::-1], y_upper[::-1])
    y_l_int = np.interp(x_check, x_lower, y_lower)
    
    diff = y_u_int - y_l_int
    min_diff = np.min(diff)
    
    print("Checking for overlap (y_upper - y_lower) near TE:")
    for i in range(len(x_check)):
        status = "OK" if diff[i] >= -1e-9 else "OVERLAP"
        print(f"x={x_check[i]:.4f}: y_u={y_u_int[i]:.6f}, y_l={y_l_int[i]:.6f}, diff={diff[i]:.6f} {status}")
        
    if min_diff < -1e-9:
        print(f"FAILURE: Overlap detected! Min diff: {min_diff}")
    else:
        print("SUCCESS: No overlap detected.")

if __name__ == "__main__":
    check_airfoil("63A006")
    check_airfoil("64A012")
    check_airfoil("65A010")
