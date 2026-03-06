import numpy as np

def naca4_coordinates(name, n_points=100):
    """
    Generates coordinates for a NACA 4-digit airfoil (e.g., naca2412).
    """
    name = name.strip()
    digits = name[4:]
    
    if len(digits) != 4:
        return None
        
    try:
        m = int(digits[0]) / 100.0
        p = int(digits[1]) / 10.0
        t = int(digits[2:]) / 100.0
    except ValueError:
        return None

    # Generate x coordinates (cosine spacing)
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p == 0:
        # Symmetric airfoil
        xu = x
        yu = yt
        xl = x
        yl = -yt
    else:
        for i, xi in enumerate(x):
            if xi <= p:
                yc[i] = (m / p**2) * (2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / p**2) * (p - xi)
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - xi)
                
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    # Combine into (N, 2) array, TE -> LE -> TE
    xu_flip = np.flip(xu)
    yu_flip = np.flip(yu)
    
    xl_cut = xl[1:]
    yl_cut = yl[1:]
    
    x_final = np.concatenate([xu_flip, xl_cut])
    y_final = np.concatenate([yu_flip, yl_cut])
    
    return np.column_stack([x_final, y_final])


def naca5_coordinates(name, n_points=100):
    """
    Generates coordinates for a NACA 5-digit airfoil.
    Supports standard 5-digit series (e.g., 23012, 24012) and reflexed (e.g., 24112).
    """
    name = name.strip()
    # Basic check, caller should dispatch
    digits = name[4:]
    if len(digits) != 5:
        return None
        
    # Parse digits
    # L: Design CL * 20/3 (approx)
    # P: Position of max camber * 20
    # Q: Reflex (0=Standard, 1=Reflex)
    # TT: Thickness
    
    try:
        L_digit = int(digits[0])
        P_digit = int(digits[1])
        Q_digit = int(digits[2])
        T_digits = int(digits[3:])
    except ValueError:
        return None
    
    p = P_digit / 20.0
    t = T_digits / 100.0
    is_reflex = (Q_digit == 1)
    
    # Constants
    # p -> (r, k1)
    standard_constants = {
        0.05: (0.0580, 361.400),
        0.10: (0.1260, 51.640),
        0.15: (0.2025, 15.957),
        0.20: (0.2900, 6.643),
        0.25: (0.3910, 3.230),
    }
    
    # p -> (r, k1, k2/k1)
    reflex_constants = {
        0.10: (0.1300, 51.990, 0.000764),
        0.15: (0.2170, 15.793, 0.00677),
        0.20: (0.3180, 6.520, 0.0303),
        0.25: (0.4410, 3.191, 0.1355),
    }
    
    if is_reflex:
        if p not in reflex_constants:
            print(f"Warning: Non-standard max camber position {p} for NACA 5-digit reflex.")
            return None
        r, k1, k2_k1 = reflex_constants[p]
    else:
        if p not in standard_constants:
            print(f"Warning: Non-standard max camber position {p} for NACA 5-digit standard.")
            return None
        r, k1 = standard_constants[p]
        k2_k1 = 0 # Not used for standard
        
    # Generate x coordinates (cosine spacing)
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Thickness distribution (same as 4-digit)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if is_reflex:
            # Reflex equations
            if xi < r:
                # Front
                # yc = k1/6 * ((x-r)^3 - k2/k1*(1-r)^3*x - r^3*x + r^3)
                term1 = (xi - r)**3
                term2 = k2_k1 * (1 - r)**3 * xi
                term3 = r**3 * xi
                term4 = r**3
                yc[i] = (k1 / 6.0) * (term1 - term2 - term3 + term4)
                
                # dyc/dx = k1/6 * (3(x-r)^2 - k2/k1*(1-r)^3 - r^3)
                d_term1 = 3 * (xi - r)**2
                d_term2 = k2_k1 * (1 - r)**3
                d_term3 = r**3
                dyc_dx[i] = (k1 / 6.0) * (d_term1 - d_term2 - d_term3)
            else:
                # Back
                # yc = k1/6 * (k2/k1*(x-r)^3 - k2/k1*(1-r)^3*x - r^3*x + r^3)
                term1 = k2_k1 * (xi - r)**3
                term2 = k2_k1 * (1 - r)**3 * xi
                term3 = r**3 * xi
                term4 = r**3
                yc[i] = (k1 / 6.0) * (term1 - term2 - term3 + term4)
                
                # dyc/dx = k1/6 * (3*k2/k1*(x-r)^2 - k2/k1*(1-r)^3 - r^3)
                d_term1 = 3 * k2_k1 * (xi - r)**2
                d_term2 = k2_k1 * (1 - r)**3
                d_term3 = r**3
                dyc_dx[i] = (k1 / 6.0) * (d_term1 - d_term2 - d_term3)
        else:
            # Standard equations
            if xi < r:
                # Front
                # yc = k1/6 * (x^3 - 3rx^2 + r^2(3-r)x)
                yc[i] = (k1 / 6.0) * (xi**3 - 3*r*xi**2 + r**2*(3-r)*xi)
                dyc_dx[i] = (k1 / 6.0) * (3*xi**2 - 6*r*xi + r**2*(3-r))
            else:
                # Back
                # yc = k1*r^3/6 * (1-x)
                yc[i] = (k1 * r**3 / 6.0) * (1 - xi)
                dyc_dx[i] = -k1 * r**3 / 6.0
            
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine into (N, 2) array, TE -> LE -> TE
    xu_flip = np.flip(xu)
    yu_flip = np.flip(yu)
    
    xl_cut = xl[1:]
    yl_cut = yl[1:]
    
    x_final = np.concatenate([xu_flip, xl_cut])
    y_final = np.concatenate([yu_flip, yl_cut])
    
    return np.column_stack([x_final, y_final])

def get_naca_points(name, n_points=100):
    """
    Dispatcher for NACA 4-digit and 5-digit airfoils.
    """
    name = name.strip().lower()
    if not name.startswith("naca"):
        return None
        
    digits = name[4:]
    if len(digits) == 4:
        return naca4_coordinates(name, n_points)
    elif len(digits) == 5:
        return naca5_coordinates(name, n_points)
    else:
        return None



if __name__ == "__main__":
    # Test standard
    coords = naca5_coordinates("naca24012")
    if coords is not None:
        print("Successfully generated naca24012")
        
    # Test reflex
    coords_reflex = naca5_coordinates("naca24112")
    if coords_reflex is not None:
        print("Successfully generated naca24112")
