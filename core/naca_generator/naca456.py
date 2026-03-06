import numpy as np
from core.naca_generator.naca_aux import (
    thickness_6, mean_line_6, mean_line_7, combine_thickness_and_camber,
    mean_line_3, mean_line_3_reflex
)

def generate_naca_airfoil(designation, n_points=100):
    """
    Generate coordinates for a NACA airfoil.
    """
    designation = designation.strip().upper()
    
    # 4-digit
    if len(designation) == 4 and designation.isdigit():
        m = int(designation[0]) / 100.0
        p = int(designation[1]) / 10.0
        t = int(designation[2:]) / 100.0
        
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        yc = np.zeros_like(x)
        dydx_c = np.zeros_like(x)
        
        if m > 0:
            p_idx = np.argmin(np.abs(x - p))
            
            yc[x <= p] = m / p**2 * (2 * p * x[x <= p] - x[x <= p]**2)
            yc[x > p] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[x > p] - x[x > p]**2)
            
            dydx_c[x <= p] = 2 * m / p**2 * (p - x[x <= p])
            dydx_c[x > p] = 2 * m / (1 - p)**2 * (p - x[x > p])
            
        xu, yu, xl, yl = combine_thickness_and_camber(x, yt, yc, dydx_c)
        x_final = np.concatenate([xu[::-1], xl[1:]])
        y_final = np.concatenate([yu[::-1], yl[1:]])
        return x_final, y_final

    # 5-digit
    elif len(designation) == 5 and designation.isdigit():
        l = int(designation[0])
        p = int(designation[1])
        q = int(designation[2])
        toc = int(designation[3:]) / 100.0
        
        cl_design = l * 0.15
        m = p / 20.0
        is_reflex = (q == 1)
        
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        # Thickness (same as 4-digit)
        yt = 5 * toc * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        if is_reflex:
            yc, dydx_c = mean_line_3_reflex(cl_design, m, x)
        else:
            yc, dydx_c = mean_line_3(cl_design, m, x)
            
        xu, yu, xl, yl = combine_thickness_and_camber(x, yt, yc, dydx_c)
        x_final = np.concatenate([xu[::-1], xl[1:]])
        y_final = np.concatenate([yu[::-1], yl[1:]])
        return x_final, y_final

    # 7-series
    elif designation.startswith('7'):
        import re
        match = re.match(r"(\d+)([A-Z]?)(\d+)", designation)
        if not match:
            raise ValueError(f"Unsupported 7-series designation: {designation}")
            
        series_name = match.group(1)
        mod_code = match.group(2)
        rest = match.group(3)
        
        if len(rest) == 3:
            cl_des = int(rest[0])
            toc_des = int(rest[1:])
            cl = cl_des / 10.0
            toc = toc_des / 100.0
        else:
            raise ValueError(f"Cannot parse parameters from: {rest}")
            
        is_type_a = (mod_code == 'A')
        
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        # Use 64A-series (Family 7) if Type A mod is requested
        family_id = 7 if is_type_a else 2
        yt = thickness_6(family_id, toc, x, is_type_a=False)
        
        yc, dydx_c = mean_line_7(series_name, cl, x)
        
        xu, yu, xl, yl = combine_thickness_and_camber(x, yt, yc, dydx_c)
        x_final = np.concatenate([xu[::-1], xl[1:]])
        y_final = np.concatenate([yu[::-1], yl[1:]])
        return x_final, y_final

    # 6-series
    elif designation.startswith('6'):
        clean_des = designation.replace('-', '')
        
        family = 0
        if clean_des.startswith('63A'): family = 6; rest = clean_des[3:]
        elif clean_des.startswith('64A'): family = 7; rest = clean_des[3:]
        elif clean_des.startswith('65A'): family = 8; rest = clean_des[3:]
        elif clean_des.startswith('63'): family = 1; rest = clean_des[2:]
        elif clean_des.startswith('64'): family = 2; rest = clean_des[2:]
        elif clean_des.startswith('65'): family = 3; rest = clean_des[2:]
        elif clean_des.startswith('66'): family = 4; rest = clean_des[2:]
        elif clean_des.startswith('67'): family = 5; rest = clean_des[2:]
        else:
            raise ValueError(f"Unsupported 6-series designation: {designation}")
            
        if len(rest) == 3:
            cl_des = int(rest[0])
            toc_des = int(rest[1:])
            cl = cl_des / 10.0
            toc = toc_des / 100.0
        else:
            raise ValueError(f"Cannot parse parameters from: {rest}")
            
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        yt = thickness_6(family, toc, x)
        yc, dydx_c = mean_line_6(cl, x)
        
        xu, yu, xl, yl = combine_thickness_and_camber(x, yt, yc, dydx_c)
        x_final = np.concatenate([xu[::-1], xl[1:]])
        y_final = np.concatenate([yu[::-1], yl[1:]])
        return x_final, y_final
        
    else:
        raise ValueError(f"Unknown NACA designation: {designation}")
