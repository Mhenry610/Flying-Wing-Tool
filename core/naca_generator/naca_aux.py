import numpy as np
import aerosandbox.numpy as asb_np
from core.naca_generator.epsilon_psi import EpsilonPsiData
from core.naca_generator.spline_utils import fmm_spline, spline_zero, pc_lookup

# Constants
TWOPI = 2.0 * np.pi
EPS = 1.0e-9

def calculate_d1(k1):
    """
    Calculate d1 for NACA 5-digit mean lines.
    """
    if k1 < 0.00001:
        return 0.0
    return 2.0 * k1 * (1.0/6.0 - k1/2.0 + k1**2 - k1**3/2.0)

def table_lookup(x_arr, y_arr, x_val):
    """
    Simple linear interpolation for table lookup.
    """
    if x_val <= x_arr[0]:
        return y_arr[0]
    if x_val >= x_arr[-1]:
        return y_arr[-1]
        
    # Find interval
    for i in range(len(x_arr) - 1):
        if x_arr[i] <= x_val <= x_arr[i+1]:
            t = (x_val - x_arr[i]) / (x_arr[i+1] - x_arr[i])
            return y_arr[i] + t * (y_arr[i+1] - y_arr[i])
    return y_arr[-1]

def get_rk1(m):
    """
    Calculate r and k1 for a given m (max camber position) for 3-digit mean line.
    Ref: nacax.f90 Subroutine GetRk1
    """
    M_TAB = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    R_TAB = np.array([0.0580, 0.126, 0.2025, 0.29, 0.391])
    K_TAB = np.array([361.4, 51.64, 15.957, 6.643, 3.23])
    
    r = table_lookup(M_TAB, R_TAB, m)
    k1 = table_lookup(M_TAB, K_TAB, m)
    return r, k1

def get_rk1_k2(m):
    """
    Calculate r, k1, k2 for reflexed mean lines.
    Ref: nacax.f90 Subroutine GetRk1k2
    """
    M_TAB = np.array([0.1, 0.15, 0.2, 0.25])
    R_TAB = np.array([0.13, 0.217, 0.318, 0.441])
    K_TAB = np.array([51.99, 15.793, 6.52, 3.191])
    
    r = table_lookup(M_TAB, R_TAB, m)
    k1 = table_lookup(M_TAB, K_TAB, m)
    
    # k2 formula from nacax.f90
    # k2=(3.0*(r-x)**2-r**3)/(1.0-r)**3
    # Note: x in the formula refers to m (max camber pos)
    k2 = (3.0 * (r - m)**2 - r**3) / (1.0 - r)**3
    
    return r, k1, k2

def mean_line_3(cl, m, x):
    """
    NACA 3-digit mean line (for 5-digit series).
    Ref: nacax.f90 Subroutine MeanLine3
    """
    r, k1 = get_rk1(m)
    
    yc = np.zeros_like(x)
    dydx = np.zeros_like(x)
    
    # Logic from nacax.f90
    # IF (xx < r) THEN
    #   ym(k)=xx*(xx*(xx-3.0*r)+r*r*(3.0-r))
    #   ymp(k)=3.0*xx*(xx-r-r)+r*r*(3.0-r)
    # ELSE
    #   ym(k)=r*r*r*(1.0-xx)  
    #   ymp(k)=-r*r*r
    # END IF
    
    mask = x < r
    x1 = x[mask]
    x2 = x[~mask]
    
    # Forward part
    # ym = x^3 - 3rx^2 + r^2(3-r)x
    yc[mask] = x1**3 - 3*r*x1**2 + r**2 * (3-r) * x1
    dydx[mask] = 3*x1**2 - 6*r*x1 + r**2 * (3-r)
    
    # Aft part
    yc[~mask] = r**3 * (1.0 - x2)
    dydx[~mask] = -r**3
    
    # Scaling
    # ym(1:n)=(k1*cl/1.8)*ym(1:n)
    factor = k1 * cl / 1.8
    
    return yc * factor, dydx * factor

def mean_line_3_reflex(cl, m, x):
    """
    NACA 3-digit reflex mean line.
    Ref: nacax.f90 Subroutine MeanLine3Reflex
    """
    r, k1, k2 = get_rk1_k2(m)
    
    yc = np.zeros_like(x)
    dydx = np.zeros_like(x)
    
    mr3 = (1.0 - r)**3
    r3 = r**3
    
    mask = x < r
    x1 = x[mask]
    x2 = x[~mask]
    
    # Forward part
    # ym(k)=(xx-r)**3-k21*mr3*xx-xx*r3+r3
    # ymp(k)=3.0*(xx-r)**2-k21*mr3-r3
    # Note: k21 in Fortran is our k2
    
    yc[mask] = (x1 - r)**3 - k2 * mr3 * x1 - x1 * r3 + r3
    dydx[mask] = 3.0 * (x1 - r)**2 - k2 * mr3 - r3
    
    # Aft part
    # ym(k)=k21*(xx-r)**3-k21*mr3*xx-xx*r3+r3
    # ymp(k)=3.0*k21*(xx-r)**2-k21*mr3-r3
    
    yc[~mask] = k2 * (x2 - r)**3 - k2 * mr3 * x2 - x2 * r3 + r3
    dydx[~mask] = 3.0 * k2 * (x2 - r)**2 - k2 * mr3 - r3
    
    # Scaling
    factor = k1 * cl / 1.8
    
    return yc * factor, dydx * factor


def thickness_6(family, toc, x, is_type_a=False):
    """
    Compute the thickness distribution of a NACA 6-series section.
    """
    # Get the appropriate epsilon and psi arrays
    if family == 1: # 63-series
        eps = EpsilonPsiData.EPS1
        psi = EpsilonPsiData.PSI1
    elif family == 2: # 64-series
        eps = EpsilonPsiData.EPS2
        psi = EpsilonPsiData.PSI2
    elif family == 3: # 65-series
        eps = EpsilonPsiData.EPS3
        psi = EpsilonPsiData.PSI3
    elif family == 4: # 66-series
        eps = EpsilonPsiData.EPS4
        psi = EpsilonPsiData.PSI4
    elif family == 5: # 67-series
        eps = EpsilonPsiData.EPS5
        psi = EpsilonPsiData.PSI5
    elif family == 6: # 63A-series
        eps = EpsilonPsiData.EPS6
        psi = EpsilonPsiData.PSI6
    elif family == 7: # 64A-series
        eps = EpsilonPsiData.EPS7
        psi = EpsilonPsiData.PSI7
    elif family == 8: # 65A-series
        eps = EpsilonPsiData.EPS8
        psi = EpsilonPsiData.PSI8
    else:
        raise ValueError(f"Unknown NACA 6-series family: {family}")
        
    if eps is None or psi is None:
        raise ValueError("Data for this family is not available.")

    # Scale factor calculation
    COEFF = np.array([
        [0.0, 8.1827699, 1.3776209,  -0.092851684, 7.5942563],   # 63
        [0.0, 4.6535511, 1.038063,   -1.5041794,   4.7882784],   # 64
        [0.0, 6.5718716, 0.49376292,  0.7319794,   1.9491474],   # 65
        [0.0, 6.7581414, 0.19253769,  0.81282621,  0.85202897],  # 66
        [0.0, 6.627289,  0.098965859, 0.96759774,  0.90537584],  # 67
        [0.0, 8.1845925, 1.0492569,   1.31150930,  4.4515579],   # 63A
        [0.0, 8.2125018, 0.76855961,  1.4922345,   3.6130133],   # 64A
        [0.0, 8.2514822, 0.46569361,  1.50113018,  2.0908904]    # 65A
    ])
    
    c = COEFF[family-1] # 0-indexed
    
    # Evaluate polynomial
    sf = c[4]
    for j in range(3, -1, -1):
        sf = sf * toc + c[j]
        
    # Scale epsilon and psi
    eps_scaled = sf * eps
    psi_scaled = sf * psi
    
    # Conformal mapping
    NP = 201
    phi = np.linspace(0, np.pi, NP)
    a = 1.0
    
    # z = a * exp(psi + 1j * phi)
    z = a * np.exp(psi_scaled[0] + 1j * phi) 
    
    # zprime = z * exp(psi - psi(1) - i*eps)
    zprime = z * np.exp((psi_scaled - psi_scaled[0]) - 1j * eps_scaled)
    
    zeta = zprime + a*a / zprime
    
    scale = np.abs(zeta[-1] - zeta[0])
    zfinal = (zeta[0] - zeta) / scale
    
    xt = np.real(zfinal)
    yt = -np.imag(zfinal) 
    
    # Parametrize by arc length s and spline
    xt_rev = xt[::-1]
    yt_rev = yt[::-1]
    
    x_loop = np.concatenate([xt_rev, xt[1:]])
    y_loop = np.concatenate([yt_rev, -yt[1:]])
    
    dx = np.diff(x_loop)
    dy = np.diff(y_loop)
    ds = np.sqrt(dx*dx + dy*dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    
    xp = fmm_spline(s, x_loop)
    yp = fmm_spline(s, y_loop)
    
    # Upper surface segment:
    s_upper = s[:len(xt)]
    x_upper = x_loop[:len(xt)]
    y_upper = y_loop[:len(xt)]
    xp_upper = xp[:len(xt)]
    yp_upper = yp[:len(xt)]
    
    y_out = np.zeros_like(x)
    
    # If Type A manual modification is requested (deprecated in favor of using 64A family)
    # We keep the logic here just in case, but naca456.py should switch families.
    
    for i in range(len(x)):
        xi = x[i]
        s_sol, err = spline_zero(s_upper, x_upper, xp_upper, xi, 1e-6)
        if err == 0:
            y_val, _, _, _ = pc_lookup(s_upper, y_upper, yp_upper, s_sol)
            y_out[i] = y_val
        else:
            y_out[i] = 0.0
            
    return y_out

def mean_line_6_general(cl, a, x):
    """
    Compute the mean line of a NACA 6-series airfoil.
    """
    n = len(x)
    ym = np.zeros(n)
    ymp = np.zeros(n)
    
    if abs(cl) < EPS:
        return ym, ymp
        
    if abs(a - 1.0) < EPS:
        # Uniform load (a=1.0)
        for k in range(n):
            xx = x[k]
            if xx < EPS or xx > 1.0 - EPS:
                ym[k] = 0.0
                ymp[k] = 0.0 # Singularity at ends
            else:
                ym[k] = -(1.0 - xx) * np.log(1.0 - xx) - xx * np.log(xx)
                ymp[k] = np.log(1.0 - xx) - np.log(xx)
        
        factor = cl / (TWOPI * 2.0) # Wait, formula?
        # Standard: y = -cli/4pi * ...
        # My formula above is missing 1/4pi?
        # Let's check nacax.f90 logic.
        # It uses: g = -1/(4pi) * ...
        # Here I'll just use the generic logic below which handles a=1.0 too.
        pass

    # Generic logic from MeanLine6
    oma = 1.0 - a
    if abs(oma) < EPS:
        # a=1.0 case
        g = -1.0
        h = 0.0
        oma = 1.0 # Avoid div by zero in loop? No, handle separately.
        # Actually, let's use the limit.
        for k in range(n):
            xx = x[k]
            if xx < EPS or xx > 1.0 - EPS:
                ym[k] = 0.0
                ymp[k] = 0.0
            else:
                ym[k] = -(1.0 - xx) * np.log(1.0 - xx) - xx * np.log(xx)
                ymp[k] = np.log(1.0 - xx) - np.log(xx)
        factor = cl / (4.0 * np.pi)
        ym *= factor
        ymp *= factor
        return ym, ymp
    else:
        g = -(a*a * (0.5 * np.log(a) - 0.25) + 0.25) / oma
        h = g + (0.5 * oma*oma * np.log(oma) - 0.25 * oma*oma) / oma
        
    for k in range(n):
        xx = x[k]
        omx = 1.0 - xx
        
        if xx < EPS or abs(omx) < EPS:
            ym[k] = 0.0
            ymp[k] = 0.0
            continue
            
        amx = a - xx
        if abs(amx) < EPS:
            term1 = 0.0
            term1p = 0.0
        else:
            term1 = amx*amx * (2.0 * np.log(abs(amx)) - 1.0)
            term1p = -amx * np.log(abs(amx))
            
        term2 = omx*omx * (1.0 - 2.0 * np.log(omx))
        term2p = omx * np.log(omx)
        
        ym[k] = 0.25 * (term1 + term2) / oma - xx * np.log(xx) + g - h * xx
        ymp[k] = (term1p + term2p) / oma - 1.0 - np.log(xx) - h
        
    factor = cl / (TWOPI * (a + 1.0))
    ym *= factor
    ymp *= factor
    
    return ym, ymp

# 7-Series Mean Line Recipes
MEAN_LINE_7_RECIPES = {
    "747": {
        'base_cl': 0.3,
        'components': [
            (0.4, 0.763),
            (0.7, -0.463)
        ]
    }
}

def calculate_ideal_angle_factor(a):
    """
    Calculate phi.
    """
    n = 1000
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2.0
    _, dydx = mean_line_6_general(1.0, a, x)
    dbeta = np.pi / (n - 1)
    integrand = dydx * (1 - np.cos(beta))
    alpha_ideal = - (1.0 / np.pi) * np.trapz(integrand, beta)
    return alpha_ideal

def solve_7_series_recipe(a1, a2, target_cl):
    """
    Solve for CL1 and CL2.
    """
    phi1 = calculate_ideal_angle_factor(a1)
    phi2 = calculate_ideal_angle_factor(a2)
    
    if abs(phi2) < 1e-9:
        if abs(phi1) > 1e-9:
            cl1 = 0.0
            cl2 = target_cl
        else:
            cl1 = target_cl / 2.0
            cl2 = target_cl / 2.0
    else:
        ratio = phi1 / phi2
        if abs(1 - ratio) < 1e-9:
            cl1 = target_cl / 2.0
            cl2 = target_cl / 2.0
        else:
            cl1 = target_cl / (1.0 - ratio)
            cl2 = target_cl - cl1
            
    return cl1, cl2

def mean_line_7(series, cl, x):
    """
    Compute the mean line for NACA 7-series.
    """
    x = asb_np.asarray(x)
    
    if series in MEAN_LINE_7_RECIPES:
        recipe = MEAN_LINE_7_RECIPES[series]
        base_cl = recipe['base_cl']
        components = recipe['components']
    else:
        if len(series) == 3 and series.startswith('7') and series[1:].isdigit():
            a1 = int(series[1]) / 10.0
            a2 = int(series[2]) / 10.0
            cl1, cl2 = solve_7_series_recipe(a1, a2, cl)
            base_cl = cl
            components = [(a1, cl1), (a2, cl2)]
        else:
            print(f"Warning: Unknown 7-series '{series}', defaulting to 747 recipe.")
            recipe = MEAN_LINE_7_RECIPES["747"]
            base_cl = recipe['base_cl']
            components = recipe['components']
        
    yc_base = np.zeros_like(x)
    ymp_base = np.zeros_like(x)
    
    for a_val, cl_val in components:
        yc_comp, ymp_comp = mean_line_6_general(cl_val, a_val, x)
        yc_base += yc_comp
        ymp_base += ymp_comp
        
    delta_cl = cl - base_cl
    
    if abs(delta_cl) > 1e-9:
        yc_off, ymp_off = mean_line_6_general(delta_cl, 1.0, x)
        yc_total = yc_base + yc_off
        ymp_total = ymp_base + ymp_off
    else:
        yc_total = yc_base
        ymp_total = ymp_base
        
    return yc_total, ymp_total

def mean_line_6(cl, x):
    """
    Wrapper for mean_line_6_general with a=1.0.
    """
    return mean_line_6_general(cl, 1.0, x)

def combine_thickness_and_camber(x, yt, yc, dydx_c):
    """
    Combine thickness and camber.
    """
    theta = np.arctan(dydx_c)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl
