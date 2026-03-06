import numpy as np
import aerosandbox.numpy as asb_np

def evaluate_cubic(a, fa, fpa, b, fb, fpb, u):
    """
    Evaluate a cubic polynomial defined by the function and the 1st derivative at two points.
    
    Args:
        a: x-coordinate of first point
        fa: function value at a
        fpa: derivative at a
        b: x-coordinate of second point
        fb: function value at b
        fpb: derivative at b
        u: point where function is to be evaluated
        
    Returns:
        Interpolated value at u
    """
    d = (fb - fa) / (b - a)
    t = (u - a) / (b - a)
    p = 1.0 - t
    
    fu = p * fa + t * fb - p * t * (b - a) * (p * (d - fpa) - t * (d - fpb))
    return fu

def evaluate_cubic_and_derivs(a, fa, fpa, b, fb, fpb, u):
    """
    Evaluate a cubic polynomial and its 1st, 2nd, and 3rd derivatives at a specified point.
    
    Returns:
        tuple: (f, fp, fpp, fppp)
    """
    # The "magic" matrix from Fortran code
    MAGIC = np.array([
        [2.0, -3.0, 0.0, 1.0],
        [-2.0, 3.0, 0.0, 0.0],
        [1.0, -2.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 0.0]
    ]).T # Transpose because Fortran is column-major but the definition was RESHAPE linear
    # Actually, let's look at the Fortran reshape:
    # (/2.0, -3.0,  0.0,  1.0, -2.0,  3.0,  0.0,  0.0, 1.0, -2.0,  1.0,  0.0,  1.0, -1.0,  0.0,  0.0/)
    # Fortran fills columns first.
    # Col 1: 2, -3, 0, 1
    # Col 2: -2, 3, 0, 0
    # Col 3: 1, -2, 1, 0
    # Col 4: 1, -1, 0, 0
    # So the matrix is:
    # [[ 2, -2,  1,  1],
    #  [-3,  3, -2, -1],
    #  [ 0,  0,  1,  0],
    #  [ 1,  0,  0,  0]]
    
    # Let's re-verify the logic from splprocs.f90
    # rhs(1)=fa
    # rhs(2)=fb
    # rhs(3)=fpa*(b-a)
    # rhs(4)=fpb*(b-a)
    # coef=MATMUL(MAGIC,rhs)
    
    # If we use the matrix above:
    # coef[0] = 2fa - 2fb + fpa*h + fpb*h
    # This looks like the standard Hermite basis coefficients.
    
    MAGIC = np.array([
        [2.0, -2.0, 1.0, 1.0],
        [-3.0, 3.0, -2.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    rhs = np.array([fa, fb, fpa * (b - a), fpb * (b - a)])
    coef = MAGIC @ rhs
    
    h_inv = 1.0 / (b - a)
    t = (u - a) * h_inv
    
    f = coef[3] + t * (coef[2] + t * (coef[1] + t * coef[0]))
    fp = h_inv * (coef[2] + t * (2.0 * coef[1] + t * 3.0 * coef[0]))
    fpp = h_inv * h_inv * (2.0 * coef[1] + t * 6.0 * coef[0])
    fppp = h_inv * h_inv * h_inv * 6.0 * coef[0]
    
    return f, fp, fpp, fppp

def fmm_spline(x, y):
    """
    Compute the cubic spline with endpoint conditions chosen by FMM 
    (Forsythe, Malcolm & Moler).
    
    Args:
        x: array of x coordinates (must be increasing)
        y: array of y coordinates
        
    Returns:
        yp: array of derivatives at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    yp = np.zeros(n)
    
    if n < 2:
        return yp
        
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    delta = dy / dx
    
    if n == 2:
        yp[:] = delta[0]
        return yp
        
    dd = delta[1:] - delta[:-1]
    
    if n == 3:
        deriv2 = dd[0] / (x[2] - x[0])
        deriv1 = delta[0] - deriv2 * dx[0]
        yp[0] = deriv1
        yp[1] = deriv1 + deriv2 * dx[0]
        yp[2] = deriv1 + deriv2 * (x[2] - x[0])
        return yp
        
    # n > 3
    alpha = np.zeros(n)
    beta = np.zeros(n)
    sigma = np.zeros(n)
    
    alpha[0] = -dx[0]
    alpha[1:n-1] = 2.0 * (dx[0:n-2] + dx[1:n-1])
    
    # Forward elimination
    for i in range(1, n-1):
        alpha[i] = alpha[i] - dx[i-1] * dx[i-1] / alpha[i-1]
        
    alpha[n-1] = -dx[n-2] - dx[n-2] * dx[n-2] / alpha[n-2]
    
    beta[0] = dd[1] / (x[3] - x[1]) - dd[0] / (x[2] - x[0])
    beta[0] = beta[0] * dx[0] * dx[0] / (x[3] - x[0])
    
    beta[1:n-1] = dd
    
    beta[n-1] = dd[n-3] / (x[n-1] - x[n-3]) - dd[n-4] / (x[n-2] - x[n-4])
    beta[n-1] = -beta[n-1] * dx[n-2] * dx[n-2] / (x[n-1] - x[n-4])
    
    # Forward elimination for beta
    for i in range(1, n):
        beta[i] = beta[i] - dx[i-1] * beta[i-1] / alpha[i-1]
        
    sigma[n-1] = beta[n-1] / alpha[n-1]
    
    # Back substitution
    for i in range(n-2, -1, -1):
        sigma[i] = (beta[i] - dx[i] * sigma[i+1]) / alpha[i]
        
    yp[0:n-1] = delta - dx * (sigma[0:n-1] + sigma[0:n-1] + sigma[1:n])
    yp[n-1] = yp[n-2] + dx[n-2] * 3.0 * (sigma[n-1] + sigma[n-2])
    
    return yp

def interpolate_polynomial(x, y, u):
    """
    Compute the value of the interpolating polynomial thru x- and y-arrays 
    at the x-value of u, using Lagrange's equation.
    """
    du = u - x
    sum_val = 0.0
    for j in range(len(x)):
        fact = 1.0
        for i in range(len(x)):
            if i != j:
                fact = fact * du[i] / (x[j] - x[i])
        sum_val += y[j] * fact
    return sum_val

def lookup(xtab, x):
    """
    Search a sorted (increasing) array to find the interval bounding a given number.
    Returns i such that xtab[i] <= x < xtab[i+1] (0-indexed).
    """
    # Using numpy searchsorted which is efficient
    # searchsorted returns the index where x should be inserted to maintain order.
    # if x is in [xtab[i], xtab[i+1]), searchsorted(side='right') returns i+1.
    # So i = idx - 1.
    
    if x < xtab[0]:
        return -1 # Fortran returned 0 (1-based) -> 0 (0-based) if < first? No, Fortran returned 0 if < a(1).
                  # Fortran indices: 1..n. 
                  # If x < a(1), return 0.
                  # If a(i) <= x < a(i+1), return i.
                  # If x > a(n), return n.
                  
    if x >= xtab[-1]:
        return len(xtab) - 1
        
    idx = np.searchsorted(xtab, x, side='right')
    return idx - 1

def pc_lookup(x, y, yp, u):
    """
    Interpolate in a cubic spline at one point.
    """
    k = lookup(x, u)
    k = max(0, min(len(x) - 2, k))
    
    a = x[k]
    fa = y[k]
    fpa = yp[k]
    b = x[k+1]
    fb = y[k+1]
    fpb = yp[k+1]
    
    return evaluate_cubic_and_derivs(a, fa, fpa, b, fb, fpb, u)

def zeroin(ax, bx, f_func, tol):
    """
    Compute a zero of f in the interval (ax,bx) using Brent's method.
    """
    # Scipy's brentq is equivalent to the Zeroin algorithm (Brent's method)
    from scipy.optimize import brentq
    try:
        return brentq(f_func, ax, bx, xtol=tol)
    except ValueError:
        # Fallback or error handling if signs are not opposite
        # The Fortran code had some specific handling, but brentq expects opposite signs.
        # If signs are same, Fortran code might fail or do something else?
        # Fortran code: "should test that fa and fb have opposite signs"
        # If not, it proceeds... actually the Fortran code seems to assume they cross or it fails.
        # Let's check f(ax) and f(bx).
        fa = f_func(ax)
        fb = f_func(bx)
        if fa * fb > 0:
            # No crossing found
            raise ValueError(f"f(a) and f(b) must have different signs: f({ax})={fa}, f({bx})={fb}")
        return brentq(f_func, ax, bx, xtol=tol)

def spline_zero(x, f, fp, fbar, tol):
    """
    Find a value of x corresponding to a value of fbar of the cubic spline.
    """
    n = len(x)
    
    # Check for exact match
    for k in range(n):
        if abs(f[k] - fbar) < tol:
            return x[k], 0
            
    f_local = f - fbar
    
    # Look for crossing
    k_cross = -1
    for k in range(1, n):
        if f_local[k-1] * f_local[k] < 0:
            k_cross = k
            break
            
    if k_cross == -1:
        return 0.0, 1 # Error code 1
        
    # Setup cubic for the interval [x[k-1], x[k]]
    k = k_cross
    idx = k - 1
    
    a = x[idx]
    fa = f_local[idx]
    fpa = fp[idx]
    b = x[k]
    fb = f_local[k]
    fpb = fp[k]
    
    def cubic_func(u):
        return evaluate_cubic(a, fa, fpa, b, fb, fpb, u)
        
    try:
        xbar = zeroin(a, b, cubic_func, tol)
        return xbar, 0
    except ValueError:
        return 0.0, 1

def table_lookup(x, y, order, u):
    """
    Use polynomial evaluation for table lookup.
    """
    m = min(order + 1, len(x))
    j = lookup(x, u)
    j = j - (m // 2 - 1)
    j = min(len(x) - m, j)
    j = max(0, j)
    
    # In python slice is j:j+m
    return interpolate_polynomial(x[j:j+m], y[j:j+m], u)
