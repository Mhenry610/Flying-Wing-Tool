import math
import re


def floats_in_line(line: str):
    return [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', line)]


def isa_density(alt_m: float) -> float:
    """ISA (troposphere) density. Valid to ~11 km."""
    g0 = 9.80665
    T0 = 288.15
    L = -0.0065
    R = 287.05287
    p0 = 101325.0
    T = T0 + L * alt_m
    if T <= 0:
        T = 170.0  # guard
    p = p0 * (T / T0) ** (-g0 / (L * R))
    return p / (R * T)


def tip_mach(rpm: float, D_m: float, a_ms: float = 343.0) -> float:
    w = rpm * 2 * math.pi / 60.0
    return (w * (D_m / 2)) / a_ms

