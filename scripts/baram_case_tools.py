from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def normalize_case_dir(case_root: Path) -> Path:
    """Accept either a .bf bundle path or the inner OpenFOAM-style case dir."""
    case_root = Path(case_root).resolve()
    if (case_root / "case").is_dir():
        return case_root / "case"
    return case_root


def find_latest_numeric_dir(parent: Path) -> Path:
    numeric_dirs = []
    for child in parent.iterdir():
        if child.is_dir():
            try:
                numeric_dirs.append((float(child.name), child))
            except ValueError:
                continue
    if not numeric_dirs:
        raise FileNotFoundError(f"No numeric subdirectories found under {parent}")
    numeric_dirs.sort(key=lambda item: item[0])
    return numeric_dirs[-1][1]


def find_monitor_file(case_dir: Path, monitor_name: str, filename: str) -> Path:
    monitor_root = case_dir / "postProcessing" / monitor_name
    latest_dir = find_latest_numeric_dir(monitor_root)
    target = Path(filename)
    candidates: list[Path] = []

    direct = latest_dir / target.name
    if direct.exists():
        candidates.append(direct)

    # Parallel functionObjects commonly emit suffixed files (e.g. coefficient_0.dat).
    suffixed = sorted(latest_dir.glob(f"{target.stem}_*{target.suffix}"))
    for path in suffixed:
        if path.is_file():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Expected monitor file not found in {latest_dir}: {target.name} or {target.stem}_*{target.suffix}"
        )

    # Prefer the freshest file so restarted/parallel runs use the latest monitor data.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_coefficient_dat(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append(
            {
                "time": float(parts[0]),
                "Cd_raw": float(parts[1]),
                "Cl_raw": float(parts[2]),
                "Cm_raw": float(parts[3]),
            }
        )
    if not rows:
        raise ValueError(f"No coefficient rows parsed from {path}")
    return rows


def parse_force_dat(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        rows.append(
            {
                "time": float(parts[0]),
                "total_x_half": float(parts[1]),
                "total_y_half": float(parts[2]),
                "total_z_half": float(parts[3]),
                "pressure_x_half": float(parts[4]),
                "pressure_y_half": float(parts[5]),
                "pressure_z_half": float(parts[6]),
                "viscous_x_half": float(parts[7]),
                "viscous_y_half": float(parts[8]),
                "viscous_z_half": float(parts[9]),
            }
        )
    if not rows:
        raise ValueError(f"No force rows parsed from {path}")
    return rows


def _parse_vector(text: str, key: str) -> tuple[float, float, float] | None:
    match = re.search(
        rf"\b{re.escape(key)}\s*\(\s*([^\s\)]+)\s+([^\s\)]+)\s+([^\s\)]+)\s*\)\s*;",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def read_force_axis_dirs(case_dir: Path, function_name: str = "force-mon-1") -> dict[str, tuple[float, float, float]]:
    control_dict = case_dir / "system" / "controlDict"
    text = control_dict.read_text(errors="ignore")
    block_match = re.search(
        rf"\b{re.escape(function_name)}\s*\{{(.*?)\n\}}",
        text,
        flags=re.DOTALL,
    )
    block = block_match.group(1) if block_match else text

    drag_dir = _parse_vector(block, "dragDir")
    lift_dir = _parse_vector(block, "liftDir")
    if drag_dir is None:
        drag_dir = (1.0, 0.0, 0.0)
    if lift_dir is None:
        lift_dir = (0.0, 0.0, 1.0)
    return {"drag_dir": drag_dir, "lift_dir": lift_dir}


def project_force_components(
    force_row: dict[str, float],
    drag_dir: tuple[float, float, float],
    lift_dir: tuple[float, float, float],
) -> dict[str, float]:
    def dot(prefix: str, direction: tuple[float, float, float]) -> float:
        return (
            force_row[f"{prefix}_x_half"] * direction[0]
            + force_row[f"{prefix}_y_half"] * direction[1]
            + force_row[f"{prefix}_z_half"] * direction[2]
        )

    return {
        "total_drag_half": dot("total", drag_dir),
        "total_lift_half": dot("total", lift_dir),
        "pressure_drag_half": dot("pressure", drag_dir),
        "viscous_drag_half": dot("viscous", drag_dir),
        "pressure_lift_half": dot("pressure", lift_dir),
        "viscous_lift_half": dot("viscous", lift_dir),
    }


def load_summary_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics", summary.get("aero", {}))
    if not metrics:
        raise KeyError("No 'metrics' or 'aero' block found in summary JSON.")
    return metrics


def extract_actual_area(summary: dict[str, Any], fallback_area: float | None = None) -> float:
    geometry = summary.get("geometry", {})
    actual_area = geometry.get("actual_area")
    if actual_area is None:
        planform = summary.get("planform", {})
        actual_area = planform.get("actual_area")
    if actual_area is None and fallback_area is not None:
        actual_area = fallback_area
    if actual_area is None:
        raise KeyError("Actual area not found in summary JSON and no fallback area supplied.")
    return float(actual_area)


def average_last(rows: list[dict[str, float]], key: str, count: int) -> float:
    subset = rows[-count:]
    return sum(row[key] for row in subset) / len(subset)


def make_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return slug.strip("_").lower() or "case"


def update_pvsm_case_path(pvsm_text: str, new_case_foam_path: str) -> str:
    return re.sub(
        r'value="[^"]+case\.foam"',
        lambda _: f'value="{new_case_foam_path}"',
        pvsm_text,
    )


def compute_dynamic_pressure(density: float, velocity: float) -> float:
    return 0.5 * density * velocity * velocity


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def discover_baram_openfoam_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root).resolve()
        if not (root / "bin").is_dir():
            raise FileNotFoundError(f"OpenFOAM bin directory not found under {root}")
        return root

    candidates = [
        Path(os.environ.get("BARAM_OPENFOAM_ROOT", "")),
        Path(r"C:\Users\Malik\AppData\Local\Programs\BARAM\solvers\openfoam"),
    ]
    for candidate in candidates:
        if str(candidate) and (candidate / "bin").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not discover BARAM OpenFOAM root. Supply --baram-openfoam-root.")


def build_openfoam_env(openfoam_root: Path) -> dict[str, str]:
    openfoam_root = Path(openfoam_root).resolve()
    mingw_bin = openfoam_root.parent / "mingw64" / "bin"
    mpi_bin = Path(r"C:\Program Files\Microsoft MPI\Bin")
    env = dict(os.environ)
    env["WM_PROJECT"] = "OpenFOAM"
    env["WM_PROJECT_VERSION"] = "v2412"
    env["WM_PROJECT_DIR"] = str(openfoam_root)
    env["FOAM_ETC"] = str(openfoam_root / "etc")
    env["FOAM_SIGFPE"] = env.get("FOAM_SIGFPE", "1")

    path_parts = [
        str(openfoam_root / "bin"),
        str(openfoam_root / "lib"),
        str(openfoam_root / "lib" / "msmpi"),
        str(mingw_bin),
        str(mpi_bin),
        env.get("PATH", ""),
    ]
    env["PATH"] = ";".join(part for part in path_parts if part)
    return env


def resolve_solver_path(openfoam_root: Path, solver_name: str) -> Path:
    solver_path = openfoam_root / "bin" / f"{solver_name}.exe"
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver executable not found: {solver_path}")
    return solver_path


def run_openfoam_utility(
    openfoam_root: Path,
    utility_name: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    utility_path = resolve_solver_path(openfoam_root, utility_name)
    env = build_openfoam_env(openfoam_root)
    return subprocess.run(
        [str(utility_path), *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        timeout=timeout,
        text=True,
        capture_output=capture_output,
    )


def read_control_dict_application(control_dict_path: Path) -> str:
    text = control_dict_path.read_text()
    match = re.search(r"^\s*application\s+([A-Za-z0-9_]+)\s*;", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse application from {control_dict_path}")
    return match.group(1)


def read_decompose_subdomains(decompose_dict_path: Path) -> int | None:
    if not decompose_dict_path.exists():
        return None
    text = decompose_dict_path.read_text()
    match = re.search(r"^\s*numberOfSubdomains\s+(\d+)\s*;", text, flags=re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def count_processor_dirs(case_dir: Path) -> int:
    return sum(1 for child in case_dir.iterdir() if child.is_dir() and re.fullmatch(r"processor\d+", child.name))


def list_numeric_time_dirs(parent: Path) -> list[Path]:
    result: list[tuple[float, Path]] = []
    for child in parent.iterdir():
        if child.is_dir():
            try:
                result.append((float(child.name), child))
            except ValueError:
                continue
    result.sort(key=lambda item: item[0])
    return [path for _, path in result]


def parse_boundary_patch_nfaces(boundary_path: Path, patch_name: str) -> int:
    text = boundary_path.read_text()
    pattern = rf"{re.escape(patch_name)}\s*\{{.*?nFaces\s+(\d+)\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find patch '{patch_name}' in {boundary_path}")
    return int(match.group(1))


def _update_bounds(bounds_min: list[float], bounds_max: list[float], values: tuple[float, float, float]) -> None:
    for index, value in enumerate(values):
        bounds_min[index] = min(bounds_min[index], value)
        bounds_max[index] = max(bounds_max[index], value)


def _scan_ascii_obj_bounds(path: Path) -> tuple[list[float], list[float]] | None:
    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]
    found_vertex = False
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            _update_bounds(bounds_min, bounds_max, (float(parts[1]), float(parts[2]), float(parts[3])))
            found_vertex = True
    return (bounds_min, bounds_max) if found_vertex else None


def _scan_ascii_stl_bounds(path: Path) -> tuple[list[float], list[float]] | None:
    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]
    found_vertex = False
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            _update_bounds(bounds_min, bounds_max, (float(parts[1]), float(parts[2]), float(parts[3])))
            found_vertex = True
    return (bounds_min, bounds_max) if found_vertex else None


def infer_geometry_length_scale_to_m(case_dir: Path) -> dict[str, Any]:
    tri_surface_dir = case_dir / "constant" / "triSurface"
    if not tri_surface_dir.is_dir():
        return {
            "source": None,
            "length_scale_to_m": 1.0,
            "area_scale_to_m2": 1.0,
            "inferred_units": "m",
            "bbox_min": None,
            "bbox_max": None,
            "bbox_span": None,
            "note": "triSurface directory not found; assuming geometry is already in meters.",
        }

    candidates = sorted(tri_surface_dir.glob("*_surface.obj")) + sorted(tri_surface_dir.glob("*_surface.stl"))
    candidates += sorted(path for path in tri_surface_dir.glob("*.obj") if path not in candidates)
    candidates += sorted(path for path in tri_surface_dir.glob("*.stl") if path not in candidates)

    for candidate in candidates:
        bounds: tuple[list[float], list[float]] | None
        if candidate.suffix.lower() == ".obj":
            bounds = _scan_ascii_obj_bounds(candidate)
        else:
            bounds = _scan_ascii_stl_bounds(candidate)
        if bounds is None:
            continue
        bounds_min, bounds_max = bounds
        span = [bounds_max[index] - bounds_min[index] for index in range(3)]
        max_span = max(span)
        length_scale = 1.0
        inferred_units = "m"
        note = "Geometry span is already meter-scale."
        if max_span > 100.0:
            length_scale = 1e-3
            inferred_units = "mm"
            note = "Geometry span is millimeter-scale; postprocessing scales forces by 1e-6 to recover physical SI values."
        return {
            "source": str(candidate),
            "length_scale_to_m": length_scale,
            "area_scale_to_m2": length_scale * length_scale,
            "inferred_units": inferred_units,
            "bbox_min": bounds_min,
            "bbox_max": bounds_max,
            "bbox_span": span,
            "note": note,
        }

    return {
        "source": None,
        "length_scale_to_m": 1.0,
        "area_scale_to_m2": 1.0,
        "inferred_units": "m",
        "bbox_min": None,
        "bbox_max": None,
        "bbox_span": None,
        "note": "No readable OBJ/STL vertices found; assuming geometry is already in meters.",
    }
