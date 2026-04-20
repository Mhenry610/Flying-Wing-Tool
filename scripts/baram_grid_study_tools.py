from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MESH_PRESETS: dict[str, dict[str, Any]] = {
    "coarse": {
        "block_scale": 0.85,
        "surface_level_min_delta": 0,
        "surface_level_max_delta": 0,
        "region_level_delta": -1,
        "layer_count": 3,
        "layer_expansion_ratio": 1.12,
        "layer_first_thickness": 0.005,
        "layer_min_thickness": 0.0005,
        "layer_feature_angle": 150,
        "layer_nGrow": 1,
        "layer_max_face_thickness_ratio": 0.5,
        "layer_max_thickness_to_medial_ratio": 0.3,
        "layer_nSmoothSurfaceNormals": 8,
        "layer_nSmoothThickness": 20,
        "layer_min_medial_axis_angle": 65,
        "layer_nSmoothNormals": 8,
        "layer_nRelaxIter": 10,
        "layer_nBufferCellsNoExtrude": 2,
        "layer_nLayerIter": 80,
        "layer_nRelaxedIter": 80,
        "layer_final_thickness": 0.15,
    },
    "medium": {
        "block_scale": 1.0,
        "surface_level_min_delta": 0,
        "surface_level_max_delta": 0,
        "region_level_delta": 0,
        "layer_count": 3,
        "layer_expansion_ratio": 1.12,
        "layer_first_thickness": 0.005,
        "layer_min_thickness": 0.0005,
        "layer_feature_angle": 150,
        "layer_nGrow": 1,
        "layer_max_face_thickness_ratio": 0.5,
        "layer_max_thickness_to_medial_ratio": 0.3,
        "layer_nSmoothSurfaceNormals": 8,
        "layer_nSmoothThickness": 20,
        "layer_min_medial_axis_angle": 65,
        "layer_nSmoothNormals": 8,
        "layer_nRelaxIter": 10,
        "layer_nBufferCellsNoExtrude": 2,
        "layer_nLayerIter": 80,
        "layer_nRelaxedIter": 80,
        "layer_final_thickness": 0.15,
    },
    "fine": {
        "block_scale": 1.15,
        "surface_level_min_delta": 0,
        "surface_level_max_delta": 1,
        "region_level_delta": 0,
        "layer_count": 3,
        "layer_expansion_ratio": 1.12,
        "layer_first_thickness": 0.005,
        "layer_min_thickness": 0.0005,
        "layer_feature_angle": 150,
        "layer_nGrow": 1,
        "layer_max_face_thickness_ratio": 0.5,
        "layer_max_thickness_to_medial_ratio": 0.3,
        "layer_nSmoothSurfaceNormals": 8,
        "layer_nSmoothThickness": 20,
        "layer_min_medial_axis_angle": 65,
        "layer_nSmoothNormals": 8,
        "layer_nRelaxIter": 10,
        "layer_nBufferCellsNoExtrude": 2,
        "layer_nLayerIter": 80,
        "layer_nRelaxedIter": 80,
        "layer_final_thickness": 0.15,
    },
    "extra_fine": {
        "block_scale": 1.0,
        "surface_level_min_delta": 0,
        "surface_level_max_delta": 0,
        "region_level_delta": 1,
        "layer_count": 3,
        "layer_expansion_ratio": 1.12,
        "layer_first_thickness": 0.005,
        "layer_min_thickness": 0.0005,
        "layer_feature_angle": 150,
        "layer_nGrow": 1,
        "layer_max_face_thickness_ratio": 0.5,
        "layer_max_thickness_to_medial_ratio": 0.3,
        "layer_nSmoothSurfaceNormals": 8,
        "layer_nSmoothThickness": 20,
        "layer_min_medial_axis_angle": 65,
        "layer_nSmoothNormals": 8,
        "layer_nRelaxIter": 10,
        "layer_nBufferCellsNoExtrude": 2,
        "layer_nLayerIter": 80,
        "layer_nRelaxedIter": 80,
        "layer_final_thickness": 0.15,
        "target_cells_note": "~5-10M cells targeted with one extra volume-refinement level beyond baseline while keeping the baseline surface recipe; verify with a mesh-only run before solving.",
    },
}

DEFAULT_PRESET_ORDER = ["coarse", "medium", "fine", "extra_fine"]
GENERIC_PATCHES = {"xMin", "xMax", "yMin", "yMax", "zMin", "zMax"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return slug.strip("_").lower() or "study"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_bundle(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def format_float(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.12g}"


def format_vector(vector: tuple[float, float, float]) -> str:
    return f"({format_float(vector[0])} {format_float(vector[1])} {format_float(vector[2])})"


def freestream_from_alpha(alpha_deg: float, speed: float) -> tuple[float, float, float]:
    alpha_rad = math.radians(alpha_deg)
    return (speed * math.cos(alpha_rad), 0.0, speed * math.sin(alpha_rad))


def drag_lift_dirs_from_alpha(alpha_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    alpha_rad = math.radians(alpha_deg)
    drag_dir = (math.cos(alpha_rad), 0.0, math.sin(alpha_rad))
    lift_dir = (-math.sin(alpha_rad), 0.0, math.cos(alpha_rad))
    return drag_dir, lift_dir


def replace_uniform_scalar(text: str, value: float) -> str:
    return re.sub(
        r"(internalField\s+uniform\s+)([-+0-9.eE]+)(\s*;)",
        lambda match: f"{match.group(1)}{format_float(value)}{match.group(3)}",
        text,
        count=1,
    )


def replace_uniform_vector(text: str, vector: tuple[float, float, float]) -> str:
    return re.sub(
        r"(internalField\s+uniform\s+)\([^)]+\)(\s*;)",
        lambda match: f"{match.group(1)}{format_vector(vector)}{match.group(2)}",
        text,
        count=1,
    )


def find_named_block(text: str, name: str) -> tuple[int, int]:
    match = re.search(rf"(^|\s){re.escape(name)}\s*\{{", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find block '{name}'")
    name_index = match.start()

    cursor = match.end() - 1
    while cursor > name_index and text[cursor] != "{":
        cursor -= 1
    if text[cursor] != "{":
        raise ValueError(f"Could not locate opening brace for '{name}'")

    depth = 0
    end = cursor
    while end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return name_index, end + 1
        end += 1
    raise ValueError(f"Unbalanced block '{name}'")


def replace_named_block(text: str, name: str, new_block: str) -> str:
    start, end = find_named_block(text, name)
    return text[:start] + new_block + text[end:]


def replace_patch_block(text: str, name: str, new_block: str) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(name)}\s*\{{.*?^\s*\}})", flags=re.MULTILINE | re.DOTALL)
    result, count = pattern.subn(new_block, text, count=1)
    if count == 0:
        raise ValueError(f"Could not replace patch block '{name}'")
    return result


def parse_xmin_speed_from_u_text(text: str) -> float | None:
    vector_match = re.search(
        r"xMin\s*\{.*?\bvalue\s+uniform\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if vector_match:
        ux, uy, uz = (float(vector_match.group(i)) for i in range(1, 4))
        return math.sqrt(ux * ux + uy * uy + uz * uz)

    scalar_match = re.search(
        r"xMin\s*\{.*?\brefValue\s+uniform\s+([-+0-9.eE]+)\s*;",
        text,
        flags=re.DOTALL,
    )
    if scalar_match:
        return abs(float(scalar_match.group(1)))
    return None


def patch_u_file(path: Path, *, velocity: float, alpha_deg: float) -> dict[str, Any]:
    text = path.read_text()
    base_speed = parse_xmin_speed_from_u_text(text) or velocity
    vector = freestream_from_alpha(alpha_deg, velocity)
    x_min_block = (
        "    xMin\n"
        "    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {format_vector(vector)};\n"
        "    }"
    )
    x_max_block = (
        "    xMax\n"
        "    {\n"
        "        type            pressureInletOutletVelocity;\n"
        f"        value           uniform {format_vector(vector)};\n"
        "    }"
    )
    text = replace_uniform_vector(text, vector)
    text = replace_patch_block(text, "xMin", x_min_block)
    text = replace_patch_block(text, "xMax", x_max_block)
    path.write_text(text)
    return {
        "path": str(path),
        "base_speed_m_s": base_speed,
        "target_speed_m_s": velocity,
        "alpha_deg": alpha_deg,
        "velocity_vector": vector,
    }


def patch_scalar_field_with_speed_ratio(path: Path, *, speed_ratio: float) -> dict[str, Any]:
    text = path.read_text()
    field_name = path.name.lower()
    exponent = 1.0
    if field_name == "k":
        exponent = 2.0
    elif field_name == "epsilon":
        exponent = 3.0
    elif field_name == "nut":
        exponent = 1.0

    ratio = speed_ratio**exponent

    internal_match = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if internal_match:
        value = float(internal_match.group(1)) * ratio
        text = replace_uniform_scalar(text, value)

    x_min_match = re.search(r"xMin\s*\{.*?\bvalue\s+uniform\s+([-+0-9.eE]+)\s*;", text, flags=re.DOTALL)
    if x_min_match:
        value = float(x_min_match.group(1)) * ratio
        block_text = re.search(r"xMin\s*\{.*?\}", text, flags=re.DOTALL)
        if block_text:
            x_min_block = re.sub(
                r"(\bvalue\s+uniform\s+)([-+0-9.eE]+)(\s*;)",
                lambda match: f"{match.group(1)}{format_float(value)}{match.group(3)}",
                block_text.group(0),
                count=1,
            )
            text = replace_patch_block(text, "xMin", x_min_block)

    path.write_text(text)
    return {
        "path": str(path),
        "speed_ratio": speed_ratio,
        "scaling_exponent": exponent,
    }


def patch_force_coeffs(control_dict_path: Path, *, velocity: float, alpha_deg: float) -> dict[str, Any]:
    text = control_dict_path.read_text()
    drag_dir, lift_dir = drag_lift_dirs_from_alpha(alpha_deg)

    def replace_scalar(entry: str, value: float, source: str) -> str:
        pattern = rf"(\b{re.escape(entry)}\s+)([-+0-9.eE]+)(\s*;)"
        result, count = re.subn(
            pattern,
            lambda match: f"{match.group(1)}{format_float(value)}{match.group(3)}",
            source,
            count=1,
        )
        if count == 0:
            raise ValueError(f"Could not patch {entry} in {control_dict_path}")
        return result

    def replace_vector(entry: str, vector: tuple[float, float, float], source: str) -> str:
        pattern = rf"(\b{re.escape(entry)}\s+)\([^)]+\)(\s*;)"
        result, count = re.subn(
            pattern,
            lambda match: f"{match.group(1)}{format_vector(vector)}{match.group(2)}",
            source,
            count=1,
        )
        if count == 0:
            raise ValueError(f"Could not patch {entry} in {control_dict_path}")
        return result

    text = replace_scalar("magUInf", velocity, text)
    text = replace_vector("dragDir", drag_dir, text)
    text = replace_vector("liftDir", lift_dir, text)
    control_dict_path.write_text(text)
    return {
        "path": str(control_dict_path),
        "magUInf": velocity,
        "dragDir": drag_dir,
        "liftDir": lift_dir,
    }


def patch_fvsolution_residual_controls(
    fv_solution_path: Path,
    *,
    residual_tolerance: float,
) -> dict[str, Any]:
    text = fv_solution_path.read_text()
    tolerance_text = format_float(residual_tolerance)

    def patch_simple_block(source: str) -> str:
        simple_start, simple_end = find_named_block(source, "SIMPLE")
        simple_block = source[simple_start:simple_end]
        residual_start, residual_end = find_named_block(simple_block, "residualControl")
        residual_block = simple_block[residual_start:residual_end]
        residual_body_start = residual_block.find("{") + 1
        residual_body_end = residual_block.rfind("}")
        residual_body = residual_block[residual_body_start:residual_body_end]
        residual_body = re.sub(
            r"(^\s*\"?[A-Za-z0-9|()._*]+\"?\s+)([-+0-9.eE]+)(\s*;)",
            lambda match: f"{match.group(1)}{tolerance_text}{match.group(3)}",
            residual_body,
            flags=re.MULTILINE,
        )
        patched_residual = residual_block[:residual_body_start] + residual_body + residual_block[residual_body_end:]
        return simple_block[:residual_start] + patched_residual + simple_block[residual_end:]

    def patch_nested_residual_block(source: str, parent_name: str) -> str:
        parent_start, parent_end = find_named_block(source, parent_name)
        parent_block = source[parent_start:parent_end]
        residual_start, residual_end = find_named_block(parent_block, "residualControl")
        residual_block = parent_block[residual_start:residual_end]
        patched_residual = re.sub(
            r"(\btolerance\s+)([-+0-9.eE]+)(\s*;)",
            lambda match: f"{match.group(1)}{tolerance_text}{match.group(3)}",
            residual_block,
        )
        if parent_name == "LU-SGS":
            patched_residual = re.sub(
                r"(^\s*\"?[A-Za-z0-9|()._*]+\"?\s+)([-+0-9.eE]+)(\s*;)",
                lambda match: f"{match.group(1)}{tolerance_text}{match.group(3)}",
                patched_residual,
                flags=re.MULTILINE,
            )
        return parent_block[:residual_start] + patched_residual + parent_block[residual_end:]

    for parent_name, patcher in (
        ("SIMPLE", patch_simple_block),
        ("PIMPLE", lambda src: patch_nested_residual_block(src, "PIMPLE")),
        ("LU-SGS", lambda src: patch_nested_residual_block(src, "LU-SGS")),
    ):
        if re.search(rf"(^|\s){re.escape(parent_name)}\s*\{{", text, flags=re.MULTILINE):
            patched_parent = patcher(text)
            parent_start, parent_end = find_named_block(text, parent_name)
            text = text[:parent_start] + patched_parent + text[parent_end:]

    fv_solution_path.write_text(text)
    return {
        "path": str(fv_solution_path),
        "residual_tolerance": residual_tolerance,
    }


def patch_flow_condition(case_dir: Path, *, velocity: float, alpha_deg: float) -> dict[str, Any]:
    u_path = case_dir / "0" / "U"
    boundary_u_path = case_dir / "0" / "boundaryFields" / "U"
    control_dict_path = case_dir / "system" / "controlDict"
    fv_solution_path = case_dir / "system" / "fvSolution"

    u_summary = patch_u_file(u_path, velocity=velocity, alpha_deg=alpha_deg)
    boundary_u_summary = patch_u_file(boundary_u_path, velocity=velocity, alpha_deg=alpha_deg) if boundary_u_path.exists() else None

    base_speed = float(u_summary["base_speed_m_s"])
    speed_ratio = velocity / base_speed if base_speed else 1.0
    scalar_summaries = []
    for field_name in ("k", "epsilon", "nut"):
        field_path = case_dir / "0" / field_name
        if field_path.exists():
            scalar_summaries.append(patch_scalar_field_with_speed_ratio(field_path, speed_ratio=speed_ratio))

    coeff_summary = patch_force_coeffs(control_dict_path, velocity=velocity, alpha_deg=alpha_deg)
    fv_solution_summary = (
        patch_fvsolution_residual_controls(fv_solution_path, residual_tolerance=1e-6)
        if fv_solution_path.exists()
        else None
    )
    return {
        "u": u_summary,
        "boundary_u": boundary_u_summary,
        "speed_ratio": speed_ratio,
        "scaled_scalar_fields": scalar_summaries,
        "force_coeffs": coeff_summary,
        "fv_solution": fv_solution_summary,
    }


def parse_block_mesh_cells(text: str) -> tuple[int, int, int]:
    match = re.search(r"\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s+simpleGrading", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse blockMesh cell counts.")
    return tuple(int(match.group(i)) for i in range(1, 4))


def patch_block_mesh_cells(path: Path, *, scale: float) -> tuple[int, int, int]:
    text = path.read_text()
    base_cells = parse_block_mesh_cells(text)
    new_cells = tuple(max(1, int(round(value * scale))) for value in base_cells)
    replacement = f"({new_cells[0]} {new_cells[1]} {new_cells[2]})  simpleGrading"
    patched, count = re.subn(
        r"\(\s*\d+\s+\d+\s+\d+\s*\)\s+simpleGrading",
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0:
        raise ValueError(f"Could not patch blockMesh cells in {path}")
    if patched != text:
        path.write_text(patched)
    return new_cells


def parse_surface_levels(text: str) -> tuple[str, int, int]:
    match = re.search(
        r"([A-Za-z0-9_]+)\s*\{\s*patchInfo\s*\{.*?\}\s*level\s*\(\s*(\d+)\s+(\d+)\s*\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not parse refinementSurfaces level tuple.")
    return match.group(1), int(match.group(2)), int(match.group(3))


def parse_region_level(text: str) -> tuple[str, int]:
    match = re.search(
        r"([A-Za-z0-9_]+)\s*\{\s*mode\s+inside\s*;\s*levels\s*\(\s*\(\s*[-+0-9.eE]+\s+(\d+)\s*\)\s*\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not parse refinementRegions level.")
    return match.group(1), int(match.group(2))


def parse_layer_patch(text: str) -> tuple[str, int]:
    match = re.search(r"([A-Za-z0-9_]+)\s*\{\s*nSurfaceLayers\s+(\d+)\s*;", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse addLayersControls block.")
    return match.group(1), int(match.group(2))


def patch_snappy_recipe(
    path: Path,
    *,
    surface_level_min_delta: int,
    surface_level_max_delta: int,
    region_level_delta: int,
    layer_count: int,
    layer_expansion_ratio: float = 1.15,
    layer_first_thickness: float = 0.02,
    layer_min_thickness: float = 0.01,
    layer_feature_angle: float = 130.0,
    layer_n_grow: int = 0,
    layer_max_face_thickness_ratio: float = 0.7,
    layer_max_thickness_to_medial_ratio: float = 0.5,
    layer_n_smooth_surface_normals: int = 3,
    layer_n_smooth_thickness: int = 15,
    layer_min_medial_axis_angle: float = 70.0,
    layer_n_smooth_normals: int = 6,
    layer_n_relax_iter: int = 15,
    layer_n_buffer_cells_no_extrude: int = 0,
    layer_n_layer_iter: int = 100,
    layer_n_relaxed_iter: int = 40,
    layer_final_thickness: float = 0.4,
) -> dict[str, Any]:
    text = path.read_text()
    surface_name, surface_min, surface_max = parse_surface_levels(text)
    region_name, region_level = parse_region_level(text)
    layer_patch_name, current_layers = parse_layer_patch(text)

    new_surface_min = max(0, surface_min + surface_level_min_delta)
    new_surface_max = max(new_surface_min, surface_max + surface_level_max_delta)
    new_region_level = max(0, region_level + region_level_delta)
    new_layer_count = max(0, layer_count)

    text, surface_count = re.subn(
        r"(level\s*\(\s*)(\d+)(\s+)(\d+)(\s*\)\s*;)",
        lambda match: f"{match.group(1)}{new_surface_min}{match.group(3)}{new_surface_max}{match.group(5)}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if surface_count == 0:
        raise ValueError(f"Could not patch refinement surface levels in {path}")

    text, region_count = re.subn(
        r"(\(\s*[-+0-9.eE]+\s+)(\d+)(\s*\)\s*\)\s*;)",
        lambda match: f"{match.group(1)}{new_region_level}{match.group(3)}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if region_count == 0:
        raise ValueError(f"Could not patch refinement region level in {path}")

    add_layers_block = (
        "addLayersControls\n"
        "{\n"
        "    layers\n"
        "    {\n"
        f"        {layer_patch_name}\n"
        "        {\n"
        f"            nSurfaceLayers  {new_layer_count};\n"
        "            thicknessModel  firstAndExpansion;\n"
        "            relativeSizes   on;\n"
        f"            expansionRatio  {format_float(layer_expansion_ratio)};\n"
        f"            firstLayerThickness {format_float(layer_first_thickness)};\n"
        f"            minThickness    {format_float(layer_min_thickness)};\n"
        "        }\n"
        "    }\n"
        f"    nGrow           {layer_n_grow};\n"
        f"    featureAngle    {format_float(layer_feature_angle)};\n"
        f"    maxFaceThicknessRatio {format_float(layer_max_face_thickness_ratio)};\n"
        f"    nSmoothSurfaceNormals {layer_n_smooth_surface_normals};\n"
        f"    nSmoothThickness {layer_n_smooth_thickness};\n"
        f"    minMedialAxisAngle {format_float(layer_min_medial_axis_angle)};\n"
        f"    maxThicknessToMedialRatio {format_float(layer_max_thickness_to_medial_ratio)};\n"
        f"    nSmoothNormals  {layer_n_smooth_normals};\n"
        f"    nRelaxIter      {layer_n_relax_iter};\n"
        f"    nBufferCellsNoExtrude {layer_n_buffer_cells_no_extrude};\n"
        f"    nLayerIter      {layer_n_layer_iter};\n"
        f"    nRelaxedIter    {layer_n_relaxed_iter};\n"
        "    thicknessModel  firstAndExpansion;\n"
        "    relativeSizes   on;\n"
        f"    expansionRatio  {format_float(layer_expansion_ratio)};\n"
        f"    firstLayerThickness {format_float(layer_first_thickness)};\n"
        f"    minThickness    {format_float(layer_min_thickness)};\n"
        "}"
    )
    text = replace_named_block(text, "addLayersControls", add_layers_block)

    path.write_text(text)
    return {
        "surface_patch": surface_name,
        "surface_levels": [new_surface_min, new_surface_max],
        "region_name": region_name,
        "region_level": new_region_level,
        "layer_patch": layer_patch_name,
        "layer_count": new_layer_count,
        "layer_expansion_ratio": layer_expansion_ratio,
        "layer_first_thickness": layer_first_thickness,
        "layer_min_thickness": layer_min_thickness,
        "layer_feature_angle": layer_feature_angle,
        "layer_nGrow": layer_n_grow,
        "layer_max_face_thickness_ratio": layer_max_face_thickness_ratio,
        "layer_max_thickness_to_medial_ratio": layer_max_thickness_to_medial_ratio,
        "layer_nSmoothSurfaceNormals": layer_n_smooth_surface_normals,
        "layer_nSmoothThickness": layer_n_smooth_thickness,
        "layer_min_medial_axis_angle": layer_min_medial_axis_angle,
        "layer_nSmoothNormals": layer_n_smooth_normals,
        "layer_nRelaxIter": layer_n_relax_iter,
        "layer_nBufferCellsNoExtrude": layer_n_buffer_cells_no_extrude,
        "layer_nLayerIter": layer_n_layer_iter,
        "layer_nRelaxedIter": layer_n_relaxed_iter,
        "layer_final_thickness": layer_final_thickness,
        "baseline_surface_levels": [surface_min, surface_max],
        "baseline_region_level": region_level,
        "baseline_layer_count": current_layers,
    }


def apply_mesh_preset(case_dir: Path, preset: dict[str, Any]) -> dict[str, Any]:
    block_mesh_path = case_dir / "system" / "blockMeshDict"
    snappy_path = case_dir / "system" / "snappyHexMeshDict"

    block_cells = patch_block_mesh_cells(block_mesh_path, scale=float(preset["block_scale"]))
    snappy_summary = patch_snappy_recipe(
        snappy_path,
        surface_level_min_delta=int(preset.get("surface_level_min_delta", 0)),
        surface_level_max_delta=int(preset.get("surface_level_max_delta", 0)),
        region_level_delta=int(preset.get("region_level_delta", 0)),
        layer_count=int(preset.get("layer_count", 0)),
        layer_expansion_ratio=float(preset.get("layer_expansion_ratio", 1.15)),
        layer_first_thickness=float(preset.get("layer_first_thickness", 0.02)),
        layer_min_thickness=float(preset.get("layer_min_thickness", 0.01)),
        layer_feature_angle=float(preset.get("layer_feature_angle", 130.0)),
        layer_n_grow=int(preset.get("layer_nGrow", 0)),
        layer_max_face_thickness_ratio=float(preset.get("layer_max_face_thickness_ratio", 0.7)),
        layer_max_thickness_to_medial_ratio=float(preset.get("layer_max_thickness_to_medial_ratio", 0.5)),
        layer_n_smooth_surface_normals=int(preset.get("layer_nSmoothSurfaceNormals", 3)),
        layer_n_smooth_thickness=int(preset.get("layer_nSmoothThickness", 15)),
        layer_min_medial_axis_angle=float(preset.get("layer_min_medial_axis_angle", 70.0)),
        layer_n_smooth_normals=int(preset.get("layer_nSmoothNormals", 6)),
        layer_n_relax_iter=int(preset.get("layer_nRelaxIter", 15)),
        layer_n_buffer_cells_no_extrude=int(preset.get("layer_nBufferCellsNoExtrude", 0)),
        layer_n_layer_iter=int(preset.get("layer_nLayerIter", 100)),
        layer_n_relaxed_iter=int(preset.get("layer_nRelaxedIter", 40)),
        layer_final_thickness=float(preset.get("layer_final_thickness", 0.4)),
    )
    return {
        "block_scale": float(preset["block_scale"]),
        "block_cells": list(block_cells),
        "snappy": snappy_summary,
    }


def parse_owner_cell_count(owner_path: Path) -> int | None:
    if not owner_path.exists():
        return None
    text = owner_path.read_text(errors="ignore")
    match = re.search(r"note\s+\"[^\"]*nCells:(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def parse_mesh_log_cell_count(case_dir: Path) -> int | None:
    log_candidates = [
        case_dir / "mesh_automation_layers_stdout.log",
        case_dir / "mesh_automation_snap_stdout.log",
        case_dir / "mesh_automation_castellated_stdout.log",
    ]
    for log_path in log_candidates:
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="ignore")
        matches = re.findall(r"mesh\s*:\s*cells:(\d+)", text)
        if matches:
            return int(matches[-1])
    return None


def summarize_variant_result(
    *,
    variant: dict[str, Any],
    postprocess_summary_path: Path | None,
    mesh_owner_path: Path | None,
) -> dict[str, Any]:
    case_dir = Path(str(variant.get("case_dir", ""))).resolve() if variant.get("case_dir") else None
    cell_count = parse_mesh_log_cell_count(case_dir) if case_dir else None
    if cell_count is None and mesh_owner_path:
        cell_count = parse_owner_cell_count(mesh_owner_path)
    result: dict[str, Any] = {
        "name": variant["name"],
        "condition": variant["condition"],
        "mesh_preset": variant["mesh_preset"],
        "cell_count": cell_count,
    }
    if postprocess_summary_path and postprocess_summary_path.exists():
        summary = load_json(postprocess_summary_path)
        scaled = summary.get("scaled_last_average", {})
        result.update(
            {
                "summary_path": str(postprocess_summary_path),
                "Cd": scaled.get("Cd"),
                "Cl": scaled.get("Cl"),
                "L_over_D": scaled.get("L_over_D"),
                "drag_full_N": scaled.get("drag_full_N"),
                "lift_full_N": scaled.get("lift_full_N"),
            }
        )
    return result
