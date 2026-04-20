from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract readable configuration text from BARAM .bf/.bm bundles.")
    parser.add_argument("--bundle-root", required=True, help="Path to a .bf or .bm bundle.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted configuration files.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the bundle stem.")
    parser.add_argument("--h5dump", help="Path to h5dump.exe. Defaults to PATH lookup.")
    return parser.parse_args()


def resolve_h5dump(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).resolve())
    discovered = shutil.which("h5dump")
    if discovered:
        return discovered
    fallback = Path(r"C:\ProgramData\anaconda3\Library\bin\h5dump.exe")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Could not locate h5dump.exe. Supply --h5dump explicitly.")


def dump_dataset(h5dump_path: str, file_path: Path, dataset: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        completed = subprocess.run(
            [h5dump_path, "-y", "-w", "0", "-o", str(tmp_path), "-d", dataset, str(file_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"h5dump failed for {file_path} {dataset}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_h5dump_scalar_string(text: str) -> str:
    first_quote = text.find('"')
    last_quote = text.rfind('"')
    if first_quote == -1 or last_quote == -1 or last_quote <= first_quote:
        raise ValueError("Could not locate quoted scalar string in h5dump output.")
    return text[first_quote + 1 : last_quote]


def parse_flow_xml(xml_text: str) -> dict[str, object]:
    root = ET.fromstring(xml_text)
    ns = {"b": "http://www.baramcfd.org/baram"}

    def find_text(path: str) -> str | None:
        node = root.find(path, ns)
        return None if node is None or node.text is None else node.text.strip()

    force_names = []
    for node in root.findall(".//b:monitors/b:forces/b:forceMonitor/b:name", ns):
        if node.text:
            force_names.append(node.text.strip())

    surface_names = []
    for node in root.findall(".//b:monitors/b:surfaces/b:surfaceMonitor/b:name", ns):
        if node.text:
            surface_names.append(node.text.strip())

    return {
        "flow_type": find_text("./b:general/b:flowType"),
        "solver_type": find_text("./b:general/b:solverType"),
        "time_transient": find_text("./b:general/b:timeTransient"),
        "reference_area": find_text("./b:referenceValues/b:area"),
        "reference_density": find_text("./b:referenceValues/b:density"),
        "reference_length": find_text("./b:referenceValues/b:length"),
        "reference_velocity": find_text("./b:referenceValues/b:velocity"),
        "turbulence_model": find_text("./b:models/b:turbulenceModels/b:model"),
        "pressure_velocity_coupling": find_text("./b:numericalConditions/b:pressureVelocityCouplingScheme"),
        "max_iterations_per_time_step": find_text("./b:numericalConditions/b:maxIterationsPerTimeStep"),
        "number_of_iterations": find_text("./b:runCalculation/b:runConditions/b:numberOfIterations"),
        "parallel_cores": find_text("./b:runCalculation/b:parallel/b:numberOfCores"),
        "force_monitors": force_names,
        "surface_monitors": surface_names,
    }


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or bundle_root.stem
    h5dump_path = resolve_h5dump(args.h5dump)

    outputs: dict[str, object] = {"bundle_root": str(bundle_root)}

    configuration_h5 = bundle_root / "configuration.h5"
    if configuration_h5.exists():
        raw = dump_dataset(h5dump_path, configuration_h5, "/configuration")
        xml_text = parse_h5dump_scalar_string(raw)
        xml_path = output_dir / f"{prefix}_configuration.xml"
        xml_path.write_text(xml_text)
        outputs["configuration_xml"] = str(xml_path)
        try:
            summary = parse_flow_xml(xml_text)
        except Exception as exc:
            summary = {"parse_error": str(exc)}
        summary_path = output_dir / f"{prefix}_configuration_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        outputs["configuration_summary"] = str(summary_path)

    configurations_h5 = bundle_root / "configurations.h5"
    if configurations_h5.exists():
        raw = dump_dataset(h5dump_path, configurations_h5, "/configurations")
        config_text = parse_h5dump_scalar_string(raw)
        config_path = output_dir / f"{prefix}_mesh_configurations.yaml"
        config_path.write_text(config_text)
        outputs["mesh_configurations"] = str(config_path)

    manifest_path = output_dir / f"{prefix}_extracted_files.json"
    manifest_path.write_text(json.dumps(outputs, indent=2))
    print(f"Wrote extraction manifest: {manifest_path}")


if __name__ == "__main__":
    main()
