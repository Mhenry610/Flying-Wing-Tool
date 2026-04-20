from __future__ import annotations

import argparse
import json
from pathlib import Path

from baram_grid_study_tools import DEFAULT_PRESET_ORDER, dump_json, load_json, summarize_variant_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare solved variants from a BARAM grid-independence study.")
    parser.add_argument("--manifest", required=True, help="Path to a grid_study_manifest.json file.")
    parser.add_argument(
        "--tolerance-percent",
        type=float,
        default=1.0,
        help="Heuristic medium-to-fine relative-change tolerance for declaring practical independence.",
    )
    return parser.parse_args()


def relative_change_percent(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if abs(a) < 1e-12:
        return None
    return 100.0 * (b - a) / a


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Variant",
        "Cells",
        "Cd",
        "Cl",
        "L/D",
        "dCd prev [%]",
        "dCl prev [%]",
        "dL/D prev [%]",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_cell(row.get("name")),
                    format_cell(row.get("cell_count")),
                    format_cell(row.get("Cd")),
                    format_cell(row.get("Cl")),
                    format_cell(row.get("L_over_D")),
                    format_cell(row.get("delta_prev_Cd_percent")),
                    format_cell(row.get("delta_prev_Cl_percent")),
                    format_cell(row.get("delta_prev_L_over_D_percent")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    report_output_dir = Path(str(manifest["report_output_dir"])).resolve()
    report_output_dir.mkdir(parents=True, exist_ok=True)

    preset_order = list(manifest.get("mesh_presets", {}).keys()) or DEFAULT_PRESET_ORDER
    preset_rank = {name: index for index, name in enumerate(preset_order)}

    grouped: dict[str, list[dict[str, object]]] = {}
    for item in manifest.get("variants", []):
        if not isinstance(item, dict):
            continue
        workspace_root = Path(str(item["workspace_root"])).resolve()
        bundle_root = workspace_root / str(item["bundle_name"])
        case_dir = bundle_root / "case"
        summary_path = report_output_dir / f"{item['name']}_summary.json"
        owner_path = case_dir / "constant" / "polyMesh" / "owner"
        grouped.setdefault(str(item["condition"]), []).append(
            summarize_variant_result(
                variant=item,
                postprocess_summary_path=summary_path if summary_path.exists() else None,
                mesh_owner_path=owner_path if owner_path.exists() else None,
            )
        )

    comparison: dict[str, object] = {
        "study_name": manifest["study_name"],
        "report_output_dir": str(report_output_dir),
        "conditions": {},
        "tolerance_percent": args.tolerance_percent,
    }

    markdown_parts = [f"# Grid Study Comparison: {manifest['study_name']}\n"]
    for condition_name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: preset_rank.get(str(row.get("mesh_preset")), 999))
        previous = None
        for row in rows:
            row["delta_prev_Cd_percent"] = relative_change_percent(previous.get("Cd"), row.get("Cd")) if previous else None
            row["delta_prev_Cl_percent"] = relative_change_percent(previous.get("Cl"), row.get("Cl")) if previous else None
            row["delta_prev_L_over_D_percent"] = (
                relative_change_percent(previous.get("L_over_D"), row.get("L_over_D")) if previous else None
            )
            previous = row

        medium = next((row for row in rows if row.get("mesh_preset") == "medium"), None)
        fine = next((row for row in rows if row.get("mesh_preset") == "fine"), None)
        independence = None
        if medium and fine:
            cd_delta = abs(relative_change_percent(medium.get("Cd"), fine.get("Cd")) or 0.0)
            cl_delta = abs(relative_change_percent(medium.get("Cl"), fine.get("Cl")) or 0.0)
            ld_delta = abs(relative_change_percent(medium.get("L_over_D"), fine.get("L_over_D")) or 0.0)
            independence = {
                "medium_to_fine_abs_delta_percent": {
                    "Cd": cd_delta,
                    "Cl": cl_delta,
                    "L_over_D": ld_delta,
                },
                "passes_tolerance": cd_delta <= args.tolerance_percent and cl_delta <= args.tolerance_percent,
                "note": "Heuristic pass uses Cd and Cl only; boundary-layer independence is still separate if layers are not being added.",
            }

        comparison["conditions"][condition_name] = {
            "rows": rows,
            "independence": independence,
        }

        markdown_parts.append(f"## {condition_name.capitalize()}\n")
        markdown_parts.append(build_markdown_table(rows))
        if independence:
            markdown_parts.append(
                "Medium-to-fine absolute deltas: "
                f"Cd={independence['medium_to_fine_abs_delta_percent']['Cd']:.3f}%, "
                f"Cl={independence['medium_to_fine_abs_delta_percent']['Cl']:.3f}%, "
                f"L/D={independence['medium_to_fine_abs_delta_percent']['L_over_D']:.3f}%. "
                f"Pass={independence['passes_tolerance']}.\n"
            )

    json_path = report_output_dir / f"{manifest['study_name']}_grid_comparison.json"
    md_path = report_output_dir / f"{manifest['study_name']}_grid_comparison.md"
    dump_json(json_path, comparison)
    md_path.write_text("\n".join(markdown_parts).strip() + "\n")
    print(f"Wrote comparison JSON: {json_path}")
    print(f"Wrote comparison Markdown: {md_path}")


if __name__ == "__main__":
    main()
