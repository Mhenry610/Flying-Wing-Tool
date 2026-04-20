from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from baram_case_tools import make_slug, update_pvsm_case_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local BARAM study workspace.")
    parser.add_argument("--source-bf", required=True, help="Path to the source .bf bundle.")
    parser.add_argument("--source-bm", help="Optional path to the source .bm bundle.")
    parser.add_argument("--source-pvsm", help="Optional path to a ParaView state.")
    parser.add_argument("--study-root", required=True, help="Directory where the study should be created.")
    parser.add_argument("--name", required=True, help="Study name.")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant name. Repeat for multiple variants. Defaults to a single baseline variant.",
    )
    parser.add_argument("--summary-json", help="Optional project summary JSON to copy into the study manifest.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing variant directory under the study root.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Register an existing variant directory in the manifest without recopying files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files.")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.copytree(src, dst)


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def remove_tree(path: Path, dry_run: bool) -> None:
    if dry_run or not path.exists():
        return
    shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    source_bf = Path(args.source_bf).resolve()
    source_bm = Path(args.source_bm).resolve() if args.source_bm else None
    source_pvsm = Path(args.source_pvsm).resolve() if args.source_pvsm else None
    study_root = Path(args.study_root).resolve()
    study_name = make_slug(args.name)
    variants = args.variants or ["baseline"]
    manifest_path = study_root / study_name / "study_manifest.json"

    manifest: dict[str, object] = {
        "study_name": study_name,
        "source_bf": str(source_bf),
        "source_bm": str(source_bm) if source_bm else None,
        "source_pvsm": str(source_pvsm) if source_pvsm else None,
        "variants": [],
        "notes": [
            "Prepared by baram_prepare_study.py",
            "This scaffold does not yet patch BARAM GUI HDF configuration files.",
            "Use it to clone case bundles, localize ParaView state paths, and track validation variants.",
        ],
    }
    if args.summary_json:
        manifest["summary_json"] = str(Path(args.summary_json).resolve())

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("study_name") == study_name:
            existing_variants = existing.get("variants", [])
            if isinstance(existing_variants, list):
                manifest["variants"] = [variant for variant in existing_variants if isinstance(variant, dict)]
            for key in ("notes", "summary_json"):
                if key in existing and key not in manifest:
                    manifest[key] = existing[key]

    for variant in variants:
        variant_slug = make_slug(variant)
        variant_root = study_root / study_name / variant_slug
        variant_bf = variant_root / source_bf.name
        variant_bm = variant_root / source_bm.name if source_bm else None
        if source_pvsm:
            pvsm_name = source_pvsm.name
            if not pvsm_name.endswith("_local.pvsm"):
                pvsm_name = source_pvsm.stem + "_local" + source_pvsm.suffix
            variant_pvsm = variant_root / pvsm_name
        else:
            variant_pvsm = variant_root / f"{source_bf.stem}_{variant_slug}.pvsm"

        print(f"[prepare] variant={variant_slug}")
        print(f"  bf:   {variant_bf}")
        if variant_bm:
            print(f"  bm:   {variant_bm}")
        if variant_pvsm:
            print(f"  pvsm: {variant_pvsm}")

        reuse_existing = variant_root.exists() and args.reuse_existing
        if variant_root.exists():
            if reuse_existing:
                pass
            elif not args.overwrite:
                raise FileExistsError(
                    f"Variant directory already exists: {variant_root}. "
                    "Pass --overwrite to replace it or --reuse-existing to register it as-is."
                )
            else:
                remove_tree(variant_root, args.dry_run)

        if not args.dry_run and not reuse_existing:
            variant_root.mkdir(parents=True, exist_ok=True)
        if not reuse_existing:
            copy_tree(source_bf, variant_bf, args.dry_run)
        if source_bm:
            if not reuse_existing:
                copy_tree(source_bm, variant_bm, args.dry_run)

        if source_pvsm and not reuse_existing:
            pvsm_text = source_pvsm.read_text()
            new_case_foam = str((variant_bf / "case" / "case.foam").resolve())
            localized_text = update_pvsm_case_path(pvsm_text, new_case_foam)
            write_text(variant_pvsm, localized_text, args.dry_run)

        variant_record = {
            "name": variant_slug,
            "root": str(variant_root),
            "bf_bundle": str(variant_bf),
            "bm_bundle": str(variant_bm) if variant_bm else None,
            "pvsm": str(variant_pvsm) if source_pvsm else None,
            "case_dir": str((variant_bf / "case").resolve()),
            "status": "prepared" if not args.dry_run else "dry-run",
        }
        existing_variants = [item for item in manifest["variants"] if item.get("name") != variant_slug]
        existing_variants.append(variant_record)
        existing_variants.sort(key=lambda item: item.get("name", ""))
        manifest["variants"] = existing_variants

    manifest_text = json.dumps(manifest, indent=2)
    write_text(manifest_path, manifest_text, args.dry_run)
    print(f"[prepare] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
