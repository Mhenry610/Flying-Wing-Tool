# BARAM Automation Scaffold

## Scripts

- `scripts/baram_prepare_study.py`
  - Clones a `.bf` bundle, optional `.bm` bundle, and localizes a `.pvsm` file into a study workspace.
  - Supports multiple variants and a `--dry-run` mode.

- `scripts/baram_postprocess.py`
  - Parses `coefficient.dat` and `force.dat`.
  - Computes cruise-scaled coefficients, drag decomposition, and optional comparison against a project summary JSON.
  - Writes JSON and plots.

- `scripts/baram_run_case.py`
  - Discovers the BARAM OpenFOAM backend.
  - Optionally patches `controlDict` entries via `foamDictionary`.
  - Runs the saved solver backend directly, including MPI-parallel restarts.
  - Writes separate automation stdout/stderr logs into the case directory.

- `scripts/baram_run_study.py`
  - Reads `study_manifest.json`.
  - Runs one or more prepared variants through `baram_run_case.py`.
  - Optionally postprocesses each solved variant into a shared report directory.
  - Updates variant statuses inside the manifest.

- `scripts/baram_run_mesh_case.py`
  - Runs the `.bm` meshing workflow headlessly.
  - Automates `blockMesh`, `decomposePar`, and the three-stage `snappyHexMesh` sequence.
  - Can compare reproduced stage patch counts against a baseline `.bm` bundle.

- `scripts/baram_run_mesh_study.py`
  - Runs `baram_run_mesh_case.py` for one or more variants from `study_manifest.json`.
  - Updates the manifest with mesh status and mesh summary paths.

- `scripts/baram_build_flow_case_from_mesh.py`
  - Promotes a reconstructed mesh time from a `.bm` case into `constant/polyMesh`.
  - Copies the flow template `0/`, `system/`, and solver-side `constant/` files onto that mesh.
  - Patches boundary types to match the flow template.

- `scripts/baram_run_flow_from_mesh.py`
  - Orchestrates the full synthetic handoff: build flow case from remeshed mesh, decompose time `0`, and run a startup solve.

- `scripts/baram_extract_config.py`
  - Extracts readable XML/YAML-style configuration text from `configuration.h5` and `configurations.h5`.
  - Writes machine-readable summaries for the flow bundle and plain-text mesh settings for the mesh bundle.

- `scripts/baram_pvbatch_export.py`
  - Run with ParaView `pvpython`.
  - Loads a `.pvsm` and exports a screenshot from the active view.

- `scripts/baram_export_cfd_geometry.py`
  - Loads a project JSON through the repo's existing geometry/export stack.
  - Builds closed spanwise strip solids from the project airfoil sections.
  - Exports CFD-oriented STEP patches grouped into `centerbody`, `junction`, and `outer_wing` for remeshing.
  - Also writes per-strip STEP files and a manifest describing the section/patch layout.

- `scripts/baram_build_split_trisurface.py`
  - Loads a project JSON through the same geometry/export stack and writes split `triSurface` STL assets directly into a `.bm` case.
  - Builds OML patch surfaces for `centerbody`, `junction`, and `outer_wing`, plus a closed `cfd_volume.stl` for `refinementRegions`.
  - Patches `system/snappyHexMeshDict` to use the split surfaces and disables the legacy explicit feature-OBJ dependency.

## Example Usage

### Prepare a study workspace

```powershell
python scripts/baram_prepare_study.py `
  --source-bf "BARAM\\IntendedValidation2.bf" `
  --source-bm "BARAM\\IntendedValidation2.bm" `
  --source-pvsm "BARAM\\IntendedValidation2_local.pvsm" `
  --study-root "BARAM\\studies" `
  --name "IntendedValidation2 Validation" `
  --variant baseline `
  --variant mesh_coarse `
  --variant mesh_medium `
  --variant mesh_fine `
  --overwrite
```

### Run a one-iteration restart on a cloned case

```powershell
python scripts/baram_run_case.py `
  --case-root "BARAM\\studies\\intendedvalidation2_validation\\smoke_restart_605\\IntendedValidation2.bf" `
  --set-end-time 605 `
  --set-write-interval 1 `
  --timeout 120
```

### Run and postprocess variants from the manifest

```powershell
python scripts/baram_run_study.py `
  --manifest "BARAM\\studies\\intendedvalidation2_validation\\study_manifest.json" `
  --variant smoke_restart_605 `
  --set-end-time 605 `
  --set-write-interval 1 `
  --timeout 120 `
  --summary-json "Reports\\Report\\intendedvalidation2_results_summary.json" `
  --report-output-dir "Reports\\Report"
```

### Rebuild the `.bm` mesh headlessly

```powershell
python scripts/baram_run_mesh_case.py `
  --case-root "BARAM\\studies\\intendedvalidation2_validation\\mesh_smoke\\IntendedValidation2.bm" `
  --clean `
  --compare-case-root "BARAM\\IntendedValidation2.bm"
```

### Run mesh automation from the study manifest

```powershell
python scripts/baram_run_mesh_study.py `
  --manifest "BARAM\\studies\\intendedvalidation2_validation\\study_manifest.json" `
  --variant mesh_script_smoke `
  --clean `
  --compare-case-root "BARAM\\IntendedValidation2.bm"
```

### Build and start a flow case from a remeshed mesh

```powershell
python scripts/baram_run_flow_from_mesh.py `
  --mesh-case-root "BARAM\\studies\\intendedvalidation2_validation\\mesh_script_smoke\\IntendedValidation2.bm" `
  --flow-template-case-root "BARAM\\studies\\intendedvalidation2_validation\\smoke_restart_605\\IntendedValidation2.bf" `
  --output-case-root "BARAM\\studies\\intendedvalidation2_validation\\synthetic_flow_script_smoke\\IntendedValidation2.bm" `
  --end-time 1 `
  --write-interval 1
```

### Extract bundle configuration text

```powershell
python scripts/baram_extract_config.py `
  --bundle-root "BARAM\\IntendedValidation2.bf" `
  --output-dir "Reports\\Report" `
  --prefix "intendedvalidation2_bf_config"
```

### Postprocess a solved case

```powershell
python scripts/baram_postprocess.py `
  --case-root "BARAM\\IntendedValidation2.bf" `
  --summary-json "Reports\\Report\\intendedvalidation2_results_summary.json" `
  --output-dir "Reports\\Report" `
  --case-name "intendedvalidation2_baram"
```

### Export a screenshot from ParaView

```powershell
& "C:\\Program Files\\ParaView 6.0.1\\bin\\pvpython.exe" scripts\\baram_pvbatch_export.py `
  --state "BARAM\\IntendedValidation2_local.pvsm" `
  --output "Reports\\Report\\baram_state_overview.png"
```

### Export split CFD geometry from a project JSON

```powershell
python scripts/baram_export_cfd_geometry.py `
  --project-json "D:\\AI code\\flying wing tool\\Full Set\\Unified Directory\\IntendedValidation2.json" `
  --output-dir "BARAM\\geometry_exports\\intendedvalidation2_patch_kit"
```

### Patch a `.bm` case with split triSurface assets

```powershell
C:\\ProgramData\\anaconda3\\envs\\flying-wing-tool\\python.exe scripts\\baram_build_split_trisurface.py `
  --project-json "D:\\AI code\\flying wing tool\\Full Set\\Unified Directory\\IntendedValidation2.json" `
  --mesh-case-root "BARAM\\studies\\intendedvalidation2_validation\\split_trisurface_smoke\\IntendedValidation2.bm"
```

## Current Limitation

- The current solved flow case still uses a single aircraft wall patch. That means per-region drag attribution still requires a new export/remesh cycle even though the split-geometry exporter now exists.
- The mesh-regeneration path is now validated headlessly for this case, including staged `snappyHexMesh`.
- The direct `triSurface` path is now partially validated too: a split-surface smoke run advanced through actual castellated meshing and baffle creation before being stopped for time. The remaining work is to carry one of those split-surface cases through a full remesh and then into the flow solve.
