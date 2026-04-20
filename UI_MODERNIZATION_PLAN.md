# UI Modernization Plan

## Goal
Replace the current primary desktop UI with a more modern, flexible interface while keeping the existing Qt GUI available as a regression and test harness.

## Current State
- `run_app.py` launches the current PyQt application through [`app/main_window.py`](D:\AI code\flying wing tool\Full Set\FWT_V1-0\app\main_window.py).
- The older monolithic Qt implementation still exists in [`app/guiqt.py`](D:\AI code\flying wing tool\Full Set\FWT_V1-0\app\guiqt.py).
- The application already has a useful separation point:
  - `core/` holds data models and project serialization.
  - `services/` holds geometry, analysis, optimization, mission, structure, export, and viewer logic.
  - `app/tabs/` still mixes UI code with direct service calls and widget-local state.
- The CLI layer is only a stub today, so there is not yet a clean non-GUI application facade.

## Recommendation
Build the new UI as a local web application and keep Python as the computation backend.

Recommended stack:
- Frontend: React + TypeScript + Vite
- Desktop shell: pywebview first, with the option to move to Tauri later if packaging becomes important
- Backend bridge: FastAPI running locally inside the Python app process or as a managed sidecar
- Visualization:
  - 2D charts: Plotly or ECharts
  - 3D geometry preview: Three.js / react-three-fiber
  - Long-running jobs: async task endpoints with progress polling or event streaming

Why this is the right fit here:
- It separates presentation from engineering logic better than continuing to grow Qt widget code.
- It is much easier to create a polished, responsive UI with a design system, animations, resizable panels, and adaptive layouts.
- The existing Python `core/` and `services/` code can stay authoritative.
- It allows the old Qt app to remain runnable with minimal disruption.

## Non-Goals
- Do not rewrite aerodynamic, geometry, structure, mission, or export logic during the UI project.
- Do not delete the current Qt UI until the new UI reaches feature parity for core workflows.
- Do not couple frontend state directly to internal dataclass layout without an adapter layer.

## Target Architecture

### 1. Stable application facade
Create a Python application layer between UI and domain logic.

Suggested package:
- `api/` or `application/`

Responsibilities:
- Load/save project files
- Return full project state in UI-safe JSON
- Accept patch/update commands for project fields
- Run analysis, optimization, export, mission, and structure workflows
- Normalize errors and progress events

### 2. Keep domain logic in Python
Continue using:
- `core.state.Project`
- `core.models.*`
- `services.*`

But stop calling them directly from UI widgets where possible.

### 3. New frontend as the primary UX layer
Create a new `frontend/` app with:
- app shell
- route or workspace layout
- geometry editor
- analysis dashboard
- mission planner
- structure workspace
- export workspace
- project file actions

### 4. Legacy UI retention
Retain both existing entrypoints during migration:
- `python run_app.py` for the current PyQt UI
- `python run_web_ui.py` for the new UI shell

Later, after parity:
- make the new UI the default entrypoint
- keep the Qt UI behind a `--legacy-ui` flag or separate script

## Migration Phases

### Phase 0: UX and technical spike
Duration: 3-5 days

Deliverables:
- screen inventory of current workflows
- list of must-keep features from each current tab
- wireframes for the new information architecture
- proof of concept for:
  - launching a local frontend
  - loading a `Project`
  - editing one geometry parameter
  - rendering one chart
  - rendering one 3D preview payload

Decision gate:
- confirm web UI direction before broader integration work

### Phase 1: Extract an app-facing backend contract
Duration: 4-7 days

Tasks:
- create a thin API layer over `Project` and key services
- define request/response schemas with Pydantic
- centralize project mutation logic
- add uniform error handling and progress reporting
- move file operations behind backend endpoints

Suggested first endpoints:
- `GET /project`
- `POST /project/new`
- `POST /project/load`
- `POST /project/save`
- `PATCH /project`
- `POST /analysis/run`
- `POST /optimization/twist`
- `POST /mission/run`
- `POST /structure/run`
- `POST /export/step`
- `POST /export/dxf`
- `GET /preview/geometry`

Success criteria:
- the new backend contract can support one UI without importing Qt
- some existing Qt actions can optionally call the same facade

### Phase 2: Frontend foundation
Duration: 4-6 days

Tasks:
- scaffold `frontend/` with React + TypeScript + Vite
- add a component library strategy
- define theme tokens, typography, spacing, and panel behavior
- create API client, project store, and job/progress store
- build app shell with:
  - left navigation
  - top project bar
  - resizable content regions
  - notification system
  - error boundary

Recommended visual direction:
- engineering workstation layout, not a form-heavy dialog app
- strong hierarchy with pinned summaries, inspector panels, and result canvases
- responsive enough for laptop use, but optimized for desktop

### Phase 3: Build feature slices in priority order
Duration: 2-4 weeks

Priority order:
1. Project open/save/new
2. Geometry workspace
3. Analysis workspace
4. Export workspace
5. Mission workspace
6. Structure workspace

Per-slice pattern:
- backend endpoint
- frontend data model
- editable form or grid
- plot or viewer
- validation
- regression test against known project files

### Phase 4: Dual-run validation
Duration: 1-2 weeks

Tasks:
- keep the Qt UI runnable for comparison testing
- use the same sample project files in both UIs
- compare:
  - saved JSON structure
  - analysis outputs
  - export artifacts
  - mission summaries
  - structure summaries
- document any intentional differences

Success criteria:
- new UI matches legacy outputs for defined reference cases
- major workflows are usable without falling back to Qt

### Phase 5: Promote the new UI
Duration: 2-3 days

Tasks:
- switch primary launcher to the new UI
- rename current Qt launcher to legacy
- update README and environment setup
- keep automated smoke coverage for both launchers until confidence is high

## Repository Changes

### New directories
- `frontend/`
- `backend_api/` or `application/`
- `tests/ui_contract/`

### New entrypoints
- `run_web_ui.py`
- optional `run_legacy_qt.py`

### Existing files likely to change
- `run_app.py`
- `README.md`
- `environment.yml`
- parts of `app/main_window.py` if it is updated to consume shared backend facades

## Testing Strategy

### Keep the old UI for:
- smoke launch tests
- output comparison on reference projects
- manual fallback during migration

### Add new tests for:
- project serialization round trips
- API contract tests
- service-level regression tests
- frontend component tests
- end-to-end workflow tests for key user paths

Reference workflows to lock down:
- create/edit geometry and save project
- run airfoil/analysis workflow
- run twist optimization
- run export workflow
- run mission workflow

## Risks

### Risk: frontend bypasses domain constraints
Mitigation:
- all writes go through backend validation and typed schemas

### Risk: long-running Python jobs freeze the UI
Mitigation:
- run heavy jobs off the UI thread
- expose progress and cancellation where practical

### Risk: 3D preview is harder in web than Qt/PyVista
Mitigation:
- start with backend-generated geometry payloads and a simple browser renderer
- keep complex export geometry generation in Python
- use Qt viewer only as fallback during early phases if needed

### Risk: two UIs diverge
Mitigation:
- share one backend contract
- compare outputs against fixed reference projects

## First Concrete Sprint

1. Add a Python backend facade layer for project load/save and geometry updates.
2. Create `run_web_ui.py` that starts a local FastAPI server plus a simple web shell.
3. Scaffold `frontend/` with one polished workspace: Geometry.
4. Support:
   - open project
   - edit core planform values
   - live derived-geometry summary
   - 2D geometry plot
5. Keep `run_app.py` untouched as the regression path.

## Definition of Done for the UI replacement
- The new UI can handle the main workflows without needing the Qt UI for normal use.
- The old Qt UI still launches for regression testing.
- Project files remain compatible.
- Core outputs match agreed reference cases.
- The UI code is clearly separated from engineering logic.
