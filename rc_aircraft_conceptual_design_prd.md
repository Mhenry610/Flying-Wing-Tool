# RC Aircraft Conceptual Design Tool — PRD / Technical Spec

**Status:** Draft 0.1  
**Date:** 2026-04-25  
**Project baseline:** Flying Wing Tool / FWT_V1-0  
**Primary goal:** Extend the current flying-wing-focused tool into a general RC aircraft conceptual design framework without throwing away the working aerodynamic, propulsion, mission, structure, and export work already in place.

---

## 1. Product thesis

The tool should become a conceptual RC aircraft design workstation that can take mission requirements, payload constraints, propulsion options, aerodynamic surfaces, structural layouts, and export targets, then produce a checked conceptual aircraft design.

The important shift is this:

> Move from “a flying wing tool with special cases” to “an aircraft project model where every aerodynamic surface is a first-class object.”

A flying wing should continue to work as the simplest case: one symmetric lifting surface with elevons. A conventional aircraft should be the same framework with additional first-class surfaces such as a horizontal tail, vertical tail, canard, winglets, or fins.

The AI layer should not replace deterministic analysis. It should operate through the same backend commands as the UI, run analyses, compare design states, explain tradeoffs, and propose project patches that the user can accept or reject.

---

## 2. Current baseline

The existing tool already supports several major workflows:

- Flying-wing geometry definition.
- Aerodynamic and performance analysis.
- Mission-oriented calculations.
- Propulsion model usage through required surrogate model files.
- Structural analysis centered on wingbox-style geometry.
- DXF / STEP export workflows.
- A project structure with `core/`, `services/`, `app/`, `cli/`, `scripts/`, and `data/propeller_models/`.

Current limitations that drive this PRD:

- The project model is still centered around a single `wing` object.
- Aerodynamic surfaces are not yet generalized as peer objects.
- Fuselage/body behavior is not mature enough to treat as a first-class aerodynamic/structural object in the first extension pass.
- Structural geometry is too wingbox-specific.
- Round spars, tube spars, braced layouts, struts, wires, and complex hardpoint load paths need a broader structure model.
- Tail/canard sizing should be determined through aerodynamic trim, stability, and control authority, not tail volume coefficient rules of thumb.
- CPACS/TiGL support is worth reintroducing as a geometry and data-exchange path, especially for future fuselage work.

---

## 3. Product goals

### 3.1 Primary goals

1. Generalize the project schema from `WingProject` toward `AircraftProject`.
2. Treat every aerodynamic surface as a first-class object.
3. Add per-surface symmetry behavior so wings, tails, canards, vertical fins, winglets, and asymmetric layouts can share the same framework.
4. Preserve flying-wing workflows as regression cases.
5. Add multi-surface aerodynamic aggregation for forces, moments, trim, and stability.
6. Rework structural geometry so it supports more than wingbox layouts.
7. Reintroduce CPACS/TiGL as an optional interoperability and geometry backend path.
8. Build AI integration around controlled backend tools, not free-form guesses.
9. Defer detailed fuselage work until the surface, structure, schema, and CPACS/TiGL layers are stable.

### 3.2 Secondary goals

1. Add configuration presets for flying wing, conventional tail, canard, twin fin, and twin boom concepts.
2. Add requirements-driven conceptual sizing.
3. Add mass and CG build-up across all components.
4. Add propulsion matching and mission trade studies.
5. Add design review reports that explain assumptions, warnings, and constraint drivers.

---

## 4. Non-goals for the initial extension

The initial extension should not attempt to solve every aircraft modeling problem.

Out of scope for the first major extension:

- Full fuselage lofting, fuselage structural analysis, or fuselage/body aerodynamic modeling as a mature feature.
- Full CFD.
- Full 6-DOF flight dynamics as a required workflow.
- Full OpenVSP replacement behavior.
- Aeroelastic optimization.
- Autopilot simulation.
- VTOL/multirotor hybrid support.
- Cloud collaboration.
- AI-generated aircraft designs without deterministic analysis support.
- Tail volume coefficient sizing as a design method.

Tail volume coefficients may be useful as legacy comparison values in some contexts, but they should not be used as sizing constraints, acceptance criteria, or the basis for tail/canard design in this tool.

---

## 5. Core design principles

### 5.1 Surfaces are first-class citizens

Every aerodynamic surface should be represented with the same core object model.

A main wing, horizontal tail, canard, vertical tail, winglet, fin, and elevon-bearing flying-wing panel should all share the same base representation:

```text
LiftingSurface
  identity
  role metadata
  transform
  symmetry mode
  planform sections
  airfoil distribution
  twist distribution
  dihedral / orientation
  control surfaces
  structural layout
  analysis settings
```

The solver should not assume that only the main wing matters. Instead, it should assemble all active surfaces, compute or approximate each contribution, then aggregate aircraft-level forces and moments about a shared reference point.

### 5.2 Role should guide interpretation, not create hard forks

Surface role is still useful:

```text
main_wing
horizontal_tail
vertical_tail
canard
fin
winglet
stabilator
custom
```

But role should not create separate geometry systems. It should mostly control defaults, labels, validation hints, control-surface defaults, and which stability/control checks apply.

Examples:

- A `main_wing` defaults to symmetric about the aircraft centerline.
- A `horizontal_tail` defaults to symmetric about the centerline.
- A `canard` defaults to symmetric about the centerline and ahead of the reference CG.
- A `vertical_tail` defaults to single centerline or mirrored twin-fin behavior depending on the selected preset.
- A `winglet` may be single, paired, or attached to another surface.

### 5.3 Symmetry is explicit

Symmetry should be a first-class toggle or enum on every surface.

Recommended symmetry modes:

```text
none
mirrored_about_xz
single_centerline
paired_explicit
```

Suggested meanings:

| Mode | Meaning | Example |
|---|---|---|
| `none` | Surface is modeled exactly as placed. No automatic mirror. | One asymmetric fin, one test surface, one side panel. |
| `mirrored_about_xz` | User defines one side; the tool creates a mirrored counterpart across the aircraft centerline plane. | Main wing, horizontal tail, canard, twin vertical fins. |
| `single_centerline` | Surface lies on or near the centerline and is not mirrored. | Centerline vertical tail. |
| `paired_explicit` | Two surfaces are linked by metadata but independently editable. | Non-identical winglets, canted fins, damaged/asymmetric test aircraft. |

This lets the flying-wing case remain clean: one `main_wing` surface with `mirrored_about_xz` symmetry and elevon control surfaces.

Vertical fins also become natural:

```yaml
surface:
  name: Center Vertical Tail
  role: vertical_tail
  symmetry: single_centerline
  local_span_axis: +Z
  origin_m: [0.85, 0.0, 0.08]
```

Twin fins can use the same surface definition with mirroring:

```yaml
surface:
  name: Twin Vertical Fin
  role: vertical_tail
  symmetry: mirrored_about_xz
  local_span_axis: +Z
  origin_m: [0.85, 0.32, 0.08]
```

### 5.4 Aerodynamic sizing beats rules of thumb

Tail, canard, and control-surface sizing should be determined from aerodynamic behavior.

Required sizing checks should include:

- Trim feasibility over the expected CG range.
- Static longitudinal stability from aerodynamic moment slope.
- Control authority at low speed, cruise speed, and landing/approach speed where applicable.
- Elevator/elevon/canard deflection required for trim.
- Stall margin on each lifting surface.
- Rudder or vertical-surface authority where applicable.
- Yaw stability from sideforce and yawing-moment derivatives where applicable.
- Roll control authority from ailerons/elevons/spoilers where applicable.

The tool should not size a horizontal tail from a target tail volume coefficient. It should size or evaluate the surface by solving the aircraft-level aerodynamic problem.

### 5.5 Fuselage work comes last in the initial extension

Detailed fuselage work should be deferred because less of the existing flying-wing-specific work transfers cleanly.

The initial extension may include a minimal body placeholder for:

- Mass and CG accounting.
- Payload volume/envelope notes.
- Drag-area estimate placeholder.
- CPACS/TiGL mapping preparation.
- Attachment references for wings, tails, booms, and landing gear.

But mature fuselage geometry, fuselage structural analysis, wing-body intersections, and body-lift/drag modeling should wait until the generalized surface and structure framework is stable.

### 5.6 CPACS/TiGL should be an adapter, not the internal source of truth at first

The internal `AircraftProject` model should remain authoritative during the first reintroduction.

CPACS/TiGL should be added through an adapter layer:

```text
AircraftProject <-> CPACSAdapter <-> CPACS XML / TiXI / TiGL
```

Expected uses:

- Export internal aircraft geometry to CPACS.
- Import selected CPACS geometry into internal project objects.
- Use TiGL as an optional geometry validation and visualization path.
- Use TiGL-derived geometry outputs to improve future fuselage behavior.
- Keep a UID map between internal objects and CPACS objects.

Do not make the solver depend on CPACS/TiGL until the adapter is validated. CPACS/TiGL should improve interoperability and future geometry depth without destabilizing the existing analysis workflow.

### 5.7 Structure must become geometry-agnostic

The current structure logic should be generalized from wingbox assumptions toward structural elements and load paths.

The structure model should support:

- Wingbox layouts.
- Round spars.
- Tube spars.
- Spar caps and webs.
- D-tube leading-edge structures.
- Ribs and formers.
- Stringers.
- Shear webs.
- Struts.
- Tension wires.
- Braced wings.
- Hardpoints and joints.
- Local reinforcements.

The structural solver should consume a generic structural layout attached to a surface or body, not a fixed set of wingbox percentages.

---

## 6. Target user workflows

### 6.1 Flying wing continuation workflow

User opens an existing flying-wing project.

The tool migrates it into:

```text
AircraftProject
  surfaces:
    - main_wing, mirrored_about_xz
```

Expected behavior:

- Existing geometry loads correctly.
- Existing elevon definitions migrate to generic control surfaces.
- Existing mission, propulsion, and structure results remain comparable.
- Existing exports still work.
- The user can run the same analysis as before.

### 6.2 Add a tail to an existing design

User starts with a flying-wing-like project and adds a horizontal tail or canard.

Expected behavior:

- The new surface is treated as another aerodynamic surface, not as a special tail object.
- The aircraft-level analysis aggregates wing and tail/canard forces and moments.
- The tool reports trim feasibility and static stability changes.
- The AI/design review explains whether the added surface improved trim, stability, or control authority.

### 6.3 Conventional aircraft preset

User creates a conventional RC aircraft from a preset.

Initial objects:

```text
surfaces:
  - main_wing, mirrored_about_xz
  - horizontal_tail, mirrored_about_xz
  - vertical_tail, single_centerline
mass_items:
  - battery
  - motor
  - ESC
  - servos
  - payload
  - receiver/autopilot
  - structure estimates
```

Expected behavior:

- The tool sizes initial geometry from requirements.
- Tail size is not selected from tail volume coefficient.
- The tool adjusts or evaluates tail size from trim, stability, and control authority calculations.
- The user can sweep tail area, tail arm, airfoil, incidence, and elevator sizing.

### 6.4 Twin vertical fins

User adds twin vertical fins to a wing or tail boom layout.

Expected behavior:

- The surface definition can use `mirrored_about_xz` symmetry.
- The local span axis can be vertical.
- The same section, airfoil, control-surface, and structure logic applies.
- The solver includes both generated surface instances in force/moment aggregation.

### 6.5 Structural layout change

User replaces a wingbox with a carbon tube spar and bracing struts.

Expected behavior:

- The structural layout stores a round spar element.
- The structure solver computes mass, bending stiffness, torsional stiffness approximation, stress margin, and deflection for supported load cases.
- Bracing elements can connect hardpoints and modify load paths at a simplified conceptual level.
- The export system can represent these features where supported.

### 6.6 AI design review

User asks:

> Review this aircraft and tell me what is limiting the design.

Expected behavior:

- AI reads project state.
- AI runs or requests required analyses.
- AI summarizes constraint drivers.
- AI distinguishes solver results from assumptions.
- AI proposes specific edits as a patch.
- User approves or rejects the patch.

---

## 7. Target architecture

Recommended package structure:

```text
core/
  aircraft/
    project.py
    surfaces.py
    bodies.py
    mass.py
    references.py
    schema.py
  geometry/
    sections.py
    transforms.py
    symmetry.py
    assembly.py
  structures/
    elements.py
    layouts.py
    materials.py
    load_cases.py
  controls/
    control_surfaces.py
  stability/
    trim_models.py
    derivatives.py

services/
  aero/
  propulsion/
  mission/
  structure/
  optimization/
  export/
  cpacs/
  validation/

application/
  project_facade.py
  commands.py
  analysis_jobs.py
  result_models.py
  migrations.py

api/
  routes_project.py
  routes_analysis.py
  routes_export.py
  routes_ai.py

ai/
  tool_registry.py
  requirements_parser.py
  design_reviewer.py
  patch_planner.py
  prompts/

frontend/  # optional web UI path
```

The important architecture boundary is the `application/` layer.

The UI, CLI, AI layer, and future web API should call the same application facade. They should not mutate internal dataclasses directly.

---

## 8. Proposed data model

### 8.1 AircraftProject

```python
@dataclass
class AircraftProject:
    schema_version: int
    metadata: ProjectMetadata
    requirements: AircraftRequirements
    reference: AircraftReferenceFrame
    surfaces: list[LiftingSurface]
    bodies: list[BodyObject]
    propulsion_systems: list[PropulsionSystem]
    mass_items: list[MassItem]
    mission: MissionProfile
    analyses: AnalysisStore
    exports: ExportSettings
    external_refs: ExternalReferenceStore
```

### 8.2 LiftingSurface

```python
@dataclass
class LiftingSurface:
    uid: str
    name: str
    role: SurfaceRole
    active: bool
    transform: SurfaceTransform
    symmetry: SymmetryMode
    local_span_axis: Axis
    planform: SurfacePlanform
    airfoils: AirfoilDistribution
    twist: TwistDistribution
    incidence_deg: float
    control_surfaces: list[ControlSurface]
    structural_layout: StructuralLayout
    analysis_settings: SurfaceAnalysisSettings
    external_refs: dict[str, str]
```

### 8.3 SurfaceTransform

```python
@dataclass
class SurfaceTransform:
    origin_m: Vec3
    orientation_euler_deg: Vec3
    parent_uid: str | None
```

The transform gives every surface a position and orientation in the aircraft frame.

For example:

- Main wing: span mostly along aircraft `Y`.
- Horizontal tail: span mostly along aircraft `Y`, behind the wing.
- Canard: span mostly along aircraft `Y`, ahead of the wing.
- Vertical tail: span mostly along aircraft `Z`.
- Winglet: span may be canted between `Y` and `Z`.

### 8.4 SymmetryMode

```python
class SymmetryMode(Enum):
    NONE = "none"
    MIRRORED_ABOUT_XZ = "mirrored_about_xz"
    SINGLE_CENTERLINE = "single_centerline"
    PAIRED_EXPLICIT = "paired_explicit"
```

### 8.5 SurfaceInstance

The authoring model stores `LiftingSurface` objects. The geometry assembly layer expands those definitions into physical instances.

```python
@dataclass
class SurfaceInstance:
    source_surface_uid: str
    instance_uid: str
    side: Literal["left", "right", "center", "explicit"]
    transform: SurfaceTransform
    expanded_geometry: ExpandedSurfaceGeometry
```

Aero, structure, visualization, and export services should consume `SurfaceInstance` objects where physical placement matters.

### 8.6 StructuralLayout

```python
@dataclass
class StructuralLayout:
    uid: str
    coordinate_system: Literal["surface_local", "aircraft"]
    elements: list[StructuralElement]
    joints: list[StructuralJoint]
    hardpoints: list[Hardpoint]
    load_cases: list[LoadCase]
    analysis_settings: StructuralAnalysisSettings
```

### 8.7 StructuralElement

```python
@dataclass
class StructuralElement:
    uid: str
    type: StructuralElementType
    material_uid: str
    start: StructuralLocation
    end: StructuralLocation
    section: StructuralSection
    connection_uids: list[str]
```

Recommended element types:

```text
wingbox
round_spar
tube_spar
spar_cap
spar_web
shear_web
rib
former
skin_panel
stringer
leading_edge_d_tube
trailing_edge_member
strut
wire
hardpoint_reinforcement
custom_beam
```

### 8.8 Fuselage/body placeholder

```python
@dataclass
class BodyObject:
    uid: str
    name: str
    role: Literal["fuselage", "pod", "boom", "payload_bay", "placeholder"]
    active: bool
    transform: BodyTransform
    mass_properties: MassProperties | None
    drag_area_estimate_m2: float | None
    envelope: BodyEnvelope | None
    attachments: list[AttachmentPoint]
    external_refs: dict[str, str]
```

For the initial extension, this should be enough for mass/CG, payload placement, and future CPACS/TiGL mapping. Detailed fuselage lofts should wait.

---

## 9. Functional requirements

### FR-001 — AircraftProject schema

The tool shall introduce an `AircraftProject` schema that can contain multiple aerodynamic surfaces, mass items, propulsion systems, mission definitions, structural layouts, analysis results, and external references.

Acceptance criteria:

- Existing flying-wing projects migrate into the new schema.
- The migrated project contains one active `main_wing` surface.
- Legacy fields remain loadable through migration logic.
- Schema versioning is explicit.

---

### FR-002 — First-class aerodynamic surfaces

The tool shall represent all aerodynamic surfaces using one common `LiftingSurface` model.

Acceptance criteria:

- A main wing, horizontal tail, canard, and vertical fin can all be represented by the same core class.
- Surface role changes defaults and validation, not the underlying geometry system.
- Each surface can have independent planform, airfoil, twist, incidence, transform, controls, and structure.

---

### FR-003 — Surface symmetry modes

The tool shall provide explicit symmetry behavior for every aerodynamic surface.

Acceptance criteria:

- A flying wing can be represented as one mirrored surface.
- A horizontal tail can be represented as one mirrored surface.
- A centerline vertical fin can be represented as one non-mirrored or centerline surface.
- Twin vertical fins can be represented with mirrored symmetry.
- The geometry assembly service can expand authoring surfaces into physical surface instances.

---

### FR-004 — Multi-surface aerodynamic assembly

The tool shall assemble all active surface instances into a common aerodynamic analysis input.

Acceptance criteria:

- Aircraft-level CL, CD, CM, and reference quantities can be computed from multiple surfaces.
- Forces and moments are resolved about a shared aircraft reference frame.
- Each surface reports its own contribution to lift, drag, and moment where supported.
- Aggregated results preserve traceability to each source surface.

---

### FR-005 — Aerodynamic trim and stability sizing

The tool shall evaluate tail, canard, and control-surface sizing through aerodynamic trim, stability, and control authority checks.

Acceptance criteria:

- Tail/canard sizing does not use tail volume coefficient targets.
- The solver can evaluate trim feasibility across a CG range.
- The solver can report required control deflection for trim.
- The solver can report static margin or equivalent aerodynamic stability metrics.
- The solver can run sensitivity sweeps for tail area, tail arm, incidence, and control-surface size.

---

### FR-006 — Control surfaces as generic objects

The tool shall support generic control surfaces attached to any lifting surface.

Acceptance criteria:

- Elevons, elevators, ailerons, flaps, rudders, spoilers, and custom controls use one shared model.
- Control surfaces can be defined by span range, chord/hinge line, deflection limits, and mixing behavior.
- Flying-wing elevons migrate into the generic control-surface model.
- Control authority can be included in trim and stability analyses where supported.

---

### FR-007 — Structural geometry overhaul

The tool shall replace wingbox-only structural assumptions with a generic structural layout model.

Acceptance criteria:

- A wingbox structure can still be represented.
- A round spar or tube spar can be represented.
- Ribs, skins, stringers, spar caps, and webs can be represented as elements.
- Bracing struts and wires can be represented as elements connected to hardpoints.
- The structure solver reports mass, load paths or simplified load sharing, margins, and deflections for supported layouts.

---

### FR-008 — Structural load cases

The tool shall support named structural load cases.

Recommended load cases:

```text
positive_g_pullup
negative_g_pushdown
landing_or_belly_impact
launch_acceleration
payload_inertia
motor_thrust
braced_wing_compression
transport_handling
custom
```

Acceptance criteria:

- Each structural layout can define one or more load cases.
- Results identify the critical load case.
- Results identify critical elements and margins.

---

### FR-009 — Mass and CG build-up

The tool shall support component-level mass and CG build-up.

Acceptance criteria:

- Components can be added with mass and x/y/z location.
- Structure-derived mass can flow into aircraft mass properties.
- Propulsion, payload, battery, receiver/autopilot, servos, and ballast can be represented.
- The tool computes total mass, CG, and CG sensitivity to movable components.
- The tool can sweep battery or payload position to satisfy CG constraints.

---

### FR-010 — Requirements-driven conceptual sizing

The tool shall provide a requirements input workflow for early aircraft sizing.

Inputs should include:

```text
payload mass
endurance or range target
cruise speed target
stall speed limit
launch method
landing method
max span
max takeoff mass
battery cell count preference
construction method
material preference
```

Acceptance criteria:

- The tool computes first-pass wing loading and power loading estimates.
- The tool proposes feasible starting geometries.
- Generated geometry remains editable.
- The tool labels estimates as conceptual assumptions, not validated results.

---

### FR-011 — Propulsion matching

The tool shall match motor, propeller, battery, and ESC choices to aircraft requirements.

Acceptance criteria:

- The tool estimates static thrust, cruise current, full-throttle current, climb power, and endurance.
- The tool reports current margin, battery C-rate margin, ESC margin, and thermal concerns where supported.
- The tool can rank propulsion combinations by endurance, climb margin, current margin, or mass.

---

### FR-012 — Mission analysis

The tool shall evaluate mission profiles using aircraft aerodynamic, mass, and propulsion data.

Acceptance criteria:

- Mission segments can include launch, climb, cruise, loiter, descent, and landing/approach assumptions.
- The tool estimates energy use and endurance.
- The tool reports which segment drives the energy or power requirement.

---

### FR-013 — CPACS export

The tool shall export supported internal aircraft geometry and metadata to CPACS.

Acceptance criteria:

- Each internal object with CPACS support receives a stable UID mapping.
- At minimum, lifting surfaces and basic control surfaces export where supported.
- Export failures return clear unsupported-feature diagnostics.
- The CPACS file validates against the chosen schema version where practical.

---

### FR-014 — CPACS import

The tool shall import selected CPACS aircraft geometry into internal project objects.

Acceptance criteria:

- Supported wings/surfaces import into `LiftingSurface` objects.
- Unsupported objects are preserved as external references or warnings where practical.
- Imported geometry does not silently overwrite existing internal project data.
- Round-trip loss is documented.

---

### FR-015 — TiGL optional geometry path

The tool shall optionally use TiGL for CPACS-based geometry generation, validation, or visualization.

Acceptance criteria:

- TiGL is optional and gracefully disabled if unavailable.
- TiGL-generated geometry can be compared against internal geometry for supported objects.
- TiGL failures do not break non-TiGL analysis workflows.
- Packaging complexity is isolated from the core tool.

---

### FR-016 — Fuselage placeholder

The tool shall include a minimal body/fuselage placeholder object during the initial extension.

Acceptance criteria:

- A fuselage/pod/boom can contribute mass and CG.
- It can define payload envelope notes or simple dimensions.
- It can define attachment points.
- It can hold CPACS/TiGL external references.
- It does not pretend to provide mature fuselage aerodynamic or structural analysis.

---

### FR-017 — AI design review

The tool shall provide a read-only AI design review mode before enabling AI project mutation.

Acceptance criteria:

- AI can read project state and analysis results.
- AI can request deterministic analyses through the application facade.
- AI findings cite the result or assumption they depend on.
- AI clearly separates solver output, assumptions, and suggestions.
- AI does not claim flight safety.

---

### FR-018 — AI patch proposal

The tool shall allow AI to propose project changes as explicit patches.

Acceptance criteria:

- AI-generated changes are displayed before application.
- The user can accept, reject, or edit the patch.
- Patches pass the same validation as UI edits.
- Applied patches can be undone.

---

### FR-019 — Trade studies

The tool shall support parameter sweeps across geometry, propulsion, CG, and structure.

Candidate sweep variables:

```text
wing area
aspect ratio
taper ratio
sweep
tail/canard area
tail/canard arm
surface incidence
battery mass
battery location
motor/prop choice
spar diameter
spar wall thickness
bracing geometry
payload mass
cruise speed
```

Acceptance criteria:

- Results show feasible/infeasible regions.
- Constraint violations are traceable.
- AI can summarize trade-study outcomes without inventing values.

---

### FR-020 — Export continuity

The tool shall preserve existing DXF/STEP export paths where possible.

Acceptance criteria:

- Existing flying-wing export workflows continue to work after migration.
- Multi-surface exports include each active surface where supported.
- Unsupported structural features are either approximated with warnings or excluded with explicit diagnostics.

---

## 10. Analysis requirements

### 10.1 Aircraft reference quantities

The tool shall maintain explicit aircraft-level reference quantities:

```text
S_ref
b_ref
c_ref
x_ref / y_ref / z_ref
x_cg / y_cg / z_cg
mass
inertia estimate where available
```

The user should be able to choose whether reference quantities come from:

- Main wing only.
- Total lifting area.
- User-defined custom values.

For stability and moment reporting, reference choices must be visible. Hidden reference assumptions will make multi-surface results hard to trust.

### 10.2 Force and moment aggregation

Each surface analysis result should include:

```text
surface_uid
instance_uid
CL contribution
CD contribution
CM contribution
force vector in aircraft frame
moment vector about aircraft reference point
alpha / beta / control state
Reynolds number estimate
validity warnings
```

Aircraft-level results are summed from physical instances.

### 10.3 Trim

Trim should solve for control deflections and/or incidence settings that satisfy aircraft-level moment balance.

Possible trim variables:

```text
elevon deflection
elevator deflection
stabilator incidence
canard incidence
main wing incidence
throttle setting
flight speed
angle of attack
```

Minimum required trim reports:

- Trim exists / does not exist.
- Required angle of attack.
- Required control deflection.
- Remaining control margin.
- Surface lift coefficients at trim.
- Stall margin per surface.
- CG location used.

### 10.4 Stability

The tool should compute stability through aerodynamic derivatives or finite-difference sweeps where possible.

Minimum useful metrics:

```text
dCm/dAlpha
neutral point estimate
static margin
control-fixed trim sensitivity
control authority margin
```

For directional/lateral work where supported:

```text
dCn/dBeta
dCl/dBeta
rudder authority
dihedral effect estimate
roll control authority
```

The tool may still display familiar RC design heuristics as informational notes later, but the core sizing logic should rely on aerodynamic calculations.

---

## 11. Structural requirements

### 11.1 Structure model stages

#### Stage A — Preserve current wingbox behavior

- Wrap current wingbox parameters in the new `StructuralLayout` model.
- Ensure existing projects still analyze.
- Keep legacy output comparable.

#### Stage B — Add beam/tube elements

- Add round spar and tube spar definitions.
- Support material properties and cross-section properties.
- Compute bending stiffness, torsional stiffness approximation, stress, and mass.

#### Stage C — Add panels/ribs/stringers as generic elements

- Convert skin, ribs, stringers, and webs into explicit or generated elements.
- Maintain conceptual-level analysis speed.
- Preserve manufacturability outputs where possible.

#### Stage D — Add braced layouts

- Add strut and wire elements.
- Add hardpoint/joint model.
- Add simplified load sharing assumptions.
- Report warnings when the model is too simplified for detailed stress validation.

### 11.2 Structural element location

Structural elements should be locatable using surface-relative coordinates.

Example:

```yaml
element:
  type: tube_spar
  start:
    eta: 0.0
    chord_fraction: 0.28
    z_offset_m: 0.0
  end:
    eta: 1.0
    chord_fraction: 0.28
    z_offset_m: 0.0
  section:
    outer_diameter_mm: 12.0
    wall_thickness_mm: 1.0
```

This is more flexible than hard-coding front and rear spar percentages into planform geometry.

### 11.3 Bracing element location

Bracing elements need aircraft-frame endpoints or hardpoint references.

Example:

```yaml
element:
  type: strut
  material_uid: carbon_tube_6mm
  start_hardpoint_uid: fuselage_lower_strut_mount
  end_hardpoint_uid: right_wing_strut_mount
  section:
    outer_diameter_mm: 6.0
    wall_thickness_mm: 0.75
```

### 11.4 Structural output

Minimum structural outputs:

```text
total structural mass
mass by surface
mass by element type
critical load case
critical element
stress margin
buckling margin where supported
tip deflection
tip twist
joint/hardpoint reactions where supported
warnings / unsupported assumptions
```

---

## 12. CPACS/TiGL integration spec

### 12.1 Integration intent

CPACS/TiGL should support interoperability, future geometry depth, and eventual fuselage behavior. It should not block the main internal workflow.

### 12.2 CPACS adapter responsibilities

```text
serialize AircraftProject subset to CPACS
parse supported CPACS objects into AircraftProject subset
maintain UID maps
validate schema where practical
record unsupported fields
support round-trip diagnostics
```

### 12.3 TiGL service responsibilities

```text
load CPACS file
generate geometry for supported surfaces/bodies
return triangulated or B-rep preview payloads where useful
compare TiGL geometry against internal geometry
support optional export/validation workflows
```

### 12.4 UID mapping

Each internal object that maps to CPACS should have:

```text
internal_uid
cpacs_uid
cpacs_xpath
last_sync_direction
last_sync_timestamp
sync_status
warnings
```

### 12.5 Versioning

The project should explicitly store:

```text
internal_schema_version
cpacs_schema_version_target
tigl_version_used, if any
```

### 12.6 Guardrails

- CPACS import should not silently discard unsupported geometry.
- TiGL should remain optional.
- CPACS/TiGL failures should produce diagnostics, not generic crashes.
- Internal project data should remain authoritative until the adapter is proven stable.

---

## 13. AI integration spec

### 13.1 AI role

The AI should act as a design assistant that can:

- Parse requirements.
- Suggest starting configurations.
- Run deterministic tool commands.
- Review project state.
- Explain analysis results.
- Identify constraint drivers.
- Propose project patches.
- Generate reports.

The AI should not invent aerodynamic, structural, or propulsion results.

### 13.2 Tool interface

Suggested AI-callable tools:

```text
get_project_state()
validate_project()
patch_project(diff)
run_initial_sizing()
run_aero_analysis()
run_trim_analysis()
run_stability_analysis()
run_propulsion_match()
run_structure_analysis()
run_mission_analysis()
run_trade_study(parameters, ranges)
compare_designs(project_ids)
generate_design_review()
generate_report()
```

### 13.3 Required AI guardrails

- AI cannot silently overwrite project state.
- AI must show project diffs before applying them.
- AI must cite solver outputs or assumptions behind each technical warning.
- AI must distinguish between “analysis result,” “rule of thumb,” and “suggestion.”
- AI must not claim the aircraft is safe to fly.
- AI should recommend validation steps when model fidelity is limited.

### 13.4 First AI feature

The first AI feature should be read-only design review.

Example output categories:

```text
geometry risks
trim/stability risks
CG risks
propulsion risks
structure risks
manufacturing/export risks
missing data
recommended next analyses
```

Only after this mode is useful should the tool support AI-generated project patches.

---

## 14. UI / UX requirements

The UI should be organized around engineering workflows, not raw dataclass fields.

Recommended workspaces:

```text
Project / Requirements
Geometry / Surfaces
Controls
Mass / CG
Aerodynamics
Trim / Stability
Propulsion
Mission
Structure
Trade Studies
CPACS / TiGL
Export
AI Review
```

### 14.1 Geometry workspace

Requirements:

- List all aerodynamic surfaces.
- Add/remove/duplicate surfaces.
- Toggle symmetry mode.
- Set role, origin, orientation, incidence, local span axis.
- Edit planform sections.
- Edit airfoil and twist distribution.
- Show generated physical instances.
- Show aircraft-level top/side/front views.

### 14.2 Structure workspace

Requirements:

- Show structural layout per surface.
- Add wingbox, round spar, tube spar, rib, skin, stringer, strut, wire, and hardpoint elements.
- Show structural element locations over geometry.
- Show mass and critical margins.
- Show warnings when structural idealization is too simple.

### 14.3 Stability workspace

Requirements:

- Show trim status over CG range.
- Show required control deflection.
- Show static margin or equivalent stability metric.
- Show surface contribution to pitching moment.
- Show stall margin by surface.
- Support sweeps over tail/canard area, incidence, arm, and CG.

### 14.4 CPACS/TiGL workspace

Requirements:

- Export CPACS.
- Import CPACS subset.
- Validate CPACS file where practical.
- Show UID mappings.
- Show unsupported objects/fields.
- Run TiGL preview/validation when available.

---

## 15. Migration plan

### Phase 0 — Baseline lock

Purpose: protect current behavior.

Tasks:

- Add regression projects for current flying-wing workflows.
- Store reference outputs for key analyses.
- Add project load/save round-trip tests.
- Add export smoke tests.

Exit criteria:

- Existing project files load.
- Existing analysis paths run.
- Existing exports do not regress unexpectedly.

### Phase 1 — AircraftProject schema and surface model

Purpose: introduce generalized surfaces without breaking flying wings.

Tasks:

- Add `AircraftProject`.
- Add `LiftingSurface`.
- Add symmetry modes.
- Add geometry assembly service.
- Migrate current `wing` into `surfaces[0]`.
- Migrate elevons into generic control surfaces.

Exit criteria:

- A migrated flying wing produces comparable geometry and analysis outputs.
- Multiple surfaces can exist in the project file.
- Surface transforms and symmetry expansion are testable.

### Phase 2 — Multi-surface aero aggregation

Purpose: support wings, tails, canards, and fins consistently.

Tasks:

- Add force/moment aggregation across surfaces.
- Add shared aircraft reference frame.
- Add trim and static stability reporting.
- Add surface-level result breakdown.
- Add conventional and canard test projects.

Exit criteria:

- Main wing + horizontal tail analysis runs.
- Main wing + canard analysis runs.
- Centerline vertical fin and twin-fin geometry assemble correctly.
- Tail/canard sizing is based on aerodynamic checks, not tail volume coefficient.

### Phase 3 — Structural geometry overhaul

Purpose: move from wingbox-only structure to generic elements.

Tasks:

- Wrap current wingbox as a structural layout.
- Add round/tube spar elements.
- Add generic ribs, skins, stringers, and webs.
- Add hardpoints.
- Add strut/wire elements.
- Add structural load case system.

Exit criteria:

- Current wingbox projects still analyze.
- A carbon tube spar layout can be analyzed.
- A braced wing layout can be represented and evaluated at conceptual fidelity.

### Phase 4 — Mass/CG and propulsion integration

Purpose: make the aircraft-level model practical for RC design.

Tasks:

- Add component mass table.
- Add structure-derived mass integration.
- Add CG envelope tools.
- Add propulsion matching interface.
- Add mission-energy integration.

Exit criteria:

- Aircraft mass and CG update from components and structure.
- Propulsion choices can be evaluated against mission requirements.
- CG sensitivity to battery/payload placement is visible.

### Phase 5 — CPACS/TiGL reintroduction

Purpose: restore interoperability and prepare for better body/fuselage behavior.

Tasks:

- Add CPACS adapter module.
- Add export of supported surfaces.
- Add import of supported surfaces.
- Add UID mapping.
- Add optional TiGL service.
- Add CPACS/TiGL diagnostics.

Exit criteria:

- Supported project subset exports to CPACS.
- Supported CPACS geometry imports into internal surfaces.
- TiGL preview/validation works when installed and fails gracefully otherwise.

### Phase 6 — AI read-only review

Purpose: add useful AI without letting it mutate project state.

Tasks:

- Add AI tool registry.
- Add project review prompt.
- Add deterministic tool calls.
- Add finding categories.
- Add report generation.

Exit criteria:

- AI can review a project and point to solver-backed issues.
- AI does not invent missing results.
- AI can recommend analyses or parameter sweeps.

### Phase 7 — AI patch proposals

Purpose: let AI propose changes after trust is established.

Tasks:

- Add patch format.
- Add patch preview UI.
- Add patch validation.
- Add undo support.

Exit criteria:

- AI can propose a surface, structure, propulsion, or CG change.
- User approval is required.
- Invalid patches are rejected by the same backend validation as UI edits.

### Phase 8 — Detailed fuselage/body work

Purpose: start mature fuselage support only after the generalized framework is stable.

Candidate tasks:

- Add fuselage cross-section model.
- Add CPACS/TiGL-based body geometry import/export.
- Add wing/body attachment behavior.
- Add fuselage wetted area and drag estimates.
- Add simple body structural elements.
- Add payload bay geometry and packaging checks.

Exit criteria:

- Fuselage geometry is useful enough to justify being first-class.
- Fuselage behavior does not destabilize surface, structure, or export workflows.

---

## 16. Validation and testing

### 16.1 Regression tests

Required regression cases:

```text
existing flying wing baseline
flying wing with elevons
flying wing with BWB/body-section style geometry, if supported
conventional aircraft: wing + horizontal tail + vertical tail
canard aircraft: wing + canard + vertical tail
twin-fin aircraft
round-spar wing
braced wing
CPACS export/import supported subset
```

### 16.2 Unit tests

Required unit test categories:

```text
schema migration
surface symmetry expansion
surface transform math
control surface migration
force/moment aggregation
CG computation
structural element section properties
structural layout serialization
CPACS UID mapping
AI patch validation
```

### 16.3 Numerical validation

The tool should compare key outputs against:

- Existing FWT results for flying wings.
- Hand calculations for simple rectangular wings/tails.
- Known beam formulas for simple spar cases.
- Known mass/CG examples.
- CPACS/TiGL geometry outputs for supported simple cases.

### 16.4 User-facing validation warnings

Any result that depends on simplified conceptual assumptions should expose warnings.

Example warning categories:

```text
low Reynolds number outside model range
airfoil polar missing or extrapolated
surface interference not modeled
fuselage/body aero not modeled
bracing load path simplified
buckling model not valid for selected structure
control authority estimated with simplified model
CPACS object partially imported
TiGL unavailable
```

---

## 17. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Generalized surfaces break flying-wing behavior | Existing tool value comes from current flying-wing workflows | Freeze regression cases before schema migration |
| Multi-surface aero becomes inconsistent | Tail/canard/fins need shared reference frames | Add explicit aircraft reference quantities and force/moment aggregation tests |
| Symmetry handling becomes confusing | Vertical fins and twin fins can expose bad assumptions | Use explicit symmetry modes and expanded `SurfaceInstance` objects |
| Structural overhaul becomes too large | Structure could swallow the entire project | Stage the work: wrap wingbox, then tubes, then panels, then bracing |
| CPACS/TiGL packaging becomes fragile | Optional geometry dependencies can complicate installs | Keep CPACS/TiGL optional and isolated in services |
| AI gives false confidence | Aircraft design has safety-critical implications | Keep AI solver-bound, require citations to results, and avoid flight-safety claims |
| Fuselage work derails the extension | Less current code transfers to fuselage behavior | Defer detailed fuselage work until surfaces/structure/CPACS are stable |

---

## 18. Open technical questions

1. What aerodynamic backend should own multi-surface force/moment aggregation first: existing logic, AeroSandbox, a VLM path, or a hybrid approach?
2. Should surface symmetry expansion happen before every service call, or should services understand symmetry directly?
3. What should the default aircraft reference area be for multi-surface aircraft: main wing area or user-defined reference area?
4. How should canted fins and winglets be represented: surface transform only, local span-axis enum, or full local coordinate basis?
5. What is the minimum structural fidelity needed for braced layouts to be useful without pretending to be detailed FEA?
6. Which CPACS schema version should be targeted first?
7. Should TiGL be installed through the main environment or treated as an optional plugin?
8. What fuselage placeholder fields are worth adding now without creating migration debt later?
9. What stability derivative targets should be used as defaults for RC aircraft, if any, given the rejection of tail volume coefficient sizing?
10. Should AI-generated designs create multiple project branches automatically, or only propose patches to the current branch?

---

## 19. Definition of done for initial extension

The initial extension is successful when:

- Existing flying-wing projects load and run through the migrated schema.
- The project can contain multiple first-class aerodynamic surfaces.
- Each surface has explicit symmetry behavior.
- A conventional aircraft can be represented without special-case tail geometry.
- A canard aircraft can be represented without special-case canard geometry.
- A centerline vertical fin and twin fins can both be represented cleanly.
- Tail/canard sizing is evaluated through aerodynamic trim/stability/control checks, not tail volume coefficients.
- Structural layouts support at least current wingbox behavior and one round/tube spar case.
- The structural model has a path toward braced layouts.
- Mass and CG are aircraft-level, component-based, and connected to structure.
- CPACS export/import exists for a supported subset.
- TiGL integration is optional and does not destabilize the core workflow.
- AI can perform a useful read-only design review using deterministic analysis results.
- Detailed fuselage work remains deferred until the core generalized framework is stable.

---

## 20. Suggested first implementation sprint

### Sprint goal

Create the generalized aircraft/surface foundation while preserving current flying-wing behavior.

### Tasks

1. Add `AircraftProject` schema skeleton.
2. Add `LiftingSurface`, `SurfaceTransform`, and `SymmetryMode` models.
3. Add geometry assembly service that expands surfaces into `SurfaceInstance` objects.
4. Add migration from current `Project.wing` to `AircraftProject.surfaces[0]`.
5. Migrate elevons into generic `ControlSurface` objects.
6. Add regression tests for existing flying-wing project load/save.
7. Add simple multi-surface project fixture: wing + horizontal tail + vertical tail.
8. Add validation warnings for unsupported multi-surface analyses.

### Sprint non-goals

- No detailed fuselage model.
- No CPACS/TiGL implementation yet.
- No AI patching yet.
- No full structural overhaul yet.

### Sprint acceptance criteria

- Existing flying-wing project opens under the new schema.
- The tool can serialize and deserialize multiple surfaces.
- Symmetry expansion produces expected physical instances.
- The old flying-wing geometry path can still be reached or matched.
- Multi-surface objects exist in project state even if some analyses still return unsupported warnings.

---

## 21. References

- Flying Wing Tool repository: https://github.com/Mhenry610/Flying-Wing-Tool
- CPACS documentation: https://dlr-sl.github.io/cpacs-website/pages/documentation.html
- CPACS schema documentation: https://www.cpacs.de/documentation/CPACS_3_4_0_Docs/html/89b6a288-0944-bd56-a1ef-8d3c8e48ad95.htm
- TiGL documentation: https://dlr-sc.github.io/tigl/pages/documentation.html
- TiGL overview: https://www.dlr.de/en/sc/research-transfer/software-solutions/tigl
