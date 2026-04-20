# BARAM Validation Plan for `IntendedValidation2`

## Purpose

This plan turns the current CFD discrepancy into a controlled validation study. The immediate goal is not to prove the low-order workflow correct or incorrect in general. The goal is to determine why the BARAM result produces substantially higher drag than the repository cruise analysis, and to do so with evidence that can be defended in the thesis.

## Current Verified State

- The current solved BARAM flow case is [D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf](D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf).
- The current ParaView state is [D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2_local.pvsm](D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2_local.pvsm).
- The mesh contains a single aircraft wall patch, `IntendedValidation2CFDmesh_surface`, in [D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf\case\constant\polyMesh\boundary](D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf\case\constant\polyMesh\boundary).
- Because the aircraft is one patch, the current case cannot isolate centerbody drag, outer-wing drag, or junction/interference drag directly.
- The solved force history in [D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf\case\postProcessing\force-mon-1_forces\439\force.dat](D:\AI code\flying wing tool\Full Set\FWT_V1-0\BARAM\IntendedValidation2.bf\case\postProcessing\force-mon-1_forces\439\force.dat) shows the drag discrepancy is pressure-dominated, not primarily viscous.

## Hypotheses To Test

### H1: The low-order workflow underpredicts pressure drag from the blended centerbody and wing-body transition.

Evidence that supports H1:
- A split-patch CFD run shows a large share of pressure drag on the centerbody or junction surfaces.
- Simplifying or removing the centerbody causes drag to drop significantly while lift remains comparable.

Evidence that weakens H1:
- Pressure drag remains concentrated on the outer wing instead of the centerbody/junction.
- Centerbody simplification produces little drag change.

### H2: Local separation or strong adverse-pressure regions are present in CFD and are not captured adequately by the low-order model.

Evidence that supports H2:
- `Cp`, wall shear, wake slices, and streamlines show separation, recirculation, or a thickened wake on the body blend or wing root region.
- Drag is sensitive to angle of attack or turbulence-model choice in a way consistent with marginal separation.

Evidence that weakens H2:
- Surface flow remains attached and wake growth is mild across the relevant regions.
- Drag remains high even with no sign of separation-sensitive behavior.

### H3: Part of the discrepancy is numerical or setup-related rather than physical.

Evidence that supports H3:
- Drag shifts materially with mesh refinement, layer settings, or turbulence model.
- Force coefficients are not stable under modest changes in numerical resolution.

Evidence that weakens H3:
- Mesh and model studies show drag is converged to a narrow band.

## Required Case Changes

### 1. Create a patch-split geometry export

The current case uses one aircraft patch. To validate centerbody and interference hypotheses, the aircraft surface must be re-exported as separate named surfaces before meshing.

Minimum patch split:
- `centerbody_surface`
- `outer_wing_surface`
- `control_surface`

Preferred patch split:
- `centerbody_surface`
- `wing_inner_surface`
- `wing_outer_surface`
- `elevon_surface`
- `wingbody_junction_surface` if the export path can isolate it cleanly

The cleanest route is to export separate STL bodies from the design tool rather than trying to reverse-engineer patch regions from the current single STL.

### 2. Match the comparison condition exactly

For every validation rerun, lock these to the same values on both sides:
- Geometry revision
- Reference area
- Air density / altitude
- Velocity
- Angle of attack
- Symmetry convention
- Reference point for moments

Do not compare a reconstructed zero-alpha CFD state to a loosely corresponding low-order state if an exact matched run can be generated.

### 3. Add monitored outputs for each surface patch

For each rerun, add:
- `forces` monitor per aircraft patch
- `forceCoeffs` monitor with thesis-level reference area and velocity
- residual/convergence history

## Validation Matrix

### Case A: Baseline matched rerun

Objective:
- Establish a clean one-to-one CFD vs repository comparison.

Changes:
- Same geometry as current `IntendedValidation2`
- Same cruise velocity
- Same angle of attack as the repository cruise point
- Same area and density references in the force coefficient monitor

Outputs:
- `CL`, `CD`, `Cm`, `L/D`
- Pressure/viscous drag breakdown
- Force convergence plot

### Case B: Mesh-convergence study

Objective:
- Determine whether the CFD drag is numerically trustworthy.

Cases:
- Coarse mesh
- Medium mesh
- Fine mesh

Hold constant:
- Domain
- turbulence model
- angle of attack
- velocity
- reference values

Track:
- `CL`
- `CD`
- pressure drag
- viscous drag
- wall-layer metrics if available

Acceptance target:
- Drag change between the two finest meshes should be small enough to argue practical convergence.

### Case C: Patch-resolved drag study

Objective:
- Locate where the excess pressure drag actually lives.

Requirements:
- Patch-split export and remesh

Outputs:
- pressure and viscous drag per patch
- lift per patch
- percent contribution to total drag

This case is the key test for the centerbody/interference hypothesis.

### Case D: Flow-structure visualization study

Objective:
- Test for separation, wake thickening, and adverse-pressure regions.

Required figures:
- `Cp` on aircraft surface
- velocity slice through center plane
- wake slice downstream of trailing edge
- streamlines around centerbody and wing root
- wall-shear or equivalent attached/separated-flow indicator if available

Interpretation:
- Large pressure plateaus, reversed-flow signatures, recirculation, or abrupt wake growth support a separation/form-drag explanation.

### Case E: Geometry ablation study

Objective:
- Identify whether the centerbody or body-wing transition is the dominant source.

Recommended geometry variants:
- baseline blended configuration
- wing-only or strongly simplified centerbody
- smoother centerbody/junction variant

Compare:
- `CL`
- `CD`
- pressure drag
- patch-resolved drag if available

### Case F: Turbulence-model sensitivity study

Objective:
- Determine whether drag is strongly model-dependent.

Minimum models:
- standard `k-epsilon` baseline
- one second RANS model if BARAM setup permits it cleanly

Interpretation:
- Large drag shifts indicate the flow may be separation-sensitive or numerically delicate.

## Deliverables

### Figures

Produce these figures into `Reports/Report`:
- `baram_baseline_convergence.png`
- `baram_mesh_study_cd.png`
- `baram_patch_drag_breakdown.png`
- `baram_cp_surface.png`
- `baram_centerplane_velocity_slice.png`
- `baram_wake_slice.png`
- `baram_streamlines_centerbody.png`
- `baram_geometry_ablation_cd.png`

### Tables

Produce these tables in the thesis:
- matched-case CFD vs repository cruise comparison
- mesh-convergence summary
- patch-resolved drag breakdown
- geometry-ablation comparison
- turbulence-model sensitivity summary

### Machine-readable summaries

Recommended JSON artifacts:
- `baram_caseA_summary.json`
- `baram_mesh_study_summary.json`
- `baram_patch_breakdown_summary.json`
- `baram_ablation_summary.json`

## Recommended Execution Order

1. Regenerate the baseline case with exact cruise-condition matching and thesis-level force references.
2. Run the three-mesh study on that matched case.
3. Export a patch-split aircraft surface and rerun the matched case.
4. Extract `Cp`, wake, and streamline figures from the patch-split case.
5. Run at least one simplified-centerbody ablation.
6. If drag remains ambiguous, run one turbulence-model sensitivity case.

## Thesis Claim Discipline

After this workflow:
- If patch-resolved pressure drag concentrates on the centerbody/junction, claim a centerbody/interference drag mechanism.
- If flow visualization shows separation in the same region, claim separation-supported pressure drag.
- If the result is highly mesh- or model-sensitive, claim the discrepancy is not yet physically isolated.
- Do not claim “viscous effects” as the main explanation unless a rerun shows viscous drag dominating the difference, which the current case does not.
