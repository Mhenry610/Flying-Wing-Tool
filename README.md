# Flying Wing Tool (FWT_V1-0)

## Purpose
Flying Wing Tool is a design and analysis application for flying-wing aircraft. It combines geometry definition, aerodynamic/performance analysis, structural analysis, mission-oriented calculations, and export workflows (including DXF/STEP).

This package is a cleaned runnable snapshot from the larger development directory.

## What is included
- GUI app entrypoint: `run_app.py`
- Batch/full analysis entrypoint: `run_full_analysis.py`
- Core code: `app/`, `cli/`, `core/`, `services/`, `scripts/`
- Required propeller surrogate models: `data/propeller_models/*.pkl`

## Requirements
Conda is required for this project because `pythonocc-core` is required for STEP/OCC geometry workflows and is most reliable through conda-forge.

Recommended:
- OS: Windows
- Python: 3.11 (recommended)
- Conda (Miniconda or Anaconda)

## Quick setup (environment.yml)
From this folder (`FWT_V1-0`):

```powershell
conda env create -f environment.yml
conda activate fwt
```

Then run:

```powershell
python run_app.py
```
## Setup (Conda + pip)
From this folder (`FWT_V1-0`):

```powershell
conda create -n fwt python=3.11 -y
conda activate fwt

# Core scientific + OCC stack (conda)
conda install -c conda-forge pythonocc-core numpy scipy pandas matplotlib casadi -y

# App/runtime libraries (pip)
pip install pyqt6 aerosandbox neuralfoil ezdxf
```

Optional (only if you use PyVista-based viewers):

```powershell
pip install pyvista pyvistaqt vtk
```

## Running the tool
### GUI
```powershell
python run_app.py
```

### Full analysis on a project JSON
```powershell
python run_full_analysis.py IntendedValidation2.json
```

If no file is passed, `run_full_analysis.py` defaults to `IntendedValidation.json` in the working directory.

## Notes about data and paths
- Keep `data/propeller_models` in place. Propulsion code expects pre-trained models there.
- Run commands from the `FWT_V1-0` root so relative imports and data paths resolve correctly.

## Common issues
- `No pre-trained propeller models found in data/propeller_models`:
  - Ensure `data/propeller_models/*.pkl` exists in this folder.
- OCC/STEP export errors:
  - Confirm you are in the conda env where `pythonocc-core` is installed.
- Qt import errors:
  - Install/repair `pyqt6` in the active environment.

## Project layout
- `app/`: GUI shell and tabs
- `core/`: domain models and exporters/generators
- `services/`: analysis, propulsion, mission, structure, export services
- `scripts/`: advanced analysis scripts and utilities
- `data/propeller_models/`: surrogate model files used by propulsion

