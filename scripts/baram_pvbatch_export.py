from __future__ import annotations

import argparse
from pathlib import Path

from paraview.simple import GetActiveViewOrCreate, LoadState, Render, ResetCamera, SaveScreenshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a ParaView state and export a screenshot.")
    parser.add_argument("--state", required=True, help="Path to the .pvsm state file.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--width", type=int, default=1800, help="Screenshot width in pixels.")
    parser.add_argument("--height", type=int, default=1000, help="Screenshot height in pixels.")
    parser.add_argument("--reset-camera", action="store_true", help="Reset the active camera before saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = str(Path(args.state).resolve())
    output_path = str(Path(args.output).resolve())

    LoadState(state_path)
    view = GetActiveViewOrCreate("RenderView")
    if args.reset_camera:
        ResetCamera(view)
    Render(view)
    SaveScreenshot(output_path, view, ImageResolution=[args.width, args.height])
    print(f"Wrote screenshot: {output_path}")


if __name__ == "__main__":
    main()
