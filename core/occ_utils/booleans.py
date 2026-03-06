"""
Thin wrappers for OCC boolean operations with defensive fallbacks.
These mirror monolith semantics: if an operation fails, return the original shape.
"""

from __future__ import annotations

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common


def cut(a: TopoDS_Shape, b: TopoDS_Shape) -> TopoDS_Shape:
    try:
        op = BRepAlgoAPI_Cut(a, b)
        return op.Shape() if op.IsDone() else a
    except Exception:
        return a


def fuse(a: TopoDS_Shape, b: TopoDS_Shape) -> TopoDS_Shape:
    try:
        op = BRepAlgoAPI_Fuse(a, b)
        return op.Shape() if op.IsDone() else a
    except Exception:
        return a


def common(a: TopoDS_Shape, b: TopoDS_Shape) -> TopoDS_Shape:
    try:
        op = BRepAlgoAPI_Common(a, b)
        return op.Shape() if op.IsDone() else TopoDS_Shape()
    except Exception:
        return TopoDS_Shape()