"""Legacy GUI entry point.

The active Qt application lives in :mod:`app.main_window`. This module remains
as a small compatibility shim for older launchers that still import
``app.guiqt``.
"""

def __getattr__(name):
    if name in {"MainWindow", "main"}:
        from app.main_window import MainWindow, main

        exports = {"MainWindow": MainWindow, "main": main}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def launch():
    """Legacy launcher alias for older scripts."""
    from app.main_window import main

    main()


__all__ = ["MainWindow", "launch", "main"]


if __name__ == "__main__":
    launch()
