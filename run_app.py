#!/usr/bin/env python3
"""
Unified Flying Wing Tool - Main Entry Point
"""

import sys
import signal
import traceback
import faulthandler
from pathlib import Path

# Enable faulthandler to catch segfaults and print tracebacks
faulthandler.enable()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def install_exception_hook():
    """
    Install a global exception hook to catch unhandled exceptions.
    
    PyQt6 silently swallows exceptions in slot handlers by default.
    This ensures they are printed to stderr so crashes are visible
    when running outside a debugger.
    """
    _original_excepthook = sys.excepthook

    def exception_hook(exc_type, exc_value, exc_tb):
        # Print the full traceback to stderr
        print("\n" + "=" * 60, file=sys.stderr, flush=True)
        print("UNHANDLED EXCEPTION:", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        sys.stderr.flush()
        print("=" * 60 + "\n", file=sys.stderr, flush=True)
        
        # Also call the original hook (for debuggers, etc.)
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_hook


def install_signal_handlers():
    """Install signal handlers to catch crashes."""
    def signal_handler(signum, frame):
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"SIGNAL RECEIVED: {signal.Signals(signum).name} ({signum})", file=sys.stderr, flush=True)
        print("="*60, file=sys.stderr, flush=True)
        print("Stack trace:", file=sys.stderr, flush=True)
        traceback.print_stack(frame, file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    
    # Handle common crash signals
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, signal_handler)
    
    # On Windows, SIGABRT may be raised by C libraries
    if hasattr(signal, 'SIGABRT'):
        signal.signal(signal.SIGABRT, signal_handler)


from app.main_window import main

if __name__ == "__main__":
    install_exception_hook()
    install_signal_handlers()
    print("[DEBUG] Application starting with faulthandler enabled", flush=True)
    main()
