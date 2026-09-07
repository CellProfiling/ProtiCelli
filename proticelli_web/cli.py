"""Command-line entry point for the ProtiCelli browser interface."""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProtiCelli Interactive Gallery locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mode",
        choices=("auto", "real", "demo"),
        default=os.getenv("PROTICELLI_WEB_MODE", "auto"),
        help="Inference engine. Auto uses real checkpoints when they are present.",
    )
    parser.add_argument("--reload", action="store_true", help="Reload after source edits")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local interface automatically")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print PyTorch/CUDA diagnostics and exit",
    )
    args = parser.parse_args()

    if args.diagnose:
        from .inference import runtime_diagnostics

        print(json.dumps(runtime_diagnostics(os.getenv("PROTICELLI_WEB_DEVICE", "auto")), indent=2))
        return

    os.environ["PROTICELLI_WEB_MODE"] = args.mode

    import uvicorn

    if not args.no_browser and not args.reload:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        opener = threading.Timer(1.2, webbrowser.open, args=(f"http://{browser_host}:{args.port}",))
        opener.daemon = True
        opener.start()

    uvicorn.run(
        "proticelli_web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
