from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.api.app import create_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the morphometric FastAPI service.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run(create_app(project_root=PROJECT_ROOT), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
