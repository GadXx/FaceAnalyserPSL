from __future__ import annotations

import os
from pathlib import Path

from src.face_aesthetics import __version__
from src.face_aesthetics import data, features, models, pipelines, services, utils
from src.face_aesthetics.utils.config import load_named_configs, resolve_project_paths


def main() -> None:
    project_root = Path.cwd()
    config_dir = Path(os.getenv("FACE_AESTHETICS_CONFIG_DIR", "configs"))

    configs = load_named_configs(config_dir)
    base_config = configs["base"]
    project_paths = resolve_project_paths(
        project_root=project_root,
        configured_paths=base_config.get("paths", {}),
    )

    print(f"face_aesthetics version: {__version__}")
    print(f"config_dir: {config_dir}")
    print("import_check:")
    print(f"  data={data.__name__}")
    print(f"  features={features.__name__}")
    print(f"  models={models.__name__}")
    print(f"  pipelines={pipelines.__name__}")
    print(f"  services={services.__name__}")
    print(f"  utils={utils.__name__}")
    print("resolved_paths:")

    for key in sorted(project_paths):
        print(f"  {key}: {project_paths[key]}")


if __name__ == "__main__":
    main()
