from __future__ import annotations

import json
from pathlib import Path

from .schema import ArtifactManifest, manifest_to_dict


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
