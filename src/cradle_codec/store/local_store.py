from __future__ import annotations

from pathlib import Path

from cradle_codec.manifest import ArtifactManifest, read_manifest, verify_part_payload

from .keys import artifact_dir_name


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def artifact_path(self, source_key: str) -> Path:
        return self.root / artifact_dir_name(source_key)

    def manifest_path(self, source_key: str) -> Path:
        return self.artifact_path(source_key) / "manifest.json"

    def has_complete_artifact(self, source_key: str) -> bool:
        manifest_path = self.manifest_path(source_key)
        if not manifest_path.is_file():
            return False
        try:
            manifest = read_manifest(manifest_path)
            for part in manifest.parts:
                verify_part_payload(manifest_path.parent, part)
        except Exception:
            return False
        return True

    def load_manifest(self, source_key: str) -> ArtifactManifest:
        return read_manifest(self.manifest_path(source_key))
