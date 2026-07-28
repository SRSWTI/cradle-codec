from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from cradle_codec.manifest import artifact_payload_path, manifest_from_dict, manifest_to_dict
from cradle_codec.store import LocalArtifactStore


class ArtifactHttpError(RuntimeError):
    """Transport or protocol failure while fetching a KVCodec artifact."""


def _single_query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if values is None or len(values) != 1 or values[0] == "":
        raise ValueError(f"expected exactly one non-empty {name!r} query value")
    return values[0]


def _send_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, content_type: str, payload: bytes) -> None:
    handler.send_response(int(status))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _json_payload(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def make_artifact_request_handler(root: str | Path) -> type[BaseHTTPRequestHandler]:
    """Build a minimal read-only HTTP handler for a LocalArtifactStore root.

    Routes:
    - ``GET /manifest?source_key=...`` -> manifest JSON.
    - ``GET /part?source_key=...&payload_path=...`` -> encoded payload bytes.
    """

    store = LocalArtifactStore(root)

    class ArtifactRequestHandler(BaseHTTPRequestHandler):
        server_version = "KVCodecArtifactHTTP/1"

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
            try:
                if parsed.path == "/manifest":
                    source_key = _single_query_value(query, "source_key")
                    manifest = store.load_manifest(source_key)
                    _send_response(self, HTTPStatus.OK, "application/json", _json_payload(manifest_to_dict(manifest)))
                    return
                if parsed.path == "/part":
                    source_key = _single_query_value(query, "source_key")
                    payload_path = _single_query_value(query, "payload_path")
                    artifact_dir = store.artifact_path(source_key)
                    data = artifact_payload_path(artifact_dir, payload_path).read_bytes()
                    _send_response(self, HTTPStatus.OK, "application/octet-stream", data)
                    return
                raise FileNotFoundError(parsed.path)
            except FileNotFoundError as exc:
                _send_response(self, HTTPStatus.NOT_FOUND, "text/plain", str(exc).encode("utf-8"))
            except ValueError as exc:
                _send_response(self, HTTPStatus.BAD_REQUEST, "text/plain", str(exc).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - inherited signature
            return

    return ArtifactRequestHandler


def serve_artifacts(root: str | Path, *, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create a read-only artifact HTTP server.

    The returned server is not started; tests and callers can run
    ``server.serve_forever`` on their own thread and call ``server.shutdown``.
    """

    return ThreadingHTTPServer((host, port), make_artifact_request_handler(root))


class ArtifactHttpClient:
    """Small stdlib HTTP client for remote KVCodec artifacts."""

    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        normalized = base_url.rstrip("/") + "/"
        self.base_url = normalized
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

    def _url(self, path: str, **params: str) -> str:
        return urljoin(self.base_url, path.lstrip("/")) + "?" + urlencode(params)

    def _get(self, path: str, **params: str) -> bytes:
        url = self._url(path, **params)
        request = Request(url, headers={"Accept": "application/octet-stream,application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - caller supplies artifact endpoint
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ArtifactHttpError(f"artifact HTTP {exc.code} for {path}: {detail}") from exc
        except URLError as exc:
            raise ArtifactHttpError(f"artifact HTTP transport failure for {path}: {exc.reason}") from exc

    def load_manifest(self, source_key: str):
        data = self._get("manifest", source_key=source_key)
        return manifest_from_dict(json.loads(data.decode("utf-8")))

    def fetch_part(self, source_key: str, payload_path: str) -> bytes:
        return self._get("part", source_key=source_key, payload_path=payload_path)
