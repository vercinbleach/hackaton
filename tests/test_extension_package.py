from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"


def test_manifest_uses_minimum_permissions_and_static_injection() -> None:
    manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 3
    assert manifest["permissions"] == ["storage"]
    assert manifest["host_permissions"] == ["http://127.0.0.1:8765/*"]
    assert manifest["content_scripts"] == [
        {
            "matches": ["https://console.cala.ai/*"],
            "js": ["content.js"],
            "css": ["styles.css"],
            "run_at": "document_idle",
        }
    ]
    assert "tabs" not in manifest["permissions"]
    assert "scripting" not in manifest["permissions"]
    assert "webRequest" not in manifest["permissions"]


def test_manifest_assets_exist() -> None:
    manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    assets = [manifest["background"]["service_worker"]]
    for content_script in manifest["content_scripts"]:
        assets.extend(content_script["js"])
        assets.extend(content_script["css"])

    assert all((EXTENSION / asset).is_file() for asset in assets)
