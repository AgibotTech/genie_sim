# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A registered bundled SDK resolves from its manifest, not inherited paths."""

from importlib import metadata
import json
from types import SimpleNamespace

from genie_sim_engine_isaacsim.kit import bootstrap


def test_registered_bundle_resolves_real_apps_and_converter(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("6.2.0")
    (tmp_path / "apps").mkdir()
    experience = tmp_path / "apps/isaacsim.exp.full.newton.kit"
    experience.touch()
    converter = tmp_path / "exts/isaacsim.asset.importer.urdf/urdf_prebundle"
    converter.mkdir(parents=True)
    (converter.parent / "pip_prebundle").mkdir()
    monkeypatch.setenv("ISAAC_PATH", "/unrelated/combo/sdk")
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda name: json.dumps({"root": str(tmp_path)})),
    )
    monkeypatch.setattr(bootstrap, "_ISAAC_PKG_LITERAL", "/unrelated/pip/sdk")
    result = bootstrap.resolve_isaac_paths()
    assert result["package_root"] == {"path": str(tmp_path), "how": "bundled", "exists": True}
    assert result["kit_experience"] == {"path": str(experience), "how": "bundled", "exists": True}
    assert result["urdf_prebundle"] == {"path": str(converter), "how": "bundled", "exists": True}


def test_pip_layout_has_no_bundled_override(monkeypatch, tmp_path):
    monkeypatch.setattr(metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda name: None))
    monkeypatch.setattr(bootstrap, "_ISAAC_PKG_LITERAL", str(tmp_path))
    monkeypatch.setattr(bootstrap, "_isaac_package_root", lambda: (str(tmp_path), "importlib"))
    experience = tmp_path / "apps/isaacsim.exp.full.newton.kit"
    experience.parent.mkdir()
    experience.touch()
    assert bootstrap.resolve_isaac_paths()["kit_experience"]["path"] == str(experience)
