# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS audit findings exclude generated trees and preserve source metadata."""

import pytest

from geniesim_cli.commands import tool


def _package(repo, relative="robot"):
    package = repo / "source/geniesim_ros/src/ros_ws/src" / relative
    package.mkdir(parents=True)
    (package / "package.xml").write_text(
        "<package><name>robot</name><version>3.2.0</version>"
        '<maintainer email="yangwendisheng@agibot.com">Genie Sim Team</maintainer>'
        '<license file="LICENSE">Mozilla Public License Version 2.0</license></package>'
    )
    (package / "LICENSE").write_text("License text\n")
    return package


def test_metadata_accepts_shared_convention(tmp_path):
    _package(tmp_path)
    assert tool._audit_metadata_ros(tmp_path) == []


@pytest.mark.parametrize("generated", ["devel", "devel_build", "devel_log", "build", "install", "external"])
def test_generated_packages_cannot_shadow_source(tmp_path, generated):
    source = _package(tmp_path)
    stale = _package(tmp_path, f"{generated}/robot")
    (stale / "package.xml").write_text("<broken")
    assert tool._enumerate_ros_packages(tmp_path) == [source]
    assert set(tool._collect_ros_packages(tmp_path)) == {"robot"}
    assert tool._audit_metadata_ros(tmp_path) == []


def test_metadata_reports_missing_license_and_maintainer(tmp_path):
    package = _package(tmp_path)
    (package / "package.xml").write_text("<package><name>robot</name><license>Other</license></package>")
    (package / "LICENSE").unlink()
    findings = tool._audit_metadata_ros(tmp_path)
    assert len(findings) == 3
    assert any("expected exactly one <maintainer>" in item for item in findings)
    assert any("<license> must be" in item for item in findings)
    assert any("missing LICENSE file" in item for item in findings)


def test_metadata_reports_malformed_xml(tmp_path):
    package = _package(tmp_path)
    (package / "package.xml").write_text("<broken")
    assert "unparseable" in tool._audit_metadata_ros(tmp_path)[0]
