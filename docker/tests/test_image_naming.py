# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pin the unified image naming convention from docker/AGENTS.md.

Images are one tag per family major — ``geniesim{2,3,4}:latest``. A
minor-suffixed name builds a tag that no downstream ``FROM`` reads, so the base
looks rebuilt while every image above it keeps the stale layer. That failure is
silent, hence these tests.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKER_PY = REPO / "source/geniesim_cli/src/geniesim_cli/commands/docker.py"

# A minor suffix is always digits-dot-digits, so require the trailing digit:
# prose that ends a sentence with a family name is not a violation. This
# pattern holds no example of the forbidden spelling, so this file — which is
# itself scanned once tracked — does not trip its own guard.
MINOR_SUFFIXED = re.compile(r"geniesim[0-9]+\.[0-9]")

# The only files allowed to quote the forbidden spelling, because they exist to
# forbid it. Keep identical to the ``git grep`` pathspec in docker/AGENTS.md.
RULE_FILES = {"AGENTS.md", "docker/AGENTS.md"}


def _tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def _variant_images():
    """Map ``dockerfile`` -> ``image`` from VARIANTS, without importing the CLI."""
    tree = ast.parse(DOCKER_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = getattr(node, "target", None)
        if not isinstance(node, ast.AnnAssign) or getattr(target, "id", None) != "VARIANTS":
            continue
        mapping = node.value.args[0]  # MappingProxyType({...})
        images = {}
        for variant in mapping.values:
            kwargs = {k.arg: k.value for k in variant.keywords}
            images[ast.literal_eval(kwargs["dockerfile"])] = ast.literal_eval(kwargs["image"])
        return images
    raise AssertionError(f"VARIANTS not found in {DOCKER_PY.relative_to(REPO)}")


def test_no_minor_suffixed_image_names():
    """The documented guard: nothing outside the rule files may use the old spelling."""
    offenders = []
    for rel in _tracked_files():
        if rel in RULE_FILES or not rel.startswith(("docker/", "source/geniesim_cli/")):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; cannot carry an image name
        for lineno, line in enumerate(text.splitlines(), 1):
            if MINOR_SUFFIXED.search(line):
                offenders.append(f"  {rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Minor-suffixed Docker image names are forbidden — use the family major "
        "only (geniesim<N>:latest). See 'Image naming' in docker/AGENTS.md.\n" + "\n".join(offenders)
    )


def test_guard_pattern_catches_the_old_spelling_and_spares_valid_strings():
    """Pin the pattern itself, so the scan above cannot silently stop matching."""
    # Concatenated so this file holds no literal instance of the old spelling.
    assert MINOR_SUFFIXED.search("geniesim4" ".0:latest")
    assert MINOR_SUFFIXED.search("geniesim3" ".5")
    assert MINOR_SUFFIXED.search("geniesim2" ".0")
    for valid in (
        "registry.agibot.com/genie-sim/geniesim3:latest",
        "registry.agibot.com/genie-sim/geniesim4:latest",
        "geniesim3-data-collection:latest",
        "rlinf-geniesim:cloud",
        "geniesim docker5.1 build",
        "docker build -f docker/Dockerfile.5.1",
        "ISAACSIM_TAG=6.2.0.1",
        "the CLI runs geniesim3.",  # prose ending a sentence on a family name
    ):
        assert not MINOR_SUFFIXED.search(valid), valid


def test_variant_images_are_family_major_tags():
    """Every CLI-dispatched image is ``geniesim<N>:latest``, family major only."""
    images = _variant_images()
    assert images, "no variants parsed from docker.py"
    canonical = re.compile(r"^registry\.agibot\.com/genie-sim/geniesim[0-9]+:latest$")
    for dockerfile, image in sorted(images.items()):
        assert canonical.match(image), f"{dockerfile} declares a non-canonical image: {image}"


def test_dockerfile_headers_build_the_tag_the_cli_runs():
    """A copy-pasted header build must not produce a tag nothing consumes."""
    for dockerfile, image in sorted(_variant_images().items()):
        text = (REPO / "docker" / dockerfile).read_text(encoding="utf-8")
        tags = re.findall(r"-t\s+(registry\.agibot\.com/\S+)", text)
        assert tags, f"docker/{dockerfile} header has no -t build tag to check"
        wrong = sorted(set(tags) - {image})
        assert not wrong, (
            f"docker/{dockerfile} header builds {wrong} but the CLI runs {image}; "
            "a hand-run build would produce an orphan tag."
        )


def test_start_sh_fallback_is_a_known_canonical_image():
    """start.sh's fallback only applies to direct callers, so nothing else pins it.

    Asserted loosely on purpose: it must be an image some variant declares, not
    a minor-suffixed or invented tag. Which family it *should* default to is a
    separate open question — it names geniesim4 while ``_env.py`` defaults to
    geniesim3.
    """
    text = (REPO / "docker/start.sh").read_text(encoding="utf-8")
    match = re.search(r'IMAGE="\$\{GENIESIM_IMAGE:-([^}"]+)\}"', text)
    assert match, "start.sh GENIESIM_IMAGE fallback not found"
    fallback = match.group(1)
    assert not MINOR_SUFFIXED.search(fallback), fallback
    assert fallback in set(_variant_images().values()), f"start.sh falls back to {fallback}, which no variant declares"


@pytest.mark.skipif(
    not all((REPO / "docker" / name).is_file() for name in ("Dockerfile.post-train", "Dockerfile.post-train-teleop")),
    reason="requires the optional post-training image recipes",
)
def test_downstream_images_pin_unsuffixed_bases():
    """Post-training layers inherit the selected family base and server image."""
    for name, argument, base in (
        ("Dockerfile.post-train", "BASE_IMAGE", _variant_images()["Dockerfile"]),
        ("Dockerfile.post-train-teleop", "SERVER_IMAGE", "geniesim-post-train:server"),
    ):
        lines = (REPO / "docker" / name).read_text(encoding="utf-8").splitlines()
        assert f"ARG {argument}={base}" in lines
        assert f"FROM ${{{argument}}}" in lines
