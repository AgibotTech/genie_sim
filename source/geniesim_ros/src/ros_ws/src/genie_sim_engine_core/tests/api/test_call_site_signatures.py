# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Every in-tree call site must bind against the callee it actually reaches.

The bug this exists for
-----------------------
``setup/model.py`` called ``mjc_contact.apply_pre_finalize(...,
pad_body_indices=..., physics_params=...)``.  ``apply_pre_finalize`` accepts
neither and has no ``**kwargs``.  The call sat unconditionally inside
``_phase_build_model`` <- ``_build()`` <- ``__init__``, so the newton backend
could not be constructed at all — and *nothing* caught it before runtime,
because the cross-module imports are lazy function-body imports: the module
pair never even meets at import time, let alone at type-check time.  There is
no mypy gate on this tree, and a unit test only finds it if the unit test
happens to construct the thing that is broken.

So this module does statically what nothing else here does: it resolves call
sites to callees *inside this repo* and binds the arguments against the
callee's real signature.  It is deliberately AST-only — importing these
modules needs ROS, warp, newton, and a GPU, none of which a signature check
should require.

Four layers, each with its own soundness rule
---------------------------------------------
Precision matters more than reach here: one false positive and the guard gets
switched off, which costs more than the drift it would have caught.  Every
layer therefore resolves the callee *positively* and skips whatever it cannot
resolve, rather than guessing from a name.

A. **Imported functions** — ``from <repo module> import f`` then ``f(...)``.
   Skipped if the name is rebound anywhere in the calling module, or defined
   more than once in the callee module.  This is the layer that catches the
   bug above.
B. **Overrides** — an override must accept every call its base permits (LSP).
   This is what turns a widened base signature into a failure at the base,
   instead of at whichever subclass a run happens to reach first.
C. **Inferred attribute receivers** — ``self._x.m(...)`` where ``self._x`` is
   assigned from exactly one in-repo constructor *anywhere in the inheritance
   family*.  The family walk is load-bearing: ``self._lifecycle`` is assigned
   in ``engine_base.py`` and used from a mixin in ``control.py``, which is the
   cross-file, cross-class shape a per-class scan misses.
   Notably NOT done by name alone: a "this method name is unique in the repo,
   so that must be the callee" heuristic reports ``ndarray.mean(axis=...)``
   against a private ``_Acc.mean()`` and the real ROS logger's ``warning``
   against ``SimpleLogger.warning``.  Both are the wrong object.
D. **Self calls** — ``self.m(...)`` bound against the class's own MRO, only
   for classes whose bases all resolve in-repo (an out-of-tree base may define
   ``m``), and only for names never assigned as an attribute (a callable
   attribute shadows the method).

Test files are not scanned.  A stub with a deliberately narrow arity is the
point of several of them (see ``pacing/test_lifecycle.py``'s
``UpstreamAritySim``), and asserting they match the tree would invert it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import pytest

_SRC = Path(__file__).resolve().parents[3]
_PKGS = ("genie_sim_engine_core", "genie_sim_engine_newton")


class Finding(NamedTuple):
    module: str
    lineno: int
    call: str
    target: str
    error: str
    signature: str

    def __str__(self) -> str:  # what a failure actually reads like
        return f"{self.module}:{self.lineno}: {self.call}(...) -> {self.target}{self.signature}: {self.error}"


# ----------------------------------------------------------------------
# Source index
# ----------------------------------------------------------------------


def _load_tree() -> Dict[str, ast.Module]:
    """Dotted module name -> parsed AST, for every shipped module present."""
    out: Dict[str, ast.Module] = {}
    for pkg in _PKGS:
        root = _SRC / pkg / pkg
        if not root.is_dir():
            continue  # sibling package not checked out; scan what is here
        for f in sorted(root.rglob("*.py")):
            if f.name == "conftest.py" or f.name.startswith("test_"):
                continue
            parts = list(f.relative_to(_SRC / pkg).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            out[".".join(parts)] = ast.parse(f.read_text(), filename=str(f))
    return out


def _signature(fn: ast.AST) -> inspect.Signature:
    """An ``inspect.Signature`` for a ``def``, faithful about what is optional.

    Defaults become ``None`` rather than their real value — this only ever
    binds placeholder arguments, so the value is irrelevant and the *presence*
    of a default is the whole point.
    """
    a = fn.args
    P, empty = inspect.Parameter, inspect.Parameter.empty
    params = []
    positional = a.posonlyargs + a.args
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    padded = [empty] * (len(positional) - len(a.defaults)) + [None] * len(a.defaults)
    for arg, default in zip(positional, padded):
        params.append(P(arg.arg, P.POSITIONAL_OR_KEYWORD, default=default))
    if a.vararg:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        params.append(P(arg.arg, P.KEYWORD_ONLY, default=(empty if default is None else None)))
    if a.kwarg:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))
    return inspect.Signature(params)


def _bindable(call: ast.Call) -> bool:
    """``f(*xs)`` / ``f(**kw)`` hide the arity — nothing to check."""
    return not any(isinstance(a, ast.Starred) for a in call.args) and all(k.arg for k in call.keywords)


def _try_bind(sig: inspect.Signature, call: ast.Call) -> str:
    try:
        sig.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})
    except TypeError as exc:
        return str(exc)
    return ""


def _base_names(cls: ast.ClassDef) -> List[str]:
    out = []
    for b in cls.bases:
        name = getattr(b, "id", getattr(b, "attr", None))
        if name:
            out.append(name)
    return out


class Index:
    """Whole-tree name resolution: functions, classes, MROs, attribute types."""

    def __init__(self, mods: Dict[str, ast.Module]):
        self.mods = mods
        self.funcs: Dict[str, Dict[str, List[ast.AST]]] = {}
        by_name: Dict[str, List[Tuple[str, ast.ClassDef]]] = {}
        for mod, tree in mods.items():
            self.funcs[mod] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.funcs[mod].setdefault(node.name, []).append(node)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    by_name.setdefault(node.name, []).append((mod, node))
        # A duplicated class name is unresolvable, so it is simply not indexed.
        self.classes = {n: v[0] for n, v in by_name.items() if len(v) == 1}
        self.family = {n: self._family(n) for n in self.classes}
        self.members: Dict[str, set] = {}
        for name, fam in self.family.items():
            for member in fam:
                self.members.setdefault(member, set()).add(name)
        self.methods = {n: self._methods(n) for n in self.classes}
        self.attrs = {n: self._attrs(n) for n in self.classes}

    def _family(self, name: str, seen=None) -> set:
        seen = set() if seen is None else seen
        if name in seen or name not in self.classes:
            return seen
        seen.add(name)
        for base in _base_names(self.classes[name][1]):
            self._family(base, seen)
        return seen

    def _methods(self, name: str) -> Dict[str, Tuple[str, ast.AST]]:
        out: Dict[str, Tuple[str, ast.AST]] = {}
        if name not in self.classes:
            return out
        for base in reversed(_base_names(self.classes[name][1])):
            out.update(self._methods(base))
        for node in self.classes[name][1].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[node.name] = (name, node)
        return out

    def _attrs(self, name: str) -> Dict[str, set]:
        """``self.x = SomeRepoClass(...)`` -> ``{"x": {"SomeRepoClass"}}``."""
        out: Dict[str, set] = {}
        for node in ast.walk(self.classes[name][1]):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and isinstance(node.value, ast.Call)
            ):
                continue
            ctor = getattr(node.value.func, "id", getattr(node.value.func, "attr", None))
            if ctor in self.classes:
                out.setdefault(target.attr, set()).add(ctor)
        return out

    def attr_type(self, user: str, attr: str):
        """Type of ``self.<attr>`` as seen from inside class ``user``.

        Widened across every concrete class that mixes ``user`` in, because
        the assignment and the use routinely live in different classes.  Two
        concrete classes disagreeing means no answer, not a guess.
        """
        found = set()
        for concrete in self.members.get(user, {user}):
            for member in self.family.get(concrete, {concrete}):
                found |= self.attrs.get(member, {}).get(attr, set())
        return next(iter(found)) if len(found) == 1 else None

    def closed(self, name: str, seen=None) -> bool:
        """True when the whole base chain is in this repo."""
        seen = set() if seen is None else seen
        if name in seen:
            return True
        seen.add(name)
        if name not in self.classes:
            return False
        return all(self.closed(b, seen) for b in _base_names(self.classes[name][1]))


# ----------------------------------------------------------------------
# The four layers
# ----------------------------------------------------------------------


def layer_imported_functions(idx: Index) -> Tuple[List[Finding], int]:
    findings, checked = [], 0
    for mod, tree in idx.mods.items():
        rebound = set()
        alias: Dict[str, Tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                rebound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rebound.add(node.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in idx.funcs:
                for n in node.names:
                    alias[n.asname or n.name] = (node.module, n.name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name in rebound or name not in alias:
                continue
            src_mod, src_name = alias[name]
            defs = idx.funcs[src_mod].get(src_name, [])
            if len(defs) != 1 or not _bindable(node):
                continue
            checked += 1
            sig = _signature(defs[0])
            err = _try_bind(sig, node)
            if err:
                findings.append(Finding(mod, node.lineno, name, f"{src_mod}.{src_name}", err, str(sig)))
    return findings, checked


def layer_overrides(idx: Index) -> Tuple[List[Finding], int]:
    findings, checked = [], 0
    for name, (mod, cls) in idx.classes.items():
        for base in _base_names(cls):
            if base not in idx.classes:
                continue
            base_methods = idx.methods[base]
            for node in cls.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name == "__init__" or node.name not in base_methods:
                    continue
                base_sig = _signature(base_methods[node.name][1])
                if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in base_sig.parameters.values()):
                    continue  # base takes anything; the override cannot narrow it observably
                over = _signature(node)
                positional = [
                    p for p in base_sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                checked += 1
                for args, kwargs in (([], {p: None for p in base_sig.parameters}), ([None] * len(positional), {})):
                    try:
                        over.bind(*args, **kwargs)
                    except TypeError as exc:
                        findings.append(
                            Finding(mod, node.lineno, f"{name}.{node.name}", f"{base}.{node.name}", str(exc), str(over))
                        )
                        break
    return findings, checked


def layer_attribute_receivers(idx: Index) -> Tuple[List[Finding], int]:
    findings, checked = [], 0
    for mod, tree in idx.mods.items():
        for cls in ast.walk(tree):
            if not (isinstance(cls, ast.ClassDef) and cls.name in idx.classes):
                continue
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                recv = node.func.value
                if not (
                    isinstance(recv, ast.Attribute) and isinstance(recv.value, ast.Name) and recv.value.id == "self"
                ):
                    continue
                ty = idx.attr_type(cls.name, recv.attr)
                if ty is None or node.func.attr not in idx.methods.get(ty, {}):
                    continue
                if not _bindable(node):
                    continue
                owner, fn = idx.methods[ty][node.func.attr]
                sig = _signature(fn)
                checked += 1
                err = _try_bind(sig, node)
                if err:
                    call = f"self.{recv.attr}.{node.func.attr}"
                    findings.append(Finding(mod, node.lineno, call, f"{owner}.{node.func.attr}", err, str(sig)))
    return findings, checked


_NOT_A_PLAIN_METHOD = ("property", "staticmethod", "classmethod", "setter", "getter", "cached_property")


def layer_self_calls(idx: Index) -> Tuple[List[Finding], int]:
    findings, checked = [], 0
    for mod, tree in idx.mods.items():
        for cls in ast.walk(tree):
            if not (isinstance(cls, ast.ClassDef) and cls.name in idx.classes and idx.closed(cls.name)):
                continue
            shadowed = {a for fam in idx.family[cls.name] for a in idx.attrs.get(fam, {})}
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
                    continue
                name = node.func.attr
                if name in shadowed or name not in idx.methods[cls.name] or not _bindable(node):
                    continue
                owner, fn = idx.methods[cls.name][name]
                decorated = [getattr(d, "id", getattr(d, "attr", "")) for d in fn.decorator_list]
                if any(d in _NOT_A_PLAIN_METHOD for d in decorated):
                    continue
                sig = _signature(fn)
                checked += 1
                err = _try_bind(sig, node)
                if err:
                    findings.append(Finding(mod, node.lineno, f"self.{name}", f"{owner}.{name}", err, str(sig)))
    return findings, checked


_LAYERS = {
    "imported functions": layer_imported_functions,
    "overrides": layer_overrides,
    "attribute receivers": layer_attribute_receivers,
    "self calls": layer_self_calls,
}

_INDEX = Index(_load_tree())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_every_resolvable_call_site_binds(layer):
    findings, _ = _LAYERS[layer](_INDEX)
    assert not findings, "\n".join(str(f) for f in findings)


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_the_layer_resolved_something(layer):
    """A scan that resolves nothing passes for the wrong reason.

    Renaming a package, or moving the tests, would silently empty the index
    and leave a green guard that checks zero call sites.
    """
    _, checked = _LAYERS[layer](_INDEX)
    assert checked > 0, f"{layer!r} bound no call sites — the scan is not reaching the tree"


def test_the_tree_index_found_both_packages_or_says_which():
    """``genie_sim_engine_newton`` is a sibling package, not a dependency."""
    assert any(m.startswith("genie_sim_engine_core") for m in _INDEX.mods)
    if not (_SRC / "genie_sim_engine_newton" / "genie_sim_engine_newton").is_dir():
        pytest.skip("genie_sim_engine_newton not checked out; core-only scan")
    assert any(m.startswith("genie_sim_engine_newton") for m in _INDEX.mods)


# The four planted breaks below are the ones the layers exist to catch, in the
# shape they took in this tree.  They keep the scanner honest: a refactor that
# quietly stops resolving callees fails here rather than going green.

_PLANTED = {
    "imported functions": (
        # The real one, minus the file names: setup/model.py -> mjc_contact.
        {
            "pkg.callee": "def apply_pre_finalize(builder, *, scene_cfg, logger):\n    pass\n",
            "pkg.caller": "from pkg.callee import apply_pre_finalize\n"
            "def go(b):\n"
            "    apply_pre_finalize(b, scene_cfg=None, logger=None, pad_body_indices=[])\n",
        },
        "pad_body_indices",
    ),
    "overrides": (
        {
            "pkg.m": "class Base:\n"
            "    def episode_reset(self, env_ids=None, seeds=None, instances=None):\n"
            "        pass\n"
            "class Sub(Base):\n"
            "    def episode_reset(self, env_ids=None, seed=None):\n"
            "        pass\n",
        },
        "seeds",  # the first parameter the narrowed override cannot accept
    ),
    "attribute receivers": (
        {
            "pkg.m": "class Coord:\n"
            "    def drain_resets(self):\n"
            "        pass\n"
            "class Node:\n"
            "    def __init__(self):\n"
            "        self._lifecycle = Coord()\n"
            "    def tick(self):\n"
            "        self._lifecycle.drain_resets(env_ids=None)\n",
        },
        "env_ids",
    ),
    "self calls": (
        {
            "pkg.m": "class Engine:\n"
            "    def teleport(self, requests):\n"
            "        pass\n"
            "    def tick(self):\n"
            "        self.teleport()\n",
        },
        "requests",
    ),
}


@pytest.mark.parametrize("layer", sorted(_PLANTED))
def test_the_layer_catches_a_planted_break(layer):
    sources, expected = _PLANTED[layer]
    idx = Index({name: ast.parse(text) for name, text in sources.items()})
    findings, checked = _LAYERS[layer](idx)
    assert checked, f"{layer!r} did not even resolve the planted call site"
    assert len(findings) == 1, [str(f) for f in findings]
    assert expected in findings[0].error, str(findings[0])
