#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Compare two benchmark model runs for a statistically significant score difference.

WHY THIS METHOD (read before changing it)
-----------------------------------------
A benchmark run = a set of rollouts. One rollout = one (layout, instruction),
i.e. a specific scene arrangement + instruction, each run once.

Two facts forced the design:

1. Run-to-run differences are real and NOT measurable from a single run.
   Re-running the SAME model on different hardware/GPU state shifts the total
   score by a non-trivial amount (~0.04 in our data). A single run cannot see
   this, so we deliberately do not try to model it.

2. We refuse any method that would flag two runs of the SAME model as
   "significantly different". A paired / 150-iid test that treats correlated
   repeats as the noise unit underestimates variance and calls same-model
   reruns different. Unacceptable.

The method:

   * Unit of analysis = one rollout = one (instruction, layout) instance. Every
     rollout is an independent layout, so there are no repeats to average.
   * Compare the two models' per-unit score means UNPAIRED with a two-sample
     Welch t-test (unequal variances). The CLT makes this exact enough; a
     bootstrap would only Monte-Carlo the same number, so we don't.

NOT pairing makes the between-layout difficulty spread act as a noise buffer.
That buffer is what keeps same-model reruns from reading as different, at the
cost of being conservative: we would rather say "not yet significant" than ever
conclude a false difference.

Consequence: a true model gap near the run-to-run floor (~0.04) may read as
"not significant". To gain power, add more layouts / instructions - they are the
independent units.

Metric: step_mean (default, == leaderboard `statistics.average`) or e2e.

Usage
-----
    python compare_models.py --a <runA_dir> --b <runB_dir>
        [--name-a A --name-b B] [--metric step_mean|e2e] [--merge]
        [--ci 0.95] [--json out.json] [--plot out.png]

--paired opts OUT of the conservative unpaired default: it aligns A and B by
(task group, position) and runs a PAIRED t-test on per-layout differences. This
removes the between-layout difficulty buffer above, so it is far more powerful —
but it is ONLY valid when the two runs replay the SAME layouts in the same order
and run-to-run noise is negligible (deterministic pipeline). With a real
run-to-run floor it WILL flag same-model reruns as different.

A and B are matched by CONTENT, never by path: each evaluate json is identified
by (task_name, set of instructions it contains). Same-signature files within a
run merge into one task group; groups pair up across runs by signature, so file
names and directory layout are irrelevant.
  * default       - one independent test per content-matched task group;
                    per-task p-values are BH (FDR) corrected across the table,
                    so the ✅ marks survive multiple comparisons
  * --merge       - pool ALL tasks from ALL files under the dir into ONE unit
                    pool and give a single verdict. This is how low-unit tasks
                    (e.g. a manip task with only ~10 units) become testable:
                    pooled together they reach a usable unit count. A per-task
                    breakdown is still printed, but it is descriptive only.

<run_dir> contains, at any depth, `evaluate*.json` files whose `details` hold the
per-rollout results. Each rollout is one unit, keyed by (file, index, instruction).
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


# --------------------------------------------------------------------------- #
# Terminal colours
# --------------------------------------------------------------------------- #
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "black": "\033[30m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "grey": "\033[90m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_red": "\033[41m",
}


class Colorizer:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, text, *names):
        if not self.enabled or not names:
            return text
        return "".join(_ANSI[n] for n in names) + text + _ANSI["reset"]

    def band(self, text, *names, width=66):
        """A full-width centred colour band (for the headline verdict)."""
        return self(f" {text} ".center(width), *names)


def _color_enabled(no_color_flag):
    if no_color_flag or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _short(path, width):
    """Left-truncate a long relpath, keeping the informative tail."""
    return path if len(path) <= width else "…" + path[-(width - 1) :]


def _disp_w(s):
    """Terminal display width (CJK fullwidth chars count as 2)."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in str(s))


def _pad(s, width, right=False):
    """Pad by DISPLAY width, so CJK names align correctly."""
    s = str(s)
    gap = " " * max(0, width - _disp_w(s))
    return gap + s if right else s + gap


class _Tee:
    """Mirror stdout while recording everything printed (incl. ANSI codes)."""

    def __init__(self, stream):
        self.stream = stream
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def text(self):
        return "".join(self.buf)

    def begin_capture(self):
        self._start = len(self.buf)

    def end_capture(self):
        self._end = len(self.buf)

    def captured_text(self):
        s = getattr(self, "_start", 0)
        e = getattr(self, "_end", len(self.buf))
        return "".join(self.buf[s:e])


# symbols PIL fonts usually lack glyphs for -> safe substitutes in the PNG
_PNG_SUBS = {"✅": "√", "✔": "√", "✘": "x", "❗": "!", "⚠️": "(!)", "⚠": "(!)", "\ufe0f": ""}


def render_terminal_png(text, path, scale=2.0):
    """Render captured terminal output (with ANSI colours) to a PNG image.

    scale: font-size multiplier (2.0 -> ~36px glyphs, crisp on hi-dpi screens).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[warn] Pillow not installed; skipping table PNG")
        return
    import re
    import unicodedata

    size = max(10, int(18 * scale))

    def _load(paths):
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return None

    mono = _load(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"])
    mono_b = _load(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]) or mono
    cjk = (
        _load(
            [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            ]
        )
        or mono
    )
    if mono is None:
        print("[warn] no usable monospace font found; skipping table PNG")
        return

    for k, v in _PNG_SUBS.items():
        text = text.replace(k, v)
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    cols = max((_disp_w(ansi_re.sub("", ln)) for ln in lines), default=80)
    cw = mono.getlength("M")
    lh = int(size * 1.5)
    pad = size
    FG = {
        "31": (255, 106, 96),
        "91": (255, 106, 96),
        "32": (87, 220, 120),
        "92": (87, 230, 120),
        "33": (240, 200, 90),
        "93": (255, 224, 102),
        "36": (100, 200, 220),
        "96": (110, 220, 240),
        "30": (30, 32, 40),
        "90": (140, 145, 155),
        "97": (245, 245, 245),
    }
    BG = {"41": (176, 58, 58), "42": (38, 145, 70), "43": (196, 160, 36)}
    default_fg = (222, 226, 230)
    img = Image.new("RGB", (int(pad * 2 + cols * cw), pad * 2 + len(lines) * lh), (24, 26, 32))
    d = ImageDraw.Draw(img)
    for row, ln in enumerate(lines):
        x = 0.0
        fg, bold, dim, bg = default_fg, False, False, None
        for part in re.split(r"(\x1b\[[0-9;]*m)", ln):
            m = re.match(r"\x1b\[([0-9;]*)m", part)
            if m:
                for code in (m.group(1) or "0").split(";"):
                    if code in ("", "0"):
                        fg, bold, dim, bg = default_fg, False, False, None
                    elif code == "1":
                        bold = True
                    elif code == "2":
                        dim = True
                    elif code in FG:
                        fg = FG[code]
                    elif code in BG:
                        bg = BG[code]
                continue
            for ch in part:
                wch = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
                px, py = pad + x * cw, pad + row * lh
                if bg:
                    d.rectangle([px, py, px + wch * cw, py + lh], fill=bg)
                col = tuple(int(v * 0.6) for v in fg) if dim else fg
                font = cjk if wch == 2 else (mono_b if bold else mono)
                d.text((px, py + 2), ch, font=font, fill=col)
                if bold and wch == 2:  # faux bold for CJK
                    d.text((px + 1, py + 2), ch, font=font, fill=col)
                x += wch
    img.save(path, dpi=(144, 144))
    print(f"Table PNG -> {path}  ({img.width}x{img.height})")


def _finish_png(args):
    """Restore stdout and render the captured output to PNG (default on)."""
    tee = getattr(args, "_tee", None)
    if tee is None:
        return
    sys.stdout = tee.stream
    if args.png and not args.no_png:
        render_terminal_png(tee.captured_text(), args.png, getattr(args, "png_scale", 2.0))


# --------------------------------------------------------------------------- #
# CSV export — mirrors the printed table exactly (same rows, columns, formats)
# --------------------------------------------------------------------------- #
def write_csv(path, fields, rows):
    import csv

    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel friendly
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {path}")


# --------------------------------------------------------------------------- #
# Per-rollout score
# --------------------------------------------------------------------------- #
def _rollout_scalar(detail, metric, layout=None):
    """Use the episode's complete layout; never infer a shorter denominator."""
    from geniesim_benchmark.plugins.output_system.eval_utils import get_task_steps, extract_scores
    from geniesim_benchmark.plugins.output_system.score_groups import episode_complete, score_layout_steps

    result = detail.get("result", {})
    if result.get("code", 0) not in (0, 2):
        raise ValueError("Evaluation contains an interrupted/invalid episode; compare complete runs")
    layout = result.get("score_layout") or get_task_steps(
        detail.get("score_layout_name") or detail.get("sub_task_name", ""), detail.get("instruction_mode", "full")
    ) or layout
    if not layout:
        raise ValueError("Missing scoring layout; cannot recover the episode denominator")
    # Old exports may contain the #200-compressed STEPS. Prefer the original
    # progress when available; ambiguous legacy grouping is rejected explicitly.
    if result.get("score_schema_version") is None and result.get("progress"):
        steps = score_layout_steps(result["progress"], layout)
    else:
        steps = extract_scores(result, layout)
    if metric == "e2e":
        return float(episode_complete(result, steps, layout))
    return sum(s["score"] for s in steps) / len(layout)


# --------------------------------------------------------------------------- #
# Task recovery
# --------------------------------------------------------------------------- #
def load_task_scores(root, metric):
    """Load one run and return per-task scores.

    Files are identified by CONTENT, not by path: a file's signature is
    (task_name, the set of distinct instructions it contains). Files with the
    same signature within a run are merged into one group; across two runs,
    groups pair up by signature — so file names / directory layout are
    irrelevant for matching.

    Each rollout is an independent (instruction, layout) instance and counts as
    one unit; there are no repeats to average.

    Returns (scores, groups, info):
      scores : np.ndarray of one score per unit
      groups : {content_label: np.ndarray of that group's unit scores}
      info   : dict with n_tasks, n_rollouts, n_files, groups
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No *.json found under {root!r}")

    tasks = defaultdict(list)  # (file, rollout_index, instruction) -> [rollout scalar]
    file_sig = {}  # file relpath -> content signature
    file_subname = {}  # file relpath -> sub_task_name from content (or None)
    n_rollouts = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] skipping unreadable {fp}: {e}")
            continue
        rel = os.path.relpath(fp, root)
        rows = []  # (instruction, scalar) in file order
        task_name = None
        # sub_task_name lives in the statistics block of newer outputs; also
        # accept it at the top level or inside details for robustness
        stats = payload.get("statistics") or {}
        sub_name = stats.get("sub_task_name") or payload.get("sub_task_name")
        for detail in payload.get("details", []):
            instr = detail.get("task_instruction")
            if instr is None:
                continue
            task_name = task_name or detail.get("task_name")
            sub_name = sub_name or detail.get("sub_task_name")
            v = _rollout_scalar(detail, metric, [s["name"] for s in stats.get("weighted_step_scores", [])])
            if v is not None:
                rows.append((instr, v))
        if not rows:
            continue
        # signature: prefer the explicit sub_task_name; fall back to instruction set
        key2 = sub_name if sub_name else frozenset(r[0] for r in rows)
        file_sig[rel] = (task_name or "?", key2)
        file_subname[rel] = sub_name
        # each rollout is an independent (instruction, layout) instance -> one unit
        for gi, (instr, v) in enumerate(rows):
            tasks[(rel, gi, instr)].append(v)
            n_rollouts += 1

    if not tasks:
        raise SystemExit(f"No scorable rollouts under {root!r} (metric={metric}).")
    scores = np.array([np.mean(v) for v in tasks.values()], dtype=float)

    # Group task scores by content signature (merging same-signature files).
    # by_sig_instr keeps the per-unit instruction in the SAME order as by_sig,
    # so a paired run can align A and B by position and sanity-check the match.
    by_sig = defaultdict(list)
    by_sig_instr = defaultdict(list)
    for (rel, _, instr), v in tasks.items():
        by_sig[file_sig[rel]].append(np.mean(v))
        by_sig_instr[file_sig[rel]].append(instr)

    # Run-independent KEY = task_name (+ content hash when several groups share
    # one task_name) — deterministic, so both runs derive the same key and
    # pairing never depends on paths.
    def _sig_hash(key2):
        s = key2 if isinstance(key2, str) else "\n".join(sorted(key2))
        return hashlib.md5(s.encode()).hexdigest()[:6]

    labels = {sig: sig[0] for sig in by_sig}
    dup = {lab for lab, n in Counter(labels.values()).items() if n > 1}
    for sig, lab in labels.items():
        if lab in dup:
            labels[sig] = f"{lab}@{_sig_hash(sig[1])}"
    groups = {labels[sig]: np.asarray(v, float) for sig, v in by_sig.items()}
    group_instr = {labels[sig]: list(by_sig_instr[sig]) for sig in by_sig}

    # DISPLAY name, in order of preference:
    #   1. sub_task_name from the json content (statistics block of newer outputs)
    #   2. the json FILE NAME (stem), e.g. pick_block_color.json -> pick_block_color,
    #      unless it is a generic name like evaluate.json / evaluate_ret_07.json
    #   3. the sub_task directory the group's files live under
    #   4. the task_name
    # Used for printing only — never for matching.
    def _file_label(rel):
        stem = os.path.splitext(os.path.basename(rel))[0]
        return None if stem.lower().startswith("evaluate") else stem

    sig_name = {}
    for rel, sig in file_sig.items():
        if sig not in sig_name:
            sig_name[sig] = (
                file_subname.get(rel) or _file_label(rel) or os.path.basename(os.path.dirname(rel)) or sig[0]
            )
    names = {labels[sig]: sig_name[sig] for sig in by_sig}

    info = {
        "n_tasks": len(tasks),
        "n_rollouts": n_rollouts,
        "n_files": len(files),
        "groups": sorted(groups),
    }
    return scores, groups, names, group_instr, info


# --------------------------------------------------------------------------- #
# Unpaired, task-as-unit comparison
# --------------------------------------------------------------------------- #
def compare(a_scores, b_scores, ci):
    """Two-sample Welch t-test of D = mean(A) - mean(B), the TASK as the unit.

    Returns the difference, its (1-α) CI and two-sided p. CI and p come from the
    SAME t/df, so 'CI excludes 0' and 'p < α' always agree. Degenerate inputs
    (n<2, or both groups constant) are not testable and stay conservative
    (p=1, not significant) rather than reporting a spurious zero-width CI.
    """
    na, nb = a_scores.size, b_scores.size
    ma, mb = float(a_scores.mean()), float(b_scores.mean())
    d = ma - mb
    out = {"score_a": ma, "score_b": mb, "diff": d, "ci_lo": d, "ci_hi": d, "p": 1.0, "significant": False, "df": 0.0}
    if na < 2 or nb < 2:
        return out
    va, vb = float(a_scores.var(ddof=1)), float(b_scores.var(ddof=1))
    se = (va / na + vb / nb) ** 0.5
    if se == 0:  # both groups constant -> not testable
        return out
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_stat = d / se
    tcrit = float(_scipy_stats.t.ppf((1 + ci) / 2, df))
    lo, hi = d - tcrit * se, d + tcrit * se
    out.update(
        ci_lo=lo,
        ci_hi=hi,
        p=float(2 * _scipy_stats.t.sf(abs(t_stat), df)),
        significant=bool(lo > 0 or hi < 0),
        t_stat=float(t_stat),
        df=float(df),
    )
    return out


def compare_paired(a_scores, b_scores, ci):
    """Paired t-test of the per-layout difference d = A - B (A and B already
    aligned position-by-position, equal length).

    Pairing removes the between-layout difficulty variance, so this is far more
    powerful than the unpaired Welch test — but ONLY valid when the two runs use
    the SAME layouts in the same order AND run-to-run noise is negligible. With a
    real run-to-run floor it is overconfident (see module docstring). Returns the
    same dict shape as compare(); n<2 or constant differences stay conservative.
    """
    a = np.asarray(a_scores, float)
    b = np.asarray(b_scores, float)
    n = int(min(a.size, b.size))
    a, b = a[:n], b[:n]
    d = a - b
    md = float(d.mean()) if n else 0.0
    out = {
        "score_a": float(a.mean()) if n else 0.0,
        "score_b": float(b.mean()) if n else 0.0,
        "diff": md,
        "ci_lo": md,
        "ci_hi": md,
        "p": 1.0,
        "significant": False,
        "df": 0.0,
        "n_pairs": n,
    }
    if n < 2:
        return out
    se = float(d.std(ddof=1)) / n**0.5
    if se == 0:  # all pairwise differences identical -> not testable
        return out
    df = n - 1
    t_stat = md / se
    tcrit = float(_scipy_stats.t.ppf((1 + ci) / 2, df))
    lo, hi = md - tcrit * se, md + tcrit * se
    out.update(
        ci_lo=lo,
        ci_hi=hi,
        p=float(2 * _scipy_stats.t.sf(abs(t_stat), df)),
        significant=bool(lo > 0 or hi < 0),
        t_stat=float(t_stat),
        df=float(df),
    )
    return out


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values), same order as input."""
    p = np.asarray(pvals, float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


def make_plot(res, name_a, name_b, ci, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, lo, hi = res["diff"], res["ci_lo"], res["ci_hi"]
    sigma = max((hi - lo) / 2 / 1.96, 1e-9)  # normal implied by the CI
    xs = np.linspace(min(lo, 0) - sigma, max(hi, 0) + sigma, 400)
    ys = np.exp(-0.5 * ((xs - d) / sigma) ** 2)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(xs, ys, color="#9fb3c8", alpha=0.5, lw=0)
    for x, ls, col, lab in [
        (0, "--", "#888", "no difference"),
        (d, "-", "#1a7f37" if res["significant"] else "#cf7a17", f"D={d:+.3f}"),
        (lo, ":", "#444", None),
        (hi, ":", "#444", None),
    ]:
        ax.axvline(x, ls=ls, color=col, lw=1.8 if lab else 1.2, label=lab)
    ax.set_title(
        f"{name_a} - {name_b}  ({int(ci*100)}% CI=[{lo:+.3f}, {hi:+.3f}], "
        f"{'significant' if res['significant'] else 'not significant'})"
    )
    ax.set_xlabel("task-mean score difference (Welch t sampling distribution)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nPlot -> {path}")


def make_forest(rows, name_a, name_b, ci, path):
    """Forest plot of per-subtask differences (per-subtask mode)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["diff"])
    labels = [_short(r["task_name"], 44) for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6))
    for yi, r in zip(y, rows):
        sig = r.get("sig_bh", r["ci_lo"] > 0 or r["ci_hi"] < 0)
        color = "#1a7f37" if (sig and r["diff"] > 0) else ("#cf222e" if sig else "#57606a")
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(r["diff"], yi, "o", color=color, ms=5)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"{name_a} − {name_b}   [{int(ci*100)}% CI per sub_task]")
    ax.set_title(f"Per-subtask comparison: {name_a} vs {name_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nForest plot -> {path}")


def make_multi_forest(labels, series, ci, path, ref_name):
    """Publication-style forest plot (multi-model): one capped CI per model per task,
    a summary diamond for the pooled OVERALL row. Legends sit outside the panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "xtick.major.size": 4,
        }
    )

    n, k = len(labels), len(series)
    y = np.arange(n)[::-1]
    spread = 0.6
    offsets = np.linspace(-spread / 2, spread / 2, k) if k > 1 else [0.0]
    palette = [
        "#1f4e79",
        "#1a7f37",
        "#b3261e",
        "#7a3ea8",
        "#c2570c",
        "#0e7490",
        "#9a7400",
        "#b21e62",
        "#3f4753",
        "#4d7c0f",
    ]
    pct = int(ci * 100)

    fig, ax = plt.subplots(figsize=(11.5, 0.6 * n + 2.4))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e8e8", lw=0.8, zorder=0)
    # faint alternating row bands for readability
    for yi in y:
        if int(yi) % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f6f7f9", zorder=0)
    lo_all, hi_all = [], []

    def overall(lab):
        return lab.upper() == "OVERALL"

    for j, (name, data) in enumerate(series):
        col = palette[j % len(palette)]
        for yi, lab in zip(y, labels):
            if lab not in data:
                continue
            d, lo, hi, sig = data[lab]
            lo_all.append(lo)
            hi_all.append(hi)
            yy = yi + offsets[j]
            if overall(lab):  # meta-analysis summary diamond for the pooled estimate
                h = spread / (2 * k) * 0.95
                ax.add_patch(
                    Polygon(
                        [(lo, yy), (d, yy + h), (hi, yy), (d, yy - h)],
                        closed=True,
                        facecolor=col,
                        edgecolor=col,
                        lw=1.0,
                        alpha=0.95,
                        zorder=5,
                    )
                )
                continue
            ax.errorbar(
                d,
                yy,
                xerr=[[d - lo], [hi - d]],
                fmt="o",
                ms=5.2,
                color=col,
                ecolor=col,
                elinewidth=1.3,
                capsize=3.2,
                capthick=1.1,
                mfc=col if sig else "white",
                mec=col,
                mew=1.3,
                alpha=1.0 if sig else 0.85,
                zorder=4,
            )

    ax.axvline(0, color="#333", ls=(0, (5, 4)), lw=1.0, zorder=2)
    # separator above the OVERALL (bottom) row
    ax.axhline(0.5, color="#cfcfcf", lw=0.9, zorder=1)
    dlo, dhi = min(lo_all), max(hi_all)
    span = max(dhi - dlo, 0.1)
    ax.set_xlim(dlo - 0.30 * span, dhi + 0.30 * span)
    ax.set_ylim(-0.6, n - 1 + 0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(rf"Effect size:  score difference (model $-$ {ref_name})    [{pct}% CI]", fontsize=10, labelpad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # "favours" cues just below the x-axis, outside the data area
    ax.annotate(
        f"◀ favours {ref_name}",
        xy=(0, -0.082),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#777",
        annotation_clip=False,
    )
    ax.annotate(
        "favours model ▶",
        xy=(0, -0.082),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#777",
        annotation_clip=False,
    )

    # legends OUTSIDE the panel (right side), stacked — no overlap with data
    model_leg = ax.legend(
        handles=[
            Line2D([0], [0], color=palette[j % len(palette)], marker="o", lw=2.0, ms=6.5, label=nm)
            for j, (nm, _) in enumerate(series)
        ],
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        title="model",
        title_fontsize=9.5,
        framealpha=1.0,
        edgecolor="#d0d0d0",
        borderpad=0.8,
        labelspacing=0.6,
    )
    ax.add_artist(model_leg)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#444",
                marker="o",
                lw=0,
                ms=6.5,
                mfc="#444",
                label=f"significant\n({pct}% CI excludes 0)",
            ),
            Line2D([0], [0], color="#444", marker="o", lw=0, ms=6.5, mfc="white", mec="#444", label="not significant"),
            Line2D([0], [0], color="#444", marker="D", lw=0, ms=7.5, mfc="#444", label="pooled (OVERALL)"),
        ],
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.015, 0.0),
        title="estimate",
        title_fontsize=9.5,
        framealpha=1.0,
        edgecolor="#d0d0d0",
        borderpad=0.8,
        labelspacing=0.9,
    )

    ax.set_title(f"Per-task effect sizes vs reference: {ref_name}", fontsize=13, fontweight="bold", pad=12)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.rcdefaults()
    print(f"\nForest plot -> {path}")


def _run_multi(args, c, pct, a_scores, a_groups, a_info, models, disp):
    """Multi-model mode: every model in --b is compared against the reference --a.

    Δ is defined as (model − reference): positive = better than the baseline.
    """
    ref = args.name_a
    all_groups = sorted(set(a_groups).union(*[set(m["groups"]) for m in models]))
    for m in models:
        only = sorted(set(m["groups"]) ^ set(a_groups))
        if only:
            print(
                c(
                    f"  [warn] {m['name']}: content mismatch with reference on "
                    f"{[disp.get(s, s) for s in only]} — those groups are skipped for it.",
                    "yellow",
                )
            )
    _names = sorted(disp.get(s, s) for s in all_groups)
    sub_disp = "; ".join(_names[:6]) + (f"; … +{len(_names)-6} more" if len(_names) > 6 else "")
    print(f"Detected {len(all_groups)} task group(s) (matched by CONTENT, paths irrelevant): {sub_disp}")
    multi_tasks = len(all_groups) > 1
    merged_mode = args.merge or not multi_tasks
    print(
        f"(mode={'merge' if merged_mode else 'per-task'}, reference={ref}, "
        f"metric={args.metric}, Welch t, ci={args.ci};  Δ = model − {ref})"
    )

    # per-model results vs the reference
    alpha = 1 - args.ci
    results = {}  # name -> {"per_task": {group: res}, "overall": res}
    for m in models:
        commons = [g for g in sorted(a_groups) if g in m["groups"]]
        per = {}
        for g in commons:
            r = compare(m["groups"][g], a_groups[g], args.ci)  # diff = model - ref
            r["reliable"] = min(m["groups"][g].size, a_groups[g].size) >= 5
            per[g] = r
        # BH (FDR) across this model's per-task family; the overall row is a
        # single test and stays uncorrected
        for g, q in zip(commons, bh_adjust([per[g]["p"] for g in commons])):
            per[g]["p_bh"] = float(q)
        ov = compare(
            np.concatenate([m["groups"][g] for g in commons]), np.concatenate([a_groups[g] for g in commons]), args.ci
        )
        ov["reliable"] = True
        results[m["name"]] = {"per_task": per, "overall": ov}

    def cell(r, bw):
        if r is None:
            return " | " + _pad("-", bw, 1) + " | " + _pad("-", 7, 1) + " | " + _pad("-", 6, 1) + " | -"
        sig = r["significant"] and r["reliable"] and r.get("p_bh", 0.0) <= alpha
        col = "green" if (sig and r["diff"] > 0) else ("red" if (sig and r["diff"] < 0) else "grey")
        dlt = c(_pad("%+.3f" % r["diff"], 7, 1), "bold", col) if sig else _pad("%+.3f" % r["diff"], 7, 1)
        mark = c("*", "bold", col) if sig else (c("!", "yellow") if not r["reliable"] else c("-", "grey"))
        return (
            " | "
            + _pad("%.3f" % r["score_a"], bw, 1)
            + " | "
            + dlt
            + " | "
            + _pad("%.4f" % r["p"], 6, 1)
            + " | "
            + mark
        )

    if merged_mode:
        # one row per model, overall only
        w = max([_disp_w(m["name"]) for m in models] + [_disp_w(ref) + 6, 12])
        args._tee.begin_capture()
        print(f"\nOverall comparison vs reference ({c('pooled over all common tasks', 'grey')}):")
        print(
            c(
                "  "
                + _pad("model", w)
                + " | "
                + _pad("units", 5, 1)
                + " | "
                + _pad("score", 8, 1)
                + " | "
                + _pad("diff", 8, 1)
                + " | "
                + _pad("%d%% CI" % pct, 18, 1)
                + " | "
                + _pad("p", 7, 1)
                + " | verdict",
                "bold",
            )
        )
        print(
            "  "
            + _pad(ref + " (ref)", w)
            + " | "
            + _pad(a_scores.size, 5, 1)
            + " | "
            + _pad("%.3f" % a_scores.mean(), 8, 1)
            + " | "
            + _pad("-", 8, 1)
            + " | "
            + _pad("-", 18, 1)
            + " | "
            + _pad("-", 7, 1)
            + " | -"
        )
        for m in models:
            r = results[m["name"]]["overall"]
            sig = r["significant"]
            if sig and r["diff"] > 0:
                tag = c("✅ better than ref", "bold", "green")
            elif sig and r["diff"] < 0:
                tag = c("❗ worse than ref", "bold", "red")
            else:
                tag = c("n.s.", "grey")
            col = "green" if (sig and r["diff"] > 0) else ("red" if (sig and r["diff"] < 0) else "grey")
            dlt = c(_pad("%+.3f" % r["diff"], 8, 1), "bold", col) if sig else _pad("%+.3f" % r["diff"], 8, 1)
            ci_str = "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"])
            print(
                "  "
                + _pad(m["name"], w)
                + " | "
                + _pad(m["scores"].size, 5, 1)
                + " | "
                + _pad("%.3f" % r["score_a"], 8, 1)
                + " | "
                + dlt
                + " | "
                + _pad(ci_str, 18, 1)
                + " | "
                + _pad("%.4f" % r["p"], 7, 1)
                + " | "
                + tag
            )
        args._tee.end_capture()
    else:
        # wide per-task table: ref column + one block per model
        rows = sorted(set(a_groups), key=lambda s: disp.get(s, s))
        w = max([_disp_w(disp.get(s, s)) for s in rows] + [_disp_w("OVERALL"), 12])
        wr = max(8, _disp_w(ref))
        bws = {m["name"]: max(7, _disp_w(m["name"])) for m in models}
        hdr = "  " + _pad("task_name", w) + " | " + _pad("units", 5, 1) + " | " + _pad(ref, wr, 1)
        for m in models:
            hdr += (
                " | "
                + _pad(m["name"], bws[m["name"]], 1)
                + " | "
                + _pad("diff", 7, 1)
                + " | "
                + _pad("p", 6, 1)
                + " | s"
            )
        args._tee.begin_capture()
        print(f"\nPer-task comparison vs reference ({c('per cell: score, Δ=model−ref, p, sig', 'grey')}):")
        print(c(hdr, "bold"))
        for g in rows:
            line = (
                "  "
                + _pad(disp.get(g, g), w)
                + " | "
                + _pad(a_groups[g].size, 5, 1)
                + " | "
                + _pad("%.3f" % a_groups[g].mean(), wr, 1)
            )
            for m in models:
                line += cell(results[m["name"]]["per_task"].get(g), bws[m["name"]])
            print(line)
        total_w = w + 5 + wr + 6 + sum(bws[m["name"]] + 26 for m in models)
        print(c("  " + "-" * total_w, "grey"))
        line = (
            "  "
            + c(_pad("OVERALL", w), "bold")
            + " | "
            + _pad(a_scores.size, 5, 1)
            + " | "
            + _pad("%.3f" % a_scores.mean(), wr, 1)
        )
        for m in models:
            line += cell(results[m["name"]]["overall"], bws[m["name"]])
        print(line)
        print(
            "  "
            + c(
                "* significant after BH(FDR) across tasks (green = better than ref, red = worse) | - n.s. | ! n<5",
                "grey",
            )
        )
        args._tee.end_capture()

    if args.plot:
        labels = [disp.get(g, g) for g in sorted(set(a_groups), key=lambda s: disp.get(s, s))] + ["OVERALL"]
        series = []
        alpha = 1 - args.ci

        def _sig(r, overall=False):
            if overall:
                return bool(r["significant"])
            return bool(r["significant"] and r["reliable"] and r.get("p_bh", 0.0) <= alpha)

        for m in models:
            data = {
                disp.get(g, g): (r["diff"], r["ci_lo"], r["ci_hi"], _sig(r))
                for g, r in results[m["name"]]["per_task"].items()
            }
            ov = results[m["name"]]["overall"]
            data["OVERALL"] = (ov["diff"], ov["ci_lo"], ov["ci_hi"], _sig(ov, overall=True))
            series.append((m["name"], data))
        make_multi_forest(labels, series, args.ci, args.plot, ref)

    if args.json:
        payload = {
            "config": {
                "reference": {"name": ref, "dir": os.path.abspath(args.a), "info": a_info},
                "models": [{"name": m["name"], "dir": os.path.abspath(m["dir"]), "info": m["info"]} for m in models],
                "metric": args.metric,
                "method": "welch_t",
                "ci": args.ci,
                "mode": "merge" if merged_mode else "per_task",
                "diff_definition": "model - reference",
            },
            "results": {
                name: {
                    "overall": res["overall"],
                    "per_task": {disp.get(g, g): r for g, r in res["per_task"].items()},
                }
                for name, res in results.items()
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nFull result JSON -> {args.json}")

    if args.csv:
        # mirror the printed table: same rows, same order, same formats
        if merged_mode:
            ci_key = "ci_%d" % pct
            csv_fields = ["model", "units", "score", "diff", ci_key, "p", "verdict"]
            csv_rows = [
                {"model": ref + " (ref)", "units": int(a_scores.size), "score": "%.3f" % a_scores.mean(), "diff": "-"}
            ]
            for m in models:
                r = results[m["name"]]["overall"]
                sig = r["significant"]
                verdict = (
                    "better than ref"
                    if sig and r["diff"] > 0
                    else "worse than ref" if sig and r["diff"] < 0 else "n.s."
                )
                csv_rows.append(
                    {
                        "model": m["name"],
                        "units": int(m["scores"].size),
                        "score": "%.3f" % r["score_a"],
                        "diff": "%+.3f" % r["diff"],
                        ci_key: "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"]),
                        "p": "%.4f" % r["p"],
                        "verdict": verdict,
                    }
                )
        else:
            csv_fields = ["task_name", "units", ref]
            for m in models:
                csv_fields += [f"{m['name']}_score", f"{m['name']}_diff", f"{m['name']}_p", f"{m['name']}_verdict"]

            def _cells(r):
                if r is None:
                    return "-", "-", "-", "-"
                sig = r["significant"] and r["reliable"] and r.get("p_bh", 0.0) <= alpha
                v = "significant" if sig else ("n<5" if not r["reliable"] else "n.s.")
                return "%.3f" % r["score_a"], "%+.3f" % r["diff"], "%.4f" % r["p"], v

            csv_rows = []
            for g in sorted(set(a_groups), key=lambda s: disp.get(s, s)):
                row = {"task_name": disp.get(g, g), "units": int(a_groups[g].size), ref: "%.3f" % a_groups[g].mean()}
                for m in models:
                    sc, d, p, v = _cells(results[m["name"]]["per_task"].get(g))
                    row.update(
                        {
                            f"{m['name']}_score": sc,
                            f"{m['name']}_diff": d,
                            f"{m['name']}_p": p,
                            f"{m['name']}_verdict": v,
                        }
                    )
                csv_rows.append(row)
            row = {"task_name": "OVERALL", "units": int(a_scores.size), ref: "%.3f" % a_scores.mean()}
            for m in models:
                sc, d, p, v = _cells(results[m["name"]]["overall"])
                row.update(
                    {f"{m['name']}_score": sc, f"{m['name']}_diff": d, f"{m['name']}_p": p, f"{m['name']}_verdict": v}
                )
            csv_rows.append(row)
        write_csv(args.csv, csv_fields, csv_rows)

    _finish_png(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run dir of the REFERENCE model")
    ap.add_argument(
        "--b",
        required=True,
        nargs="+",
        help="run dir(s) of the model(s) to compare against the reference; "
        "pass several dirs to get a multi-model table",
    )
    ap.add_argument("--name-a", default=None, help="reference display name (default: dir basename)")
    ap.add_argument(
        "--name-b", nargs="+", default=None, help="display names matching --b order (default: dir basenames)"
    )
    ap.add_argument("--metric", choices=["step_mean", "e2e"], default="step_mean")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument(
        "--merge",
        action="store_true",
        help="pool ALL tasks under the run dir into one unit pool and give a "
        "single verdict (default: one independent test per content-matched task group)",
    )
    ap.add_argument(
        "--paired",
        action="store_true",
        help="pair rollouts by (task group, position) across the two runs and run a "
        "PAIRED t-test on per-layout differences. Requires a single --b and "
        "layouts emitted in the same order (verified via per-position "
        "instruction match). ONLY valid when layouts are deterministic across "
        "runs (run-to-run noise negligible); otherwise it is overconfident.",
    )
    ap.add_argument("--json", help="write full result JSON here")
    ap.add_argument(
        "--csv",
        help="write the comparison table as CSV here (long format, "
        "one row per task x model; diff = model - reference)",
    )
    ap.add_argument("--plot", help="write a forest/distribution plot PNG here")
    ap.add_argument(
        "--png",
        default="compare_result.png",
        help="render the terminal output (tables, colours) to this PNG " "(default: compare_result.png)",
    )
    ap.add_argument("--no-png", action="store_true", help="disable the terminal-output PNG")
    ap.add_argument(
        "--png-scale",
        type=float,
        default=2.0,
        help="PNG render scale / sharpness (default 2.0; bigger = crisper & larger file)",
    )
    ap.add_argument("--no-color", action="store_true", help="disable coloured output")
    args = ap.parse_args()

    if _scipy_stats is None:
        raise SystemExit("scipy is required (Welch t-test). Install: pip install scipy")
    pct = int(args.ci * 100)
    # Enable ANSI when stdout is a tty, OR when a table PNG will be rendered
    # from the captured text (so the cmp PNG carries colour even when piped).
    _want_color = not (args.no_color or os.environ.get("NO_COLOR"))
    _png_on = args.png and not args.no_png
    c = Colorizer(_want_color and (sys.stdout.isatty() or _png_on))
    args._tee = _Tee(sys.stdout)
    sys.stdout = args._tee

    if args.name_a is None:
        args.name_a = os.path.basename(os.path.normpath(args.a)) or "A"
    names_b = args.name_b or [os.path.basename(os.path.normpath(d)) or f"B{i+1}" for i, d in enumerate(args.b)]
    if len(names_b) != len(args.b):
        raise SystemExit("--name-b count must match --b count")

    print(f"Loading reference {args.name_a} <- {args.a}")
    a_scores, a_groups, a_names, a_ginstr, a_info = load_task_scores(args.a, args.metric)
    models = []
    for nm, d in zip(names_b, args.b):
        print(f"Loading {nm} <- {d}")
        s, g, n, gi, i = load_task_scores(d, args.metric)
        models.append({"name": nm, "dir": d, "scores": s, "groups": g, "names": n, "ginstr": gi, "info": i})

    # display names (sub_task dir); prefer the reference's naming, dedupe collisions
    disp = {}
    for m in models[::-1]:
        disp.update(m["names"])
    disp.update(a_names)
    _cnt = Counter(disp.values())
    for k, v in disp.items():
        if _cnt[v] > 1:
            disp[k] = f"{v}@{hashlib.md5(k.encode()).hexdigest()[:6]}"

    if args.paired and len(models) > 1:
        raise SystemExit("--paired supports exactly one --b (pairing is between two runs).")

    if len(models) > 1:
        _run_multi(args, c, pct, a_scores, a_groups, a_info, models, disp)
        return

    # ---- single comparison model: original two-model flow --------------------
    m0 = models[0]
    args.name_b = m0["name"]
    b_scores, b_groups, b_ginstr, b_info = m0["scores"], m0["groups"], m0["ginstr"], m0["info"]

    # paired mode: align A and B by (group, position); warn on count/order mismatch
    method_lbl = "paired t" if args.paired else "unpaired Welch t"
    if args.paired:
        for g in sorted(set(a_groups) & set(b_groups)):
            na_, nb_ = a_groups[g].size, b_groups[g].size
            ai, bi = a_ginstr.get(g, []), b_ginstr.get(g, [])
            nmis = sum(1 for k in range(min(len(ai), len(bi))) if ai[k] != bi[k])
            if na_ != nb_ or nmis:
                print(
                    c(
                        f"  [warn] paired align '{disp.get(g, g)}': {na_} vs {nb_} units"
                        + (f", {nmis} position(s) with mismatched instruction" if nmis else "")
                        + " — pairing truncates to the shorter and may be misaligned.",
                        "yellow",
                    )
                )

    def _cmp(av, bv):
        return compare_paired(av, bv, args.ci) if args.paired else compare(av, bv, args.ci)

    def _pool(groups, common):
        """Concatenate group arrays; in paired mode truncate each to the shared length."""
        if args.paired:
            return np.concatenate([groups[s][: min(a_groups[s].size, b_groups[s].size)] for s in common])
        return np.concatenate([groups[s] for s in common])

    only_a = sorted(set(a_groups) - set(b_groups))
    only_b = sorted(set(b_groups) - set(a_groups))
    if only_a or only_b:
        print(
            c(
                f"  [warn] task contents differ (matched by task_name + sub_task_name) — "
                f"only in {args.name_a}: {[disp.get(s, s) for s in only_a] or '-'}; "
                f"only in {args.name_b}: {[disp.get(s, s) for s in only_b] or '-'}. "
                f"These are skipped (per-task mode) or bias the pool (--merge).",
                "yellow",
            )
        )
    multi = len(set(a_groups) | set(b_groups)) > 1
    merged_mode = args.merge or not multi
    print(
        f"\nUnit = rollout / (instruction, layout) ({method_lbl}).  "
        f"{args.name_a}: {a_info['n_tasks']} units / {a_info['n_rollouts']} rollouts;  "
        f"{args.name_b}: {b_info['n_tasks']} units / {b_info['n_rollouts']} rollouts."
    )
    subs = sorted(set(a_groups) | set(b_groups))
    _names = sorted(disp.get(s, s) for s in subs)
    sub_disp = "; ".join(_names[:6]) + (f"; … +{len(_names)-6} more" if len(_names) > 6 else "")
    print(
        f"Detected {len(subs)} task group(s) (matched by CONTENT = task_name + sub_task_name (fallback: instruction set),"
        f" paths irrelevant): {sub_disp}"
    )
    if not multi and args.merge:
        print(
            c(
                "  [note] only ONE task group detected — --merge has no effect; "
                "per-task and merge modes are identical here.",
                "yellow",
            )
        )
    print(f"(mode={'merge' if merged_mode else 'per-task'}, metric={args.metric}, " f"{method_lbl}, ci={args.ci})")

    res = None
    breakdown = []
    if merged_mode:
        # ---- ONE pooled verdict over every task under the dir ----------------
        if args.paired:
            common_all = sorted(set(a_groups) & set(b_groups))
            res = compare_paired(_pool(a_groups, common_all), _pool(b_groups, common_all), args.ci)
        else:
            res = compare(a_scores, b_scores, args.ci)
        args._tee.begin_capture()

        arrow = "▲" if res["diff"] > 0 else ("▼" if res["diff"] < 0 else "=")
        lead = args.name_a if res["diff"] >= 0 else args.name_b
        sa = c("%.4f" % res["score_a"], "cyan")
        sb = c("%.4f" % res["score_b"], "cyan")
        sd = c("%+.4f" % res["diff"], "bold", "cyan")
        ci_line = "%d%% CI [%+.4f, %+.4f]    %s p = %.4f" % (pct, res["ci_lo"], res["ci_hi"], method_lbl, res["p"])

        print("\n" + c("─" * 66, "grey"))
        print(f"  {c(args.name_a, 'bold')} = {sa}      {c(args.name_b, 'bold')} = {sb}")
        print(f"  Δ = {sd} {c(arrow, 'cyan')}   ({args.name_a} − {args.name_b}, {c(lead, 'bold')} ahead)")
        print(f"  {c(ci_line, 'grey')}")
        print()
        if res["significant"]:
            winner = args.name_a if res["diff"] > 0 else args.name_b
            print(c.band(f"✅  SIGNIFICANT at {pct}%   —   {winner} is better", "bold", "black", "bg_green"))
        else:
            print(c.band(f"⚠️   NOT SIGNIFICANT at {pct}%   —   within noise floor", "bold", "black", "bg_yellow"))
        print(c("─" * 66, "grey"))

        # Descriptive breakdown of the pool (no per-row significance: the
        # verdict above is for the POOL; single groups rarely have the units).
        if multi:
            common = sorted(set(a_groups) & set(b_groups), key=lambda s: disp.get(s, s))
            w = max([_disp_w(disp.get(s, s)) for s in common] + [16])
            wa = max(9, _disp_w(args.name_a))
            wb = max(9, _disp_w(args.name_b))
            print(f"\nPer-task breakdown ({c('descriptive only — no per-row significance', 'grey')}):")
            print(
                c(
                    "  "
                    + _pad("task_name", w)
                    + " | "
                    + _pad("units", 5, 1)
                    + " | "
                    + _pad(args.name_a, wa, 1)
                    + " | "
                    + _pad(args.name_b, wb, 1)
                    + " | "
                    + _pad("diff", 8, 1),
                    "bold",
                )
            )
            for sub in common:
                ga, gb = a_groups[sub], b_groups[sub]
                dlt = ga.mean() - gb.mean()
                breakdown.append(
                    {
                        "task_name": disp.get(sub, sub),
                        "units_a": int(ga.size),
                        "units_b": int(gb.size),
                        "score_a": float(ga.mean()),
                        "score_b": float(gb.mean()),
                        "diff": float(dlt),
                    }
                )
                mark = c(_pad("%+.3f" % dlt, 8, 1), "green" if dlt > 0 else ("red" if dlt < 0 else "grey"))
                units = "%d" % ga.size if ga.size == gb.size else "%d/%d" % (ga.size, gb.size)
                print(
                    "  "
                    + _pad(disp.get(sub, sub), w)
                    + " | "
                    + _pad(units, 5, 1)
                    + " | "
                    + _pad("%.3f" % ga.mean(), wa, 1)
                    + " | "
                    + _pad("%.3f" % gb.mean(), wb, 1)
                    + " | "
                    + mark
                )
        args._tee.end_capture()
    else:
        # ---- one INDEPENDENT test per content-matched task group -------------
        common = sorted(set(a_groups) & set(b_groups))
        w = max([_disp_w(disp.get(s, s)) for s in common] + [16])
        wa = max(8, _disp_w(args.name_a))
        wb = max(8, _disp_w(args.name_b))
        args._tee.begin_capture()
        print(f"\nPer-task comparison ({c('each row = independent test at %d%%, BH(FDR)-corrected' % pct, 'grey')}):")
        print(
            c(
                "  "
                + _pad("task_name", w)
                + " | "
                + _pad("units", 5, 1)
                + " | "
                + _pad(args.name_a, wa, 1)
                + " | "
                + _pad(args.name_b, wb, 1)
                + " | "
                + _pad("diff", 8, 1)
                + " | "
                + _pad("%d%% CI" % pct, 18, 1)
                + " | "
                + _pad("p", 7, 1)
                + " | "
                + _pad("p(BH)", 7, 1)
                + " | verdict",
                "bold",
            )
        )

        def _row(name, units, sa, sb, dlt, ci_str, p, q, tag, bold_name=False):
            nm = c(_pad(name, w), "bold") if bold_name else _pad(name, w)
            print(
                "  "
                + nm
                + " | "
                + _pad(units, 5, 1)
                + " | "
                + _pad("%.3f" % sa, wa, 1)
                + " | "
                + _pad("%.3f" % sb, wb, 1)
                + " | "
                + dlt
                + " | "
                + _pad(ci_str, 18, 1)
                + " | "
                + _pad("%.4f" % p, 7, 1)
                + " | "
                + _pad(q, 7, 1)
                + " | "
                + tag
            )

        rows = []
        for sub in common:
            r = _cmp(a_groups[sub], b_groups[sub])
            rows.append((sub, r))
        # BH (FDR) across the per-task family; OVERALL is a single test, uncorrected
        alpha = 1 - args.ci
        for (_, rr), q in zip(rows, bh_adjust([rr["p"] for _, rr in rows])):
            rr["p_bh"] = float(q)
        n_sig = 0
        for sub, r in sorted(rows, key=lambda kv: disp.get(kv[0], kv[0])):
            n_min = min(a_groups[sub].size, b_groups[sub].size)
            reliable = n_min >= 5  # below this the t CI is meaningless
            sig = r["significant"] and reliable and r["p_bh"] <= alpha
            r["sig_bh"] = bool(sig)
            n_sig += sig
            color = "green" if (sig and r["diff"] > 0) else ("red" if (sig and r["diff"] < 0) else "grey")
            dlt = c(_pad("%+.3f" % r["diff"], 8, 1), "bold", color) if sig else _pad("%+.3f" % r["diff"], 8, 1)
            ci_str = "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"])
            units = (
                "%d" % a_groups[sub].size
                if a_groups[sub].size == b_groups[sub].size
                else "%d/%d" % (a_groups[sub].size, b_groups[sub].size)
            )
            if not reliable:
                tag = c("(!) n<5", "yellow")
            elif sig:
                tag = c("✅ significant", "bold", "green")
            else:
                tag = c("n.s.", "grey")
            _row(disp.get(sub, sub), units, r["score_a"], r["score_b"], dlt, ci_str, r["p"], "%.4f" % r["p_bh"], tag)
            breakdown.append(
                {
                    "task_name": disp.get(sub, sub),
                    "units_a": int(a_groups[sub].size),
                    "units_b": int(b_groups[sub].size),
                    "reliable": reliable,
                    **r,
                }
            )

        # ---- bottom row: OVERALL = merge-style pooled verdict over all rows --
        a_pool = _pool(a_groups, common)
        b_pool = _pool(b_groups, common)
        res = _cmp(a_pool, b_pool)
        o_sig = res["significant"]
        o_color = "green" if (o_sig and res["diff"] > 0) else ("red" if (o_sig and res["diff"] < 0) else "grey")
        o_dlt = c(_pad("%+.3f" % res["diff"], 8, 1), "bold", o_color) if o_sig else _pad("%+.3f" % res["diff"], 8, 1)
        o_ci = "[%+.3f,%+.3f]" % (res["ci_lo"], res["ci_hi"])
        o_units = "%d" % a_pool.size if a_pool.size == b_pool.size else "%d/%d" % (a_pool.size, b_pool.size)
        o_tag = c("✅ significant", "bold", "green") if o_sig else c("n.s.", "grey")
        print(c("  " + "-" * (w + wa + wb + 69), "grey"))
        _row("OVERALL", o_units, res["score_a"], res["score_b"], o_dlt, o_ci, res["p"], "-", o_tag, bold_name=True)
        breakdown.append(
            {"task_name": "OVERALL", "units_a": int(a_pool.size), "units_b": int(b_pool.size), "reliable": True, **res}
        )

        print(
            f"\n  {n_sig}/{len(rows)} tasks significant at {pct}% (BH/FDR-corrected).  "
            + c(
                "Per-row power is limited by each task's units; the OVERALL row " "pools all of them (= --merge).",
                "grey",
            )
        )
        args._tee.end_capture()

    if args.plot:
        if merged_mode:
            make_plot(res, args.name_a, args.name_b, args.ci, args.plot)
        else:
            make_forest(breakdown, args.name_a, args.name_b, args.ci, args.plot)

    if args.json:
        res_json = dict(res) if res else None
        payload = {
            "config": {
                "run_a": os.path.abspath(args.a),
                "run_b": os.path.abspath(args.b[0]),
                "name_a": args.name_a,
                "name_b": args.name_b,
                "metric": args.metric,
                "ci": args.ci,
                "method": "paired per-layout t-test" if args.paired else "unpaired task-unit Welch t-test",
                "mode": "merge" if merged_mode else "per_subtask",
                "a_info": a_info,
                "b_info": b_info,
            },
            "result": res_json,
            "per_subtask": breakdown,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nFull result JSON -> {args.json}")

    if args.csv:
        # mirror the printed table: same rows, same order, same formats
        ci_key = "ci_%d" % pct
        csv_fields = ["task_name", "units", args.name_a, args.name_b, "diff", ci_key, "p", "p_bh", "verdict"]
        csv_rows = []
        for r in breakdown:
            ua, ub = r["units_a"], r["units_b"]
            row = {
                "task_name": r["task_name"],
                "units": "%d" % ua if ua == ub else "%d/%d" % (ua, ub),
                args.name_a: "%.3f" % r["score_a"],
                args.name_b: "%.3f" % r["score_b"],
                "diff": "%+.3f" % r["diff"],
            }
            if "ci_lo" in r:
                row[ci_key] = "[%+.3f,%+.3f]" % (r["ci_lo"], r["ci_hi"])
                row["p"] = "%.4f" % r["p"]
                if "p_bh" in r:
                    row["p_bh"] = "%.4f" % r["p_bh"]
                reliable = r.get("reliable", True)
                sig = r.get("sig_bh", r["significant"] and reliable)
                row["verdict"] = "significant" if sig else ("n<5" if not reliable else "n.s.")
            csv_rows.append(row)
        if merged_mode and res:  # merge mode prints OVERALL in the banner; append it last
            csv_rows.append(
                {
                    "task_name": "OVERALL",
                    "units": (
                        "%d" % a_scores.size
                        if a_scores.size == b_scores.size
                        else "%d/%d" % (a_scores.size, b_scores.size)
                    ),
                    args.name_a: "%.4f" % res["score_a"],
                    args.name_b: "%.4f" % res["score_b"],
                    "diff": "%+.4f" % res["diff"],
                    ci_key: "[%+.4f,%+.4f]" % (res["ci_lo"], res["ci_hi"]),
                    "p": "%.4f" % res["p"],
                    "verdict": "significant" if res["significant"] else "n.s.",
                }
            )
        write_csv(args.csv, csv_fields, csv_rows)

    _finish_png(args)


if __name__ == "__main__":
    main()
