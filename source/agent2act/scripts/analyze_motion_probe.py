# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Summarize measured immediate/held motion-probe response; never infer task success."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(path: Path) -> list[dict]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {}
    definitions = {}
    episodes = {}
    for e in events:
        session = e.get("session_id", "unknown")
        if e.get("message") == "motion_probe_started":
            episodes[session] = episodes.get(session, 0) + 1
            definitions[(session, episodes[session])] = {t["name"]: t for t in e["trials"]}
        if e.get("message") == "motion_probe_sample" and e["phase"] in ("pulse", "target_hold"):
            groups.setdefault((session, episodes[session], e["trial"]), {})[e["phase"]] = e
    result = []
    for (session, episode, name), samples in groups.items():
        pulse = samples.get("pulse")
        if not pulse:
            continue
        trial = definitions[(session, episode)][name]
        arm = trial["arm"]
        held = samples.get("target_hold")
        if "translation_mm" in trial:
            axis = next(i for i, v in enumerate(trial["translation_mm"]) if v)
            baseline = pulse["before"][f"{arm}_pose"][axis]
            command = (pulse["target"][f"{arm}_pose"][axis] - baseline) * 1000
            actual = (pulse["observed"][f"{arm}_pose"][axis] - baseline) * 1000
            settled = (held["observed"][f"{arm}_pose"][axis] - baseline) * 1000 if held else None
            residual = (
                math.dist(pulse["target"][f"{arm}_pose"][:3], pulse["observed"][f"{arm}_pose"][:3]) * 1000
            )
            held_error = (
                math.dist(pulse["target"][f"{arm}_pose"][:3], held["observed"][f"{arm}_pose"][:3]) * 1000
                if held
                else None
            )
            unit = "mm"
        else:
            i = 0 if arm == "left" else 1
            baseline = pulse["before"]["grippers"][i]
            command = pulse["target"]["grippers"][i] - baseline
            actual = pulse["observed"]["grippers"][i] - baseline
            settled = held["observed"]["grippers"][i] - baseline if held else None
            residual = abs(pulse["target"]["grippers"][i] - pulse["observed"]["grippers"][i])
            held_error = (
                abs(pulse["target"]["grippers"][i] - held["observed"]["grippers"][i]) if held else None
            )
            unit = "command units"
        result.append(
            {
                "session": session,
                "diagnostic_episode": episode,
                "trial": name,
                "units": unit,
                "requested_duration_s": pulse["duration_s"],
                "issued_duration_s": pulse["feedback"]["issued_steps"] * pulse["dt_s"],
                "command_delta": command,
                "immediate_delta": actual,
                "held_delta": settled,
                "immediate_residual": residual,
                "held_residual": held_error,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = summarize(args.log)
    if not rows:
        parser.error("no observed motion probe pulses in log")
    output = args.log.with_suffix(".probe.csv")
    with output.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
