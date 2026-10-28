# Identification overlays

An **overlay** is a sparse YAML record of the parameters one offline
identification run actually measured. The generated URDF is the
*baseline*; an overlay carries only the fields that were measured, and
nothing else.

Absent fields retain their baseline or class fallback; present fields carry
measured or explicitly authored values. Ordered stacks combine sparse records.

This file documents the **data**: where overlays live here, the
conventions they follow, and what ships today. The schema, the
validator, the application channels and the invariants are the
**method**, and they live in `genie_sim_engine_core`. The [schema and CLI](../../genie_sim_engine_core/genie_sim_engine_core/schema/joint_params.py)
define the supported fields and read-only inspection commands.

## Where they live

```
robots/<vendor>/<model>/overlays/<id>.yaml
```

The filename stem and the `id:` field inside the file must match. The
convention is `<target>.<kind>-<YYYY-MM-DD>`, where the date is the
identification date — not the authoring date, and not a version number.
Re-identifying the same target produces a **new file**, never an edit to
the old one.

## What ships today

| overlay | target | carries | fields |
|---|---|---|---|
| `genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml` | g2 · crsB arm | 14 links: `mass` `com` `inertia`<br>14 joints: `damping` `frictionloss` `armature` | 182 |
| `genie/g2/overlays/g2_wuji.sysid-2026-09-24.yaml` | g2 · crsB · wuji hand | 40 joints: `armature` `damping` `frictionloss` `stiffness` `damping_drive` | 200 |
| `genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml` | g2 · crsB · swiftpicker | 12 joints, the same five families | 60 |
| `arx/acone/overlays/ac1.sysid-2026-09-30.yaml` | arx · acone | 12 joints, those five + `command_delay` | 72 |
| `arx/x5/overlays/x5a.sysid-2026-09-30.yaml` | arx · x5 | 6 joints, those five + `command_delay` | 36 |

All five are `kind: sysid`. Read each file's `meta.caveats` before
adopting it — every one of them has caveats that change how the numbers
should be used.

## Selection

Select link inertials and URDF joint parameters with `inertia_source:=sysid`
for G2 crsB or `param_source:=sysid` for ARX. Select `default` for the stock
parameter files. The generated `*_sysid.urdf` files are explicit alternatives;
generating them does not select them in a running simulation.

Core's `JointParamResolver` validates ordered overlay paths and resolves sparse
values with last-wins precedence, including explicit zeros. The active
`genie_sim_engine` launch path does not consume `robot.joint_overlays`.
Consequently, validation or resolution does not establish runtime application,
and class tuning can overwrite values authored in the URDF. Armature has no
standard URDF field; command delay and torque constant have no runtime consumer.

## Conventions these files follow

- **Joint and link names are URDF names, verbatim.** Including the
  `idxNN_` prefixes, which are part of the name as generated — e.g.
  `idx21_arm_l_joint1`, not `arm_l_joint1`. A name that does not exist in
  the target URDF is a **hard load error**, not a warning: the overlay is
  a claim about a specific robot, and a typo that silently does nothing
  is worse than a crash.
- **Zero is a value, not an absence.** `frictionloss: 0.0` means the run
  measured zero friction. Presence decides whether a field applies —
  never magnitude.
- **No overlay publishes a limit.** Two different fields look like one
  and neither is identifiable. `effort` and `velocity` are the vendor's
  ratings and round-trip through URDF `<limit>` — that is where they
  belong, as one copy every backend can see. `max_effort` is not a
  limit at all: it is the runtime drive clamp, with no URDF
  representation, so it is a *condition of the run*. All three are
  declared in `meta.not_identified`, and the clamp the fit actually ran
  under is named in `meta.caveats` with its value. The class fallback only raises caps; selected runtime overlays can
  supply explicit drive limits. Neither mechanism makes an unmeasured limit
  an identification result. If a vendor figure is missing from the URDF, fix the URDF —
  and if no vendor figure exists, say so in `caveats` and publish
  nothing.
- **SI units throughout.** kg, m, kg·m², N·m, N·m·s/rad. No degrees, no
  grams, no mm. A per-backend unit quirk is the adapter's problem; never
  pre-divide a value here to compensate for one.
- **Four fields may carry one value per engine.** `stiffness`,
  `damping_drive`, `damping` and `frictionloss` are consumed by a
  control law, and the backends do not share one. Every other field —
  inertials, `armature`, the limits, `command_delay`, `torque_constant`
  — is one number for every engine, and an engine key on one of them is
  a hard load error. See *Engine-specific values* below.
- **An overlay beats the class tables.** That is its purpose.
- **`applies_to` is a filter, not a path.** It names
  `robot_model` / `body` / `arm` / `ee` so a loader can refuse an overlay
  pointed at the wrong robot.

## The `meta` block

| key | what it is for |
|---|---|
| `kind` | `sysid` — output of an identification run; **do not hand-edit**. `correction` — a deliberate fix to the baseline that is not a measurement. `operator` — a scratch overlay for one experiment. |
| `identified_on` | The date the measurement was taken. |
| `source_run` | Identity of the measurement run. |
| `not_identified` | Families this run did **not** measure, named explicitly. This is the companion to sparseness: absent-and-listed means "not measured", and the list is what makes that readable without diffing against the baseline. A field cannot be in both places: present in `joints:` *and* listed here is a load error, which is the right asymmetry — declaring absence is a claim about the run, and a published value contradicts it. The three limit fields are always listed here, and the clamp the run applied goes in `caveats`. |
| `caveats` | Known limits on the numbers — unidentifiable families, regularization artifacts, fields carried over rather than refit. Prose, and the most important key in the block. |

## Authoring

**Adopting a run.** Validate against the URDF it targets, then query the
fields you expect:

```
M=genie_sim_engine_core.schema.joint_params
python -m $M validate robots/genie/g2/overlays/<id>.yaml --urdf <urdf>
python -m $M resolve  --overlay <same> --urdf <urdf> --joint <joint>
```

`validate` alone is a schema check; `--urdf` additionally checks every
joint and link name against that robot. `resolve` prints each field with
the layer that won — a field you expected to see as `overlay` but that
reads `baseline` is one the overlay does not carry; check
`meta.not_identified`. `diff` reports baseline-vs-overlaid per field,
and `bilateral` reports left/right symmetry, which is the quickest smell
test on an arm run.

**Changing one joint for an experiment.** Author a `kind: operator`
overlay with that single field and append it to the stack. Do **not**
edit a `sysid` overlay; it is a record of a measurement, and editing it
destroys the only copy of what the run said.

**Engine-specific values.** Write a field flat when one number applies
everywhere. Split it only when a run **identified a different number**
against a particular backend:

```yaml
joints:
  idx21_arm_l_joint1:
    armature: 1.51643658513          # flat: same quantity on every engine
    stiffness:                       # one branch per engine
      mujoco: {value: 526.311, note: "fitted on mjwarp, run A"}
      physx:  {value: 498.02,  note: "refitted on physx, run B"}
```

Keys are `mujoco`, `newton` and `physx`. Each branch takes an ordinary
scalar, terse or annotated, and **must** carry a `note` saying what
identified it — a split with no provenance is indistinguishable from a
workaround for a backend bug, and the two call for opposite fixes.
The caller selects the schema engine key when resolving values. A schema
engine key does not establish runtime solver support. Three things to know before using an engine map:

- **A split is a measurement, not a fix.** If a backend consumes the
  number wrongly, that is a bug in the engine adapter and belongs there.
  Writing a smaller branch to cancel it out makes the file physically
  meaningless and breaks silently the moment the adapter is corrected.
- **An unnamed engine gets nothing**, not a sibling's value. Name all
  three, or keep the field flat. `validate` warns about a partial split
  and `resolve` prints `[NO <engine> BRANCH]` for the gap.
- **`mujoco` and `newton` share one control law**, so those two normally
  carry the same number; all three carrying the same number means it is
  a flat value written three times, and `validate` says so.

Every shipped overlay uses flat values. Read `meta.identified_on_engine` and
`meta.caveats` for the fitted law; a flat number is not cross-engine validation.

**Combining runs.** Overlays stack, last-wins-per-field, which is how
"new inertials over previous friction" is expressed — rather than
hand-carrying numbers between files.

## Runtime consumers

The current launch path uses `genie_sim_engine` and its class-level tuning.
Core provides the overlay schema and resolver, but validation does not configure
a solver. Inspect initialized values to establish which parameters a runtime
actually applies. The YAML files remain authoritative for measurement values
and fit conditions; see [SYSID_PARAMS.md](SYSID_PARAMS.md) for their caveats.
