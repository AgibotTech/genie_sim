# scoop_popcorn rule evaluation

`g2op_manip_scoop_popcorn` uses `ScoopPopcorn` instead of a VLM in all 20
foreground instances. The instruction and two equally weighted scoring points
are unchanged:

| Current observation | Score |
|---|---:|
| No popcorn inside the bucket | 0 |
| At least one inside, with tabletop spills | 0.5 |
| At least one inside, with no tabletop spills | 1 |

“Tabletop spills” means grains on the tabletop **outside both the popcorn
machine and bucket**. There is no distance-from-bucket cutoff. Grains remaining
inside the machine, airborne grains and floor drops do not count as tabletop
spills. The rule does not require all source grains to enter the bucket or
validate the right-arm motion separately.

The checker samples every 10 updates, retains the highest observed score and
finishes at 1, matching the previous VLM checker's timing and score retention.
Both scoring conditions are evaluated in the same snapshot. The existing
900-step `StepOut` is retained.

## Geometry and readback

- Read the 1200 rigid-body `popcornInstancer` instance transforms using the
  prototype's **bind transform captured before physics starts**, followed by
  the current instancer transform. PhysX stores instance positions relative
  to that bind transform. Applying the independently moving prototype's live
  transform would shift all instances and miss real spills. APICore captures
  the bind transforms on background load and before vectorized world reset,
  preserving them across workspace-only scene changes.
  The prototype is also a visible,
  independent rigid grain outside the instancer, so it is evaluated too.
  Missing/inactive instances or invalid geometry raise an error rather than
  awarding success for a vanished population.
- Derive the bucket's cavity from `entity/body/collider_body_00`. Vertical
  rays in the bucket's own frame distinguish the open interior from its
  collision shell and exterior; an AABB corner is not inside the bucket.
  The entity's local +Y is its opening axis. Its current transform is read at
  each sample, so translations and tilts follow the simulated body.
- Exclude the popcorn machine's table-contact pedestal collider
  `entity/body/collider_body_003`. Its box geometry covers the occupied
  tabletop area. Using the whole machine's bounding envelope would wrongly
  ignore spills under its roof/tray overhangs. Source contents above the table
  are excluded by the surface-height check.
- Use the popcorn cart's rectangular tabletop collider
  `entity/body/collider_body_000` for the tabletop XY extent and surface height.
  A grain is near that surface when its center's vertical distance is at most
  its bounding-sphere collision radius plus 2 mm of contact tolerance. A grain
  inside the bucket or its solid wall is excluded from tabletop spills.

These paths and the opening axis belong to the delivered popcorn assets;
changing those assets requires updating and revalidating this checker. The
ordinary Isaac task uses USD rigid-instance readback; this path was checked by
stepping the real background for 120 physics steps and observing changed
instance positions with an unchanged population. Fabric-only rigid transforms
without USD writeback are not supported by this checker.

## Verification

From the repository root, with the benchmark source on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/source/geniesim_benchmark/src${PYTHONPATH:+:$PYTHONPATH}"
export GENIESIM_ASSETS_PATH=/path/to/geniesim_assets
python3 -m unittest discover -s source/geniesim_benchmark/tests -p test_popcorn.py -v
```

This covers the 20 configurations, parser, result aggregation, no-VLM imports,
three scores, peak-score retention, actual bucket geometry, moving/tilted
buckets, independently moving prototypes, roof overhangs and machine/air/floor/tabletop cases.
When `trimesh` and `rtree` are available, 5000 seeded points are compared with
an independent closed-mesh volume reference for the bucket cavity.
Asset cases skip when
`GENIESIM_ASSETS_PATH` is not set.

An optional Isaac SDK smoke uses the actual background and bucket without a
policy server, validates live instance readback and then checks controlled
0/0.5/1 layouts on the paused stage. It also restarts with a reduced population
to isolate six physical scenarios: source-only grains, bucket plus floor,
bucket plus tabletop spill, bucket plus exposed-table spill beneath the
machine overhang, empty bucket with a tabletop spill, and a drop outside the
bucket corner. Each case runs 180 physics steps before checking counts and
score; no expected grain position is authored after the drop:

```bash
python3.12 source/geniesim_benchmark/tests/isaac_popcorn_smoke.py
```

A full policy episode remains the usual operator command:

```bash
geniesim benchmark run g2op_manip_scoop_popcorn --infer-host=HOST:PORT
```

## Accuracy validation (2026-11-09)

21 unit/asset tests passed, including an independent closed-mesh comparison
at 5000 reproducible random points with zero containment disagreements. The
Isaac smoke verified the full 1200-instance population plus the independent
prototype, then ran the following physical drops on the real assets:

| Physical result after 180 steps | Inside | Tabletop spills | Score |
|---|---:|---:|---:|
| Both grains remain in the machine | 0 | 0 | 0 |
| One in bucket, one on floor | 1 | 0 | 1 |
| One in bucket, one on exposed table | 1 | 1 | 0.5 |
| One in bucket, one on table beneath roof overhang | 1 | 1 | 0.5 |
| Empty bucket, one on table | 0 | 1 | 0 |
| Drop misses bucket at an AABB corner and lands on table | 0 | 1 | 0 |

All six physical outcomes matched the expected counts and scores. This
validation found and corrected two false-full-score cases: excluding exposed
table beneath the machine overhang, and shifting instance poses when the
standalone prototype moves. The latter caused a tabletop grain's computed Z
to be 0.572 m instead of its physical 0.907 m before the bind-transform fix.

These are controlled geometry/physics checks, not a complete robot-policy
episode. Peak-score retention and immediate completion at full credit remain
the original task's behavior.

## Isaac 6.2 migration validation (2026-11-10)

On `benchmark_migration6.2`, the same rules are connected to the branch's
current checker parser, score layouts and scene initialization hooks. The
smoke uses system Python 3.12, matching the branch's CLI runtime selection.

Validation ran against Isaac Sim `6.2.0.0`: the full 1200-instance population
plus standalone prototype, all three controlled scores and all six physical
drop scenarios passed. Physics ray queries independently confirmed the
reported tabletop instance locations. The 21 rule/asset tests (including the
5000-point mesh reference) and 10 scene lifecycle regression tests passed.
