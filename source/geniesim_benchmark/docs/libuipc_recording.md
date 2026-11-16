# IPC observation recordings

The cloth-hanger, folding and drawstring-bag tasks reuse the synchronized
observations already requested by the policy loop when `benchmark.record=true`.
Recording never requests another physics snapshot or camera render. The native
`physical_state` archive and scoring retain their independent sampling contracts.

Each episode keeps three labelled WebM streams and, when `record_state` is
enabled, a paired `state.jsonl`. The head video shows the instruction and
zero-based episode index through the common `LocalRecorder` overlays.

## Simulation-time video

Video uses a fixed frame rate anchored at the first observed
`snapshot.sim_time`. An observed image is held until a newer observation becomes
available. The recorder repeats previously decoded camera pixels for intervening
video grid ticks; those repetitions do not represent new measurements. Inference
wall time does not extend the video. The final observation is included at the
next grid tick when its time falls between ticks, adding at most one interval of
quantization before its final display interval.

For example, observations at simulation times 1.25 s and 2.25 s produce eleven
frames at 10 fps: ten copies of the first observation followed by the final
observation. All three cameras use the same timeline. Only the current decoded
camera set and bounded encoder queues are retained in memory. Encoder queues use
IPC-local backpressure so sparse intervals cannot silently shorten the video by
dropping repeated frames. Before closing, IPC waits until every encoder has
written every submitted frame, including any frame already held by its worker.
This preserves frame order when the common recorder closes its encoder pipes.
Encoder failures stop recording explicitly.

## Paired sparse state

The JSONL header retains `geniesim.local-recorder-state.v1` and adds
`sampling=unique_observation_snapshot`, `video_time_origin` and
`frame_index_semantics=first_grid_tick_at_or_after_observation`.

Each subsequent row contains the observed action `step`, `sim_time`, measured
joint positions, welded joint positions, base pose, and the exact
`source_id`, `epoch`, `snapshot_id`. Available request, physics tick, cloth tick
and capture wall-time provenance are also retained. Video repetitions produce
no additional state rows.

Rows are deduplicated by `(source_id, epoch, snapshot_id)`, independently of the
action step and image bytes. Distinct snapshots at the same simulation time are
retained separately. Their `frame_index` may be shared: it names the earliest
candidate video grid tick, and does not promise display if a newer observation
supersedes that snapshot before that tick. Exact snapshot identity is therefore
the association for sparse measurements, rather than a claim that every state
row has a unique encoded frame.

One recording accepts one source/epoch pair. Foreign identities, missing snapshot
IDs, nonfinite or negative simulation times, and backwards simulation times fail
before adding a state row or new video frames. Exact duplicates at the current
timestamp are ignored; packets older than the latest accepted simulation time
are rejected. At an advancing timestamp the prior timestamp's deduplication
index is released, while all measurements remain on disk.

`record_state=false` suppresses the paired state file and retains the same video
timeline. Closing is idempotent. Episode directories and existing recording
artifacts are preserved; the IPC adapter opens its state file exclusively and
uses the common recorder's unique episode directory.

## Implementation boundary

`libuipc.recording.EpisodeRecording` retains its existing constructor,
`capture(packet, step)` and `close()` APIs. It uses the common recorder's public
video-writing method with state disabled for this instance. Queue backpressure
and final frame-count checks inspect that instance's encoder queues and counters;
the shared `LocalRecorder` defaults and other runtimes are unchanged.

## Publishing paired state files

The IPC-owned `publish_observation_states(bags, output_dir)` helper exports each
committed bag's `state.jsonl` as `output_dir/{bag.name}.state.jsonl` and returns
the ordered destination paths. Every source stays in its episode directory.
Each copy first writes a unique staging file in the output directory, flushes
and synchronizes it, then publishes the complete file using an exclusive hard
link. An interrupted copy leaves only its staging file, allowing a fresh retry.
Successful and interrupted staging files are retained. Repeated publication
accepts byte-identical existing destinations and rejects conflicting contents
without replacing them. Missing committed state files fail explicitly before
publication starts.

The shared recorder remains video-only. This IPC component owns its state schema
constant and publishing helper, so other task runtimes keep their existing
recording behavior.
