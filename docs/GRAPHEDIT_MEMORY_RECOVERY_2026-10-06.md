# Host-memory recovery and diagnostic gate (2026-10-06, CST)

## Confirmed failure, not a guessed leak

The step-650 continuation completed updates 651–672 and was killed at 13:47:13.
The user journal explicitly records `systemd-oomd killed 722 process(es) in this
unit` and signal 9. This is separate from the prior GCS RPC failure and from
the new-account API authorization errors. Full system oomd logs are not readable
by this user; no sudo, protection changes, swap changes or other-user process
operations were performed. GPU 0/1 belong to another user's training workload.

The last half-hour sample before exit, at 13:32, had approximately 61.2 GiB
MemAvailable and almost no free swap. The exact pressure at the kill instant
was not sampled. Ray's Plasma log at 13:46:59 reports zero current usage within
its configured 8 GiB capacity, so a full Ray object store is not the demonstrated
cause. The former run's whole-host shared-memory value was approximately 80 GiB
during startup, but no per-process/cgroup breakdown exists for it. Do not claim
that this proves a particular checkpoint, offload or pinned-memory leak.

The latest durable checkpoint remains the preserved model/optimizer/loader and
six-graph snapshot at
`expr_mm/evqa_graphedit_full1891_3b_epoch1_resume400_v1/checkpoints/global_step_650`.
The failed step-650 output and all 22 unsaved updates remain untouched.

## Changes within project code, not installed-library patches

- Optional native PyTorch CPU/mmap checkpoint reading via
  `EVOGRAPH_CHECKPOINT_MMAP_LOAD=true`. The model temporary state is applied and
  released before deserializing the optimizer. RNG and LR scheduler state still
  restore after both. This reduces avoidable staging/copy risk; it is not proof
  of the failure's root cause or a guarantee of long-run stability.
- `EVOGRAPH_MEMORY_DIAGNOSTICS_DIR` enables per-process JSONL records at model
  initialization, checkpoint reading, rollout wake/sleep, actor update/offload,
  trainer batch boundaries and checkpoint saves. Records include proportional
  anonymous/shared memory, VmLck, cgroup anon/file/shmem and pressure, host
  availability/swap, CUDA allocations, and live vLLM CPU backup bytes. No
  prompts, environment values, command lines or credentials are recorded.
- Resume preparation can select an explicitly preserved external checkpoint
  when the failed run produced no local save. Its normal shard/hash/graph and
  metric-contiguity checks still apply, including step-number validation.
- The launcher exposes native `remove_previous_ckpt_in_save` so the diagnostic
  stage can retain both normal and terminal checkpoints. No dependency version,
  CUDA setting, driver, reward or search backend was changed.

## Measured pinned-host allocation mechanism

The first probe started at15:35:49 and was deliberately stopped after real
updates651–652. Its artifacts remain under `memory_probe_v1`; they are not used
as a durable resume point. CPU/mmap checkpoint restore worked but did not solve
host-memory pressure. Each worker's proportional shared memory rose from about
15.45 GiB to33.45 GiB specifically at `checkpoint_actor_offloaded`; the combined
training-cgroup shmem rose from30.9 GiB to66.9 GiB. The model/optimizer checkpoint
files were CPU/mmap staged, not responsible for that shmem classification.

Native PyTorch2.5.1 `TensorConversions.cpp` creates a pinned CPU destination for
CUDA `.to('cpu', non_blocking=True)`. Its caching host allocator rounds allocation
sizes up to powers of two and keeps freed blocks for reuse. This is confirmed in
the versioned upstream source:

- https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/TensorConversions.cpp
- https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/core/CachingHostAllocator.h

A32 MiB real CUDA transfer on physicalGPU2 verified asynchronous output is pinned,
blocking output is not pinned, and both have identical values. Project manual
model/optimizer offload now has opt-in
`EVOGRAPH_FSDP_CPU_OFFLOAD_NON_BLOCKING=false`, invoking native synchronous `.to()`.
The original async default is preserved outside this recovery workflow. Native
FSDP reference offload, GPU reload and vLLM sleep1 are unchanged. No native library
is patched, no allocator symbol is injected and no system protection is weakened.
Blocking copies may cost transfer time; their full-run memory benefit still needs
the new real-update probe. This mechanism explains the measured offload increment,
not necessarily every allocation or the entire earlier oomd event.

The v2 first launch stopped before creating workers because Ray's UNIX socket path
exceeded107 bytes. No optimizer update occurred in that attempt. The Ray scratch
paths were shortened to `/home/data/dataset/wjz/.ray/gmp2` and `gmr2`; the traceback
is preserved in v2 train.log. No training state or graph edits existed to rewind.

## Bounded validation before full continuation

Diagnostic output:
`expr_mm/evqa_graphedit_full1891_3b_epoch1_memory_probe_v2`.
The preparer retained 324 updates (327–650) and recorded 22 unsaved updates to
replay. Six independent, hash-verified graph copies come from checkpoint650,
not the later failed live graphs. The loader position is restored without reset.
The actor remains Qwen2.5-VL-3B-Instruct on physical GPUs 2/3, batch2, repeats2,
BF16 FSDP, native vLLM sleep1, pageable manual CPU offload and dataset1862/validation16.
Expandable segments remain unset. The replacement API key was verified to call
both the existing Flash and Max models; the key itself is excluded from this doc.

The managed workflow runs exactly 20 new optimizer updates (651–670), using the
full dataset's restored next batches. It saves every10 updates and preserves
normal checkpoint670 while final validation produces terminal counter671. The
terminal counter is advanced after update670, so full continuation must restore
normal670 and next log671; resuming terminal671 would skip a metric label.

The gate requires all 20 finite-gradient updates, complete normal670 shards and
paired graphs, diagnostic final validation, and 20 post-update records from each
GPU worker. Conservative diagnostic acceptance requires at least40 GiB sampled
whole-host available memory, at most80 GiB training-cgroup anon+shmem, PSI
some.avg10 below40 percent, and at most4 GiB per-worker anonymous growth between
the first and last two post-update samples. These are operational guardrails,
not learned thresholds or proof against future spikes. They do not disable
systemd or Ray protection and do not alter other users' allocations.

Only if the process exits successfully and the gate accepts does the workflow
prepare a new independent full output,
`expr_mm/evqa_graphedit_full1891_3b_epoch1_memory_resume670_v2`, restore normal670,
verify retrieval coverage, and start continuation through the original final
counter1258. Full-stage metrics remain contiguous327–1257, totaling931 updates.
Full continuation also saves every10 updates, retaining only the latest actor
shards but retaining earlier graph snapshots. About60 additional graph snapshots
can consume approximately240 GiB; the data disk currently has about2.9 TiB free.
Any process/gate/preparation/health failure stops the workflow without silently
restarting a failed model or claiming a completed epoch.

Units use this user's systemd manager. The bounded workflow is
`evograph-ge-full1891-memory-probe-v2-train.service`; six CPU services are
`evograph-ge-full1891-memory-probe-v2-services.service`. After acceptance, the full
units are `evograph-ge-full1891-memory-resume670-v2-train.service` and
`evograph-ge-full1891-memory-resume670-v2-services.service`. Read-only status timers
record every30 minutes. The workflow writes `recovery_workflow.json`, the gate
writes `memory_probe_report.json`, and the normal full reporter still requires
final validation plus the final checkpoint before declaring completion.

Full unit suite after these changes: 198 passed, two existing deprecation warnings.
Actual model loading, optimizer updates, memory peaks, checkpoint saving and
automatic continuation must still be confirmed from runtime artifacts.
