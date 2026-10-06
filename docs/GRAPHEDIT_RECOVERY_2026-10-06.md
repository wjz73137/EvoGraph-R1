# GraphEdit full-epoch recovery (2026-10-06, CST)

## What stopped

The step-400 continuation stopped at 09:04:03 with exit status 1. Its combined
journal contains steps 327–686 (360 optimizer updates), but the latest complete
model/optimizer/dataloader checkpoint is 650. Neither final validation nor the
final epoch checkpoint completed. The failed output is preserved unchanged:
`/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_graphedit_full1891_3b_epoch1_resume400_v1`.

Ray logs show GCS RPC warnings around 08:50, approximately 12.5 minutes of
event-loop queue delay, and a keepalive timeout at 09:02:36. The GCS actor-owner
long-poll callback failed and Ray force-killed both GPU actors. The main task
subsequently failed with `ActorDiedError`. The generic actor reference-deletion
message is not evidence that training code intentionally dropped actor handles:
the main task remained alive after the cleanup began. Ray's later SIGTERM was
shutdown cleanup, not the initial failure.

This is a confirmed Ray control-plane/RPC failure chain, not a proven GPU OOM.
The precise cause of the long scheduling/RPC interruption remains unknown.
There is no direct host-memory OOM evidence for this continuation. External
search requests were slow and some Wikipedia calls returned 429, but that
alone does not establish the cause of Ray's actor cleanup. Ray 2.40's native
C++ gRPC proxy setting defaults to false; the HTTP proxy is not a demonstrated
cause either. No driver, library version, system setting or dependency patch
was changed for this recovery.

## Consistent recovery at checkpoint 650

New output:
`/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_graphedit_full1891_3b_epoch1_resume650_v1`.

The existing resume preparer verified both ranks' model, optimizer and extra
state shards, and selected checkpoint 650's paired six-graph snapshot rather
than the failed run's later live graphs. Core graph hashes matched the saved
manifest. It created independent graph copies with matching hashes and kept
only metric/trajectory steps 327–650 in the new output. This retains 324 of
the planned 931 stage updates; steps 651–686 (36 unsaved updates) must replay,
leaving 607 updates to finish the epoch. Old records are not double-counted.

The loader checkpoint records 324 yielded batches / 648 question presentations.
With seed 1 and batch size 2, the next indices are 363 and 1270, matching the
two original step-651 questions and their two repeats. The saved trajectory
order differs because training balances rows between workers; the question
multiset, not output ordering, is the loader-continuity check. The resume flag
is `EVOGRAPH_RESET_DATALOADER_ON_RESUME=false`.

Only this user's verified six-service unit and old status timer were stopped.
The replacement services use checkpoint-650 graph copies with native cached
image vectors plus BGE text retrieval. Other users' jobs, the old port-8005
read-only API, the failed output, and the immutable API graph are untouched.

The actor remains Qwen2.5-VL-3B-Instruct on physical GPUs 2 and 3, BF16 FSDP,
GRPO repeats 2, batch size 2, learning rate 5e-7. Search backend, tool policy,
reward, full 1,862-row training data, and 16 validation rows are unchanged.
Native vLLM sleep level 1 and CPU offload remain enabled; expandable segments
remain unset. Paired graph/model checkpoints are scheduled every 50 updates.

Two additional regression tests cover selecting an intact paired snapshot
despite later live edits, and refusing a corrupted paired snapshot. They change
tests only, not training behavior. Resuming is not proof that the underlying
Ray stall is fixed, and startup is not proof of epoch completion.

## Launch and checks

All six services passed health checks. Each of four training graphs covers all
1,862 training image IDs and each validation graph covers all 16 validation
image IDs, with no missing vectors. Four visual and four text searches per
service returned nonempty, error-free results (48 sample query results). This
is a restored-graph preflight, not a comparison against the later failed live
graphs, which may legitimately differ due to post-checkpoint edits. The report
is `retrieval_preflight.json` in the new output.

The full test suite passed 181 tests with two existing deprecation warnings.
Physical GPUs 2 and 3 were checked empty before startup. The continuation was
launched at 12:32:52 CST under
`evograph-ge-full1891-resume650-train.service`. The six CPU services are under
`evograph-ge-full1891-resume650-services.service`; read-only status recording
runs every 30 minutes under `evograph-ge-full1891-resume650-status.timer`.
The timer does not automatically recover failed training. The launcher still
generates a completion report and requires the full metric range, final
validation and final checkpoint before claiming success.
