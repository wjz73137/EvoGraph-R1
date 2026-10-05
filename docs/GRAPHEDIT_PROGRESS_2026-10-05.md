# GraphEdit reproduction: evidence and handoff (2026-10-05)

## Host RAM failure and step-400 recovery

The first full-stage process failed at 19:13 CST with Ray's host-memory OOM,
not a CUDA out-of-memory error: host use reached 187.75/188.48 GiB and exceeded
the native 95% guard. Its metrics contain 104 optimizer updates (327–430), but
only 74 updates (327–400) survived in the model/optimizer checkpoint. Final
validation and the full-epoch checkpoint did not finish. All failed-run logs,
metrics, graph copies and the checkpoint remain intact at
`expr_mm/evqa_graphedit_full1891_3b_epoch1_v1`.

Six independent retrieval services unnecessarily held six runtime GME models.
The existing retriever already uses indexed image vectors for all 1,862 train
and 16 validation rows, and BGE for text. A launcher option now selects this
native cached-vector/BGE path without loading runtime GME. Full image-index
coverage was checked for every row in all six copies. Four visual and four
text queries per service (48 sampled results) were compared before/after:
the response hashes were identical. This does not prove equality for arbitrary
unindexed images or a future broken BGE index; those no longer have GME fallback.
The full dataset is not covered by the small legacy fused-question cache; it
was already using indexed image vectors before this change.

Service RSS fell from approximately 41 GiB to 6.7 GiB. This restores substantial
host-memory headroom but is not proof that worker heap growth cannot recur.
Ray's native memory protection remains enabled. Actor parameter/optimizer
offload, vLLM sleep level 1, model, dataset, tool policy, API backend and rewards
were not changed. No dependency version or installed library was modified.

Resume output is separate:
`expr_mm/evqa_graphedit_full1891_3b_epoch1_resume400_v1`.
Its journal starts with only the 74 retained updates; steps 401–430 must be
replayed and are not double-counted. `reset_dataloader_on_resume=false` restores
the saved position of 148 question presentations. A stateful-loader check
reproduced indices 1288 and 68 and the exact two original step-401 questions.
The graph mutation journals contain no accepted mutations after checkpoint400,
and later logged successful-edit counts are zero. Failed edits have rollback
support, explaining later file mtimes. This evidence supports the graph resume,
but the old checkpoint had no paired graph snapshot and API cache histories are
not rewound; this limitation is explicit in `resume_manifest.json`. The six
resume graphs are independent, hash-verified copies, not the preserved originals.

Checkpoint frequency is now 50 rather than 200 updates. An optional project-code
hook also saves six graph copies with model/dataloader checkpoints, after
synchronous tool calls have drained and before advancing the latest-checkpoint
marker. These disk-backed graph snapshots avoid accumulating whole-file byte
backups in worker RAM. They are retained separately from actor shard cleanup
and may consume roughly 4 GiB per save, approximately 70 GiB over the remainder
of this epoch. The resume preparer uses paired graph snapshots when available.

Background units:

- `evograph-ge-full1891-resume400-train.service`: full-epoch continuation.
- `evograph-ge-full1891-cached-services.service`: six isolated CPU services.
- `evograph-ge-full1891-resume400-status.timer`: read-only checks every 30 minutes,
  including host MemAvailable, swap and GPU 2/3 status.

Old full-run services/timer were stopped only after verifying their exact targets;
the older read-only API on port8005 and other users' GPU0/1 jobs were left alone.
Successful launch is not yet evidence of a completed optimizer update or epoch.
The continuation was launched at 19:44 CST. The full unit-test suite passed
179 tests after these recovery changes (plus two existing deprecation warnings).

## Scope and resources

- Actor: `/home/data/dataset/wjz/models/Qwen2.5-VL-3B-Instruct`.
- Resume source: `expr_mm/evqa_grpo_graph_covered_full_3b_epoch1_v1/checkpoints/global_step_326` under the data root below.
- Code: `/home/wjz/projects/EvoGraph-R1`.
- Data root: `/home/data/dataset/wjz/EvoGraph-R1`.
- Python: `/home/data/env/wjz/evograph-r1/bin/python`.
- Approved physical GPUs: 2 and 3 only. GPUs 0 and 1 belong to another running workload and were not touched.
- Runtime versions were not changed; no installed-library monkey patch was introduced for this repair.

## Confirmed rollout corruption cause

An isolated deterministic experiment compared the same model and question before sleep,
after native vLLM sleep level 1, after sleep level 2 followed by parameter restoration,
and after additionally restoring buffers. Its artifact is:

`expr_mm/evqa_graphedit_controlled_missing_v1/diagnostics/sleep_buffers.json`

Before sleep and after level 1, the model produced the same correct Paris answer.
After level 2 with parameters restored, the answer was corrupted. Two buffers differed:

- `visual.rotary_pos_emb.inv_freq`
- `language_model.model.layers.0.self_attn.rotary_emb.cos_sin_cache`

Restoring these buffers restored the exact original output tokens. In installed vLLM
0.7.3, native deep sleep discards allocations containing these buffers; this training
integration synchronizes parameters, not those nonpersistent buffers. Therefore the
prior corrupted trajectories are not sound evidence that the 3B model cannot learn
GraphEdit. Both experiment launchers now use native `sleep_level=1`. The previously
identified `expandable_segments` setting remains unset so the native CuMemAllocator
can release rollout GPU memory during backward.

## Autonomous validation, not a training result

Artifact root:

`expr_mm/evqa_graphedit_controlled_missing_v1/val_sleep1_autonomous_v1`

Two targeted held-out questions were evaluated from checkpoint 326. Both called
`websearch`. Hotel Bohema completed native graph insertion and subsequent KB retrieval;
Jyväsjärvi answered in prose without performing an edit. Aggregate edit and verified-edit
counts were each 0.5 per question; web-search count was 1.0. F1 and EM were still zero,
and format score was 0.5. This establishes a working edit/verification path, not good
answer quality or completed training. Validation graph writes were confined to disposable
copies under `isolated_smoke_v21`; the immutable baseline was not modified.

First visual retrieval and post-successful-edit verification are controller-forced.
Text search, web search, and the edit proposal were model-generated. There is no forced
second text-search implementation in the current code. Response guidance remains active;
this is a constrained tool-policy experiment, not an unconstrained policy benchmark.

## Separate trajectory truncation repair

`ToolGenerationManager._update_right_side` incorrectly capped the accumulated response
using `max_prompt_length` and retained its prefix, potentially deleting the final answer.
It now uses `max_response_length`, and overflow retains complete vision spans and recent
text. Truncation is logged rather than silently hidden. Three regression tests cover the
response budget, final-answer/vision-span retention, and padding of shorter batch rows.
This is a project-code correctness repair, not an installed dependency patch.

The next optimizer smoke test uses a fresh six-copy graph set:

`expr_mm/evqa_graphedit_controlled_missing_v1/isolated_sleep1_train_v1`

Its output root is `expr_mm/evqa_graphedit_controlled_missing_v1/train_sleep1_complete_v1`.
Configuration: two questions, batch 2, repeats 2, BF16 FSDP, native sleep level 1,
prompt budget 2048, accumulated response budget 2560, tool-response budget 384,
per-turn generation budget 320, vLLM model length 4096, learning rate 5e-7.
Do not count this test as completed until optimizer metrics and checkpoint evidence exist.

This first training test subsequently completed successfully (process exit 0):

- One actual optimizer update, logged at step 327; final counter/checkpoint 328.
- Four rollout trajectories from two questions, not four independent questions.
- Training mean reward -0.625, format 0.375, F1/EM 0.25.
- Advantage range approximately -0.707 to +0.707; gradient norm 9.6875.
- Policy-gradient loss was numerically zero; KL loss 0.017411. A zero scalar PG loss
  alone does not prove a zero gradient, and the update must not be portrayed as large
  or as evidence of meaningful convergence.
- Rollout 182.37 s, reference 17.90 s, actor update 15.41 s, checkpoint 97.01 s.
- No successful training edit; all four trajectories called websearch.
- Final validation F1/EM 0.5, format 0.75, mean reward 0.25, successful edits zero.
- Final checkpoint: `train_sleep1_complete_v1/checkpoints/global_step_328`, 14 files,
  approximately 22 GiB, under the controlled experiment root.
- The existing checkpoint manager automatically removed the intermediate step 327
  actor save after writing the final checkpoint; the final checkpoint is retained.

A second targeted training test, `train_sleep1_edit_evidence_v1`, starts again from
326, not from the held-out-trained smoke checkpoint. It uses temperature 0.3, response
budget 3072, and tool-response budget 512, with the other settings unchanged. Before
reuse, all six graph hyperedge files matched the controlled source SHA-256 exactly.
The training launcher now persists balanced-batch-aligned tool histories and decoded
trajectories so failed or rejected edits can be inspected, rather than inferred from
aggregate scores. These records remain in experiment output, not Git.

The second test also completed one update without OOM: gradient norm 40.5, scalar
PG loss 0.7071, mean reward -0.0625, training F1/EM 0.5, format 0.6875. It still had
zero successful training edits. Its final validation had raw successful-edit count
1.5/question and verified-edit count 1.0/question, but F1/EM only 0.5 and format 0.375.
These counters are tool-success/query-result counters, not a semantic proof that every
mutated edge is correct; successful no-op or redundant operations can inflate them.

Inspection of the saved training histories found two independent causes:

1. The targeted builder labeled training `E-VQA/graphedit_targeted_smoke`, but tool
   query enrichment and reward shaping recognize the canonical `graph_edit` marker.
   The validation split already used that marker. Thus the two splits unintentionally
   took different paths. The builder now uses `E-VQA/graph_edit_targeted_smoke`; a
   launcher preflight and a regression test guard this. The official full-stage dataset
   already had the correct marker and was not changed.
2. The actor proposed a three-star fact about a Lithuanian namesake for the Polish
   Hotel Bohema. The API gate correctly rejected it (entity-identity mismatch with
   confidence 0.95–0.99), but the guidance requested repeated edits even though new
   evidence was needed. The guidance now requests location-specific retrieval after
   rejection, removes copyable value placeholders, and requests plain factual sentences.
   The gate threshold remains 0.85. No gold answer was added to the prompt or query.

The corrected smoke dataset is `datasets_targeted_graph_edit_v1` and its fresh graph
copies are `isolated_canonical_smoke_v1`, both under the controlled experiment root.
The full launcher now uses the GPU-tested response/tool budgets 3072/512; its rollout
temperature remains 1.0, while the diagnostic test can override temperature separately.

## Full-stage preparation

Prepared dataset:

`datasets_mm/E-VQA/processed/paper_grpo_graph_edit_full1891_v1`

- 1,862 training rows: 1,717 images and 1,221 documents.
- 16 validation rows: 16 images and 16 documents.
- 13 original rows excluded to prevent shared validation images/documents.
- No shared image or document between the two splits.

Graph copies:

`expr_mm/evqa_graphedit_full1891_3b_epoch1_v1/isolated_graphs`

Four training copies match the original API-extracted graph; two validation copies
match the controlled graph with eight deliberately hidden facts each. The auditor
verified parquet hashes, image readability, source/target core index hashes, independent
inodes, metadata paths, and hidden-fact counts. Its passed report is:

`expr_mm/evqa_graphedit_full1891_3b_epoch1_v1/preflight_audit.json`

These counts do not mean the full training stage has finished. Starting from 326, the
full stage is intended to process 931 two-question batches (1,862 rows) with two rollouts
per question. Step-counter naming and actual optimizer updates must be reported separately.

## Quality and provenance limitations

- Native graph extraction after an insert may expand one candidate sentence into multiple
  hyperedges; one observed insert added 20 hyperedges and 9 entities. A successful HTTP
  insert does not certify that every extracted edge is correct.
- Websearch uses the configured API model with forced search enabled and optional live
  Wikipedia augmentation. Wikipedia returned HTTP 429 for one query. Provider-generated
  source titles alone are not independently verified citations.
- The API pre-commit gate uses a confidence threshold of 0.85. This is an added safety
  mechanism, not a claim that it is an exact original-paper component.
- The immutable knowledge graph includes the held-out knowledge documents as retrieval
  corpus; split isolation prevents policy-training overlap, not closed-book evaluation.
- Strict answer formatting can yield zero scored F1 even when untagged prose names the
  correct answer. Do not conflate scoring/format failures with factual ignorance.

## Verification and GitHub status

After dataset-routing, guidance, search-evidence and multi-batch validation repairs:
`175 passed` in `evograph_mm/tests`, with two deprecation warnings;
`git diff --check` passed. Test results do not replace a real GPU optimizer smoke test.

Local research branch: `research/evograph-mm-graphedit`.
Local commits include `92f5510` and `1121805`; later repairs may be in newer commits.
`.env`, data, models, indexes, logs, and checkpoints are excluded from submission.

The connected GitHub account is `wjz73137`; `wjz73137/EvoGraph-R1` exists and the
account has repository write permission. The plugin's actual branch-create operation
repeatedly returned HTTP 403, `Resource not accessible by integration`. Repository
ownership, ChatGPT approval settings, and connector authorization are separate.
The plugin route did not upload the code.

With the user's browser authorization, official GitHub CLI 2.102.0 was installed
under `/home/wjz/tools/github-cli`, after verifying its release checksum. Terminal
authentication succeeded as `wjz73137`, with credentials stored in the system keyring.
A repository-local HTTPS credential helper was configured; no token was placed in
the repository or displayed unmasked. New remote `personal` targets the user's
`wjz73137/EvoGraph-R1` repository. Original `origin` still points to the author's
`ninjaX2o/EvoGraph-R1` repository and was not overwritten.

Terminal `git push -u personal research/evograph-mm-graphedit` succeeded, publishing
the research branch and its local commit history. The existing remote `main` was
not modified. Before publication, 175 tests passed again; 157 new historical file
objects were scanned with no credential-pattern findings, sensitive credential
paths, or files exceeding 50 MB. `.env`, models, datasets, graphs, logs and
checkpoints were excluded. Unrelated local `Dave` was preserved and not committed.

## Final preflight repairs and persistent full-stage execution

The canonical smoke completed one optimizer update: training F1/EM 0.5, format 0.6875,
mean reward -0.1625, PG loss approximately -0.7071, gradient norm 6.5. It still had
zero successful training edits; final validation F1/EM 0.5, format 0.875, edits zero.
This is not evidence that the edit policy has been learned. The original targeted run
also initially failed because its Ray Unix socket path exceeded 107 bytes; the failure
log was retained and the temporary path shortened to `.ray/gce`.

Final project-code repairs before full-scale training:

- Grounding a web query now retains the actor's distinct search intent instead of
  replacing every refinement with an identical query. Angle-bracket placeholders are
  stripped from the hint. Identical retries are still penalized.
- Provider responses consisting only of unexecuted JSON search commands are marked
  as non-evidence, including cached copies. A search-enable request flag alone does
  not prove that the provider actually performed search.
- Insert guidance again includes the concrete JSON argument shape, but no gold fact
  or answer value. The actor still chooses the proposed content; API gating remains.
- Multi-batch validation metrics now aggregate all saved rows. The old code indexed
  the final two-row batch with indexes from the full validation set and would fail
  when validating sixteen rows. Two regression tests cover cross-batch alignment.

The full preflight audit was rerun and passed. The stage-1 resume source's 650 training
rows also have zero shared image/document with the full-stage sixteen-row validation set.
The immutable graph's completed report and owner both identify construction as
`api/qwen3.7-max-2026-06-08`, not a local language-model extractor.

Full retrieval services run as the user-only transient unit
`evograph-ge-full1891-services.service`, started at 15:05:23 CST. Full training was
started at 15:13:51 CST on 2026-10-05 as `evograph-ge-full1891-train.service`,
using `.ray/gf1` and physical GPUs 2/3 only. Launch-time source commit: `fdf5cfa`.
The real training log confirms all 1,862 training rows and 16 validation rows were
retained by filtering, with 931 training batches. At 15:20 CST initialization had
finished and the first batch was making tool calls; no full-stage optimizer update
had yet been recorded. Wikipedia augmentation returned HTTP 429 for some queries.
GPU utilization can be zero while the actor waits for CPU retrieval or external API
responses; a zero utilization snapshot alone is not evidence of a failed process.

The first real full-stage optimizer update, logged step 327, subsequently completed:
154.30 seconds total, 119.30 seconds rollout, 17.24 seconds reference log-probability,
13.70 seconds actor update, gradient norm 5.375, learning rate `5e-7`, PG loss 0.35355.
Mean reward was -0.5625, format 0.4375, F1/EM zero, and successful/verified edits zero.
Four trajectories expanded their retrieved images with zero trimmed or dropped image
spans. Native sleep released 7.66 GiB on the logged rank. These are evidence of a real
update without OOM, not evidence of improved quality or learned editing. Step 327
does not mean 327 full-stage updates: it is the first update after resuming counter 326.

Inspecting that batch also exposed a retrieval-quality limitation: for the national
park question, the exact query-image match appeared second (Tusheti National Park),
while the first returned entity was Tianluokeng Tulou cluster. Current controller
guidance fixes the first returned candidate as the anchor, so subsequent searches
were grounded on the wrong entity. This is an observed single-query failure, not a
measured corpus-wide error rate. It prevents attributing the batch's zero F1 solely
to the 3B actor. The full run's retrieval/anchor configuration is being retained;
changing it mid-run would mix experimental configurations. Evaluate retrieval rank
and anchoring independently before interpreting final model quality.
These services do not depend on a terminal or chat connection staying open, but they
are transient user units, not a reboot-persistence guarantee. No other user's units,
GPU processes, proxy settings, drivers or system packages were changed.

The read-only user timer `evograph-ge-full1891-status.timer` records status every
30 minutes through `scripts/record_evqa_training_status.py`. Its first scheduled
check is 15:45:32 CST; a manual check at 15:20 confirmed the training unit was
active and that no completed-step metric existed yet. Recorded status is appended
to `periodic_status.jsonl` in the full experiment directory. The timer does not
restart jobs or change GPU processes. After the experiment finishes, stop this
experiment's timer and retrieval unit to release its own background resources;
no other user's process is a cleanup target.

`run_evqa_graphedit_full_reported.sh` runs the full stage and then writes
`completion_report.json` / `completion_report.md` under the experiment output. The
report requires the complete expected optimizer-step set, final validation, nonempty
checkpoint shards, and a successful training exit. An exited process with missing
updates is explicitly marked incomplete. A full epoch with zero edits must still be
reported as no demonstrated edit-policy learning. Full training is a 3B/two-3090
resource adaptation, not a claim of matching the paper's 7B/four-80GB-A100 setup.
