# GraphEdit reproduction: evidence and handoff (2026-10-05)

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

After the repairs: `166 passed` in `evograph_mm/tests`, with two deprecation warnings;
`git diff --check` passed. Test results do not replace a real GPU optimizer smoke test.

Local research branch: `research/evograph-mm-graphedit`.
Local commits include `92f5510` and `1121805`; later repairs may be in newer commits.
`.env`, data, models, indexes, logs, and checkpoints are excluded from submission.

The connected GitHub account is `wjz73137`; `wjz73137/EvoGraph-R1` exists and the
account has repository write permission. However, the connector's actual branch-create
operation returned HTTP 403, `Resource not accessible by integration`. Repository
ownership and connector authorization are separate. Upload has NOT succeeded; do not
present the local commits as published. The original `origin` still points to the
author's `ninjaX2o/EvoGraph-R1` repository and was not overwritten.
