# PR-02: optional ARA recording

This fork is based on `662e4ba27ecaf8878b0938c184a78f75a9116f82`. Default behavior
has recording disabled. There is no Guardrails runtime import, external patch to
apply, optimizer resume, new ARA objective or model download in this implementation task.

Use a separate Heretic environment and launch from this repository's root.
Paths below are relative to the launch working directory, so siblings can move
as a unit between this PC and the four-A100 machine. Normal Heretic arguments and
settings remain in its existing config; use approved datasets and a pinned model
commit and explicit seed. This document does not prescribe an editing experiment.

Add to the existing ARA modifier settings (normally `[modifier.ARA]`):

```toml
research_output = "../experiments/run_001"
research_metadata = "research-metadata.json"
research_trial = 0
research_intermediate_module = 1
```

`research_trial` is the zero-based Optuna number, not its presentation index.
The output directory must not exist. Only parameters from that trial's actual
suggestion are recorded; menu reproductions and other trials are left alone.
Intermediate is captured after the stated number of completed selected modules.
It must be in range; exactly three snapshots are published on success.

The provenance JSON requires `run_id`, exact `lab_git_sha`, and exact
`tokenizer_revision` (the same pinned commit as Heretic `model_commit`). Example
placeholders, to replace with real values before a run:

```json
{"run_id":"run_001","lab_git_sha":"LAB_COMMIT_SHA","tokenizer_revision":"MODEL_COMMIT_SHA"}
```

Synthetic fixtures explicitly set `synthetic: true` in their provenance JSON;
real experiments default to false. This does not infer scientific validity.

Dataset hashes are computed from the actual ordered good/bad Prompt records,
using UTF-8 `json.dumps(..., sort_keys=True, ensure_ascii=False)` including system
and user text. Only hashes are persisted. An optional `dataset_hashes` map in the
provenance JSON is checked against them. No sensitive prompt text is dumped.
Commit/version metadata describes the loaded stack; source hashes supplement Git
HEAD for uncommitted changes. Commit the research code before a real recorded run
for ordinary Git reproducibility. No commit or GitHub publication was automated.

## Artifacts and boundaries

```text
../experiments/run_001/
  manifest.json
  trials/t000000/
    manifest.json
    events.jsonl
    snapshots/{s000,s001,final}/
      manifest.json
      adapter.safetensors
```

The v1 envelope follows Guardrails Lab `docs/ARTIFACT_SCHEMA.md`. It records the
full cumulative `lora_` parameter inventory, PEFT config, base parameter fingerprint,
model config, revisions, quantization, dtype and runtime versions. Safetensors
stores detached copies in their original dtype; files have SHA256 and byte size.
The adapter representation is `peft_lora`, format `safetensors`. The Lab's toy
loader remains separate; PR-03 adds an independent Qwen2/PEFT loader. Its synthetic
checks do not certify pretrained/A100 or cross-version runtime compatibility.

Base fingerprint hashes sorted non-LoRA named parameters as float32 CPU values,
with name/shape/float32 headers and little-endian bytes. Its explicit
`base_fingerprint_definition` distinguishes this real-PEFT representation from
the toy state's full state_dict fingerprint; buffers are not included. PR-03 must
verify the representation-specific definition, not assume toy equivalence. Names are the actual PEFT model parameter names (including base_layer paths).
This can be expensive and is done only for the selected recorded trial. It is not
an optimizer state or proof of equivalence to the normalized objective.

Events: trial_start, module_start, lbfgs_outer_step_complete, module_complete,
snapshot_written, trial_complete, failure. Outer events occur AFTER
`optimizer.step(closure)`. No callback or saving inside the closure.
The optional `loss` is the value returned by LBFGS (not necessarily post-step loss).
`trial_complete` here means the ARA modification finished; the main evaluator runs
later and remains unchanged. It is not a claim that Optuna evaluation completed.

Snapshots are staged, file-fsynced and published by rename; catchable write failures
remove staging, emit failure when possible and propagate the original exception.
Existing artifacts are never reused. A hard kill may leave hidden staging, not a
published snapshot. Single writer, no power-loss durability guarantee. Hashes of
closed events appear in the trial manifest. Event append does not imply atomicity
of an entire run; a truncated final line is an incomplete journal.

## Supported scope and checks

Recording currently accepts only unquantized models, explicit pinned model/seed,
ordinary LoRA with bias=none and without DoRA/RSLoRA. Unsupported states fail before
optimization. No architecture or adapter behavior is assumed from synthetic tests.
Read safetensors through the installed library; no custom tensor serializer.

On a complete Heretic environment:

```bash
python -m pytest -q
```

For isolated CPU research tests (without full model/runtime dependencies):

```bash
GUARDRAILS_SCHEMA_REPO=../Research-Dream-Team PYTHONPATH=src \
  python -m pytest -q tests/test_ara_research.py
```

The optional external-contract test calls Lab validation in a separate process.
Without its path it skips that test. No producer runtime depends on the Lab folder
name. The CPU tests compile the actual ARA method/closure/loss AST while bypassing
unavailable unrelated imports. They are not a full import/integration test.
Real four-A100 validation still needs logging on/off runs, intermediate live/reload
logits, configuration and environment recording. See PR02_VALIDATION_REPORT.md.

## PR-03: explicit live replay references

Snapshots now record the actual `attention_implementation`. Previous snapshots
missing this field are rejected by the real loader; regenerate them with the
producer rather than guessing the backend. No optimizer or closure changes.

`save_replay_reference(snapshot_directory, reference_directory, model, input_ids,
attention_mask=None)` is a manual artifact helper. Call it in the producer process
on the exact snapshot state in eval mode, using one approved unpadded input. The
model is the live PEFT model, not the Heretic wrapper. It checks base fingerprint,
all adapters and attention backend, then atomically saves token IDs/mask plus
last-token full-vocabulary float32 logits (use_cache=False), bound to the exact
snapshot manifest SHA. An already updated model is rejected.

This helper is not called by ARA recording or within the closure. Intermediate
references require access to the corresponding live boundary; references from
the consumer's own replay are not independent evidence. Artifacts include token
IDs, so use approved inputs. Lab commands and numerical acceptance are in its
README and docs/PR03_REPORT.md. No pretrained reference was captured here.
