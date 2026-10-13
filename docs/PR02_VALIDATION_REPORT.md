# PR-02 validation — 2026-10-13

**Implemented, not validated on a real model. PR-02 acceptance remains incomplete.**
No A100, real ARA trial, model download or GitHub publication in this task.

Files changed: `src/heretic/modifiers/ara.py`, added `src/heretic/ara_research.py`,
`tests/test_ara_research.py`, `docs/ARA_INSTRUMENTATION.md` and this report.
Upstream AGPL notices preserved. No changes to main.py, model.py, loss, closure,
optimizer configuration, module order, Guardrails Lab code or its public CLI.
Most of the ara.py diff is indentation for an exception-safe recording context.

Prepared and checked against pinned upstream
`662e4ba27ecaf8878b0938c184a78f75a9116f82`. The work branch is
`research/ara-instrumentation`; no commit or remote fork was published.

## Actual evidence

Environment: local Python 3.10.12, torch 2.0.1+cu117, pytest 8.4.2, CPU.
Only safetensors 0.8.0 was installed with permission, in the existing test virtualenv.
This is not the production Heretic stack or a benchmark of the four-A100 machine.

Using the Lab test interpreter, source path pointing to Heretic:

- Isolated `python -m pytest -q tests/test_ara_research.py`: preliminary 3 passed;
  expanded 5 passed; six tests with the external contract passed in the working
  copy and after applying the fork changes (6 passed, 2.76 s).
- With `GUARDRAILS_SCHEMA_REPO` set: the six-test suite includes an independent
  process validating run/trial/snapshot manifests, SHA tables and events with the
  existing Lab v1 validator. The producer does not import the Lab at runtime.
- Existing `tests/test_config.py`: collection failed, missing `pydantic_settings`.
  Full upstream suite/model import therefore NOT verified; tests were not altered
  or skipped to hide this limitation. No full upstream stack installed here.

Synthetic on/off checks execute the actual edited `modify_model`, ara_loss, objective
and closure bodies through AST compilation. Only model/runtime imports are bypassed.
Two CPU LoRA-like modules, identical fixed parameters and two LBFGS outer steps per
module: weights, logits, returned loss histories and Torch RNG state are exactly
equal. This is evidence for the fixture, not a claim of GPU bitwise determinism.
AST checks also compare objective/closure/loss with the pinned Git source.

Three cumulative snapshots are safe-loaded independently onto copied toy bases.
Initial, intermediate and final logits exactly equal their corresponding toy states;
intermediate includes the still-unmodified second adapter. Finite tensors, key
inventory and write-read equality checked. Corrupted file detected by SHA and
safetensors. Simulated interrupted intermediate publication leaves no final target
or staging and records failure. Other trials/menu reproduction do not record.

Exact final test invocation (from the Heretic root):

```bash
GUARDRAILS_SCHEMA_REPO=../Research-Dream-Team PYTHONPATH=src \
  ../Research-Dream-Team/.venv/bin/python -m pytest -q tests/test_ara_research.py
```

The full stack was not installed. New tests and artifacts declare themselves
synthetic; their metadata versions and revisions are fixture values, not model evidence.

## Assumptions and limits

Opt-in, selected trial only. Relative paths resolve against launch CWD. Existing
output paths fail; snapshots contain all LoRA parameters, not just the current
module. Ordinary unquantized PEFT LoRA only. Model commit must match loaded config;
tokenizer revision records the same pinned loader request, not independent file
attestation. Actual prompt materializations are hashed; no behavioral labels inferred.

Snapshot envelope remains v1; real representation is explicitly peft_lora/safetensors.
Lab toy replay is not extended. Effective objective/inference/export weights are
not declared equal; optional row normalization remains untouched. The stored loss
is LBFGS's returned value. Trial completion marks modifier completion, not later
Optuna evaluation. No optimizer history/RNG resume or power-loss durability promise.

## Remaining real acceptance / next PR only

On the intended four-A100 machine, use separate Heretic and Lab environments.
Verify full imports/config tests; record one approved, fixed-configuration trial
with initial/intermediate/final snapshots; compare logging on/off objective values,
effective adapters and fixed-input logits, then independently reload the intermediate
real adapter. Report tolerances and any discrepancy. Until those checks pass,
PR-02 is implemented but not validated. Do not start PR-03 in this task.
