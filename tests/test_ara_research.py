# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU tests execute the actual ARA method AST without unrelated runtime imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from heretic.ara_research import (adapter_state, record_event, record_trial, save_snapshot)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(2, 2, bias=False)
        self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(1, 2, bias=False)})
        torch.nn.init.zeros_(self.lora_B['default'].weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B['default'](self.lora_A['default'](x))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.device = torch.device('cpu')
        self.config = SimpleNamespace(_commit_hash='d' * 40, to_dict=lambda: {'architecture': 'test_linear_lora'})

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def actual_method():
    path = Path(__file__).parents[1] / 'src/heretic/modifiers/ara.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('ara_loss', 'mean_distances_to_knn')]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ARA')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'modify_model')
    namespace = {'torch': torch, 'LA': torch.linalg, 'F': torch.nn.functional, 'LBFGS': torch.optim.LBFGS,
                 'cast': cast, 'Linear': Layer, 'Tensor': torch.Tensor, 'Context': object,
                 'Parameters': object, 'record_trial': record_trial, 'record_event': record_event}
    exec(compile(ast.Module(body=functions + [method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['modify_model']


def setup_trial(tmp_path, monkeypatch, enabled):
    import heretic.ara_research as research
    monkeypatch.setattr(research.importlib.metadata, 'version', lambda name: 'synthetic-test')
    torch.manual_seed(21)
    toy = Toy()
    wrapper = SimpleNamespace(model=toy, dtype=torch.float32,
                              tokenizer=SimpleNamespace(chat_template='test-template'))
    wrapper.get_layer_modules = lambda i: {'linear': [toy.layers[i]]}
    config = {'r': 1, 'lora_alpha': 1, 'lora_dropout': 0, 'target_modules': ['layers.0', 'layers.1'], 'bias': 'none'}
    wrapper.peft_config = SimpleNamespace(to_dict=lambda: config)
    metadata = tmp_path / 'metadata.json'
    hashes = {'edit_train_good': 'a' * 64, 'edit_train_bad': 'b' * 64}
    metadata.write_text(json.dumps({'synthetic': True, 'run_id': 'test-run', 'lab_git_sha': 'c' * 40,
                                    'tokenizer_revision': 'd' * 40, 'dataset_hashes': hashes}))
    options = dict(research_output=str(tmp_path / 'run') if enabled else None,
                   research_metadata=str(metadata), research_trial=0, research_intermediate_module=1,
                   preserve_row_magnitudes=True, learning_rate=0.1, max_iter=3, history_size=3,
                   n_optimization_steps=2, print_loss=False)
    settings = SimpleNamespace(**options)
    settings.model_dump = lambda mode: options
    parameters = SimpleNamespace(start_layer_index=0, end_layer_index=2, neighbor_count=1,
                                 overcorrect_relative_weight=0.1, preserve_good_behavior_weight=0.5,
                                 steer_bad_behavior_weight=0.5)
    parameters.to_dict = lambda: {k: v for k, v in vars(parameters).items() if k != 'to_dict'}
    inputs = torch.tensor([[1.0, 0.5], [-0.5, 1.0]])
    good, bad = {}, {}
    for i, layer in enumerate(toy.layers):
        with torch.no_grad():
            good[i] = {'linear': [(inputs.clone(), layer(inputs).clone())]}
            bad[i] = {'linear': [(-inputs.clone(), layer(-inputs).clone())]}
    modifier = SimpleNamespace(settings=settings, good_module_io=good, bad_module_io=bad,
                               _research_trial=(0, parameters), _research_dataset_hashes=hashes)
    ctx = SimpleNamespace(get_model=lambda: wrapper,
                          settings=SimpleNamespace(quantization='none', model_commit='d' * 40, seed=21, model='synthetic/test'))
    return toy, modifier, ctx, parameters, inputs


def test_enabled_disabled_equivalence_events_and_replay(tmp_path, monkeypatch):
    disabled = tmp_path / 'disabled'
    enabled = tmp_path / 'enabled'
    disabled.mkdir()
    enabled.mkdir()
    base, off, ctx_off, params_off, inputs = setup_trial(disabled, monkeypatch, False)
    on_model, on, ctx_on, params_on, _ = setup_trial(enabled, monkeypatch, True)
    original = copy.deepcopy(on_model)
    method = actual_method()
    losses = []
    optimizer_type = torch.optim.LBFGS
    class ObservedLBFGS(optimizer_type):
        def step(self, closure):
            result = super().step(closure)
            losses.append(float(result.detach()))
            return result
    method.__globals__['LBFGS'] = ObservedLBFGS
    method(off, ctx_off, params_off)
    losses_off = losses.copy()
    losses.clear()
    rng_off = torch.get_rng_state().clone()
    method(on, ctx_on, params_on)
    assert losses == losses_off
    assert torch.equal(rng_off, torch.get_rng_state())
    for name, value in base.state_dict().items():
        assert torch.equal(value, on_model.state_dict()[name])
    assert torch.equal(base(inputs), on_model(inputs))
    assert not (disabled / 'run').exists()
    trial = enabled / 'run/trials/t000000'
    events = [json.loads(line) for line in (trial / 'events.jsonl').read_text().splitlines()]
    assert [row['sequence_number'] for row in events] == list(range(len(events)))
    assert [row['event_type'] for row in events].count('lbfgs_outer_step_complete') == 4
    assert [row['event_type'] for row in events].count('module_complete') == 2
    assert events[-1]['event_type'] == 'trial_complete'
    for name in ('s000', 's001', 'final'):
        directory = trial / 'snapshots' / name
        manifest = json.loads((directory / 'manifest.json').read_text())
        content = (directory / 'adapter.safetensors').read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest['files']['adapter.safetensors']['sha256']
        saved = load_file(str(directory / 'adapter.safetensors'))
        assert set(saved) == set(adapter_state(original))
        replayed = copy.deepcopy(original)
        with torch.no_grad():
            for key, value in saved.items():
                dict(replayed.named_parameters())[key].copy_(value)
        if name == 's000':
            assert torch.equal(replayed(inputs), original(inputs))
        if name == 'final':
            assert torch.equal(replayed(inputs), on_model(inputs))
        if name == 's001':
            # First module updated, second module still original; cumulative state includes both.
            expected = copy.deepcopy(original)
            expected.layers[0].load_state_dict(on_model.layers[0].state_dict())
            assert torch.equal(replayed(inputs), expected(inputs))


def test_save_failure_leaves_no_partial_snapshot(tmp_path, monkeypatch):
    import heretic.ara_research as research
    _, modifier, ctx, parameters, _ = setup_trial(tmp_path, monkeypatch, True)
    rename = research.os.rename
    def fail(source, target):
        if Path(target).name == 's001':
            raise OSError('simulated publication failure')
        return rename(source, target)
    monkeypatch.setattr(research.os, 'rename', fail)
    with pytest.raises(OSError, match='publication'):
        actual_method()(modifier, ctx, parameters)
    trial = tmp_path / 'run/trials/t000000'
    assert not (trial / 'snapshots/s001').exists()
    assert not list((trial / 'snapshots').glob('.s001-*'))
    assert json.loads((trial / 'manifest.json').read_text())['status'] == 'failed'
    events = [json.loads(line) for line in (trial / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['event_type'] == 'failure'


def test_reproduction_and_other_trials_do_not_record(tmp_path, monkeypatch):
    _, modifier, ctx, parameters, _ = setup_trial(tmp_path, monkeypatch, True)
    modifier._research_trial = (1, parameters)
    with record_trial(modifier, ctx, parameters) as record:
        assert record is None
    modifier._research_trial = (0, copy.copy(parameters))
    with record_trial(modifier, ctx, parameters) as record:
        assert record is None
    assert not (tmp_path / 'run').exists()


def test_objective_and_closure_unchanged_from_pinned_upstream():
    import subprocess
    repository = Path(__file__).parents[1]
    original = subprocess.check_output(['git', '-C', str(repository), 'show',
                                       '662e4ba27ecaf8878b0938c184a78f75a9116f82:src/heretic/modifiers/ara.py'], text=True)
    current = (repository / 'src/heretic/modifiers/ara.py').read_text()
    for name in ('objective', 'closure', 'ara_loss', 'mean_distances_to_knn'):
        before = next(n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in ast.walk(ast.parse(current)) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before) == ast.dump(after)


def test_corrupted_safetensor_rejected(tmp_path, monkeypatch):
    _, modifier, ctx, parameters, _ = setup_trial(tmp_path, monkeypatch, True)
    actual_method()(modifier, ctx, parameters)
    path = tmp_path / 'run/trials/t000000/snapshots/s001/adapter.safetensors'
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    path.write_bytes(b'corrupted')
    assert hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][path.name]['sha256']
    with pytest.raises(SafetensorError):
        load_file(str(path))


def test_guardrails_contract_in_separate_process(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    lab = os.environ.get('GUARDRAILS_SCHEMA_REPO')
    if not lab:
        pytest.skip('Set GUARDRAILS_SCHEMA_REPO to validate the external contract')
    _, modifier, ctx, parameters, _ = setup_trial(tmp_path, monkeypatch, True)
    actual_method()(modifier, ctx, parameters)
    code = '''
import json, pathlib, sys
from guardrails_lab.artifacts import validate_manifest, verify_files, validate_events
root = pathlib.Path(sys.argv[1])
for path in root.rglob('manifest.json'):
    data = validate_manifest(json.loads(path.read_text()))
    verify_files(path.parent, data)
trial = root / 'trials/t000000'
validate_events(trial / 'events.jsonl', run_id='test-run', trial_id='t000000')
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(lab) / 'src'))
    subprocess.run([sys.executable, '-c', code, str(tmp_path / 'run')],
                   env=environment, check=True, capture_output=True, text=True)
