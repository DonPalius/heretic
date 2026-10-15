# SPDX-License-Identifier: AGPL-3.0-or-later
"""Actual PEFT/Qwen2 CPU audit through the pinned ARA method AST, not full runtime."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from heretic.ara_research import record_trial
from test_ara_research import actual_method


def trial(root, recording, normalize):
    root.mkdir()
    torch.manual_seed(19)
    config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=32, attn_implementation='eager')
    config._commit_hash = 'd' * 40
    targets = ['model.layers.0.self_attn.o_proj', 'model.layers.0.mlp.down_proj']
    peft = LoraConfig(r=2, lora_alpha=2, lora_dropout=0, bias='none', task_type='CAUSAL_LM',
                      target_modules=targets)
    model = get_peft_model(Qwen2ForCausalLM(config), peft, autocast_adapter_dtype=False)
    wrapper = SimpleNamespace(model=model, peft_config=peft, dtype=torch.float32,
                              tokenizer=SimpleNamespace(chat_template='audit-test'))
    wrapper.get_layer_modules = lambda i: {'attn.o_proj':[model.get_base_model().model.layers[i].self_attn.o_proj],
                                          'mlp.down_proj':[model.get_base_model().model.layers[i].mlp.down_proj]}
    inputs = root / 'inputs.safetensors'
    save_file({'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones((1,3),dtype=torch.long)},
              str(inputs),metadata={'schema_version':'1','artifact_type':'audit_input',
                                    'model_id':'synthetic/qwen2','model_revision':'d'*40,
                                    'chat_template_sha256':hashlib.sha256(b'audit-test').hexdigest()})
    metadata = {'run_id':'audit-test','synthetic':True,'lab_git_sha':'c'*40,
                'tokenizer_revision':'d'*40,'audit_inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest()}
    path = root / 'metadata.json'
    path.write_text(json.dumps(metadata))
    options = dict(research_output=str(root/'run') if recording else None,
                   research_metadata=str(path),research_trial=0,research_intermediate_module=1,
                   research_audit_modules=targets,research_reference_inputs=str(inputs),
                   preserve_row_magnitudes=normalize,learning_rate=0.1,max_iter=2,history_size=2,
                   n_optimization_steps=1,print_loss=False)
    settings = SimpleNamespace(**options)
    settings.model_dump = lambda mode: options
    parameters = SimpleNamespace(start_layer_index=0,end_layer_index=1,neighbor_count=1,
                                 overcorrect_relative_weight=0.1,preserve_good_behavior_weight=0.5,
                                 steer_bad_behavior_weight=0.5)
    parameters.to_dict = lambda: {k:v for k,v in vars(parameters).items() if k != 'to_dict'}
    good,bad = {0:{}},{0:{}}
    for name, modules in wrapper.get_layer_modules(0).items():
        module = modules[0]
        x = torch.linspace(-1,1,2*module.in_features).reshape(2,-1)
        good[0][name] = [(x,module(x).detach())]
        bad[0][name] = [(-x,module(-x).detach())]
    modifier = SimpleNamespace(settings=settings,good_module_io=good,bad_module_io=bad,
                               _research_trial=(0,parameters),_research_dataset_hashes={'edit_train':'a'*64})
    context = SimpleNamespace(get_model=lambda:wrapper,
                              settings=SimpleNamespace(quantization='none',model_commit='d'*40,
                                                       seed=19,model='synthetic/qwen2'))
    return model,modifier,context,parameters,path


@pytest.mark.parametrize('normalize',[False,True])
def test_recording_preserves_actual_ara_weights_rng_modes_and_losses(tmp_path, normalize):
    off,modifier_off,ctx_off,p_off,_ = trial(tmp_path/'off',False,normalize)
    on,modifier_on,ctx_on,p_on,_ = trial(tmp_path/'on',True,normalize)
    method = actual_method()
    losses = []
    class Observed(torch.optim.LBFGS):
        def step(self, closure):
            value = super().step(closure)
            losses.append(float(value.detach()))
            return value
    method.__globals__['LBFGS'] = Observed
    method(modifier_off,ctx_off,p_off)
    expected_losses = losses.copy()
    losses.clear()
    expected_rng = torch.get_rng_state().clone()
    modes = [m.training for m in on.modules()]
    method(modifier_on,ctx_on,p_on)
    assert losses == expected_losses
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert modes == [m.training for m in on.modules()]
    assert all(torch.equal(v,on.state_dict()[k]) for k,v in off.state_dict().items())
    assert all(not m._forward_hooks for m in on.modules())
    snapshots = tmp_path/'on/run/trials/t000000/snapshots'
    for name in ('s000','s001','final'):
        assert (snapshots/name/'audit/manifest.json').is_file()
        assert (snapshots/name/'reference/manifest.json').is_file()


@pytest.mark.parametrize('invalid',['hash','module','architecture','dispatch'])
def test_invalid_capture_stops_before_trial_publication(tmp_path,invalid):
    model,modifier,ctx,parameters,path = trial(tmp_path/'input',True,True)
    if invalid == 'hash':
        metadata = json.loads(path.read_text())
        metadata['audit_inputs_sha256'] = '0'*64
        path.write_text(json.dumps(metadata))
    elif invalid == 'module':
        modifier.settings.research_audit_modules = ['model.layers.999.mlp.down_proj']
    elif invalid == 'architecture':
        model.config.model_type = 'llama'
    else:
        model.get_base_model().get_submodule('model.layers.0.self_attn.o_proj')._hf_hook = object()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        with record_trial(modifier,ctx,parameters):
            raise AssertionError('Invalid audit entered optimization')
    assert not (tmp_path/'input/run').exists()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_preflight_rejects_search_and_existing_output(tmp_path,monkeypatch):
    from heretic.ara_audit import validate_recording_config
    root = tmp_path/'check'
    trial(root,True,True)
    monkeypatch.chdir(root)
    text = '\n'.join([
        'model="synthetic/qwen2"', 'model_commit="'+'d'*40+'"', 'seed=19', 'quantization="none"',
        'n_trials=1', 'study_checkpoint_dir="study-checkpoints"', 'modifiers=[{plugin="heretic.modifiers.ara.ARA"}]', '[modifier.ARA]',
        'research_output="run"', 'research_metadata="metadata.json"',
        'research_reference_inputs="inputs.safetensors"',
        'research_audit_modules=["model.layers.0.mlp.down_proj"]'])
    path = root/'config.toml'
    path.write_text(text)
    assert validate_recording_config()['snapshot_ids'] == ['s000','s001','final']
    # Execute the actual --check shell in an isolated layout; never launch Heretic.
    import os
    from pathlib import Path
    import shlex
    import shutil
    import subprocess
    import sys
    scripts = root/'scripts'
    scripts.mkdir()
    repository = Path(__file__).parents[1]
    shutil.copyfile(repository/'scripts/pr04_03_record.sh',scripts/'pr04_03_record.sh')
    (root/'src').symlink_to(repository/'src',target_is_directory=True)
    python = root/'.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' "$@"\n')
    python.chmod(0o755)
    checked = subprocess.run(['bash',str(scripts/'pr04_03_record.sh'),'--check'],
                             capture_output=True,text=True,env=dict(os.environ))
    assert checked.returncode == 0,checked.stderr
    assert json.loads(checked.stdout)['snapshot_ids'] == ['s000','s001','final']
    path.write_text(text.replace('n_trials=1','n_trials=10'))
    with pytest.raises(ValueError,match='n_trials'):
        validate_recording_config()
    path.write_text(text)
    monkeypatch.setenv('HERETIC_SEED','999')
    with pytest.raises(ValueError,match='environment overrides'):
        validate_recording_config()
    monkeypatch.delenv('HERETIC_SEED')
    path.write_text(text.replace('study-checkpoints','run'))
    with pytest.raises(ValueError,match='Study directory'):
        validate_recording_config()
    path.write_text(text)
    (root/'study-checkpoints').mkdir()
    with pytest.raises(FileExistsError):
        validate_recording_config()
    (root/'study-checkpoints').rmdir()
    (root/'run').mkdir()
    with pytest.raises(FileExistsError):
        validate_recording_config()
