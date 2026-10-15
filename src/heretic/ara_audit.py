# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in checkpoint observations. No optimizer callbacks, edits, or Lab imports."""

from contextlib import contextmanager
import copy
import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def prepare_audit(settings, wrapper, metadata, parameters):
    from safetensors import safe_open
    from safetensors.torch import load

    modules = getattr(settings, 'research_audit_modules', [])
    input_path = getattr(settings, 'research_reference_inputs', None)
    if not modules and input_path is None:
        return None
    if not modules or input_path is None or len(set(modules)) != len(modules):
        raise ValueError('Audit requires unique modules and research_reference_inputs')
    model = wrapper.model
    rope = getattr(model.config, 'rope_scaling', None)
    static_rope = (
        rope is None
        or (
            isinstance(rope, dict)
            and rope.get('rope_type') == 'default'
            and set(rope) <= {'rope_type', 'rope_theta'}
        )
    )
    if model.config.model_type != 'qwen2' or not static_rope:
        raise ValueError('PR-04 capture supports Qwen2 with static RoPE only')
    path = Path(input_path)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if metadata.get('audit_inputs_sha256') != digest:
        raise ValueError('Audit input SHA256 missing or mismatched')
    with safe_open(path, framework='pt', device='cpu') as stream:
        provenance = stream.metadata() or {}
    if (provenance.get('schema_version') != '1' or provenance.get('artifact_type') != 'audit_input'
            or provenance.get('model_id') != metadata['model_id']
            or provenance.get('model_revision') != metadata['tokenizer_revision']
            or provenance.get('chat_template_sha256') != hashlib.sha256(wrapper.tokenizer.chat_template.encode()).hexdigest()):
        raise ValueError('Audit input model/tokenizer/template provenance mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Audit input changed during loading')
    inputs = load(content)
    if set(inputs) != {'input_ids', 'attention_mask'}:
        raise ValueError('Expected only audit input_ids and attention_mask')
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    if (ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0
            or ids.shape[1] > model.config.max_position_embeddings
            or ids.min() < 0 or ids.max() >= model.config.vocab_size
            or mask.shape != ids.shape or mask.dtype != torch.long or not torch.all(mask == 1)):
        raise ValueError('Expected one valid, unpadded audit token sequence within context')
    names = {id(module): name for name, module in model.get_base_model().named_modules()}
    selected = {}
    for layer in range(parameters.start_layer_index, parameters.end_layer_index):
        for component, values in wrapper.get_layer_modules(layer).items():
            for index, module in enumerate(values):
                selected[names[id(module)]] = f'L{layer}.{component}.M{index}'
    if not set(modules) <= selected.keys():
        raise ValueError('Audit modules must belong to the selected ARA module inventory')
    for name in modules:
        module = model.get_base_model().get_submodule(name)
        if hasattr(module, '_hf_hook') or hasattr(module.base_layer, '_hf_hook'):
            raise ValueError('Dispatched/offloaded audit modules are unsupported; use a single device without offload')
        if (not isinstance(module.base_layer, torch.nn.Linear) or module.merged or module.disable_adapters
                or module.active_adapters != ['default'] or module.scaling['default'] != 1
                or getattr(module, 'fan_in_fan_out', False) or getattr(module, 'lora_variant', {})
                or any(getattr(module, 'lora_bias', {}).values())
                or not (isinstance(module.lora_dropout['default'], torch.nn.Identity)
                        or isinstance(module.lora_dropout['default'], torch.nn.Dropout)
                        and module.lora_dropout['default'].p == 0)):
            raise ValueError('Audit supports only active, ordinary default LoRA Linear modules')
    return {'modules': {name: selected[name] for name in modules}, 'inputs': inputs,
            'input_sha256': digest, 'preserve_row_magnitudes': settings.preserve_row_magnitudes}


@contextmanager
def observation_state(model):
    """Restore mixed training flags, buffers and Torch RNG even after capture failure."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            with torch.no_grad():
                for buffer, original in buffers:
                    buffer.copy_(original)
            for module, training in modes:
                module.training = training


def capture_audit(record, snapshot):
    from safetensors.torch import save_file
    from .ara_research import file_record, save_replay_reference, utc_now, write_json

    settings, model = record['audit'], record['model']
    tensors, modules = {}, []
    with observation_state(model):
        save_replay_reference(snapshot, snapshot / 'reference', model, **settings['inputs'])
        for index, (name, module_id) in enumerate(settings['modules'].items()):
            module = model.get_base_model().get_submodule(name)
            base = module.base_layer.weight.detach().float()
            a, b = module.lora_A['default'].weight, module.lora_B['default'].weight
            objective = base + b @ a
            if settings['preserve_row_magnitudes']:
                objective = F.normalize(objective, p=2, dim=1) * torch.linalg.vector_norm(base, dim=1, keepdim=True)
            # PEFT get_delta_weight can replace adapter storage on CPU half:
            # use a disposable module copy, never the producer's live parameters.
            clone = copy.deepcopy(module)
            live = base + clone.get_delta_weight('default').float()
            probe = torch.linspace(-0.5, 0.5, 2 * base.shape[1], device=module.base_layer.weight.device,
                                   dtype=module.base_layer.weight.dtype).reshape(2, -1)
            output = module(probe)
            clone.merge(safe_merge=True)
            prefix = f'm{index:03d}'
            values = {'base': base, 'objective': objective, 'live': live,
                      'merged': clone.base_layer.weight, 'input': probe,
                      'live_output': output, 'merged_output': clone(probe)}
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite audit tensor: {name}/{key}')
                tensors[prefix + '.' + key] = value.detach().float().cpu().contiguous().clone()
            modules.append({'name': name, 'module_id': module_id, 'prefix': prefix,
                            'base_dtype': str(module.base_layer.weight.dtype), 'adapter_dtype': str(a.dtype)})
            del clone
    target = snapshot / 'audit'
    target.mkdir()
    save_file(tensors, str(target / 'weights.safetensors'))
    # Snapshot publication (including these children) occurs only after capture completes.
    with (target / 'weights.safetensors').open('rb') as stream:
        os.fsync(stream.fileno())
    manifest = {'schema_version': 1, 'artifact_type': 'weight_audit_reference',
                'profile': 'objective-inference-v1', 'creation_time_utc': utc_now(),
                **{key: record['metadata'][key] for key in ('run_id', 'trial_id', 'synthetic')},
                'snapshot_id': json.loads((snapshot / 'manifest.json').read_text())['snapshot_id'],
                'snapshot_manifest_sha256': file_record(snapshot / 'manifest.json')['sha256'],
                'reference_manifest_sha256': file_record(snapshot / 'reference/manifest.json')['sha256'],
                'input_sha256': settings['input_sha256'], 'modules': modules,
                'producer_audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'preserve_row_magnitudes': settings['preserve_row_magnitudes'],
                'objective_definition': 'pinned_ara_formula_reconstructed_at_snapshot_without_bias',
                'live_definition': 'float32_base_plus_peft_delta_with_native_forward_observed_separately',
                'merged_definition': 'peft_module_merge_on_disposable_copy',
                'files': {'weights.safetensors': file_record(target / 'weights.safetensors')}}
    write_json(target / 'manifest.json', manifest)


def validate_recording_config(path='config.toml'):
    """Preflight a user-supplied, fixed-seed single trial; no search or model loading."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    if any(key.startswith('HERETIC_') and key != 'HERETIC_RESEARCH_REPO' for key in os.environ):
        raise ValueError('Remove HERETIC_ environment overrides; this launcher validates config.toml only')
    data = tomllib.loads(Path(path).read_text())
    if (not isinstance(data.get('model'), str) or type(data.get('seed')) is not int
            or not re.fullmatch('[0-9a-f]{40}', data.get('model_commit', ''))
            or data.get('quantization') != 'none' or type(data.get('n_trials')) is not int or data['n_trials'] != 1
            or data.get('evaluate_model') is not None or data.get('reproduce') is not None):
        raise ValueError('PR-04 requires model, pinned commit, explicit seed, quantization=none and n_trials=1')
    modifiers = data.get('modifiers', [])
    if len(modifiers) != 1 or modifiers[0].get('plugin') != 'heretic.modifiers.ara.ARA':
        raise ValueError('Expected one explicit ARA modifier')
    settings = data.get('modifier', {}).get('ARA', {})
    if (not settings.get('research_audit_modules') or not settings.get('research_reference_inputs')
            or not settings.get('research_output') or not settings.get('research_metadata')
            or settings.get('research_trial', 0) != 0):
        raise ValueError('Missing PR-04 audit recording configuration for trial zero')
    study = data.get('study_checkpoint_dir')
    if not isinstance(study, str) or not study:
        raise ValueError('Set an explicit new study_checkpoint_dir for trial zero')
    if Path(study).exists():
        raise FileExistsError(study)
    output_path, study_path = Path(settings['research_output']).resolve(), Path(study).resolve()
    if output_path == study_path or output_path in study_path.parents:
        raise ValueError('Study directory must be outside the new research_output directory')
    if Path(settings['research_output']).exists():
        raise FileExistsError(settings['research_output'])
    metadata = json.loads(Path(settings['research_metadata']).read_text())
    digest = hashlib.sha256(Path(settings['research_reference_inputs']).read_bytes()).hexdigest()
    if metadata.get('audit_inputs_sha256') != digest or metadata.get('tokenizer_revision') != data['model_commit']:
        raise ValueError('Audit input SHA/tokenizer revision mismatch')
    return {'config': str(Path(path).resolve()), 'model': data['model'], 'model_revision': data['model_commit'],
            'seed': data['seed'], 'audit_modules': settings['research_audit_modules'],
            'snapshot_ids': ['s000', 's001', 'final'], 'output': settings['research_output']}
