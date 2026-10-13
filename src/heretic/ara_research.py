# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in ARA observation. Files only; no Guardrails imports or optimizer changes."""

from contextlib import contextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path):
    content = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}


def base_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if 'lora_' in name:
            continue
        value = parameter.detach().float().cpu().reshape(-1)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite base parameter')
        digest.update(json.dumps([name, list(parameter.shape), 'torch.float32']).encode())
        # Stream blocks so fingerprinting does not materialize a full Python list of a large weight.
        for chunk in value.split(65536):
            values = chunk.tolist()
            digest.update(struct.pack(f'<{len(values)}f', *values))
    return digest.hexdigest()


def adapter_state(model):
    tensors = {name: value.detach().cpu().contiguous().clone()
               for name, value in model.named_parameters() if 'lora_' in name}
    if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError('Missing or nonfinite cumulative adapter')
    return tensors


def save_snapshot(record, snapshot_id, checkpoint_kind):
    from safetensors.torch import load_file, save_file

    target = record['directory'] / 'snapshots' / snapshot_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    staging = Path(tempfile.mkdtemp(prefix='.' + snapshot_id + '-', dir=target.parent))
    try:
        tensors = adapter_state(record['model'])
        if set(tensors) != record['adapter_names']:
            raise ValueError('Cumulative adapter inventory changed')
        filename = 'adapter.safetensors'
        save_file(tensors, str(staging / filename))
        with (staging / filename).open('rb') as stream:
            os.fsync(stream.fileno())
        loaded = load_file(str(staging / filename))
        if set(loaded) != set(tensors) or any(not torch.equal(tensors[n], loaded[n]) for n in tensors):
            raise ValueError('Snapshot round trip failed')
        manifest = dict(record['metadata'], artifact_type='snapshot', snapshot_id=snapshot_id,
                        checkpoint_kind=checkpoint_kind, cumulative=True,
                        optimizer_boundary=record['boundary'].copy(), creation_time_utc=utc_now(),
                        base_state_sha256=record['base_hash'],
                        base_fingerprint_definition='sorted non-LoRA named parameters, PEFT names, float32 CPU little-endian with JSON name/shape/torch.float32 header',
                        model_config=record['model_config'],
                        attention_implementation=record['model'].config._attn_implementation,
                        files={filename: file_record(staging / filename)},
                        adapter={'representation': 'peft_lora', 'format': 'safetensors',
                                 'tensors': {name: {'file': filename, 'shape': list(value.shape),
                                                    'dtype': str(value.dtype).removeprefix('torch.')}
                                             for name, value in tensors.items()}})
        write_json(staging / 'manifest.json', manifest)
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    record_event(record, 'snapshot_written', snapshot_id=snapshot_id)


def record_event(record, event_type, *, module_id=None, outer_step=None, snapshot_id=None, **details):
    if module_id is not None:
        record['boundary']['module_id'] = module_id
    if outer_step is not None:
        record['boundary']['outer_step'] = outer_step
    kinds = {'trial_start': 'trial_start', 'module_start': 'outer_step_complete',
             'lbfgs_outer_step_complete': 'outer_step_complete', 'module_complete': 'module_complete',
             'trial_complete': 'trial_complete'}
    if event_type in kinds:
        record['boundary']['kind'] = kinds[event_type]
    if event_type == 'module_complete':
        record['boundary']['completed_modules'].append(record['boundary']['module_id'])
    sequence = record['sequence']
    event = {'schema_version': 1, 'run_id': record['metadata']['run_id'],
             'trial_id': record['metadata']['trial_id'], 'event_id': f'e{sequence:06d}',
             'sequence_number': sequence, 'event_type': event_type, 'creation_time_utc': utc_now(),
             'snapshot_id': snapshot_id, 'optimizer_boundary': record['boundary'].copy(), **details}
    with (record['directory'] / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    record['sequence'] += 1
    if (event_type == 'module_complete'
            and len(record['boundary']['completed_modules']) == record['intermediate_module']):
        save_snapshot(record, 's001', 'partial')


def close_trial(record, status):
    manifest = dict(record['metadata'], artifact_type='trial', status=status,
                    optimizer_boundary=record['boundary'], creation_time_utc=utc_now(),
                    files={'events.jsonl': file_record(record['directory'] / 'events.jsonl')})
    write_json(record['directory'] / 'manifest.json', manifest)


@contextmanager
def record_trial(modifier, ctx, parameters):
    settings = modifier.settings
    trial = getattr(modifier, '_research_trial', None)
    if (settings.research_output is None or trial is None or trial[1] is not parameters
            or trial[0] != settings.research_trial):
        yield None
        return
    import safetensors.torch  # Fail before optimization if unavailable.

    model = ctx.get_model()
    if str(ctx.settings.quantization) not in ('none', 'QuantizationMethod.NONE'):
        raise ValueError('PR-02 currently records only unquantized models')
    revision = ctx.settings.model_commit
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Research recording requires an exact model_commit SHA')
    if type(ctx.settings.seed) is not int:
        raise ValueError('Research recording requires an explicit seed')
    count = sum(len(modules) for layer in range(parameters.start_layer_index, parameters.end_layer_index)
                for modules in model.get_layer_modules(layer).values())
    if not 1 <= settings.research_intermediate_module <= count:
        raise ValueError('Intermediate module must be within the selected module inventory')
    if settings.research_metadata is None:
        raise ValueError('Research recording requires research_metadata JSON')
    metadata = json.loads(Path(settings.research_metadata).read_text(encoding='utf-8'))
    if type(metadata.get('synthetic', False)) is not bool:
        raise ValueError('synthetic must be an explicit boolean')
    for key in ('run_id', 'lab_git_sha', 'tokenizer_revision'):
        value = metadata.get(key)
        pattern = '[A-Za-z0-9][A-Za-z0-9_.-]*' if key == 'run_id' else '[0-9a-f]{40}'
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f'Missing or invalid research metadata: {key}')
    if getattr(model.model.config, '_commit_hash', None) != revision:
        raise ValueError('Loaded model commit does not match pinned revision')
    if metadata['tokenizer_revision'] != revision:
        raise ValueError('Tokenizer must use the pinned model commit')
    if 'dataset_hashes' in metadata and metadata['dataset_hashes'] != modifier._research_dataset_hashes:
        raise ValueError('Research dataset hashes do not match loaded prompts')
    root = Path(settings.research_output).resolve()
    if root.exists():
        raise FileExistsError(root)
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(repository), 'diff', 'HEAD', '--', 'src/heretic'], text=True)
    # An uncommitted implementation is not mislabeled as the upstream commit alone.
    config = model.peft_config.to_dict()
    common = {'schema_version': 1, 'synthetic': metadata.get('synthetic', False), 'run_id': metadata['run_id'],
              'heretic_git_sha': commit, 'lab_git_sha': metadata['lab_git_sha'],
              'heretic_source_diff_sha256': hashlib.sha256(changes.encode()).hexdigest(),
              'research_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'ara_source_sha256': hashlib.sha256((Path(__file__).parent / 'modifiers/ara.py').read_bytes()).hexdigest(),
              'model_id': ctx.settings.model, 'model_revision': revision,
              'tokenizer_revision': metadata['tokenizer_revision'],
              'chat_template_sha256': hashlib.sha256(model.tokenizer.chat_template.encode()).hexdigest(),
              'dataset_hashes': modifier._research_dataset_hashes,
              'dataset_hash_definition': 'sha256 of UTF-8 json.dumps ordered Prompt records, sort_keys=True, ensure_ascii=False',
              'seed': ctx.settings.seed,
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'transformers': importlib.metadata.version('transformers'),
                           'peft': importlib.metadata.version('peft')},
              'dtype': str(model.dtype).removeprefix('torch.'), 'device': str(model.model.device),
              'quantization': {'method': 'none'},
              'lora_settings': {'enabled': True, 'rank': config['r'], 'alpha': config['lora_alpha'],
                                'dropout': config['lora_dropout'], 'target_modules': sorted(config['target_modules']),
                                'peft_config': json.loads(json.dumps(config, default=lambda value: sorted(value) if isinstance(value, set) else str(value)))},
              'creation_time_utc': utc_now(), 'files': {}}
    if config.get('use_dora') or config.get('use_rslora') or config.get('bias') != 'none':
        raise ValueError('Unsupported research LoRA settings')
    record = {'metadata': dict(common, trial_id=f't{trial[0]:06d}',
                              ara_settings=settings.model_dump(mode='json'), ara_parameters=parameters.to_dict()),
              'directory': root / 'trials' / f't{trial[0]:06d}', 'model': model.model,
              'model_config': model.model.config.to_dict(), 'base_hash': base_fingerprint(model.model),
              'adapter_names': set(adapter_state(model.model)), 'intermediate_module': settings.research_intermediate_module,
              'sequence': 0, 'boundary': {'kind': 'trial_start', 'module_id': None,
                                        'outer_step': 0, 'completed_modules': []}}
    record['directory'].mkdir(parents=True)
    write_json(root / 'manifest.json', dict(common, artifact_type='run'))
    try:
        record_event(record, 'trial_start')
        save_snapshot(record, 's000', 'base')
        yield record
        save_snapshot(record, 'final', 'final')
        record_event(record, 'trial_complete')
        close_trial(record, 'complete')
    except BaseException as error:
        with suppress(Exception):
            record_event(record, 'failure', error_type=type(error).__name__)
            close_trial(record, 'failed')
        raise


def prompt_hashes(good_prompts, bad_prompts):
    return {name: hashlib.sha256(json.dumps([asdict(prompt) for prompt in prompts],
                                           sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            for name, prompts in (('edit_train_good', good_prompts), ('edit_train_bad', bad_prompts))}



def save_replay_reference(snapshot_directory, reference_directory, model, input_ids, attention_mask=None):
    """Explicit inference reference capture; never called from ARA or an optimizer closure."""
    from safetensors.torch import load_file, save_file

    if model.training:
        raise ValueError('Capture references in eval mode')
    snapshot = Path(snapshot_directory)
    target = Path(reference_directory)
    if target.exists():
        raise FileExistsError(target)
    data = json.loads((snapshot / 'manifest.json').read_text())
    if data.get('attention_implementation') != model.config._attn_implementation:
        raise ValueError('Reference attention implementation mismatch')
    descriptor = data['files']['adapter.safetensors']
    if file_record(snapshot / 'adapter.safetensors') != descriptor:
        raise ValueError('Snapshot adapter hash mismatch')
    saved = load_file(str(snapshot / 'adapter.safetensors'))
    current = adapter_state(model)
    if set(saved) != set(current) or any(not torch.equal(saved[n], current[n]) for n in saved):
        raise ValueError('Live adapters differ from the referenced snapshot')
    if base_fingerprint(model) != data['base_state_sha256']:
        raise ValueError('Live base differs from the referenced snapshot')
    if input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Expected one nonempty, unpadded int64 token sequence')
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    if mask.shape != input_ids.shape or not torch.all(mask == 1):
        raise ValueError('Only unpadded references are supported')
    with torch.inference_mode():
        logits = model(input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device),
                       use_cache=False).logits[0, -1].detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite reference logits')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '-', dir=target.parent))
    try:
        save_file({'input_ids': input_ids.detach().cpu().contiguous(),
                   'attention_mask': mask.detach().cpu().contiguous(), 'logits': logits.contiguous()},
                  str(staging / 'reference.safetensors'))
        with (staging / 'reference.safetensors').open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'schema_version': 1, 'artifact_type': 'replay_reference',
                    'synthetic': data['synthetic'], 'run_id': data['run_id'], 'trial_id': data['trial_id'],
                    'snapshot_id': data['snapshot_id'], 'creation_time_utc': utc_now(),
                    'snapshot_manifest_sha256': hashlib.sha256((snapshot / 'manifest.json').read_bytes()).hexdigest(),
                    'output_definition': 'last_token_logits_unpadded_use_cache_false',
                    'files': {'reference.safetensors': file_record(staging / 'reference.safetensors')}}
        write_json(staging / 'manifest.json', manifest)
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target
