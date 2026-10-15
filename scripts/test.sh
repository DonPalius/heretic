#!/usr/bin/env bash
# Preparazione riproducibile di PR-04 dal fork Heretic.
# Uso: bash scripts/pr04_prepare_and_record.sh [--run] [--download]
# Senza --run: prepara i file e invoca soltanto --check.
set -euo pipefail

run=0
download=0
for arg in "$@"; do
    case "$arg" in
        --run) run=1 ;;
        --download) download=1 ;;
        -h|--help)
            echo 'Uso: bash scripts/pr04_prepare_and_record.sh [--run] [--download]'
            exit 0 ;;
        *) echo "Argomento sconosciuto: $arg" >&2; exit 2 ;;
    esac
done
if (( download && ! run )); then
    echo '--download ha senso solo insieme a --run.' >&2
    exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
lab="$root/../Research-Dream-Team"
input="$root/../experiments/pr04_inputs.safetensors"
output="$root/../experiments/run_001"
study="$root/checkpoints/pr04_run_001"
revision='7ae557604adf67be50417f59c2c2f167def9a775'
expected_input_sha='9b4b1b816ff4cae5352fe75edc084f00e7768676273fad50069b02db9fe8ab23'

die() { echo "ERRORE: $*" >&2; exit 1; }
[[ -f pyproject.toml && -f config.default.toml && -f scripts/pr04_03_record.sh ]] || die 'Esegui questo script dal fork Heretic (in scripts/).'
[[ -d "$lab/.git" || -f "$lab/.git" ]] || die "Repository Guardrails Lab non trovato: $lab"
[[ -s "$input" ]] || die "Input audit non trovato: $input"
[[ ! -e "$output" && ! -L "$output" ]] || die "Output già esistente: $output. Non sovrascriverlo."
[[ ! -e "$study" && ! -L "$study" ]] || die "Studio già esistente: $study. Non riprendere uno studio precedente."

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
    command -v uv >/dev/null || die 'uv non disponibile per preparare Python 3.11.'
    env -u VIRTUAL_ENV uv sync --python 3.11
fi

actual_sha="$(sha256sum "$input" | cut -d' ' -f1)"
[[ "$actual_sha" == "$expected_input_sha" ]] || die "SHA input diverso da quello verificato in PR-04-02: $actual_sha"
lab_sha="$(git -C "$lab" rev-parse HEAD)"

if [[ ! -f config.toml ]]; then
    cp -- config.default.toml config.toml
    echo 'Creato config.toml da config.default.toml.'
fi
if [[ ! -e config.toml.pr04.bak ]]; then
    cp -p -- config.toml config.toml.pr04.bak
fi

# Modifica solo i tre parametri top-level e aggiunge le impostazioni ARA.
# In caso di impostazioni di registrazione preesistenti e differenti, si ferma.
.venv/bin/python - config.toml <<'PY'
import json
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding='utf-8')
old = tomllib.loads(src)
head, sep, tail = src.partition('\n[')
if not sep:
    raise SystemExit('config.toml: nessuna sezione TOML trovata')
for key, value in (
    ('n_trials', '1'),
    ('n_startup_trials', '1'),
    ('study_checkpoint_dir', '"checkpoints/pr04_run_001"'),
):
    pattern = rf'(?m)^{re.escape(key)}\s*=\s*[^\n]*$'
    head, count = re.subn(pattern, f'{key} = {value}', head)
    if count != 1:
        raise SystemExit(f'Attesa esattamente una chiave top-level {key}, trovate {count}')
src = head + sep + tail

expected = {
    'research_output': '../experiments/run_001',
    'research_metadata': 'research-metadata.json',
    'research_trial': 0,
    'research_intermediate_module': 1,
    'research_reference_inputs': '../experiments/pr04_inputs.safetensors',
    'research_audit_modules': [
        'model.layers.12.self_attn.o_proj',
        'model.layers.12.mlp.down_proj',
    ],
}
ara = old.get('modifier', {}).get('ARA', {})
for key, value in expected.items():
    if key in ara and ara[key] != value:
        raise SystemExit(f'{key} è già impostato diversamente: {ara[key]!r}. Modifica manualmente dopo aver verificato il trial.')

m = re.search(r'(?m)^\[modifier\.ARA\]\s*$', src)
if not m:
    raise SystemExit('Manca la sezione [modifier.ARA]')
next_section = re.search(r'(?m)^\[', src[m.end():])
end = m.end() + next_section.start() if next_section else len(src)
missing = [(k, v) for k, v in expected.items() if k not in ara]
if missing:
    inserted = '\n' + '\n'.join(f'{k} = {json.dumps(v, ensure_ascii=False)}' for k, v in missing) + '\n'
    src = src[:end].rstrip() + '\n' + inserted + '\n' + src[end:]

parsed = tomllib.loads(src)
assert parsed['n_trials'] == parsed['n_startup_trials'] == 1
assert all(parsed['modifier']['ARA'][k] == v for k, v in expected.items())
if src != p.read_text(encoding='utf-8'):
    fd, name = tempfile.mkstemp(prefix='.config-pr04-', dir=str(p.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(src)
        os.chmod(name, p.stat().st_mode & 0o777)
        os.replace(name, p)
    finally:
        if os.path.exists(name):
            os.unlink(name)
print('config.toml: impostazioni PR-04 applicate e TOML verificato.')
PY

.venv/bin/python - research-metadata.json "$lab_sha" "$revision" "$actual_sha" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = {
    'run_id': 'run_001',
    'lab_git_sha': sys.argv[2],
    'tokenizer_revision': sys.argv[3],
    'audit_inputs_sha256': sys.argv[4],
}
if p.exists():
    if json.loads(p.read_text(encoding='utf-8')) != data:
        raise SystemExit('research-metadata.json esiste con valori diversi: non sovrascritto')
else:
    p.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print('research-metadata.json:', json.dumps(data))
PY

if [[ -n "$(git -C "$lab" status --porcelain)" ]]; then
    echo 'ATTENZIONE: Guardrails Lab ha modifiche non committate; il suo HEAD non descrive tutto il codice.' >&2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'ATTENZIONE: Heretic ha modifiche non committate; verifica la provenienza prima del trial.' >&2
fi

echo 'Controllo nativo del recorder (nessuna ottimizzazione):'
bash scripts/pr04_03_record.sh --check

if (( run )); then
    echo 'Avvio del trial: assicurati che modello, model_commit, seed, dataset e moduli ARA siano quelli approvati.'
    args=()
    if (( download )); then args+=(--download); fi
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash scripts/pr04_03_record.sh "${args[@]}"
fi

echo 'Preparazione terminata. Verifica modello/commit/seed/dataset e poi usa --run (eventualmente --download).'
