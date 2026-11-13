from pathlib import Path
import json
import hashlib

TEXT_SUFFIXES = {'.json', '.txt', '.jsonl', '.md', '.csv', '.py'}

def sha(path):
    # Normalize CRLF for text files so checksums survive Windows/Linux checkouts.
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(str(k)): sanitize(v) for k, v in value.items() if k not in ('authorized_attempt_id', 'ple_training_provenance', 'external_training_window')}
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, str) and (value.startswith('/') or ':\\' in value):
        return Path(value.replace('\\', '/')).name
    return value

def manifest(model_dir):
    write(model_dir / 'manifest.json', {'format_version': 1, 'files_sha256': {path.relative_to(model_dir).as_posix(): sha(path) for path in sorted(model_dir.rglob('*')) if path.is_file() and path.name != 'manifest.json'}})
