"""Resolve model archives and select inference hardware."""
from typing import Any
from pathlib import Path
import hashlib
import zipfile
import shutil


def resolve_weights(path: str | Path) -> Path:
    """Resolve weights, caching an archive's best.pt without unsafe extraction."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing weights: {path}")
    if path.suffix.lower() != '.zip':
        return path
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    target = Path('.model_cache') / digest.hexdigest() / 'best.pt'
    if not target.is_file():
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if n.endswith('best.pt') and not n.endswith('/'))
            if not members:
                raise ValueError(f"Zip contains no best.pt: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp')
            with archive.open(members[0]) as source, temporary.open('wb') as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(target)
    return target


def select_device(device: str) -> str:
    """Select CUDA, MPS, then CPU when automatic selection is requested."""
    if device != 'auto':
        return device
    import torch
    if torch.cuda.is_available():
        return 'cuda:0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(weights: str | Path) -> Any:
    """Load a YOLO pose model, including version information on failure."""
    resolved = resolve_weights(weights)
    import ultralytics
    try:
        return ultralytics.YOLO(str(resolved))
    except Exception as exc:
        raise ValueError(f"Model load failed (ultralytics {ultralytics.__version__}): {exc}") from exc
