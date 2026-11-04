"""Resolve the monorepo and optionally an older sibling checkout layout."""
import os
from pathlib import Path

UNIFIED_ROOT = Path(__file__).resolve().parents[1]

def rdd_root():
    override = os.environ.get('UMDIFF_RDD_ROOT')
    candidates = [Path(override).expanduser()] if override else [
        UNIFIED_ROOT.parent, UNIFIED_ROOT.parent/'multiscale_diffusion']
    for candidate in candidates:
        if (candidate/'DiT/models.py').is_file() and (candidate/'diffusion/__init__.py').is_file():
            return candidate.resolve()
    raise FileNotFoundError('Cannot locate legacy RDD: use the full multiscale_diffusion checkout '
                            'with unified_multires inside it, or set UMDIFF_RDD_ROOT.')

def workspace_path(relative):
    """Resolve historical source-pin paths without requiring a repo basename."""
    path = Path(relative)
    if path.is_absolute():
        return path
    if '..' in path.parts:
        raise ValueError('Workspace references must not contain parent traversal')
    if path.parts and path.parts[0] == 'multiscale_diffusion':
        return rdd_root().joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == 'unified_multires':
        return UNIFIED_ROOT.joinpath(*path.parts[1:])
    return rdd_root().parent/path
