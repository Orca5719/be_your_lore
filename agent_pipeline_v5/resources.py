"""Disk gates only; never installs dependencies or cleans files."""
from pathlib import Path
import shutil
import tempfile


class ResourceLimit(RuntimeError):
    pass


def _existing(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def check_resources(output: Path, temp_dir: Path | None = None, *, initial=False,
                    report_budget=512 * 1024**2) -> dict:
    output = Path(output).resolve()
    paths = [output, Path(temp_dir or tempfile.gettempdir()).resolve()]
    threshold = (5 if initial else 2) * 1024**3
    free = {str(p): shutil.disk_usage(_existing(p)).free for p in paths}
    if any(size < threshold for size in free.values()):
        raise ResourceLimit(f'Less than {threshold / 1024**3:g} GiB free on output/TEMP volume; no space is reserved')
    total = 0
    if output.exists():
        for p in output.rglob('*'):
            if p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()):
                raise ResourceLimit('report directory contains a filesystem link')
            if p.is_file():
                total += p.stat().st_size
    if total >= report_budget:
        raise ResourceLimit('report budget reached; checkpoint preserved')
    return {'free_bytes': free, 'report_bytes': total, 'report_budget': report_budget}
