from __future__ import annotations

from pathlib import Path


def format_tilde_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    home = Path.home().resolve(strict=False)
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    if relative == Path("."):
        return "~"
    return str(Path("~") / relative)
