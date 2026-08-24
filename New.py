#!/usr/bin/env python3
"""
Clean up common Python “garbage” files beneath a directory.
Usage: python clean.py [root_dir]
"""
import sys
from pathlib import Path
import shutil

PATTERNS = [
    "__pycache__",
    "*.py[cod]",
    "*.pyo",
    "*.egg-info",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    "*.log",
    "*.swp",
    "*~",
]

def matches(path: Path) -> bool:
    for pat in PATTERNS:
        if path.match(pat):
            return True
    return False

def clean(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_dir() and matches(p):
            print("rmdir:", p)
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file() and matches(p):
            print("unlink:", p)
            try:
                p.unlink()
            except OSError:
                pass

if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    clean(base)