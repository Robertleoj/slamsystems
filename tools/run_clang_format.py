#!/usr/bin/env python3

"""A tool to run clang-format on our source code.

Note that this is assuming this is run from the repo root.
"""

import subprocess
from itertools import chain
from pathlib import Path

repo_paths = ["cpp"]


def run_clang_format() -> None:
    """Run Clang format on our CPP files in the repository."""
    assert Path(".git").exists, "This command should run in repo root."

    all_paths = []
    for name in repo_paths:
        source_paths = Path(name).rglob("*.cpp")
        header_paths = Path(name).rglob("*.hpp")
        all_paths.extend([str(el) for el in chain(source_paths, header_paths)])

    subprocess.run(["clang-format", "-i"] + all_paths)

    print(f"Formatted {len(all_paths)} files")


if __name__ == "__main__":
    run_clang_format()
