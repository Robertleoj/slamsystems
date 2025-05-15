#!/usr/bin/env python3

"""A script to build the C++ backend and install the bindings into the source tree."""

import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

from fire import Fire

BUILD_DIR = "build"

MODULE_NAME = "foundation.cpython-311-x86_64-linux-gnu.so"


def check_in_repo() -> None:
    """Check that we are executing this from repo root."""
    assert Path(".git").exists(), "This command should run in repo root."


def build(debug: bool) -> None:
    """(Re)build the C++ backend."""
    check_in_repo()

    build_path = Path("build")
    build_path.mkdir(exist_ok=True)

    compile_cmd = [
        "cmake",
        "-B",
        str(build_path),
        "-G",
        "Ninja",
        # "-DCMAKE_C_COMPILER=/usr/bin/gcc-13",
        # "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13",
        '-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"',
        '-DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld"',
        '-DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"',
    ]

    if debug:
        compile_cmd.append("-DCMAKE_BUILD_TYPE=Debug")
    else:
        compile_cmd.append("-DCMAKE_BUILD_TYPE=Release")

    subprocess.run(
        compile_cmd,
        check=True,
    )

    subprocess.run(["ninja", "-C", str(build_path)])

    # Make sure that target was built
    target_path = build_path / "cpp" / "src" / MODULE_NAME
    assert target_path.exists()

    # Replace or create symlink
    deploy_path = Path("src/project") / MODULE_NAME
    if deploy_path.is_symlink():
        deploy_path.unlink()

    deploy_path.symlink_to(target_path.resolve())

    subprocess.run(
        ["pybind11-stubgen", "foundation", "-o", "src/project", "--numpy-array-remove-parameters"],
        env=dict(os.environ, PYTHONPATH="./src/project"),
    )

    subprocess.run(["ruff", "format", "src/project"])
    subprocess.run(["ruff", "check", "--fix-only", "src/project"])


def clean() -> None:
    """Clean the build folder and remove the symlink, if any."""
    check_in_repo()
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.rmtree("cpp_executables", ignore_errors=True)

    # Remove the symlink, if any
    deploy_path = Path(f"src/project/{MODULE_NAME}")
    if deploy_path.is_symlink():
        deploy_path.unlink()


def clean_build() -> None:
    """First clean and then build."""
    clean()
    build(False)


def debug_build():
    build(True)


def clean_debug_build():
    clean()
    build(True)


if __name__ == "__main__":
    Fire(
        {
            "build": partial(build, False),
            "clean": clean,
            "clean_build": clean_build,
            "debug_build": debug_build,
            "clean_debug_build": clean_debug_build,
        }
    )
