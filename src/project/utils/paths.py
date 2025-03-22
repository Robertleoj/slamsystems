from pathlib import Path


def repo_root() -> Path:
    file_path = Path(__file__).resolve()

    root = file_path
    while not (root / ".git").exists():
        root = root.parent

    return root


def symforce_codegen_path():
    return repo_root() / "cpp/include/foundation/symforce_generated"


def symforce_general_factor_path():
    return symforce_codegen_path() / "general_factors"
