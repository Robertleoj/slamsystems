import os
import shutil
import subprocess
from pathlib import Path

from fire import Fire

INSTALL_DIR = Path("cpp_installs")
BUILD_DIR = Path(".dep_build")
EXTERNAL_DIR = Path("external")
VCPKG_INSTALLED_DIR = Path("vcpkg_installed")


def build_symforce():
    build_dir = BUILD_DIR / "symforce"
    source_dir = EXTERNAL_DIR / "symforce"

    compile_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
        "-DSYMFORCE_BUILD_OPT=ON",
        "-DSYMFORCE_BUILD_CC_SYM=ON",  # 🔥 No Python bindings
        "-DSYMFORCE_ADD_PYTHON_TESTS=OFF",  # 🔥 No Python tests
        "-DSYMFORCE_BUILD_SYMENGINE=OFF",  # 🔥 No symenginepy junk
        "-DSYMFORCE_BUILD_EXAMPLES=OFF",
        "-DSYMFORCE_BUILD_TESTS=OFF",
        "-DSYMFORCE_GENERATE_MANIFEST=OFF",
        f"-DCMAKE_TOOLCHAIN_FILE={os.environ['VCPKG_ROOT']}/scripts/buildsystems/vcpkg.cmake",
        f"-DCMAKE_PREFIX_PATH={str(VCPKG_INSTALLED_DIR)}/x64-linux",
        "-DVCPKG_TARGET_TRIPLET=x64-linux",
        "-DSYMFORCE_USE_EXTERNAL_LCM=ON",
    ]

    subprocess.run(compile_cmd, check=True)

    build_cmd = ["cmake", "--build", str(build_dir)]

    subprocess.run(build_cmd, check=True)

    install_cmd = ["cmake", "--install", str(build_dir)]

    subprocess.run(install_cmd, check=True)


def build_slamdunk():
    build_dir = BUILD_DIR / "slam_dunk"
    source_dir = EXTERNAL_DIR / "slam_dunk/slamd"

    compile_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
        "-DSLAMD_ENABLE_INSTALL=ON",
        "-DSLAMD_VENDOR_DEPS=ON",
        "-DBUILD_SHARED_LIBS=OFF",
    ]

    subprocess.run(compile_cmd, check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)
    subprocess.run(["cmake", "--install", str(build_dir)], check=True)


def check_in_repo() -> None:
    """Check that we are executing this from repo root."""
    assert Path(".git").exists(), "This command should run in repo root."


def build() -> None:
    # build_slamdunk()
    build_symforce()


def clean() -> None:
    """Clean the build folder and remove the symlink, if any."""
    check_in_repo()
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.rmtree(INSTALL_DIR, ignore_errors=True)


def clean_build() -> None:
    """First clean and then build."""
    clean()
    build()


if __name__ == "__main__":
    Fire(
        {
            "build": build,
            "clean": clean,
            "clean_build": clean_build,
        }
    )
