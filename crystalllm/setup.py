# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Building using setuptools with NCCL compilation."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Command, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.dist import Distribution


def _get_nccl_version():
    r"""Parse NCCL version from vendor/nccl/makefiles/version.mk.

    Returns:
        str: NCCL version string in 5-digit format "major_minor_patch" (e.g., "21203")
    """
    version_file = Path("vendor/nccl/makefiles/version.mk")
    if not version_file.exists():
        return "unknown"

    version = {"major": 0, "minor": 0, "patch": 0}
    with Path(version_file).open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("NCCL_MAJOR"):
                match = re.search(r":=\s*(\d+)", line)
                if match:
                    version["major"] = int(match.group(1))
            elif line.startswith("NCCL_MINOR"):
                match = re.search(r":=\s*(\d+)", line)
                if match:
                    version["minor"] = int(match.group(1))
            elif line.startswith("NCCL_PATCH"):
                match = re.search(r":=\s*(\d+)", line)
                if match:
                    version["patch"] = int(match.group(1))

    return f"{version['major']}{version['minor']:02d}{version['patch']:02d}"


def _get_sm_version():
    r"""Get SM version from environment variable or CUDA device capability.

    Returns:
        str: SM version string (e.g., "80", "90")
    """
    sm_ver = os.environ.get("CRYSTALLLM_SM_VERSION")
    if sm_ver:
        return sm_ver

    try:
        cap = torch.cuda.get_device_capability(0)
        return f"{cap[0]}{cap[1]}"
    except Exception:
        return "unknown"


def _run_nccl_build_subprocess(name, cmd, cwd, env):
    r"""Run a subprocess with real-time output streaming.

    Args:
        name: Build configuration name for logging.
        cmd: Command list to execute.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.

    Returns:
        bool: True if the subprocess completed successfully (return code 0).
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output in real-time
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()

    # Wait for process to complete and get return code
    return_code = process.wait()

    print(f"{'-' * 60}")
    if return_code != 0:
        print(f"NCCL {name} build failed with return code: {return_code}.")
        print(f"{'=' * 60}\n")
        sys.stdout.flush()
        return False

    print(f"NCCL {name} build completed successfully.")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()
    return True


def _copy_nccl_build_output(name, builddir, dest_base_dir):
    r"""Copy NCCL build output to destination directory.

    Copies libnccl.so to the destination directory, resolving symlinks
    to copy the actual file content.

    Args:
        name: Build configuration name for logging.
        builddir: Build output directory containing lib/libnccl.so.
        dest_base_dir: Base destination directory for copied libraries.

    Returns:
        bool: True if the copy completed successfully.
    """
    destdir = dest_base_dir / name
    destdir.mkdir(parents=True, exist_ok=True)

    lib_source = builddir / "lib" / "libnccl.so"
    lib_dest = destdir / "libnccl.so"

    if not lib_source.exists():
        print(f"Error: Expected library not found at {lib_source}")
        return False

    # Resolve symlink to get the real file path
    real_lib_path = lib_source.resolve()
    print(f"Copying {real_lib_path} to {lib_dest}")

    # Copy the actual file content (not the symlink)
    shutil.copy2(real_lib_path, lib_dest)
    print(f"Successfully copied {name}")
    sys.stdout.flush()
    return True


def _build_nccl():
    r"""Build NCCL with three different configurations.

    Builds three NCCL versions in order:
    - baseBuild: No EMU_FAKE or EMU_MOCK
    - fakeBuild: EMU_FAKE=1
    - mockBuild: EMU_FAKE=1 and EMU_MOCK=1

    The build is performed in vendor/nccl/build/{name} directories, then the resulting
    libraries are copied to src/crystalllm/nccl/lib/ with their real content
    (symlinks are resolved).
    """
    # NCCL source directory
    nccl_src_dir = Path("vendor/nccl")
    # Build output base directory (in project root build directory)
    build_base_dir = Path("vendor/nccl/build")
    # Destination directory for .so files (in src)
    dest_base_dir = Path("src/crystalllm/nccl/lib")

    # Ensure destination directory exists
    dest_base_dir.mkdir(parents=True, exist_ok=True)

    # Common build parameters
    nvcc_gencode = f"-gencode=arch=compute_{_get_sm_version()},code=sm_{_get_sm_version()}"
    make_cmd = "make"

    # Build configurations
    build_configs = [
        {
            "name": "baseBuild",
            "env": {},
        },
        {
            "name": "fakeBuild",
            "env": {"EMU_FAKE": "1"},
        },
        {
            "name": "mockBuild",
            "env": {"EMU_FAKE": "1", "EMU_MOCK": "1"},
        },
    ]

    def run_build(config):
        r"""Run a single NCCL build configuration with detailed output.

        After successful build, copies libnccl.so to destination directory,
        resolving symlinks to copy the actual file content.
        """
        name = config["name"]
        builddir = build_base_dir / name

        # Prepare environment variables
        env = os.environ.copy()
        env.update(config["env"])

        # Prepare make command
        jobs = os.environ.get("CRYSTALLLM_BUILD_JOBS")
        if jobs is not None and (not jobs.isdigit() or int(jobs) < 1):
            raise ValueError("CRYSTALLLM_BUILD_JOBS must be a positive integer")
        cmd = [
            make_cmd,
            f"-j{jobs}" if jobs is not None else "-j",
            "src.build",
            f'NVCC_GENCODE="{nvcc_gencode}"',
            f"BUILDDIR={builddir.absolute()}",
        ]

        print(f"\n{'=' * 60}")
        print(f"Starting NCCL {name} build...")
        print(f"{'=' * 60}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {nccl_src_dir}")
        print(f"Build directory: {builddir.absolute()}")
        print(f"Extra environment variables: {config['env']}")
        print(f"{'-' * 60}")
        sys.stdout.flush()

        try:
            success = _run_nccl_build_subprocess(name, cmd, nccl_src_dir, env)
        except Exception as e:
            print(f"NCCL {name} build failed with exception: {e}")
            print(f"{'=' * 60}\n")
            sys.stdout.flush()
            return False

        if not success:
            return False

        return _copy_nccl_build_output(name, builddir, dest_base_dir)

    # Run builds sequentially
    print("\n" + "#" * 60)
    print("# Starting NCCL builds")
    print("#" * 60)
    sys.stdout.flush()

    results = []
    for config in build_configs:
        result = run_build(config)
        results.append(result)
        if not result:
            # Stop on first failure
            break

    print("\n" + "#" * 60)
    print("# NCCL Build Summary")
    print("#" * 60)
    for config, result in zip(build_configs, results, strict=True):
        status = "SUCCESS" if result else "FAILED"
        print(f"# {config['name']}: {status}")
    print("#" * 60)
    sys.stdout.flush()

    # Check if all builds succeeded
    if not all(results):
        print("\nOne or more NCCL builds failed!")
        sys.exit(1)

    print("\nAll NCCL builds completed successfully.")


class CustomBuildPy(_build_py):
    r"""Custom build_py command that builds NCCL first."""

    def run(self):
        r"""Run the build process."""
        _build_nccl()
        super().run()


class CustomDevelop(_develop):
    r"""Custom develop command that builds NCCL first."""

    def run(self):
        r"""Run the develop process."""
        _build_nccl()
        super().run()


class CleanNCCL(Command):
    r"""Clean NCCL build artifacts command."""

    description = "Clean NCCL build artifacts"
    user_options = []

    def initialize_options(self):
        r"""Initialize options for the command."""

    def finalize_options(self):
        r"""Finalize options for the command."""

    def run(self):  # noqa: PLR6301
        r"""Run the clean NCCL command."""
        print("\n" + "#" * 60)
        print("# Cleaning NCCL build artifacts")
        print("#" * 60)

        # Clean vendor/nccl/build/ directory (build output)
        nccl_build_dir = Path("vendor/nccl/build")
        if nccl_build_dir.exists():
            print(f"Removing directory: {nccl_build_dir}")
            shutil.rmtree(nccl_build_dir)
            print(f"Successfully removed {nccl_build_dir}")
        else:
            print(f"Directory not found, skipping: {nccl_build_dir}")

        # Clean src/crystalllm/nccl/lib/ directory (copied .so files)
        nccl_lib_dir = Path("src/crystalllm/nccl/lib")
        if nccl_lib_dir.exists():
            print(f"Removing directory: {nccl_lib_dir}")
            shutil.rmtree(nccl_lib_dir)
            print(f"Successfully removed {nccl_lib_dir}")
        else:
            print(f"Directory not found, skipping: {nccl_lib_dir}")

        print("#" * 60)
        print("# NCCL clean completed")
        print("#" * 60 + "\n")


class CustomBdistWheel(_bdist_wheel):
    r"""Custom bdist_wheel command with architecture, NCCL version and SM info in filename."""

    def run(self):
        r"""Run bdist_wheel and set custom wheel filename."""
        super().run()

        # Get build info
        nccl_ver = _get_nccl_version()
        sm_ver = _get_sm_version()

        # Build platform tag with custom info (Format: ncclX.Y.Z.smXX)
        build_tag = f"nccl{nccl_ver}.sm{sm_ver}"

        # Set build number to include our custom info
        # This will be part of the wheel filename
        old_path = next(Path(self.dist_dir).glob(f"{self.wheel_dist_name}-*.whl"), None)
        if not old_path:
            return

        new_path = old_path.with_name(
            old_path.name.replace(
                f"-{self.distribution.get_version()}-", f"-{self.distribution.get_version()}+{build_tag}-"
            )
        )
        if old_path == new_path:
            return

        old_path.rename(new_path)
        old_path_str = str(old_path)
        new_path_str = str(new_path)

        for i, dist_file in enumerate(self.distribution.dist_files):
            if dist_file[2] == old_path_str:
                self.distribution.dist_files[i] = (dist_file[0], dist_file[1], new_path_str)
                break


class BinaryDistribution(Distribution):
    """Binary distribution."""

    def has_ext_modules(self):  # noqa: PLR6301
        """Whether has ext modules."""
        return True


setup(
    package_data={
        "crystalllm.nccl": [
            "lib/*/libnccl.so",
        ],
    },
    scripts=[
        "scripts/crystalllm",
        "scripts/crystalllm-init",
        "scripts/crystalllm-numarun",
        "scripts/crystalllm-pytorch",
        "scripts/crystalllm-torchrun",
    ],
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        "build_py": CustomBuildPy,
        "bdist_wheel": CustomBdistWheel,
        "develop": CustomDevelop,
        "clean_nccl": CleanNCCL,
    },
    distclass=BinaryDistribution,
)
