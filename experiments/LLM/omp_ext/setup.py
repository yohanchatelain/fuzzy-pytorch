"""Build the omp_ext extension.

PRISM ships two dispatch builds, libprism-static.so and libprism-dynamic.so.
omp_ext writes PRISM's thread-local state directly, so it has to bind to the
same one libtorch_cpu.so links against; binding to the other yields a distinct
TLS block and precision changes are silently ignored.  The library is therefore
resolved from libtorch_cpu.so rather than hardcoded.
"""

import os
import re
import subprocess
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

PRISM_LIB_DIR = "/usr/local/lib"


def prism_library():
    """Return the PRISM library name libtorch_cpu.so is linked against."""
    libtorch = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so")
    if not os.path.exists(libtorch):
        sys.exit(f"omp_ext: cannot find {libtorch}")

    try:
        dynamic_section = subprocess.run(
            ["readelf", "-d", libtorch], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"omp_ext: could not read the dynamic section of {libtorch}: {exc}")

    match = re.search(r"libprism-(static|dynamic)\.so", dynamic_section)
    if match is None:
        sys.exit(
            f"omp_ext: {libtorch} is not linked against PRISM. This PyTorch build "
            "is not Verificarlo-instrumented with the PRISM backend, so per-module "
            "precision scoping cannot work."
        )
    return f"prism-{match.group(1)}"


setup(
    name="omp_ext",
    version="0.1.0",
    description="Broadcast PRISM virtual precision and rounding mode across OpenMP threads",
    ext_modules=[
        CppExtension(
            name="omp_ext",
            sources=["omp_ext.cpp"],
            libraries=[prism_library()],
            library_dirs=[PRISM_LIB_DIR],
            runtime_library_dirs=[PRISM_LIB_DIR],
            extra_compile_args=["-O2", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
