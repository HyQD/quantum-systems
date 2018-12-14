from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import os
import glob


os.environ["CFLAGS"] = "-std=c++11"


base_path = ["quantum_systems"]

quantum_dot_path = base_path + ["quantum_dots"]

two_dim_qd_path = quantum_dot_path + ["two_dim"]
two_dim_qd_source_path = two_dim_qd_path + ["src"]

source_files = {
    "two_dim_interface": [
        *glob.glob(os.path.join(*two_dim_qd_path, "*.pyx")),
        *glob.glob(os.path.join(*two_dim_qd_source_path, "*.cpp")),
    ],
}

include_dirs = {
    "two_dim_interface": [os.path.join(*two_dim_qd_source_path)],
}

for key in include_dirs:
    include_dirs[key] += [np.get_include()]

extensions = [
    Extension(
        name="quantum_systems.quantum_dots.two_dim.two_dim_interface",
        sources=source_files["two_dim_interface"],
        language="c++",
        include_dirs=include_dirs["two_dim_interface"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="Quantum systems",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)
