from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import os
import glob


os.environ["CFLAGS"] = "-std=c++11"


base_path = ["quantum_systems"]

source_files = {
    "system_helper": [os.path.join(*base_path, "system_helper.pyx")],
}

include_dirs = {
    "system_helper": [],
}

for key in include_dirs:
    include_dirs[key] += [np.get_include()]

extensions = [
    Extension(
        name="quantum_systems.system_helper",
        sources=source_files["system_helper"],
        language="c++",
        include_dirs=include_dirs["system_helper"],
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
