from setuptools import setup, find_packages

setup(
    name="quantum-systems",
    version="0.2.6",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "numba",
        "pyscf",
        "sympy",
    ],
)
