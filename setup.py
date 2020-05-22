from setuptools import setup, find_packages

setup(
    name="quantum-systems",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "pandas", "numba", "pyscf", "sympy",],
)
