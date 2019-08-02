# Quantum systems

[![Build Status](https://travis-ci.com/Schoyen/quantum-systems.svg?token=MvgH7xLNL8iVfczJpp8Q&branch=master)](https://travis-ci.com/Schoyen/quantum-systems)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

See the [documentation](https://schoyen.github.io/quantum-systems/) for usage.

## Installation
This project can be installed by running:
```bash
# From Github
pip install git+https://github.com/Schoyen/quantum-systems.git
# From repo quantum-systems
pip install .
```
To update the current version of `quantum-systems` run:
```bash
# From Github
pip install git+https://github.com/Schoyen/quantum-systems.git -U
# From repo quantum-systems
pip install . -U
```

### Environment
During development it is a good idea to create a _conda environment_ such that all dependencies gets installed correctly. This is easiest done by executing:

```bash
conda env create -f environment.yml
source activate quantum-systems
```

Once you are done, you can deactivate the environment by:

```bash
source deactivate
```

If the environment changes you can update the new changes by:

```bash
conda env update -f environment.yml
```
