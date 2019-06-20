Getting started
===============

Installation
------------

This module can be installed from github with ``pip`` by running::

    pip install git+https://github.com/Schoyen/quantum-systems.git

Alternatively, the same task can be accomplished using three commands::

    git clone https://github.com/Schoeyn/quantum-system.git
    cd quantum-systems
    pip install .

In order to update to the latest version use::

    pip install -U git+https://github.com/Schoyen/quantum-systems.git

or, whilst inside the cloned repo::

    pip install -U .

Conda Environment
-----------------

It can be useful to setup a conda environment for using this package. We have
included an environment specification file for this purpose::

    conda environment create -f environment.yml
    conda activate quantum-systems

Deactivating the ``conda`` environment is done with::

    conda deactivate

The environment can be updated with::

    conda env update -f environment.yml
