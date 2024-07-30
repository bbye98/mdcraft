# MDCraft

<img src="https://raw.githubusercontent.com/bbye98/mdcraft/main/assets/logo.png"
 align="right" width="256"/>

[![continuous-integration](
https://github.com/bbye98/mdcraft/actions/workflows/ci.yml/badge.svg)](
https://github.com/bbye98/mdcraft/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mdcraft/badge/?version=latest)](
https://mdcraft.readthedocs.io/en/latest/?badge=latest)

MDCraft is a toolkit of analysis modules and helper functions for
molecular dynamics (MD) simulations.

* **Documentation**: https://mdcraft.readthedocs.io/
* **Conda**: https://anaconda.org/bbye98/mdcraft
* **Python Package Index (PyPI)**: https://pypi.org/project/mdcraft/

## Features

* [`algorithm`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/algorithm):
Efficient Numba, NumPy, and SciPy algorithms for data wrangling and
evaluating structural and dynamical properties.
* [`analysis`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/analysis):
Serial and parallel data analysis tools built on top of the MDAnalysis
framework.
* [`fit`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/fit):
Two-dimensional curve fitting models for use with SciPy.
* [`lammps`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/lammps):
Helper functions for setting up LAMMPS simulations.
* [`openmm`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/openmm):
Extensions to the high-performance OpenMM toolkit, such as custom
bond/pair potentials, support for NetCDF trajectories, and much more.
* [`plot`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/plot):
Settings and additional functionality for Matplotlib figures.

### Benchmarks

The novel forcefield provided by MDCraft, Gaussian Core Model with smeared electrostatics (GCMe), provides multiple benefits discussed in more detail within our recent [publication](https://doi.org/10.1021/acs.jctc.4c00603). Of note is the computational speed-up obtained from GCMe, especially when used in OpenMM:

![benchmarks](/assets/benchmarks.png)

The codes used to generate these benchmarks are provided in the associated [repository](https://github.com/bbye98/gcme).
## Getting started

### Prerequisites

If you use pip to manage your Python packages and plan on using the
OpenMM simulation toolkit, you must compile and install OpenMM manually
since OpenMM is not available in PyPI. See the
["Compiling OpenMM from Source Code"](
http://docs.openmm.org/latest/userguide/library/02_compiling.html)
section of the OpenMM User Guide for more information.

If you use Conda, it is recommended that you use the conda-forge
channel to install dependencies. To make conda-forge the default
channel, use

    conda config --add channels conda-forge

### Installation

MDCraft requires Python 3.9 or later.

For the most up-to-date version of MDCraft, clone the repository and
install the package using pip:

    git clone https://github.com/bbye98/mdcraft.git
    cd mdcraft
    python3 -m pip install -e .

Alternatively, MDCraft is available on Conda:

    conda install bbye98::mdcraft

and PyPI:

    python3 -m pip install mdcraft

### Postrequisites

To use the method of image charges
(`mdcraft.openmm.system.add_image_charges()`) in your OpenMM
simulations, you must compile and install [`constvplugin`](
https://github.com/scychon/openmm_constV) or [`openmm-ic-plugin`](
https://github.com/bbye98/mdcraft/tree/main/lib/openmm-ic-plugin).

### Tests

After installing, to run the MDCraft tests locally, use `pytest`:

    pip install pytest
    cd mdcraft
    pytest
