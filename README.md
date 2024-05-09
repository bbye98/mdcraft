<img src="https://raw.githubusercontent.com/bbye98/mdcraft/main/assets/logo.png" align="right" width="256"/>

# MDCraft

[![continuous-integration](https://github.com/bbye98/mdcraft/actions/workflows/ci.yml/badge.svg)](https://github.com/bbye98/mdcraft/actions/workflows/ci.yml)

MDCraft is a toolkit of analysis modules and helper functions for
molecular dynamics (MD) simulations.

* **Documentation**: https://bbye98.github.io/mdcraft/
* **Conda**: https://anaconda.org/bbye98/mdcraft
* **Python Package Index (PyPI)**: https://pypi.org/project/mdcraft/

Note that MDCraft is currently an *experimental* library that has
only been tested on Linux and may contain bugs and issues. If you come
across one or would like to request new features, please
[submit a new issue](https://github.com/bbye98/mdcraft/issues/new).

## Features

* [`algorithm`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/algorithm):
Efficient NumPy and SciPy algorithms for data wrangling and evaluating
structural and dynamical properties.
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

## Installation

MDCraft requires Python 3.9 or later.

For the most up-to-date version of MDCraft, clone the repository and
install the package using pip:

    git clone https://github.com/bbye98/mdcraft.git
    cd mdcraft
    python -m pip install -e .

MDCraft is also available on Conda and PyPI:

    conda install -c bbye98 mdcraft
    python -m pip install mdcraft

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

### Postrequisites

To use the image of method charges
(`mdcraft.openmm.system.add_image_charges()`) in your OpenMM
simulations, you must compile and install [`constvplugin`](
https://github.com/scychon/openmm_constV) or [`openmm-ic-plugin`](
https://github.com/bbye98/mdcraft/tree/main/lib/openmm-ic-plugin).