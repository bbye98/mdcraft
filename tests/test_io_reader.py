import pathlib
import sys

import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.io.reader import LAMMPSDumpReader  # noqa: E402

DATA_DIRECTORY = pathlib.Path(__file__).parents[0].resolve() / "data"


def test_class_LAMMPSDumpReader():

    pass

    # Empty trajectory

    # Basic properties of trajectory with constant time step
    # and number of atoms
    # reader = LAMMPSDumpReader(DATA_DIRECTORY / "trajectories/subset_1_1.lammpstrj")

    # Basic properties of trajectory with variable time step
    # and number of atoms

    # Trajectory with unit header

    # Trajectory with time header

    # Trajectory with unit and time headers

    # Trajectory with extra attributes


# test_class_LAMMPSDumpReader()
