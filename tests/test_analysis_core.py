import pathlib
import sys

import numpy as np

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft.core import Trajectory

DATA_DIRECTORY = pathlib.Path(__file__).parents[0].resolve() / "data"


class TestClassTrajectory:

    def test_1(self):
        """
        FILE:  subset_1_1.lammpstrj
        UNITS: lj

        INDEX:      012345
        READER 0:   nnnnnn
        TRAJECTORY: ffffff

        _start_frames   = [0]
        _offset_frames  = [0]
        _overlap_frames = {}

        dt        = 0.005
        time_step = 0.005
        n_frames  = 6
        times     = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
        timesteps = [0, 1, 2, 3, 4, 5]
        """

        trajectory = Trajectory(
            DATA_DIRECTORY / "trajectories/subset_1_1.lammpstrj"
        )
        assert trajectory.dt == 0.005
        assert trajectory.time_step == 0.005
        assert trajectory.n_frames == 6

    def test_2(self):
        """
        FILES: subset_1_1.lammpstrj,
               subset_1_2.lammpstrj,
               subset_1_3.lammpstrj
        UNITS: lj

        INDEX:      01234567890123
        READER 0:   nnxxxx
        READER 1:     oooonnxx
        READER 2:           oonnnn
        TRAJECTORY: ffffffffffffff

        _start_frames   = [0, 6, 10]
        _offset_frames  = [0, 4, 2]
        _overlap_frames = {2: (1, 0), 3: (1, 1), 4: (1, 2), 5: (1, 3),
                           8: (2, 0), 9: (2, 1)}

        dt        = 0.005
        time_step = 0.005
        n_frames  = 14
        times     = [0, 0.005, 0.01, ..., 0.055, 0.06, 0.065]
        timesteps = [0, 1, 2, ..., 11, 12, 13]
        """

        trajectory = Trajectory(
            [
                DATA_DIRECTORY / "trajectories/subset_1_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_1_2.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_1_3.lammpstrj",
            ]
        )
        assert trajectory.dt == 0.005
        assert trajectory.time_step == 0.005
        assert trajectory.n_frames == 14
        assert np.allclose(
            trajectory.times,
            np.arange(
                0,
                (trajectory.n_frames - 1) * trajectory.dt
                + 10 * np.finfo(float).eps,
                trajectory.dt,
            ),
        )
        assert np.allclose(trajectory.timesteps, np.arange(trajectory.n_frames))
        frame_indices = range(1, 11)
        for frame_index, frame in zip(frame_indices, trajectory[frame_indices]):
            assert np.isclose(frame.positions[0, 0], 0.1 * frame_index)

    def test_3(self):
        """
        FILES: subset_2_1.lammpstrj,
               subset_2_2.lammpstrj,
               subset_1_3.lammpstrj
        UNITS: real

        INDEX:      01234567890123
        READER 0:   n o o o
        READER 1:    o o o n
        READER 2:           nnnnnn
        TRAJECTORY: ffffffffffffff

        _start_frames   = [0, 7, 8]
        _offset_frames  = [0, 3, 0]
        _overlap_frames = {1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0, 2),
                           5: (1, 2), 6: (0, 3)}

        dt        = 0.005
        time_step = 0.005
        n_frames  = 14
        times     = [0, 0.005, 0.01, ..., 0.055, 0.06, 0.065]
        timesteps = [0, 1, 2, ..., 11, 12, 13]
        """

        trajectory = Trajectory(
            [
                DATA_DIRECTORY / "trajectories/subset_2_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_2_2.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_1_3.lammpstrj",
            ],
            units_style="real",
        )
        assert trajectory.dt == 1
        assert trajectory.time_step == 1
        assert trajectory.n_frames == 14
        assert np.allclose(
            trajectory.times,
            np.arange(
                0,
                (trajectory.n_frames - 1) * trajectory.dt
                + 10 * np.finfo(float).eps,
                trajectory.dt,
            ),
        )
        assert np.allclose(trajectory.timesteps, np.arange(trajectory.n_frames))
        frame_indices = range(1, 9)
        for frame_index, frame in zip(
            frame_indices, trajectory.get_frames(frame_indices)
        ):
            assert np.isclose(frame.positions[0, 0], 0.01 * frame_index)

    def test_4(self):
        """
        FILES: subset_1_1.lammpstrj,
               subset_3_1.lammpstrj,
               subset_3_2.lammpstrj
        UNITS: lj

        INDEX:      0123456  7 890 1 234
        READER 0:   xooxoo
        READER 1:   o  o  n  x  o  x  o
        READER 2:            o o o o o n
        TRAJECTORY: fffffff  f fff f fff

        _start_frames   = [?, 6, 14]
        _offset_frames  = [?, 2, 5]
        _overlap_frames = {0: (1, 0), 1: (0, 1), 2: (0, 2), 3: (1, 1),
                           4: (0, 4), 5: (0, 5), 7: (2, 0), 8: (2, 1),
                           9: (1, 4), 10: (2, 2), 11: (2, 3), 12: (2, 4),
                           13: (2, 5)}

        dt        = 0.005
        time_step = None
        n_frames  = 15
        times     = [0, 0.005, 0.01, ..., 0.03, 0.045, 0.055, 0.06, 0.065,
                     0.075, 0.085, 0.09, 0.095]
        timesteps = [0, 1, 2, ..., 6, 9, 11, 12, 13, 15, 17, 18, 19]
        """

        trajectory = Trajectory(
            [
                DATA_DIRECTORY / "trajectories/subset_1_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_3_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_3_2.lammpstrj",
            ]
        )
        assert trajectory.dt == 0.005
        assert trajectory.time_step is None
        assert trajectory.n_frames == 15
        subset = trajectory[1:7]
        for frame in subset:
            assert np.isclose(frame.positions[0, 0], 0.1 * frame.timestep)

        assert subset.dt == 0.005
        assert subset.time_step == 0.005
        assert subset.n_frames == 6

    def test_5(self):
        """
        FILES: subset_4_1.lammpstrj,
               subset_3_1.lammpstrj,
               subset_2_2.lammpstrj,
               subset_3_2.lammpstrj
        UNITS: lj

        INDEX:      01234567890123 4 567
        READER 0:   x o o x o o
        READER 1:   o  x  o  x  o  x  o
        READER 2:    o o o o
        READER 3:            o o o o o n
        trajectory: ffffffffffffff f fff

        _start_frames   = [?, ?, ?, 17]
        _offset_frames  = [?, ?, ?, 5]
        _overlap_frames = {}

        dt        = 0.005
        time_step = None
        n_frames  = 18
        times     = [0, 0.005, 0.01, ..., 0.055, 0.06, 0.065, 0.075,
                     0.085, 0.09, 0.095]
        timesteps = [0, 1, 2, ..., 11, 12, 13, 15, 17, 18, 19]
        """

        trajectory = Trajectory(
            [
                DATA_DIRECTORY / "trajectories/subset_4_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_3_1.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_2_2.lammpstrj",
                DATA_DIRECTORY / "trajectories/subset_3_2.lammpstrj",
            ]
        )
        assert trajectory.dt == 0.005
        assert trajectory.time_step is None
        assert trajectory.n_frames == 18
        for frame in trajectory:
            assert np.isclose(frame.positions[0, 0], 0.1 * frame.timestep)
