"""
Analysis base classes
=====================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains custom base classes for serial and multithreaded
data analysis with support for the multiprocessing, Dask, Joblib, and
Numba libraries for parallelization.
"""

from abc import abstractmethod
from collections.abc import Generator, Iterable
import contextlib
from datetime import datetime
import logging
import multiprocessing
import os
from typing import Any, Callable, TextIO, Union
import warnings

try:
    import dask
    from dask import distributed
    FOUND_DASK = True
except ImportError:
    FOUND_DASK = False

try:
    import joblib
    FOUND_JOBLIB = True
except ImportError:
    FOUND_JOBLIB = False

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.base import ReaderBase
import numba
import numpy as np
from tqdm import tqdm

@contextlib.contextmanager
def _tqdm_joblib(tqdm_obj: tqdm) -> Generator:

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs) -> None:
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_obj.close()

def _istarmap(
        self: multiprocessing.pool.Pool, func: Callable[[Any], Any],
        iterable: Iterable, chunk_size: int = 1) -> Iterable:

    self._check_running()
    if chunk_size < 1:
        raise ValueError("Chunk size must be greater than 1.")

    task_batches = multiprocessing.pool.Pool._get_tasks(func, iterable,
                                                        chunk_size)
    result = multiprocessing.pool.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(
            result._job,
            multiprocessing.pool.starmapstar,
            task_batches
        ),
        result._set_length
    ))
    return (item for chunk in result for item in chunk)

multiprocessing.pool.Pool.istarmap = _istarmap

class Hash(dict):

    """
    A hash table, or an extension of the built-in `dict` with dot
    notation for accessing properties.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
            else:
                raise TypeError("Positional arguments must be dictionaries.")
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]

class SerialAnalysisBase(AnalysisBase):

    """
    A serial analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.

    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, verbose: bool = False, **kwargs):
        super().__init__(trajectory, verbose, **kwargs)

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, np.ndarray[int]] = None, n_jobs: int = 1,
            verbose: bool = None, **kwargs) -> "SerialAnalysisBase":

        """
        Performs the calculation in serial.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.

        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.

        **kwargs
            Additional keyword arguments to pass to
            :class:`MDAnalysis.lib.log.ProgressBar`.

        Returns
        -------
        self : `SerialAnalysisBase`
            Serial analysis base object.
        """

        return super().run(start, stop, step, frames, verbose, **kwargs)

    def save(
            self, file: Union[str, TextIO], archive: bool = True,
            compress: bool = True, **kwargs) -> None:

        """
        Saves results to a binary or archive file in NumPy format.

        Parameters
        ----------
        file : `str` or `file`
            Filename or file-like object where the data will be saved.
            If `file` is a `str`, the :code:`.npy` or :code:`.npz`
            extension will be appended automatically if not already
            present.

        archive : `bool`, default: :code:`True`
            Determines whether the results are saved to a single archive
            file. If `True`, the data is stored in a :code:`.npz` file.
            Otherwise, the data is saved to multiple :code:`.npy` files.

        compress : `bool`, default: :code:`True`
            Determines whether the :code:`.npz` file is compressed. Has
            no effect when :code:`archive=False`.

        **kwargs
            Additional keyword arguments to pass to :func:`numpy.save`,
            :func:`numpy.savez`, or :func:`numpy.savez_compressed`,
            depending on the values of `archive` and `compress`.
        """

        if archive and compress:
            np.savez_compressed(file, **self.results, **kwargs)
        elif archive:
            np.savez(file, **self.results, **kwargs)
        else:
            for data in self.results:
                np.save(f"{file}_{data}", self.results[data], **kwargs)

class NumbaAnalysisBase(SerialAnalysisBase):

    """
    A Numba-accelerated analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.

    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, verbose: bool = False, **kwargs):
        super().__init__(trajectory, verbose, **kwargs)

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, np.ndarray[int]] = None, n_threads: int = None,
            verbose: bool = None, **kwargs
        ) -> "NumbaAnalysisBase":

        """
        Performs the calculation.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.

        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        n_threads : `int`, keyword-only, optional
            Number of threads to use for analysis.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.

        **kwargs
            Additional keyword arguments to pass to
            :class:`MDAnalysis.lib.log.ProgressBar`.

        Returns
        -------
        self : `NumbaAnalysisBase`
            Analysis object with results.
        """

        if n_threads is not None:
            numba.set_num_threads(n_threads)

        return super().run(
            start=start, stop=stop, step=step, frames=frames,
            verbose=verbose, **kwargs
        )

class ParallelAnalysisBase(SerialAnalysisBase):

    """
    A multithreaded analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.

    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, verbose: bool = False, **kwargs):
        super().__init__(trajectory, verbose, **kwargs)

    def _dask_job_block(
            self, frames: Union[slice, np.ndarray[int]],
            indices: np.ndarray[int]) -> list:
        return [self._single_frame_parallel(f, i) for f, i in zip(frames, indices)]

    @abstractmethod
    def _single_frame_parallel(self, frame: int, index: int):
        pass

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, np.ndarray[int]] = None, verbose: bool = None,
            n_jobs: int = None, module: str = "multiprocessing",
            block: bool = True, method: str = None, **kwargs
        ) -> "ParallelAnalysisBase":

        """
        Performs the calculation in parallel.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.

        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.

        n_jobs : `int`, keyword-only, optional
            Number of workers. If not specified, it is automatically
            set to either the minimum number of workers required to
            fully analyze the trajectory or the maximum number of CPU
            threads available.

        module : `str`, keyword-only, default: :code:`"multiprocessing"`
            Parallelization module to use for analysis.

            **Valid values**: :code:`"dask"`, :code:`"joblib"`, and
            :code:`"multiprocessing"`.

        block : `bool`, keyword-only, default: :code:`True`
            Determines whether the trajectory is split into smaller
            blocks that are processed serially in parallel with other
            blocks. This "split–apply–combine" approach is generally
            faster since the trajectory attributes do not have to be
            packaged for each analysis run. Has no effect if
            :code:`module="multiprocessing"`.

        method : `str`, keyword-only, optional
            Specifies which Dask scheduler, Joblib backend, or
            multiprocessing start method is used.

        **kwargs
            Additional keyword arguments to pass to
            :func:`dask.compute`, :class:`joblib.Parallel`, or
            :class:`multiprocessing.pool.Pool`, depending on the value of
            `module`.

        Returns
        -------
        self : `ParallelAnalysisBase`
            Parallel analysis base object.
        """

        if verbose is None:
            verbose = getattr(self, '_verbose', False)
        logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}",
                            style="{",
                            level=logging.INFO if verbose else logging.WARNING)

        self._setup_frames(self._trajectory, start=start, stop=stop,
                           step=step, frames=frames)
        self._prepare()

        n_jobs = min(n_jobs or np.inf, self.n_frames,
                     len(os.sched_getaffinity(0)))
        frames = (frames if frames
                  else np.arange(self.start or 0, self.stop or self.n_frames,
                                 self.step))
        n_frames = len(frames)
        indices = np.arange(n_frames)

        if verbose:
            time_start = datetime.now()

        if module == "dask" and FOUND_DASK:
            try:
                config = {"scheduler": distributed.worker.get_client(),
                          **kwargs}
                n_jobs = min(len(config["scheduler"].get_worker_logs()),
                             n_jobs)
            except ValueError:
                if method is None:
                    method = "processes"
                elif method not in {"distributed", "processes", "threading",
                                    "threads", "single-threaded", "sync",
                                    "synchronous"}:
                    raise ValueError("Invalid Dask scheduler.")

                if method == "distributed":
                    emsg = ("The Dask distributed client "
                            "(client = dask.distributed.Client(...)) "
                            "should be instantiated in the main "
                            "program (__name__ = '__main__') of "
                            "your script.")
                    raise RuntimeError(emsg)
                elif method in {"threading", "threads"}:
                    emsg = ("The threaded Dask scheduler is not "
                            "compatible with MDAnalysis.")
                    raise ValueError(emsg)
                elif n_jobs == 1 and method not in {"single-threaded", "sync",
                                                    "synchronous"}:
                    method = "synchronous"
                    logging.warning(f"Since {n_jobs=}, the synchronous "
                                    "Dask scheduler will be used instead.")
                config = {"scheduler": method} | kwargs
                if method == "processes":
                    config["num_workers"] = n_jobs

            logging.info(f"Starting analysis using Dask ({n_jobs=}, "
                         f"scheduler={config['scheduler']})...")

            jobs = []
            if block:
                for frame, index in zip(np.array_split(frames, n_jobs),
                                        np.array_split(indices, n_jobs)):
                    jobs.append(dask.delayed(self._dask_job_block)(frame, index))
            else:
                for frame, index in zip(frames, indices):
                    jobs.append(dask.delayed(self._single_frame_parallel)
                                (frame, index))

            blocks = dask.delayed(jobs).persist(**config)
            if verbose:
                distributed.progress(blocks)
            self._results = blocks.compute(**config)
            if block:
                self._results = [r for b in self._results for r in b]

        elif module == "joblib" and FOUND_JOBLIB:
            if method is not None and method not in {"loky", "multiprocessing",
                                                     "threading", None}:
                raise ValueError("Invalid Joblib backend.")

            logging.info("Starting analysis using Joblib "
                         f"({n_jobs=}, backend={method})...")
            with (_tqdm_joblib(tqdm(total=n_frames)) if verbose
                  else contextlib.suppress()):
                if block:
                    self._results = joblib.Parallel(
                        n_jobs=n_jobs, backend=method, **kwargs
                    )(
                        joblib.delayed(self._single_frame_parallel)(f, i)
                        for frames_, indices_ in zip(
                            np.array_split(frames, n_jobs),
                            np.array_split(indices, n_jobs)
                        ) for f, i in zip(frames_, indices_)
                    )
                else:
                    self._results = joblib.Parallel(
                        n_jobs=n_jobs, prefer=method, **kwargs
                    )(
                        joblib.delayed(self._single_frame_parallel)(f, i)
                        for f, i in zip(frames, indices)
                    )

        else:
            if module != "multiprocessing":
                wmsg = ("The Dask or Joblib library was not found, so "
                        "the native multiprocessing module will be"
                        "used instead.")
                warnings.warn(wmsg)

            if method is None:
                method = multiprocessing.get_start_method()
            elif method not in {"fork", "forkserver", "spawn"}:
                raise ValueError("Invalid multiprocessing start method.")

            logging.info("Starting analysis using multiprocessing "
                         f"({n_jobs=}, {method=})...")
            with multiprocessing.get_context(method).Pool(n_jobs, **kwargs) as p:
                self._results = (
                    tuple(
                        tqdm(
                            p.istarmap(self._single_frame_parallel,
                                       zip(frames, indices)),
                            total=n_frames
                        )
                    ) if verbose else p.starmap(self._single_frame_parallel,
                                                zip(frames, indices))
                )

        if verbose:
            logging.info(f"Analysis finished in {datetime.now() - time_start}.")

        self._conclude()
        return self

class DynamicAnalysisBase(ParallelAnalysisBase, SerialAnalysisBase):

    """
    A dynamic analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.

    parallel : `bool`
        Determines whether the analysis is performed in parallel.

    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, parallel: bool,
            verbose: bool = False, **kwargs) -> None:

        self._parallel = parallel
        (ParallelAnalysisBase if parallel else SerialAnalysisBase).__init__(
            self, trajectory, verbose=verbose, **kwargs
        )

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, np.ndarray[int]] = None,
            verbose: bool = None, **kwargs
        ) -> Union[SerialAnalysisBase, ParallelAnalysisBase]:

        """
        Performs the calculation.

        .. seealso::

           For parallel-specific keyword arguments, see
           :meth:`ParallelAnalysisBase.run`.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.

        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.

        **kwargs
            Additional keyword arguments to pass to
            :class:`MDAnalysis.lib.log.ProgressBar`.

        Returns
        -------
        self : `SerialAnalysisBase` or `ParallelAnalysisBase`
            Analysis object with results.
        """

        return (ParallelAnalysisBase if self._parallel
                else SerialAnalysisBase).run(
            self, start=start, stop=stop, step=step, frames=frames,
            verbose=verbose, **kwargs
        )