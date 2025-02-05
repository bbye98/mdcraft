from abc import abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self
import weakref


class BaseWriter:
    """
    Base class for topology and trajectory writers.

    ...
    """

    def __init__(self, filename: str | Path) -> None:
        # Resolve full path to file
        self._filename = Path(filename).resolve(True)

        # Create finalizer
        self._finalizer = weakref.finalize(self, self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._finalizer()

    @abstractmethod
    def open(self) -> None:
        """
        Opens the topology or trajectory file and stores a handle to it.
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the topology and trajectory file and deletes the handle.
        """

        pass


class BaseTrajectoryWriter(BaseWriter):
    pass


class LAMMPSDumpWriter(BaseTrajectoryWriter):
    pass
