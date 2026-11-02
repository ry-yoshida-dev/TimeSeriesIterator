import os

from ..iterator import TimeSeriesIterator
from ..parameters import TimeSeriesIterationParameters

class FileTimeSeriesIterator(TimeSeriesIterator):
    """
    Base class for iterators reading their samples from a list of files.

    Attributes:
    ----------
    paths: list[str]
        The paths to the time series data.
    """
    def __init__(
        self,
        params: TimeSeriesIterationParameters,
        paths: list[str]
        ):
        """
        Initialize the FileTimeSeriesIterator.

        Parameters:
        ----------
        params: TimeSeriesIterationParameters
            The parameters for the time series iterator.
        paths: list[str]
            The paths to the time series data.
        """
        super().__init__(params=params)
        self._validate_paths(paths)
        self.paths = paths

    def _validate_paths(
        self,
        paths: list[str]
        ) -> None:
        """
        Validate the paths.

        Parameters:
        ----------
        paths: list[str]
            The paths to the time series data.

        Raises:
        -------
        ValueError: If the paths is empty.
        FileNotFoundError: If the file is not found.
        """
        if len(paths) == 0:
            raise ValueError("Paths is empty")
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
