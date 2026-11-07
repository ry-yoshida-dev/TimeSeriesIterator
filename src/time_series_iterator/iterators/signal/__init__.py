from __future__ import annotations


from id_manager import IDManager

from ...iterator import TimeSeriesIterator
from ...parameters import TimeSeriesIterationParameters
from ...types import NumericArray
from ...utils import SourceType
from .file_format import SignalFileFormat
from .source import SignalSource
from .sources import (
    ArraySignalSource,
    ConcatenatedSignalSource,
    DelimitedTextSignalSource,
    NumpyFileSignalSource,
)

__all__ = [
    "ArraySignalSource",
    "ConcatenatedSignalSource",
    "DelimitedTextSignalSource",
    "NumpyFileSignalSource",
    "SignalFileFormat",
    "SignalIterator",
    "SignalSource",
]

class SignalIterator(TimeSeriesIterator):
    """
    Iterator for numeric signals, reading each sample from a `SignalSource`.

    The first axis of the signal is time, and samples are assumed evenly spaced.
    A single-channel signal yields zero-dimensional samples, and a
    `(time, channels)` signal yields `(channels,)` ones.

    Attributes:
    ----------
    source: SignalSource
        The storage of the signal.
    sample_id_manager: IDManager
        The manager for the index of the next stored sample.
    """
    def __init__(
        self,
        source: SignalSource,
        params: TimeSeriesIterationParameters,
        ):
        """
        Initialize the SignalIterator.

        Parameters:
        ----------
        source: SignalSource
            The storage of the signal.
        params: TimeSeriesIterationParameters
            The parameters for the signal iterator.

        Raises:
        -------
        ValueError: If the source holds no samples.
        """
        super().__init__(params=params)
        if len(source) == 0:
            raise ValueError("Signal source has no samples")
        self.source = source
        self.sample_id_manager = IDManager(
            current_id=self.params.offset_start_id,
            step=self.params.sampling_freq,
            )

    @classmethod
    def from_paths(
        cls,
        paths: list[str],
        params: TimeSeriesIterationParameters,
        ) -> SignalIterator:
        """
        Build an iterator reading files as one signal, in the given order.

        Each file is opened with the default options of its suffix. Build the
        sources directly for anything else, such as a header or column choice.

        Parameters:
        ----------
        paths: list[str]
            The paths to the signal files.
        params: TimeSeriesIterationParameters
            The parameters for the signal iterator.

        Returns:
        -------
        SignalIterator: The iterator.

        Raises:
        -------
        ValueError: If the paths is empty or a suffix is not recognized.
        FileNotFoundError: If a file is not found.
        """
        if len(paths) == 0:
            raise ValueError("Paths is empty")
        file_formats = [SignalFileFormat.from_path(path) for path in paths]
        sources = [file_format.open(path) for file_format, path in zip(file_formats, paths, strict=True)]
        return cls(source=ConcatenatedSignalSource(sources), params=params)

    def _next_data(self) -> NumericArray | None:
        """
        Get the next sample from the source.

        Returns:
        ----------
        NumericArray | None: The next sample, or None past the last one.
        """
        index = self.sample_id_manager.next_id
        if index >= len(self.source):
            return None
        return self.source.read(index)

    def _skip_data(self) -> bool:
        """
        Advance past one sample without reading it.

        Returns:
        ----------
        bool: Whether a sample remained to skip.
        """
        return self.sample_id_manager.next_id < len(self.source)

    def get_sample(self, time_id: int) -> NumericArray:
        """
        Read the sample at an arbitrary time id, without advancing iteration.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        ----------
        NumericArray: The sample stored at `time_id`.

        Raises:
        ----------
        ValueError: If the time id addresses no stored sample.
        """
        return self.source.read(self.sample_index_of(time_id))

    def close(self) -> None:
        """Release resources. The signal source holds no open handles, so there is nothing to release."""

    def __len__(self) -> int:
        return len(self.source)

    def __str__(self) -> str:
        return f"SignalIterator(source={type(self.source).__name__}, params={self.params})"

    @property
    def source_type(self) -> SourceType:
        return SourceType.SIGNAL

    @property
    def end_time_id(self) -> int:
        return (len(self.source) - 1)*self.params.pre_sampled_freq + self.params.index_base.value
