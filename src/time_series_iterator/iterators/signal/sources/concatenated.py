from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ....types import NumericArray
from ..source import SignalSource

class ConcatenatedSignalSource(SignalSource):
    """
    Several signal sources read as one continuous signal, in the given order.

    Attributes:
    ----------
    sources: list[SignalSource]
        The sources, in time order.
    """
    def __init__(self, sources: Sequence[SignalSource]):
        """
        Initialize the ConcatenatedSignalSource.

        Parameters:
        ----------
        sources: Sequence[SignalSource]
            The sources, in time order.

        Raises:
        -------
        ValueError: If there is no source, one is empty, or sample shapes differ.
        """
        if len(sources) == 0:
            raise ValueError("Sources is empty")
        self.sources: list[SignalSource] = list(sources)
        self._end_indices: NDArray[np.int64] = np.cumsum([len(source) for source in self.sources], dtype=np.int64)
        if np.any(np.diff(self._end_indices, prepend=0) == 0):
            raise ValueError("Every source must hold at least one sample")

        sample_shapes = {source.sample_shape for source in self.sources}
        if len(sample_shapes) != 1:
            raise ValueError(f"Sources must share one sample shape, got {sorted(sample_shapes)}")

    def _read_sample(self, index: int) -> NumericArray:
        source_index = int(np.searchsorted(self._end_indices, index, side="right"))
        previous_end_index = int(self._end_indices[source_index - 1]) if source_index > 0 else 0
        return self.sources[source_index].read(index - previous_end_index)

    def __len__(self) -> int:
        return int(self._end_indices[-1])
