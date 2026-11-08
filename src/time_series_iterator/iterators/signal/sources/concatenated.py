from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate
from typing import override

from ....types import NumericArray
from ..source import SignalSource


class ConcatenatedSignalSource(SignalSource):
    """
    Several signal sources read as one continuous signal, in the given order.

    Attributes
    ----------
    sources: list[SignalSource]
        The sources, in time order.
    """

    def __init__(self, sources: Sequence[SignalSource]):
        """
        Initialize the ConcatenatedSignalSource.

        Parameters
        ----------
        sources: Sequence[SignalSource]
            The sources, in time order.

        Raises
        ------
        ValueError: If there is no source, one is empty, or sample shapes differ.
        """
        if len(sources) == 0:
            raise ValueError("Sources is empty")
        self.sources: list[SignalSource] = list(sources)
        if any(len(source) == 0 for source in self.sources):
            raise ValueError("Every source must hold at least one sample")
        self._end_indices: list[int] = list(
            accumulate(len(source) for source in self.sources)
        )

        sample_shapes = {source.sample_shape for source in self.sources}
        if len(sample_shapes) != 1:
            raise ValueError(
                f"Sources must share one sample shape, got {sorted(sample_shapes)}"
            )

    @override
    def _read_sample(self, index: int) -> NumericArray:
        source_index = bisect_right(self._end_indices, index)
        previous_end_index = (
            self._end_indices[source_index - 1] if source_index > 0 else 0
        )
        return self.sources[source_index].read(index - previous_end_index)

    @override
    def __len__(self) -> int:
        return self._end_indices[-1]
