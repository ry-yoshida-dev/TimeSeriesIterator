from abc import ABC, abstractmethod

from ...types import NumericArray

class SignalSource(ABC):
    """
    Random-access storage of a numeric signal, addressed by sample index.

    Subclass this to iterate storage the package does not cover (e.g. Parquet,
    HDF5, a database), implementing `__len__` and `_read_sample`.
    """
    def read(self, index: int) -> NumericArray:
        """
        Read one sample.

        Parameters:
        ----------
        index: int
            Zero-based sample index.

        Returns:
        -------
        NumericArray: The sample, zero-dimensional for a single-channel signal.

        Raises:
        -------
        IndexError: If the index addresses no sample.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Sample index must be between 0 and {len(self) - 1}, given: {index}")
        return self._read_sample(index)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """
        Shape of one sample.

        Returns:
        -------
        tuple[int, ...]: The shape, empty for a single-channel signal.
        """
        return tuple(self.read(0).shape)

    @abstractmethod
    def _read_sample(self, index: int) -> NumericArray:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
