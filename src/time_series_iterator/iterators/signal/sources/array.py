from typing import override

import numpy as np
import torch
from numpy.typing import ArrayLike

from ....types import NumericArray
from ..source import SignalSource


class ArraySignalSource(SignalSource):
    """
    Signal held in an array whose first axis is time.

    Anything NumPy can convert is accepted (lists, pandas objects, memory maps,
    ...) and tensors are kept as tensors, so samples stay on their device.

    Attributes
    ----------
    data: NumericArray
        The signal, shaped `(time,)` or `(time, *sample_shape)`.
    """

    def __init__(self, data: ArrayLike | torch.Tensor):
        """
        Initialize the ArraySignalSource.

        Parameters
        ----------
        data: ArrayLike | torch.Tensor
            The signal, shaped `(time,)` or `(time, *sample_shape)`.

        Raises
        ------
        TypeError: If the elements are not integers or real floats.
        ValueError: If the data has no time axis or no samples.
        """
        array: NumericArray = (
            data if isinstance(data, torch.Tensor) else np.asarray(data)
        )
        self._validate(array)
        self.data: NumericArray = array

    def _validate(self, array: NumericArray) -> None:
        """
        Validate the element type and the time axis.

        Parameters
        ----------
        array: NumericArray
            The signal to validate.

        Raises
        ------
        TypeError: If the elements are not integers or real floats.
        ValueError: If the data has no time axis or no samples.
        """
        is_numeric = (
            not (array.is_complex() or array.dtype == torch.bool)
            if isinstance(array, torch.Tensor)
            else array.dtype.kind in "iuf"
        )
        if not is_numeric:
            raise TypeError(
                f"Signal elements must be integers or real floats, got {array.dtype}"
            )
        if array.ndim == 0:
            raise ValueError(
                "Signal must have a time axis, got a zero-dimensional array"
            )
        if array.shape[0] == 0:
            raise ValueError("Signal has no samples")

    @override
    def _read_sample(self, index: int) -> NumericArray:
        # The trailing ellipsis keeps a one-dimensional NumPy signal yielding
        # zero-dimensional arrays rather than NumPy scalars.
        return self.data[index, ...]

    @override
    def __len__(self) -> int:
        return self.data.shape[0]
