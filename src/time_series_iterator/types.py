"""Type aliases for numeric arrays yielded by time series iterators."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

NumericArray = NDArray[np.integer[Any] | np.floating[Any]] | torch.Tensor
