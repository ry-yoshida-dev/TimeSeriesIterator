"""Type aliases for numeric arrays yielded by time series iterators."""

import numpy as np
import torch
from numpy.typing import NDArray

NumericArray = NDArray[np.integer | np.floating] | torch.Tensor
