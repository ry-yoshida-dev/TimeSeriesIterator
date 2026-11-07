# signal

## Overview

`SignalIterator` iterates numeric signals whose first axis is time, reading every sample through a `SignalSource`, so a new storage format needs only a new source.

## Components

| Component | Description |
|-----------|-------------|
| [\_\_init\_\_.py](./__init__.py) | `SignalIterator`, iterating the samples of a `SignalSource` |
| [source.py](./source.py) | `SignalSource`, the interface of random-access signal storage |
| [file_format.py](./file_format.py) | `SignalFileFormat`, opening a file from its suffix with default options |
| [sources/](./sources/README.md) | Concrete sources for arrays, NumPy files, text tables and concatenation |

## Examples

```python
from time_series_iterator import (
    ArraySignalSource,
    ConcatenatedSignalSource,
    DelimitedTextSignalSource,
    SignalIterator,
    SourceType,
    TimeSeriesIterationParameters,
    TimeSeriesIterator,
)

params = TimeSeriesIterationParameters(raw_sampling_rate=100, sampling_freq=2)

# Files, opened by suffix: .npy, .npz, .csv, .tsv, .txt
iterator = TimeSeriesIterator.build(SourceType.SIGNAL, ["day1.npy", "day2.npy"], params)

# In-memory data: lists, NumPy arrays, tensors, pandas objects, ...
iterator = SignalIterator(ArraySignalSource([1.0, 2.0, 3.0]), params)

# Tables with options, joined over several files
source = ConcatenatedSignalSource([
    DelimitedTextSignalSource(path, is_header_present=True, columns=["x", "y"])
    for path in ["day1.csv", "day2.csv"]
])
iterator = SignalIterator(source, params)
```

Any other storage needs only the sample count and a read by index:

```python
import h5py
import numpy as np

from time_series_iterator import SignalSource
from time_series_iterator.types import NumericArray


class Hdf5SignalSource(SignalSource):
    def __init__(self, path: str, dataset_name: str):
        self.dataset = h5py.File(path, "r")[dataset_name]

    def _read_sample(self, index: int) -> NumericArray:
        return np.asarray(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)
```
