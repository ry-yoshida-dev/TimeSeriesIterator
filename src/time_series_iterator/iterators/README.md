# iterators

## Overview

Concrete `TimeSeriesIterator` implementations, one per source type.

## Components

| Component | Description |
|-----------|-------------|
| [file.py](file.py) | `FileTimeSeriesIterator`, the base of iterators reading from a list of files |
| [image.py](image.py) | `ImageIterator`, iterating image files |
| [signal/](signal/README.md) | `SignalIterator`, iterating numeric signals from arrays, files or a custom source |
| [video/](video/README.md) | `VideoIterator`, iterating frames across video files |

## Examples

```python
import glob

from time_series_iterator import ImageIterator, TimeSeriesIterationParameters

params = TimeSeriesIterationParameters(
    sampling_freq=2, start_time_id=1, end_time_id=100
)
iterator = ImageIterator(paths=sorted(glob.glob("data/images/*.png")), params=params)

for time_id, image in iterator:
    pass
```
