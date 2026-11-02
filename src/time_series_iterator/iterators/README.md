
# iterators


## Overview

This module provides specialized iterators for processing time series data. 
The iterators are designed to handle sequential data processing with configurable frequency, time ranges, and indexing schemes.

## Components

| Component | Description |
|-----------|-------------|
| [file.py](file.py) | `FileTimeSeriesIterator`, the base of iterators reading from a list of files |
| [image.py](image.py) | Program for iterating through multiple image files |
| [signal/](signal/README.md) | Program for iterating numeric signals from arrays, files or a custom source |
| [video/](video/README.md) | Program for iterating through multiple video files, over a choice of decode backends |

## Example Usage

### ImageIterator

```python
import glob
from time_series_iterator import ImageIterator, TimeSeriesIterationParameters

params = TimeSeriesIterationParameters(
    pre_sampled_freq=1,
    sampling_freq=2,
    start_time_id=1,
    end_time_id=100,
)

image_paths = sorted(glob.glob("data/images/*.png"))
image_iterator = ImageIterator(paths=image_paths, params=params)

for frame_id, image in image_iterator.with_progress_bar(description="images"):
    pass
```

### VideoIterator

```python
import glob
import cv2
from contextlib import closing
from time_series_iterator import VideoIterator, VideoIterationParameters, IndexBase

paths = sorted(glob.glob("data/videos/*.mp4"))
params = VideoIterationParameters(
    pre_sampled_freq=1,
    sampling_freq=1,
    raw_sampling_rate=30,
    index_base=IndexBase.ZERO,
    start_time_id=0,
    end_time_id=-1,
)
iterator = VideoIterator(paths=paths, params=params)

with closing(iterator.with_progress_bar(description="frames")) as progress_iterator:
    next(iterator)
    next(progress_iterator)

    for frame_id, frame in progress_iterator:
        if frame_id > 10:
            break

frame = iterator.get_sample(30)
cv2.imwrite("output/frame_0030.jpg", frame)
```

### SignalIterator

```python
import numpy as np
from time_series_iterator import (
    ArraySignalSource,
    SignalIterator,
    TimeSeriesIterationParameters,
)

iterator = SignalIterator(
    source=ArraySignalSource(np.sin(np.linspace(0, 10, 1000))),
    params=TimeSeriesIterationParameters(sampling_freq=10),
)

for time_id, value in iterator:
    pass
```

## Random Access

`get_sample(time_id)` reads one sample without advancing iteration, and takes
the same time id the loop yields, so `iterator.get_sample(time_id)` returns what
that step of the loop returned. `sampling_freq` does not restrict it: a time id
the loop steps over still resolves, since it only skips stored samples rather
than removing them. A time id outside the stored samples, or off the pre-sampled grid
described in the [package README](../README.md), raises `ValueError`.