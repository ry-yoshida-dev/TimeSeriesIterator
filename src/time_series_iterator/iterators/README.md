
# iterators


## Overview

This module provides specialized iterators for processing time series data. 
The iterators are designed to handle sequential data processing with configurable frequency, time ranges, and indexing schemes.

## Components

| Component | Description |
|-----------|-------------|
| [image.py](image.py) | Program for iterating through multiple image files |
| [video/](video/README.md) | Program for iterating through multiple video files, over a choice of decode backends |

## Example Usage

### ImageIterator

```python
import glob
from tqdm import tqdm
from time_series_iterator import ImageIterator, TimeSeriesIterationParameters

params = TimeSeriesIterationParameters(
    pre_sampled_freq=1,
    sampling_freq=2,
    start_time_id=1,
    end_time_id=100,
)

image_paths = sorted(glob.glob("data/images/*.png"))
image_iterator = ImageIterator(paths=image_paths, params=params)

for frame_id, image in tqdm(image_iterator):
    pass
```

### VideoIterator

```python
import glob
import cv2
from tqdm import tqdm
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

tqdm_iterator = iter(tqdm(iterator))
next(iterator)
next(tqdm_iterator)

for frame_id, frame in tqdm_iterator:
    if frame_id > 10:
        break

frame = iterator.get_image(30)
cv2.imwrite("output/frame_0030.jpg", frame)
```