# TimeSeriesIterator

## Overview

TimeSeriesIterator (`time_series_iterator`) iterates time-series data in ascending order of time id, with configurable sampling, start and end ids, and index base. Every iterator yields `(time_id, sample)` pairs and can read any sample at random.

| Source type | Iterator | Parameters | Input |
|---|---|---|---|
| `SourceType.IMAGE` | `ImageIterator` | `TimeSeriesIterationParameters` | Image files, one per sample |
| `SourceType.VIDEO` | `VideoIterator` | `VideoIterationParameters` | Video files of one scene, decoded with OpenCV or torchcodec |
| `SourceType.SIGNAL` | `SignalIterator` | `TimeSeriesIterationParameters` | Numeric arrays, tensors, `.npy`/`.npz`/`.csv`/`.tsv`/`.txt` files, or a custom `SignalSource` |

See [src/time_series_iterator/iterators/](src/time_series_iterator/iterators/README.md) for each iterator.

## Installation

```bash
pip install .
```

`torchcodec` resolves to the CUDA-enabled wheel under `wheels/` on Linux x86_64 with Python 3.14, and to the published CPU release elsewhere. It loads the system FFmpeg (versions 4 to 9), so install FFmpeg before using `VideoBackend.TORCHCODEC`.

## Usage

```python
import glob

from time_series_iterator import IndexBase, SourceType, TimeSeriesIterator, VideoIterationParameters

params = VideoIterationParameters(
    sampling_freq=5,
    raw_sampling_rate=30,
    index_base=IndexBase.ONE,
)

with TimeSeriesIterator.build(SourceType.VIDEO, sorted(glob.glob("videos/*.mp4")), params) as iterator:
    for time_id, frame in iterator.with_progress_bar(description="frames"):
        if time_id >= 20:
            break

    frame = iterator.get_sample(30)
```
