# TimeSeriesIterator

## Overview

TimeSeriesIterator (`time_series_iterator`) is a Python package for iterating time-series media data (images and videos) with configurable sampling intervals.
Use `TimeSeriesIterationParameters` to control start/end IDs, sampling frequency, and index base, then build an iterator from `MediaType`. `MediaType.VIDEO` requires `VideoIterationParameters`, a subclass adding video-only settings such as the decode backend.

For package-level details, see [src/time_series_iterator/README.md](src/time_series_iterator/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

If you only need dependencies:

```bash
pip install -r requirements.txt
```

`torchcodec` (`VideoBackend.TORCHCODEC`) is a required dependency, but which
build gets installed depends on the platform. On Linux, `uv` resolves it to the
CUDA-enabled wheel committed under `wheels/`, which decodes on the GPU's NVDEC
hardware. Every other platform, and plain `pip` on any platform, resolves it to
the published `torchcodec` release, which decodes on the CPU.

Video decoding goes through
[VideoHandler](https://github.com/ry-yoshida-dev/VideoHandler), which provides
both backends. `torchcodec` loads the system FFmpeg (versions 4 to 9) at the
first decode, so install FFmpeg before using `VideoBackend.TORCHCODEC`.

## Progress Bar

`TimeSeriesIterator.with_progress_bar` wraps the iterator with
[ProgressBar](https://github.com/ry-yoshida-dev/ProgressBar), which picks an
installed progress bar library at runtime and falls back to a dependency-free
text bar when none is available or the output is redirected.

```python
from progress_bar import ProgressBarBackend

for time_id, frame in iterator.with_progress_bar(description="frames"):
    ...

for time_id, frame in iterator.with_progress_bar(
    description="frames",
    unit="frame",
    backend=ProgressBarBackend.TQDM,
    is_enabled=True,
):
    ...
```

`total` defaults to `iterator.remaining_step_count`, the number of pairs the
loop actually yields under `start_time_id`, `end_time_id` and the sampling
frequencies. That differs from `len(iterator)`, which is the size of the
underlying media. `is_enabled=False` silences the bar without changing the
call site.

Only `tqdm` is installed with this package, so `backend` selects it or one of
the dependency-free `PLAIN` and `SILENT` bars. Another backend
(`ProgressBarBackend.RICH`, `PROGRESSBAR2`, `ALIVE_PROGRESS`) renders once its
package is installed, and silently falls back otherwise.

The bar is finalized when the returned iterator is exhausted or closed. Close
it explicitly when a loop breaks early while a reference to the iterator
survives:

```python
from contextlib import closing

with closing(iterator.with_progress_bar(description="frames")) as frames:
    for time_id, frame in frames:
        if time_id > 100:
            break
```

## Example

```python
import glob
import cv2
from time_series_iterator import (
    IndexBase,
    MediaType,
    TimeSeriesIterator,
    VideoIterationParameters,
)

paths = sorted(glob.glob("videos/*.mp4"))

params = VideoIterationParameters(
    sampling_freq=5,
    raw_sampling_rate=30,
    index_base=IndexBase.ONE,
    start_time_id=1,
    end_time_id=-1,
)

iterator = TimeSeriesIterator.build(
    media_type=MediaType.VIDEO,
    paths=paths,
    parameters=params,
)

print(f"Total frames: {len(iterator)}")

for time_id, frame in iterator:
    if time_id == 10:
        cv2.imwrite("frame_0010.jpg", frame)
    if time_id >= 20:
        break

frame = iterator.get_image(30)
cv2.imwrite("frame_0030.jpg", frame)

with TimeSeriesIterator.build(MediaType.VIDEO, paths, params) as it:
    for time_id, frame in it:
        if time_id >= 5:
            break
```
