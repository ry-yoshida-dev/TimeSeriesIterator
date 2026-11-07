# video

## Overview

`VideoIterator` reads frames sequentially across a scene's video files. Decoding is delegated to [VideoHandler](https://github.com/ry-yoshida-dev/VideoHandler) through the backend selected in `VideoIterationParameters`.

## Components

| Component | Description |
|-----------|-------------|
| [iterator.py](./iterator.py) | `VideoIterator`, iterating frames across a scene's video files with the configured backend, and reading any one of them at random |
| [parameters.py](./parameters.py) | `VideoIterationParameters`, adding the video backend, decode device and start file to `TimeSeriesIterationParameters` |
| [frame_location.py](./frame_location.py) | `VideoFrameLocation`, the file and in-file index one frame of the scene resolves to |

## Examples

```python
from torch_modules import Device

from time_series_iterator import VideoBackend, VideoIterationParameters, VideoIterator

params = VideoIterationParameters(
    video_backend=VideoBackend.TORCHCODEC, decode_device=Device.CPU
)
iterator = VideoIterator(paths=["video.mp4"], params=params)
```

`get_sample(time_id)` reads one frame at random without disturbing iteration. Each file's random-access reader is opened on first use and kept until `close()`.
