# video

## Overview

`VideoIterator` reads frames sequentially across a scene's video files. Frame
decoding itself is delegated to [VideoHandler](https://github.com/ry-yoshida-dev/VideoHandler):
`VideoIterationParameters.reader_factory` builds a `video_handler.VideoReaderFactory`
for the selected `VideoBackend`, and every reader it returns implements
`video_handler.VideoFrameReader`, so `VideoIterator` never depends on which
backend is behind it. `VideoIterationParameters` extends the package's
general `TimeSeriesIterationParameters` with the video-only settings
(`video_backend`, `decode_device`, `start_video_file_index`), so
`TimeSeriesIterationParameters` itself stays media-agnostic.

## Components

| Component | Description |
|-----------|-------------|
| [iterator.py](./iterator.py) | `VideoIterator`, iterating frames across a scene's video files against whichever backend is configured |
| [parameters.py](./parameters.py) | `VideoIterationParameters`, extending `TimeSeriesIterationParameters` with video-only settings and building the `VideoReaderFactory` |
| [frame_location.py](./frame_location.py) | `VideoFrameLocation`, the file and in-file index one frame of the scene resolves to |

## Example

```python
from torch_modules import Device

from time_series_iterator import VideoBackend, VideoIterationParameters, VideoIterator

params = VideoIterationParameters(video_backend=VideoBackend.TORCHCODEC)
iterator = VideoIterator(paths=["video.mp4"], params=params)
```

`VideoBackend` is re-exported from `video_handler`. `VideoBackend.OPENCV` (the
default) yields BGR `(H, W, 3)` NumPy frames; `VideoBackend.TORCHCODEC` yields
RGB `(3, H, W)` tensors on the decode device. `torchcodec` is always installed,
but only the Linux build is CUDA-enabled.

`decode_device` picks which device that backend decodes on, and defaults to
`Device.detect()`, so the same configuration runs on a GPU host and a CPU-only
one: CUDA uses the GPU's NVDEC hardware, CPU decodes in software and still
yields tensors, unlike `VideoBackend.OPENCV`. A device this machine cannot
decode on resolves down to `Device.CPU` with a warning, and the reader's
`device` attribute reports the one actually in use. Set it explicitly to pin
the decode somewhere:

```python
params = VideoIterationParameters(
    video_backend=VideoBackend.TORCHCODEC,
    decode_device=Device.CPU,
)
```
