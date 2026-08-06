# video

## Overview

`VideoIterator` reads frames sequentially across a scene's video files. Frame
decoding itself is delegated to a backend selected by `VideoBackend`
(`VideoIterationParameters.video_backend`), so `VideoIterator` never depends
on which one is behind it. `VideoIterationParameters` extends the package's
general `TimeSeriesIterationParameters` with the video-only settings
(`video_backend`, `start_video_file_index`), so `TimeSeriesIterationParameters`
itself stays media-agnostic.

## Components

| Component | Description |
|-----------|-------------|
| [\_\_init\_\_.py](./__init__.py) | `VideoIterator`, iterating frames across a scene's video files against whichever backend is configured |
| [backend.py](./backend.py) | `VideoBackend` enum selecting the decode backend (`OPENCV` / `TORCHCODEC`) |
| [parameters.py](./parameters.py) | `VideoIterationParameters`, extending `TimeSeriesIterationParameters` with video-only settings |
| [reader.py](./reader.py) | `VideoFrameReader` protocol every backend implements |
| [factory.py](./factory.py) | Builds the reader for a given `VideoBackend`, importing a backend's module only when it is selected |
| [readers/](./readers/) | Per-backend `VideoFrameReader` implementations that need adapter logic (`torchcodec.py`); `VideoBackend.OPENCV` needs none, so `factory.py` constructs `opencv_video.VideoReader` directly |

## Example

```python
from time_series_iterator import VideoBackend, VideoIterationParameters, VideoIterator

params = VideoIterationParameters(video_backend=VideoBackend.TORCHCODEC)
iterator = VideoIterator(paths=["video.mp4"], params=params)
```

`VideoBackend.TORCHCODEC` requires the optional `torchcodec` extra
(`pip install time-series-iterator[torchcodec]`) and a CUDA-capable GPU;
`VideoBackend.OPENCV` (the default) needs neither.
