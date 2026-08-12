# readers

## Overview

Per-backend implementations of the `VideoFrameReader` protocol defined in
[reader.py](../reader.py), for backends that need adapter logic beyond
constructing another package's reader. `factory.py` imports a module in this
directory only when its corresponding `VideoBackend` is selected, so an
unused backend's dependencies never need to be installed. `VideoBackend.OPENCV`
has no module here: `opencv_video.VideoReader` already satisfies
`VideoFrameReader` directly, so `factory.py` constructs it inline.

## Components

| Component | Description |
|-----------|-------------|
| [torchcodec.py](./torchcodec.py) | `VideoBackend.TORCHCODEC` backend: decode via `torchcodec` on a CUDA device, using the GPU's NVDEC hardware instead of the CPU, a run of consecutive frames per decode call while iterating |
