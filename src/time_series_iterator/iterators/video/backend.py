from enum import Enum

class VideoBackend(Enum):
    """
    Decoder a `VideoIterator` reads video frames with.

    Attributes:
    ----------
    OPENCV: Software decode via OpenCV's `cv2.VideoCapture`. Works everywhere, CPU-only.
    TORCHCODEC: Decode via `torchcodec` on the device named by
        `VideoIterationParameters.decode_device`, which defaults to CUDA and there
        offloads decode to the GPU's NVDEC hardware instead of the CPU.
    """
    OPENCV = "opencv"
    TORCHCODEC = "torchcodec"
