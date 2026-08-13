from dataclasses import dataclass, field

from torch_modules import Device

from ...parameters import TimeSeriesIterationParameters
from .backend import VideoBackend

@dataclass
class VideoIterationParameters(TimeSeriesIterationParameters):
    """
    Parameters for iterating videos with a `VideoIterator`.

    Parameters:
    ----------
    video_backend: VideoBackend
        The backend a VideoIterator reads frames with.
    decode_device: Device
        Device the decode runs on, defaulting to the best one this machine has.
        Only `VideoBackend.TORCHCODEC` reads this; `VideoBackend.OPENCV` always
        decodes on the CPU.
    start_video_file_index: int
        The starting index into a scene's video files.
    """
    video_backend: VideoBackend = VideoBackend.OPENCV
    decode_device: Device = field(default_factory=Device.detect)
    start_video_file_index: int = 0
