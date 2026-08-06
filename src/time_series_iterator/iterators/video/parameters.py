from dataclasses import dataclass

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
    start_video_file_index: int
        The starting index into a scene's video files.
    """
    video_backend: VideoBackend = VideoBackend.OPENCV
    start_video_file_index: int = 0
