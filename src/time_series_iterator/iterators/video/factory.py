from .backend import VideoBackend
from .reader import VideoFrameReader

def build_video_reader(
    backend: VideoBackend,
    video_path: str,
    iter_start_frame: int,
    freq: int,
) -> VideoFrameReader:
    """
    Build the frame reader a backend provides.

    `torchcodec` is only imported for `VideoBackend.TORCHCODEC`, so a caller
    that never selects it does not need the package installed.

    Parameters:
    ----------
    backend: VideoBackend
        Backend to read the video with.
    video_path: str
        Path to the video file.
    iter_start_frame: int
        Frame index to start reading from.
    freq: int
        Step size between yielded frames.

    Returns:
    ----------
    VideoFrameReader: Reader backed by `backend`.
    """
    match backend:
        case VideoBackend.OPENCV:
            from opencv_video import VideoReader
            return VideoReader(
                video_path=video_path,
                iter_start_frame=iter_start_frame,
                freq=freq,
            )
        case VideoBackend.TORCHCODEC:
            from .readers.torchcodec import TorchCodecVideoReader
            return TorchCodecVideoReader(
                video_path=video_path,
                iter_start_frame=iter_start_frame,
                freq=freq,
            )
