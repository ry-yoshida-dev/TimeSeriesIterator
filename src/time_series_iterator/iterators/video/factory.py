from torch_modules import Device

from .backend import VideoBackend
from .reader import VideoFrameReader

def build_video_reader(
    backend: VideoBackend,
    video_path: str,
    iter_start_frame: int,
    freq: int,
    device: Device,
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
    device: Device
        Device the decode runs on, which `VideoBackend.TORCHCODEC` resolves
        against what this machine offers. Ignored by `VideoBackend.OPENCV`,
        which decodes on the CPU whatever this says.

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
                device=device,
            )
