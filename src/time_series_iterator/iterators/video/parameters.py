from dataclasses import dataclass, field

from torch_modules import Device
from video_handler import TorchCodecReadOptions, VideoBackend, VideoReaderFactory

from ...parameters import TimeSeriesIterationParameters

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

    @property
    def reader_factory(self) -> VideoReaderFactory:
        """
        Factory building the frame readers these parameters describe.

        Returns:
        --------
        VideoReaderFactory: Factory for `video_backend`, decoding on
            `decode_device` when the backend is `VideoBackend.TORCHCODEC`.
        """
        return VideoReaderFactory(
            backend=self.video_backend,
            torchcodec_options=TorchCodecReadOptions(device=self.decode_device),
            )
