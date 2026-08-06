from typing import Protocol, runtime_checkable

import torch
from opencv_video import BGRFrame

type VideoFrame = BGRFrame | torch.Tensor

@runtime_checkable
class VideoFrameReader(Protocol):
    """
    Protocol for a backend that reads frames sequentially from one video file.

    Every backend `VideoIterator` can use -- OpenCV, torchcodec, or any other --
    implements this surface, so `VideoIterator` itself never depends on which one
    is behind it. A backend that decodes straight to a GPU tensor (torchcodec)
    yields `torch.Tensor`; one that decodes through OpenCV yields `BGRFrame`.
    """

    @property
    def total_frame(self) -> int:
        """
        Return the total number of frames in the video.

        Returns:
        ----------
        int: Total number of frames.
        """
        ...

    @property
    def is_reach_end_of_video(self) -> bool:
        """
        Return whether every frame has already been read.

        Returns:
        ----------
        bool: True once the reader has no more frames to yield.
        """
        ...

    def __next__(self) -> VideoFrame:
        """
        Return the next frame and advance the reader's position.

        Returns:
        ----------
        VideoFrame: The next frame.

        Raises:
        ----------
        StopIteration: If the reader has reached the end of the video.
        """
        ...

    def extract_frame(self, frame_number: int) -> VideoFrame:
        """
        Return one frame at an arbitrary index, without advancing iteration.

        Parameters:
        ----------
        frame_number: int
            Frame index to read.

        Returns:
        ----------
        VideoFrame: The frame at `frame_number`.
        """
        ...

    def release(self) -> None:
        """
        Release the resources the reader holds.
        """
        ...
