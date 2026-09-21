from dataclasses import dataclass

@dataclass(frozen=True)
class VideoFrameLocation:
    """
    Position of one frame within a scene's video files.

    Attributes:
    ----------
    file_index: int
        Index into the iterator's paths of the file holding the frame.
    frame_index: int
        Zero-based index of the frame within that file.
    """
    file_index: int
    frame_index: int
