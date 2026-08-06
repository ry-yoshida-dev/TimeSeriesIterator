import torch
from torchcodec.decoders import VideoDecoder

class TorchCodecVideoReader:
    """
    Frame reader backed by `torchcodec`, decoding on a CUDA device.

    `torchcodec` hands frames back as an RGB `(3, H, W)` uint8 tensor on the
    decode device, and this reader returns that tensor as-is -- no channel
    flip, no device transfer, no NumPy conversion -- so a frame this reader
    yields never leaves the GPU on its way to `image_container.TensorImageContainer`.
    Decoding happens on the GPU's NVDEC hardware rather than the CPU, which is
    the point of this backend: choose `VideoBackend.OPENCV` instead for
    CPU-only decoding, which yields `BGRFrame` (NumPy) rather than a tensor.

    Attributes:
    ----------
    total_frame: int
        Total number of frames in the video.
    """
    def __init__(
        self,
        video_path: str,
        iter_start_frame: int = 0,
        freq: int = 1,
    ) -> None:
        """
        Initialize the TorchCodecVideoReader.

        Parameters:
        ----------
        video_path: str
            Path to the video file.
        iter_start_frame: int
            Frame index to start reading from.
        freq: int
            Step size between yielded frames.
        """
        self._decoder: VideoDecoder = VideoDecoder(video_path, device="cuda")
        self._freq: int = freq
        self._next_frame_id: int = iter_start_frame
        num_frames = self._decoder.metadata.num_frames
        if num_frames is None:
            raise RuntimeError(
                f"torchcodec could not determine the frame count for '{video_path}'."
            )
        self.total_frame: int = num_frames

    @property
    def is_reach_end_of_video(self) -> bool:
        """
        Return whether every frame has already been read.

        Returns:
        ----------
        bool: True once the next frame index is past the last frame.
        """
        return self._next_frame_id >= self.total_frame

    def __next__(self) -> torch.Tensor:
        """
        Return the next frame and advance the reader's position by `freq`.

        Returns:
        ----------
        torch.Tensor: The frame at the current position, shape (3, H, W),
            uint8, RGB, on the CUDA device.

        Raises:
        ----------
        StopIteration: If the current position is past the last frame.
        """
        if self.is_reach_end_of_video:
            raise StopIteration
        frame = self.extract_frame(frame_number=self._next_frame_id)
        self._next_frame_id += self._freq
        return frame

    def extract_frame(self, frame_number: int) -> torch.Tensor:
        """
        Return one frame at an arbitrary index, without advancing iteration.

        Parameters:
        ----------
        frame_number: int
            Frame index to read.

        Returns:
        ----------
        torch.Tensor: The frame at `frame_number`, shape (3, H, W), uint8,
            RGB, on the CUDA device.
        """
        return self._decoder[frame_number].data

    def release(self) -> None:
        """
        Release the decoder.

        `torchcodec` exposes no explicit close; dropping the reference lets the
        garbage collector free the decode context.
        """
        del self._decoder
