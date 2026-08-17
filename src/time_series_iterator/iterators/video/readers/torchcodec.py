import warnings

import torch
from torch_modules import Device
from torchcodec.decoders import VideoDecoder

DEFAULT_DECODE_RUN_LENGTH: int = 16
"""Frames one decode call reads ahead for, when iterating."""


class TorchCodecVideoReader:
    """
    Frame reader backed by `torchcodec`, decoding on a chosen torch device.

    `torchcodec` hands frames back as an RGB `(3, H, W)` uint8 tensor on the
    decode device, and this reader returns that tensor as-is -- no channel
    flip, no device transfer, no NumPy conversion -- so a frame this reader
    yields never leaves that device on its way to
    `image_container.TensorImageContainer`.

    On a CUDA device, decoding runs on the GPU's NVDEC hardware rather than on
    the CPU, which is the point of this backend. `torchcodec` also decodes in
    software on `Device.CPU` though, and still yields tensors there, so this
    reader runs on any machine: a device it cannot decode on resolves down to
    `Device.CPU` rather than failing, and `device` reports the one in use.
    `VideoBackend.OPENCV` is the other CPU-only option, but it yields
    `BGRFrame` (NumPy) rather than a tensor.

    Iteration decodes a run of consecutive frames per call and serves them one
    at a time. Asking for one frame at a time is a seek and a decode call each,
    which for a reader walking a video start to finish pays a per-frame fixed
    cost for a stream the decoder would otherwise read straight through; a run
    amortizes it. A run is a decoded buffer held on the decode device, so
    `decode_run_length` trades that cost against device memory.

    Attributes:
    ----------
    total_frame: int
        Total number of frames in the video.
    device: Device
        Device the decode actually runs on and the frames come back on, which
        is the requested one resolved against what this machine offers.
    decode_run_length: int
        Frames one decode call reads ahead for while iterating.
    """
    def __init__(
        self,
        video_path: str,
        device: Device,
        iter_start_frame: int = 0,
        freq: int = 1,
        decode_run_length: int = DEFAULT_DECODE_RUN_LENGTH,
    ) -> None:
        """
        Initialize the TorchCodecVideoReader.

        Parameters:
        ----------
        video_path: str
            Path to the video file.
        device: Device
            Device to decode on, resolved down to `Device.CPU` when this machine
            cannot decode there.
        iter_start_frame: int
            Frame index to start reading from.
        freq: int
            Step size between yielded frames.
        decode_run_length: int
            Frames one decode call reads ahead for while iterating. One decodes
            frame by frame, as an arbitrary-index read does.

        Raises:
        ----------
        ValueError: If `decode_run_length` is not positive.
        RuntimeError: If the frame count cannot be determined.
        """
        if decode_run_length < 1:
            raise ValueError(
                f"decode_run_length must be at least 1, got {decode_run_length}"
            )
        self.device: Device = self._resolve_device(device=device)
        self._decoder: VideoDecoder = VideoDecoder(video_path, device=self.device.torch_device)
        self._freq: int = freq
        self._next_frame_id: int = iter_start_frame
        self.decode_run_length: int = decode_run_length
        self._run: torch.Tensor | None = None
        self._run_start_frame_id: int = 0
        num_frames = self._decoder.metadata.num_frames
        if num_frames is None:
            raise RuntimeError(
                f"torchcodec could not determine the frame count for '{video_path}'."
            )
        self.total_frame: int = num_frames

    @staticmethod
    def _resolve_device(device: Device) -> Device:
        """
        Return the device `torchcodec` can actually decode on here.

        Resolved before the decoder is constructed rather than left to
        `torchcodec`, which reaches `torch` through the stable ABI and reports a
        failed allocation on an absent CUDA device as an opaque
        `torch_call_dispatcher` error naming neither the device nor the reason.

        Parameters:
        ----------
        device: Device
            Device the decode was asked to run on.

        Returns:
        ----------
        Device: `device` itself when `torchcodec` decodes there on this machine,
            and `Device.CPU` otherwise -- for a CUDA request torch reports no
            device for, and for `Device.MPS`, which `torchcodec` has no decode
            path for at all.
        """
        match device:
            case Device.CPU:
                return device
            case Device.CUDA if torch.cuda.is_available():
                return device
            case _:
                warnings.warn(
                    f"torchcodec cannot decode on {device} here, falling back to {Device.CPU}. "
                    + "Decoding runs in software instead of on NVDEC, which is slower.",
                    stacklevel=3,
                )
                return Device.CPU

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
            uint8, RGB, on the decode device.

        Raises:
        ----------
        StopIteration: If the current position is past the last frame.
        """
        if self.is_reach_end_of_video:
            raise StopIteration
        frame = self._from_run(frame_number=self._next_frame_id)
        self._next_frame_id += self._freq
        return frame

    def _from_run(self, frame_number: int) -> torch.Tensor:
        """
        Return one frame out of the current decoded run, decoding a new one first if needed.

        Parameters:
        ----------
        frame_number: int
            Frame index to read, which iteration reaches in order.

        Returns:
        ----------
        torch.Tensor: The frame at `frame_number`, shape (3, H, W), uint8, RGB,
            on the decode device, as a view into the run's buffer.
        """
        run = self._run
        offset = (frame_number - self._run_start_frame_id) // self._freq
        is_held = (
            run is not None
            and 0 <= offset < len(run)
            and (frame_number - self._run_start_frame_id) % self._freq == 0
        )
        if run is None or not is_held:
            run = self._decode_run(start_frame_id=frame_number)
            offset = 0
        return run[offset]

    def _decode_run(self, start_frame_id: int) -> torch.Tensor:
        """
        Decode the run of frames beginning at `start_frame_id`, and hold it.

        Parameters:
        ----------
        start_frame_id: int
            First frame index of the run.

        Returns:
        ----------
        torch.Tensor: The run, shape (n, 3, H, W), uint8, RGB, on the decode
            device, where n is at most `decode_run_length` frames of `freq`
            apart and never reaches past the end of the video.
        """
        stop = min(
            start_frame_id + self._freq * self.decode_run_length, self.total_frame
        )
        run = self._decoder.get_frames_in_range(
            start=start_frame_id, stop=stop, step=self._freq
        ).data
        self._run = run
        self._run_start_frame_id = start_frame_id
        return run

    def skip(self) -> None:
        """
        Advance past the current frame without decoding it.

        Only moves `_next_frame_id`, so a decode requested later still starts
        a fresh run from wherever iteration then stands -- the held run's
        buffer never contains a skipped frame's pixels.

        Raises:
        ----------
        StopIteration: If the current position is past the last frame.
        """
        if self.is_reach_end_of_video:
            raise StopIteration
        self._next_frame_id += self._freq

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
            RGB, on the decode device.
        """
        return self._decoder[frame_number].data

    def release(self) -> None:
        """
        Release the decoder and the run it decoded ahead.

        `torchcodec` exposes no explicit close; dropping the reference lets the
        garbage collector free the decode context.
        """
        self._run = None
        del self._decoder
