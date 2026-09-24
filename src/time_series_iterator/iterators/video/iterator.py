from __future__ import annotations

import numpy as np
from types import TracebackType

from id_manager import IDManager
from video_handler import VideoFrame, VideoFrameReader, VideoReadParameters
from .parameters import VideoIterationParameters
from .frame_location import VideoFrameLocation
from ...iterator import TimeSeriesIterator
from ...types import NumericArray
from ...utils import MediaType

class VideoIterator(TimeSeriesIterator):
    """
    Iterator for videos saved in a directory.

    Attributes:
    ----------
    file_id_manager: IDManager
        The manager for the file index of the video files.
    start_frame_index: int
        The start frame index of the video files.
    _end_frame_ids: list[int]
        The end frame ids of the video files.
    _cumulative_end_frame_ids: list[int]
        The cumulative end frame ids of the video files.
    """
    def __init__(
        self,
        paths: list[str],
        params: VideoIterationParameters
        ) -> None:
        """
        Initialize the VideoIterator.

        Parameters:
        ----------
        paths: list[str]
            The paths to the video files.
        params: VideoIterationParameters
            The parameters for the iteration of the time series data.
        """
        super().__init__(
            params=params,
            paths=paths
            )
        self.params: VideoIterationParameters = params
        # manager for the file index of the video files.
        self.file_id_manager = IDManager(
            current_id=self.params.start_video_file_index,
            step=1,
            )
        # manager for the frame index of the video files.
        self._reader_factory = self.params.reader_factory
        self.video_reader: VideoFrameReader[VideoFrame] | None = None
        self.start_frame_index = self.params.offset_start_id
        self._end_frame_ids: list[int] = self._get_end_frame_ids()
        self._cumulative_end_frame_ids: list[int] = [int(value) for value in np.cumsum(self._end_frame_ids)]

    def _get_end_frame_ids(self) -> list[int]:
        """
        Get the end frame ids of the video files to access the each frame of the video files.

        Returns:
        ----------
        list[int]: The end frame ids of the video files.
        """
        end_frame_ids: list[int] = []
        for path in self.paths:
            with self._reader_factory.build(path) as video_reader:
                end_frame_ids.append(video_reader.metadata.frame_count)
        return end_frame_ids

    def _open_next_reader(self) -> bool:
        """
        Replace the current video reader with one for the next video file.

        Returns:
        ----------
        bool: Whether a video file remained to open.
        """
        file_index = self.file_id_manager.next_id
        if file_index >= len(self.paths):
            return False

        if self.video_reader is not None:
            self.video_reader.release()

        self.video_reader = self._reader_factory.build(
            self.paths[file_index],
            VideoReadParameters(
                start_frame=self.start_frame_index,
                frame_step=self.params.sampling_freq,
                ),
            )

        # update the start index of the video reader to ensure that the reading the frame of next video file is correct.
        self._update_start_index()
        return True

    def _next_data(self) -> NumericArray | None:
        """
        Get the next data from the video iterator.

        Returns:
        ----------
        NumericArray | None: The next data from the video iterator.

        Raises:
        ----------
        StopIteration: If the end of the video is reached.
        """
        while True:
            if self.video_reader is None or self.video_reader.is_exhausted:
                if not self._open_next_reader():
                    return None
                continue

            frame = next(self.video_reader, None)
            if frame is not None:
                return frame

    def _skip_data(self) -> bool:
        """
        Advance one frame without decoding it.

        Mirrors `_next_data`'s file-rollover loop exactly, but asks the reader
        to skip the frame instead of decoding it.

        Returns:
        ----------
        bool: Whether a frame remained to skip.
        """
        while True:
            if self.video_reader is None or self.video_reader.is_exhausted:
                if not self._open_next_reader():
                    return False
                continue

            try:
                self.video_reader.skip()
            except StopIteration:
                continue
            return True

    def _update_start_index(self) -> None:
        """
        Update the start index of the video reader to ensure that the reading the frame of next video file is correct.
        """
        if self.video_reader is None:
            return
        remaining = (self.video_reader.metadata.frame_count - self.start_frame_index) % self.params.sampling_freq
        self.start_frame_index = (self.params.sampling_freq - remaining) % self.params.sampling_freq

    def close(self) -> None:
        if self.video_reader is not None:
            self.video_reader.release()
            self.video_reader = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> VideoIterator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        return self.end_frame_id

    @property
    def media_type(self) -> MediaType:
        return MediaType.VIDEO

    @property
    def end_frame_id(self) -> int:
        return self._cumulative_end_frame_ids[-1]

    @property
    def end_time_id(self) -> int:
        return (self.end_frame_id - 1)*self.params.pre_sampled_freq + self.params.index_base.value

    def _locate_frame(self, frame_index: int) -> VideoFrameLocation:
        """
        Resolve a frame index spanning every video file into one file's frame.

        Parameters:
        ----------
        frame_index: int
            Zero-based index counted across the video files in order.

        Returns:
        ----------
        VideoFrameLocation: The file holding the frame, and its index there.

        Raises:
        ----------
        ValueError: If the index lies past the last frame of the last file.
        """
        previous_end_frame_id = 0
        for file_index, end_frame_id in enumerate(self._cumulative_end_frame_ids):
            if frame_index < end_frame_id:
                return VideoFrameLocation(
                    file_index=file_index,
                    frame_index=frame_index - previous_end_frame_id,
                    )
            previous_end_frame_id = end_frame_id
        raise ValueError(
            f"Frame index is out of range, given: {frame_index}, "
            + f"max: {self.end_frame_id - 1}"
            )

    def get_image(self, time_id: int) -> NumericArray:
        """
        Read the frame at an arbitrary time id, without advancing iteration.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        ----------
        NumericArray: The frame stored at `time_id`.

        Raises:
        ----------
        ValueError: If the time id addresses no stored frame.
        """
        location = self._locate_frame(self.media_index_of(time_id))
        with self._reader_factory.build(self.paths[location.file_index]) as video_reader:
            return video_reader.read_frame_at(location.frame_index)

    def __str__(self) -> str:
        return f"VideoIterator(paths[0]={self.paths[0]}, params={self.params})"

    @property
    def fps(self) -> float:
        return self.params.raw_sampling_rate
