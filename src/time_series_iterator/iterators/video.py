
from __future__ import annotations

import numpy as np
from types import TracebackType

from id_manager import IDManager
from opencv_video import VideoReader
from ..iterator import TimeSeriesIterator
from ..parameters import TimeSeriesIterationParameters
from ..utils import MediaType

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
        params: TimeSeriesIterationParameters
        ) -> None:
        """
        Initialize the VideoIterator.

        Parameters:
        ----------
        paths: list[str]
            The paths to the video files.
        params: TimeSeriesIterationParameters
            The parameters for the iteration of the time series data.
        """
        super().__init__(
            params=params, 
            paths=paths
            )
        # manager for the file index of the video files.
        self.file_id_manager = IDManager(
            current_id=self.params.start_video_file_index,
            step=1,
            )
        # manager for the frame index of the video files.
        self.video_reader: VideoReader | None = None
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
            video_reader = VideoReader(
                video_path=path, 
                iter_start_frame=0
                )
            total_frame = int(video_reader.total_frame)
            end_frame_ids.append(total_frame)
            video_reader.release()
        return end_frame_ids

    def _next_data(self) -> np.ndarray | None:
        """
        Get the next data from the video iterator.

        Returns:
        ----------
        np.ndarray | None: The next data from the video iterator.

        Raises:
        ----------
        StopIteration: If the end of the video is reached.
        """
        while True:
            if self.video_reader is None or self.video_reader.is_reach_end_of_video:
                file_index = self.file_id_manager.next_id
                if file_index >= len(self.paths):
                    return None

                if self.video_reader is not None:
                    self.video_reader.release()

                self.video_reader = VideoReader(
                    video_path=self.paths[file_index], 
                    iter_start_frame=self.start_frame_index, 
                    freq=self.params.sampling_freq
                    )

                # update the start index of the video reader to ensure that the reading the frame of next video file is correct.
                self._update_start_index()

            frame = next(self.video_reader, None)
            if frame is not None:
                return frame

    def _update_start_index(self) -> None:
        """
        Update the start index of the video reader to ensure that the reading the frame of next video file is correct.
        """
        if self.video_reader is None:
            return
        remaining = (self.video_reader.total_frame - self.start_frame_index) % self.params.sampling_freq
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
        return self.end_frame_id + self.params.index_base.value - 1

    def get_image(self, time_id: int) -> np.ndarray:
        """
        Get the frame from the video.

        Parameters:
        ----------
        frame_id: int
            The frame id of the video.

        Returns:
        ----------
        np.ndarray: The frame from the video.

        Raises:
        ----------
        ValueError: If the frame id is out of range.
        """
        if time_id > self.end_frame_id:
            raise ValueError(f"FrameID: {time_id} is out of range")
        
        video_file_index = 0
        end_frame_id = 0 # sum of the frame ids of the previous video files.
        for tmp_video_file_index, tmp_end_frame_id in enumerate(self._cumulative_end_frame_ids):
            if time_id > tmp_end_frame_id:
                video_file_index = tmp_video_file_index + 1
                end_frame_id = self._cumulative_end_frame_ids[tmp_video_file_index]
                break
            
        video_reader = VideoReader(
            video_path=self.paths[video_file_index],
            iter_start_frame=0
            )
        frame = video_reader.extract_frame(frame_number=(time_id - end_frame_id))
        video_reader.release()
        return frame

    def __str__(self) -> str:
        return f"VideoIterator(paths[0]={self.paths[0]}, params={self.params})"

    @property
    def fps(self) -> float:
        return self.params.raw_sampling_rate