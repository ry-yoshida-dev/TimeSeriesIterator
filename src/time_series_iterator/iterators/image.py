from __future__ import annotations

from typing import Self, override

import cv2
from id_manager import IDManager

from ..parameters import TimeSeriesIterationParameters
from ..types import NumericArray
from ..utils import SourceType
from .file import FileTimeSeriesIterator


class ImageIterator(FileTimeSeriesIterator, source_type=SourceType.IMAGE):
    """
    Iterator for image data.

    Attributes
    ----------
    paths: list[str]
        The paths to the images.
    params: TimeSeriesIterationParameters
        The parameters for the image iterator.
    """

    def __init__(self, paths: list[str], params: TimeSeriesIterationParameters):
        super().__init__(params=params, paths=paths)

        self.file_id_manager: IDManager = IDManager(
            current_id=self.params.offset_start_id,
            step=self.params.sampling_freq,
        )

    @classmethod
    @override
    def from_paths(
        cls,
        paths: list[str],
        params: TimeSeriesIterationParameters | None = None,
    ) -> Self:
        """
        Build an iterator reading one image file per sample, in order.

        Parameters
        ----------
        paths: list[str]
            The paths to the images.
        params: TimeSeriesIterationParameters | None
            The parameters for the image iterator, or None for the defaults.

        Returns
        -------
        Self: The iterator.
        """
        return cls(paths=paths, params=params or TimeSeriesIterationParameters())

    @override
    def _next_data(self) -> NumericArray | None:
        """
        Get the next data from the image iterator.

        Returns
        -------
        NumericArray | None: The next data from the image iterator.
        """
        index = self.file_id_manager.next_id
        if index >= len(self.paths):
            return None
        return cv2.imread(self.paths[index])

    @override
    def _skip_data(self) -> bool:
        """
        Advance past one image without reading it from disk.

        Returns
        -------
        bool: Whether an image remained to skip.
        """
        index = self.file_id_manager.next_id
        return index < len(self.paths)

    @override
    def close(self) -> None:
        """Release resources. Image files are opened per read, so there is nothing to release."""

    def __del__(self) -> None:
        self.close()

    @override
    def __len__(self) -> int:
        return len(self.paths)

    @override
    def get_sample(self, time_id: int) -> NumericArray:
        """
        Read the image at an arbitrary time id, without advancing iteration.

        Parameters
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns
        -------
        NumericArray: The image stored at `time_id`.

        Raises
        ------
        ValueError: If the time id addresses no stored image, or the image
            cannot be read.
        """
        path = self.paths[self.sample_index_of(time_id)]
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {path}")
        return image

    @override
    def __str__(self) -> str:
        return f"ImageIterator(paths[0]={self.paths[0]}, params={self.params})"

    @property
    def fps(self) -> float:
        """Frame rate of the image sequence, from ``raw_sampling_rate``."""
        return self.params.raw_sampling_rate

    @property
    @override
    def end_time_id(self) -> int:
        return (
            len(self.paths) - 1
        ) * self.params.pre_sampled_freq + self.params.index_base.value
