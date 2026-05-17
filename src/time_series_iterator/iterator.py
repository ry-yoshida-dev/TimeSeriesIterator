from __future__ import annotations
import os
from typing import Any

import numpy as np
from abc import ABC, abstractmethod
from id_manager import IDManager
from tqdm import tqdm

from .parameters import TimeSeriesIterationParameters
from .utils import MediaType

class TimeSeriesIterator(ABC):
    """
    Base class for time series data iterator.

    Attributes:
    ----------
    params: TimeSeriesIterationParameters
        The parameters for the time series iterator.
    paths: list[str]
        The paths to the time series data.
    time_id_manager: IDManager
        The manager for the time id.
    """
    def __init__(
        self, 
        params: TimeSeriesIterationParameters, 
        paths: list[str]
        ):
        """
        Initialize the TimeSeriesIterator.

        Parameters:
        ----------
        params: TimeSeriesIterationParameters
            The parameters for the time series iterator.
        paths: list[str]
            The paths to the time series data.
        """

        self.params = params
        self._validate_paths(paths)
        self.paths = paths

        self.time_id_manager = IDManager(
            current_id=self.params.start_time_id,
            step=self.params.sampling_freq * self.params.pre_sampled_freq,
            )

    def _validate_paths(
        self, 
        paths: list[str]
        ) -> None:
        """
        Validate the paths.

        Parameters:
        ----------
        paths: list[str]
            The paths to the time series data.
        
        Raises:
        -------
        ValueError: If the paths is empty.
        FileNotFoundError: If the file is not found.
        """
        if len(paths) == 0:
            raise ValueError("Paths is empty")
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
 
    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        """
        Get the next data from the time series iterator.

        Returns:
        -------
        tuple[int, np.ndarray]
            The next data from the time series iterator.
            int: The frame id of the next data.
            np.ndarray: The next data.
        
        Raises:
        -------
        StopIteration: If the end of the time series data is reached.
        """
        data = self._next_data()

        if data is None:
            raise StopIteration

        self.time_id = self.time_id_manager.next_id

        if self.params.is_exceeded_end_time_id(self.time_id):
            raise StopIteration

        return self.time_id, data

    def with_tqdm(self, *, total: int | None = None, **tqdm_kwargs: Any) -> tqdm[tuple[int, np.ndarray]]:
        """
        Wrap this iterator with tqdm for use in a for loop.

        Parameters
        ----------
        total:
            Bar length. Defaults to len(self).
        **tqdm_kwargs:
            Forwarded to tqdm (e.g. desc, unit, leave).

        Example
        -------
        >>> for time_id, data in iterator.with_tqdm(desc="frames"):
        ...     ...
        """
        if total is None:
            total = len(self)
        return tqdm(self, total=total, **tqdm_kwargs)

    @abstractmethod
    def _next_data(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def media_type(self) -> MediaType:
        pass

    @property
    @abstractmethod
    def end_time_id(self) -> int:
        """
        Get the end time id of the time series iterator.

        Returns:
        -------
        int: The end time id of the time series iterator.
        """

    @classmethod
    def build(
        cls,
        media_type: MediaType,
        paths: list[str],
        parameters: TimeSeriesIterationParameters | None = None
        ) -> TimeSeriesIterator:
        """
        Build the time series iterator based on the media type.

        Parameters:
        ----------
        media_type: MediaType
            The media type of the time series data.
        paths: list[str]
            The paths to the time series data.
        parameters: TimeSeriesIterationParameters | None
            The parameters for the time series iterator.

        Returns:
        ----------
        TimeSeriesIterator: The time series iterator.

        Raises:
        ----------
        ValueError: If the media type is not supported.
        """
        if parameters is None:
            parameters = TimeSeriesIterationParameters()

        match media_type:
            case MediaType.IMAGE:
                from .iterators.image import ImageIterator
                return ImageIterator(paths=paths, params=parameters)
            case MediaType.VIDEO:
                from .iterators.video import VideoIterator
                return VideoIterator(paths=paths, params=parameters)
            case _:
                raise ValueError(f"Unsupported media type: {media_type}")
