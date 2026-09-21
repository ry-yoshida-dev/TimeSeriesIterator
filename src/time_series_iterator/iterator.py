from __future__ import annotations
import os

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from id_manager import IDManager
from progress_bar import ProgressBarBackend, ProgressBarReporter

from .parameters import TimeSeriesIterationParameters
from .types import NumericArray
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
 
    def __iter__(self) -> Iterator[tuple[int, NumericArray]]:
        return self

    def __next__(self) -> tuple[int, NumericArray]:
        """
        Get the next data from the time series iterator.

        Returns:
        -------
        tuple[int, NumericArray]
            The next data from the time series iterator.
            int: The frame id of the next data.
            NumericArray: The next data.

        Raises:
        -------
        StopIteration: If the end of the time series data is reached.
        """
        data = self._next_data()

        if data is None:
            raise StopIteration

        return self._advance_time_id_or_stop(), data

    def skip(self) -> int:
        """
        Advance one step without keeping its data, when the backend allows it.

        Falls back to fully decoding and discarding the step's data when the
        backend exposes no cheaper path, so a caller may always call this
        instead of `next()` and still see the same end-of-stream behavior --
        the id bookkeeping is identical to `__next__`.

        Returns:
        -------
        int: The time id of the step just skipped.

        Raises:
        -------
        StopIteration: If the end of the time series data is reached.
        """
        if not self._skip_data():
            raise StopIteration

        return self._advance_time_id_or_stop()

    def _advance_time_id_or_stop(self) -> int:
        """
        Resolve and record the time id the step just taken corresponds to.

        Returns:
        -------
        int: The time id.

        Raises:
        -------
        StopIteration: If the time id lies past the configured end.
        """
        self.time_id = self.time_id_manager.next_id

        if self.params.is_exceeded_end_time_id(self.time_id):
            raise StopIteration

        return self.time_id

    def _skip_data(self) -> bool:
        """
        Advance one step without keeping its data.

        Default implementation still decodes through `_next_data`, since the
        base class has no backend-specific cheaper path; a subclass whose
        backend can skip more cheaply overrides this.

        Returns:
        -------
        bool: Whether a step remained to skip.
        """
        return self._next_data() is not None

    @property
    def remaining_step_count(self) -> int:
        """
        Number of steps this iterator still yields under its parameters.

        Counted from the next time id the iterator will issue, so a partially
        consumed iterator reports what is left rather than its full size.
        `len(self)` is the size of the underlying media instead, and ignores
        `start_time_id`, `end_time_id` and the sampling frequencies.

        Returns:
        -------
        int: The number of steps left, or 0 once the iterator is exhausted.
        """
        last_time_id = self.end_time_id
        if self.params.is_set_end_time_id:
            last_time_id = min(last_time_id, self.params.end_time_id)

        next_time_id = self.time_id_manager.current_id
        if next_time_id > last_time_id:
            return 0

        return (last_time_id - next_time_id) // self.params.actual_sampling_freq + 1

    def with_progress_bar(
        self,
        *,
        total: int | None = None,
        description: str = "",
        unit: str = "it",
        is_leave_visible: bool = True,
        backend: ProgressBarBackend | None = None,
        is_enabled: bool = True,
        ) -> Generator[tuple[int, NumericArray], None, None]:
        """
        Wrap this iterator with a progress bar for use in a for loop.

        The backend is resolved at runtime by `progress_bar`, so the bar falls
        back to a dependency-free text bar when no progress bar library is
        installed or the output is redirected.

        Parameters:
        ----------
        total: int | None
            Bar length. Defaults to `remaining_step_count`, which is the
            number of pairs the loop actually yields.
        description: str
            Label rendered next to the bar.
        unit: str
            Name of a single step.
        is_leave_visible: bool
            Whether the finished bar stays on screen.
        backend: ProgressBarBackend | None
            Backend to render with, or None to pick the best installed one.
        is_enabled: bool
            Whether the bar is displayed at all.

        Returns:
        -------
        Generator[tuple[int, NumericArray], None, None]: The pairs yielded by
            this iterator, unchanged, with the bar advanced once per pair.

        Notes:
        -----
        The bar is finalized once the returned iterator is exhausted or
        closed. A loop that breaks early while a reference to the iterator
        survives leaves the bar open until that reference is dropped, so
        wrap the iterator in `contextlib.closing` in that case.

        Example
        -------
        >>> for time_id, data in iterator.with_progress_bar(description="frames"):
        ...     ...
        """
        if total is None:
            total = self.remaining_step_count

        reporter = ProgressBarReporter(
            total=total,
            description=description,
            unit=unit,
            is_leave_visible=is_leave_visible,
            backend=backend,
            is_enabled=is_enabled,
            )

        return reporter.report(self)

    def media_index_of(self, time_id: int) -> int:
        """
        Convert a time id into an index into the stored media.

        The stored media holds one element per `pre_sampled_freq` time ids,
        so only the ids on the grid `index_base`, `index_base +
        pre_sampled_freq`, ... address an element. `sampling_freq` is the
        stride of iteration rather than a property of the media, so an id
        this iterator's loop skips still converts.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        -------
        int: Zero-based index into the stored media.

        Raises:
        -------
        ValueError: If the time id is outside the media or off the
            pre-sampled grid.
        """
        offset = time_id - self.params.index_base.value
        if offset < 0 or time_id > self.end_time_id:
            raise ValueError(
                f"time_id must be between {self.params.index_base.value} "
                + f"and {self.end_time_id}, given: {time_id}"
                )
        if offset % self.params.pre_sampled_freq != 0:
            raise ValueError(
                "time_id must fall on the pre-sampled grid: "
                + f"(time_id - {self.params.index_base.value}) must be a multiple "
                + f"of pre_sampled_freq ({self.params.pre_sampled_freq}), "
                + f"given: {time_id}"
                )
        return offset // self.params.pre_sampled_freq

    @abstractmethod
    def get_image(self, time_id: int) -> NumericArray:
        """
        Read one element at an arbitrary time id, without advancing iteration.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        -------
        NumericArray: The element stored at `time_id`.

        Raises:
        -------
        ValueError: If the time id addresses no stored element.
        """

    @abstractmethod
    def _next_data(self) -> NumericArray | None:
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
        TypeError: If media_type is MediaType.VIDEO and parameters is not a VideoIterationParameters.
        """
        match media_type:
            case MediaType.IMAGE:
                from .iterators.image import ImageIterator
                if parameters is None:
                    parameters = TimeSeriesIterationParameters()
                return ImageIterator(paths=paths, params=parameters)
            case MediaType.VIDEO:
                from .iterators.video import VideoIterator
                from .iterators.video.parameters import VideoIterationParameters
                if parameters is None:
                    parameters = VideoIterationParameters()
                elif not isinstance(parameters, VideoIterationParameters):
                    raise TypeError(
                        "MediaType.VIDEO requires VideoIterationParameters, "
                        f"got {type(parameters).__name__}"
                        )
                return VideoIterator(paths=paths, params=parameters)
            case _:
                raise ValueError(f"Unsupported media type: {media_type}")
