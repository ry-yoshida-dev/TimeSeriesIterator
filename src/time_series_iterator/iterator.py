from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from types import TracebackType
from typing import Self
from id_manager import IDManager
from progress_bar import ProgressBarBackend, ProgressBarReporter

from .parameters import TimeSeriesIterationParameters
from .types import NumericArray
from .utils import SourceType

class TimeSeriesIterator(ABC):
    """
    Base class for time series data iterator.

    Attributes:
    ----------
    params: TimeSeriesIterationParameters
        The parameters for the time series iterator.
    time_id_manager: IDManager
        The manager for the time id.
    """
    def __init__(
        self,
        params: TimeSeriesIterationParameters,
        ):
        """
        Initialize the TimeSeriesIterator.

        Parameters:
        ----------
        params: TimeSeriesIterationParameters
            The parameters for the time series iterator.
        """

        self.params = params

        self.time_id_manager = IDManager(
            current_id=self.params.start_time_id,
            step=self.params.sampling_freq * self.params.pre_sampled_freq,
            )

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
        `len(self)` is the number of stored samples instead, and ignores
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

    def sample_index_of(self, time_id: int) -> int:
        """
        Convert a time id into an index into the stored samples.

        One sample is stored per `pre_sampled_freq` time ids, so only the ids
        on the grid `index_base`, `index_base + pre_sampled_freq`, ... address
        a sample. `sampling_freq` is the stride of iteration rather than a
        property of the stored data, so an id this iterator's loop skips still
        converts.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        -------
        int: Zero-based index into the stored samples.

        Raises:
        -------
        ValueError: If the time id is outside the stored samples or off the
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
    def get_sample(self, time_id: int) -> NumericArray:
        """
        Read one sample at an arbitrary time id, without advancing iteration.

        `sampling_freq` does not restrict it: a time id the loop steps over
        still resolves, since sampling only skips stored samples.

        Parameters:
        ----------
        time_id: int
            A time id as yielded by `__next__`.

        Returns:
        -------
        NumericArray: The sample stored at `time_id`.

        Raises:
        -------
        ValueError: If the time id addresses no stored sample or lies off the
            pre-sampled grid.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the resources held by the iterator."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @abstractmethod
    def _next_data(self) -> NumericArray | None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
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
        source_type: SourceType,
        paths: list[str],
        parameters: TimeSeriesIterationParameters | None = None
        ) -> TimeSeriesIterator:
        """
        Build the time series iterator based on the source type.

        Parameters:
        ----------
        source_type: SourceType
            The source type of the time series data.
        paths: list[str]
            The paths to the time series data.
        parameters: TimeSeriesIterationParameters | None
            The parameters for the time series iterator.

        Returns:
        ----------
        TimeSeriesIterator: The time series iterator.

        Raises:
        ----------
        ValueError: If the source type is not supported, or a signal file's suffix is unknown.
        TypeError: If source_type is SourceType.VIDEO and parameters is not a VideoIterationParameters.
        """
        match source_type:
            case SourceType.IMAGE:
                from .iterators.image import ImageIterator
                if parameters is None:
                    parameters = TimeSeriesIterationParameters()
                return ImageIterator(paths=paths, params=parameters)
            case SourceType.VIDEO:
                from .iterators.video import VideoIterator
                from .iterators.video.parameters import VideoIterationParameters
                if parameters is None:
                    parameters = VideoIterationParameters()
                elif not isinstance(parameters, VideoIterationParameters):
                    raise TypeError(
                        "SourceType.VIDEO requires VideoIterationParameters, "
                        f"got {type(parameters).__name__}"
                        )
                return VideoIterator(paths=paths, params=parameters)
            case SourceType.SIGNAL:
                from .iterators.signal import SignalIterator
                if parameters is None:
                    parameters = TimeSeriesIterationParameters()
                return SignalIterator.from_paths(paths=paths, params=parameters)
            case _:
                raise ValueError(f"Unsupported source type: {source_type}")
