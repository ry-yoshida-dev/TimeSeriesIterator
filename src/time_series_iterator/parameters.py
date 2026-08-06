from dataclasses import dataclass

from .utils import IndexBase

@dataclass
class TimeSeriesIterationParameters:
    """
    Parameters for the time series iterator.
    * start_time_id and end_time_id are counted from the index_base.
    -> If index_base is IndexBase.One, start_time_id and end_time_id should be counted from 1.

    Parameters:
    ----------
    pre_sampled_freq: int
        The sampling frequency of the pre-sampled time series data.
    sampling_freq: int
        The sampling frequency from input data of the time series data.
    raw_sampling_rate: float
        The sampling rate of the raw time series data (e.g. fps).
    index_base: IndexBase
        The base index of the time series data.
    start_time_id: int
        The start time id of the time series data.
    end_time_id: int
        The end time id of the time series data.
        * -1 indicates iteration until the last time.
    """
    pre_sampled_freq: int = 1
    sampling_freq: int = 1
    raw_sampling_rate: float | int = 30
    index_base: IndexBase = IndexBase.ONE
    start_time_id: int = 1
    end_time_id: int = -1

    def __post_init__(self):
        self._validate_parameters()

    def _validate_parameters(self):
        if self.start_time_id < self.index_base.value:
            raise ValueError(
                f"start_step_id must be greater than or equal to the index base ({self.index_base.value})"
                )
        if self.end_time_id <= 0 and self.end_time_id != -1:
            raise ValueError(
                "end_time_id must be -1 (iterate until last time) or greater than 0"
                )
        if self.end_time_id != -1 and self.end_time_id < self.start_time_id:
            raise ValueError(
                "end_time_id must be -1 (iterate until last time) or greater than start_time_id"
                )

    @property
    def offset_start_id(self) -> int:
        """
        Start offset for sequence access: ``start_time_id`` minus ``index_base``.

        Returns:
        --------
        int: Zero-based index into the underlying time series sequence.
        """
        return self.start_time_id - self.index_base.value

    @property
    def actual_sampling_freq(self) -> int:
        """
        The sampling frequency of the actual time series data.

        Returns:
        --------
        int: The sampling frequency of the actual time series data.
        """
        return self.sampling_freq*self.pre_sampled_freq

    @property
    def is_set_end_time_id(self) -> bool:
        """
        Check if the end time id is set.
        
        Returns:
        --------
        bool: True if the end time id is set, False otherwise.
        """
        return self.end_time_id != -1

    def is_exceeded_end_time_id(self, time_id: int) -> bool:
        """
        Check if the time id is exceeded the end time id.
        
        Parameters:
        ----------
        time_id: int
            The time id to check.
        """
        if not self.is_set_end_time_id:
            return False
        return time_id > self.end_time_id