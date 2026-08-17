import cv2
from types import TracebackType

from id_manager import IDManager
from ..iterator import TimeSeriesIterator
from ..parameters import TimeSeriesIterationParameters
from ..types import NumericArray
from ..utils import MediaType

class ImageIterator(TimeSeriesIterator):
    """
    Iterator for image data.

    Attributes:
    ----------
    paths: list[str]
        The paths to the images.
    params: TimeSeriesIterationParameters
        The parameters for the image iterator.
    """
    def __init__(
        self, 
        paths: list[str], 
        params: TimeSeriesIterationParameters
        ):
        super().__init__(
            params=params, 
            paths=paths
            )

        self.file_id_manager = IDManager(
            current_id=self.params.offset_start_id,
            step=self.params.sampling_freq,
            )

    def _next_data(self) -> NumericArray | None:
        """
        Get the next data from the image iterator.

        Returns:
        ----------
        NumericArray | None: The next data from the image iterator.
        """
        index = self.file_id_manager.next_id
        if index >= len(self.paths):
            return None
        return cv2.imread(self.paths[index])

    def _skip_data(self) -> bool:
        """
        Advance past one image without reading it from disk.

        Returns:
        ----------
        bool: Whether an image remained to skip.
        """
        index = self.file_id_manager.next_id
        return index < len(self.paths)

    def close(self):
        pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def media_type(self) -> MediaType:
        return MediaType.IMAGE

    def get_image(self, frame_id: int) -> NumericArray:
        """
        Get the image from the image iterator.

        Parameters:
        ----------
        frame_id: int
            The frame id of the image.

        Returns:
        ----------
        NumericArray: The image from the image iterator.

        Raises:
        ----------
        ValueError: If the frame id is out of range.
        """
        if frame_id < 0:
            raise ValueError(f"Frame id must be greater than or equal to 0, given: {frame_id}")
        if frame_id >= len(self.paths) + self.params.index_base.value:
            raise ValueError(f"Frame id is out of range, given: {frame_id}, max: {len(self.paths) + self.params.index_base.value - 1}")
        target_index = frame_id-self.params.index_base.value
        image = cv2.imread(self.paths[target_index])
        if image is None:
            raise ValueError(f"Failed to load image from path: {self.paths[target_index]}")
        return image

    def __str__(self) -> str:
        return f"ImageIterator(paths[0]={self.paths[0]}, params={self.params})"

    @property
    def fps(self) -> float:
        return self.params.raw_sampling_rate

    @property
    def end_time_id(self) -> int:
        return (len(self.paths)-1)*self.params.pre_sampled_freq*self.params.sampling_freq + self.params.index_base.value