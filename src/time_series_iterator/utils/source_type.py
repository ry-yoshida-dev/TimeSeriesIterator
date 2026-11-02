from enum import Enum

class SourceType(Enum):
    """
    Kind of source a time series iterator reads its samples from.

    Attributes:
    ----------
    IMAGE: One image file per sample.
    VIDEO: Video files holding one frame per sample.
    SIGNAL: Numeric arrays or tables holding one value or vector per sample.
    """
    IMAGE = "image"
    VIDEO = "video"
    SIGNAL = "signal"
