"""
Time series iterator module for handling image, video and numeric signal iteration.

This module provides iterators for time series data including images, videos and
numeric signals, with support for configurable parameters and different indexing schemes.
"""

from .iterator import TimeSeriesIterator
from .iterators import (
    ArraySignalSource,
    ConcatenatedSignalSource,
    DelimitedTextSignalSource,
    ImageIterator,
    NumpyFileSignalSource,
    SignalFileFormat,
    SignalIterator,
    SignalSource,
    VideoBackend,
    VideoIterationParameters,
    VideoIterator,
)
from .parameters import TimeSeriesIterationParameters
from .utils import IndexBase, SourceType

__all__ = [
    "TimeSeriesIterator",
    "TimeSeriesIterationParameters",
    "ArraySignalSource",
    "ConcatenatedSignalSource",
    "DelimitedTextSignalSource",
    "IndexBase",
    "ImageIterator",
    "NumpyFileSignalSource",
    "SignalFileFormat",
    "SignalIterator",
    "SignalSource",
    "VideoBackend",
    "VideoIterationParameters",
    "VideoIterator",
    "SourceType",
]
