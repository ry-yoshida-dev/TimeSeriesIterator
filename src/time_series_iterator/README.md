# time_series_iterator

## Overview

Package for iterating time-series media data in ascending order with configurable sampling and indexing.

## Components

| Component | Description |
|-----------|-------------|
| [iterator.py](./iterator.py) | Base iterator class and factory method (`TimeSeriesIterator.build`) |
| [parameters.py](./parameters.py) | Dataclass for iteration parameters (sampling rate, start/end IDs, index base) |
| [types.py](./types.py) | Type aliases for numeric arrays yielded by iterators (`NumericArray`) |
| [utils/](./utils/README.md) | Utility enums such as media type and index base |
| [iterators/](./iterators/README.md) | Concrete iterator implementations for image/video inputs |

## Parameters

`TimeSeriesIterationParameters` holds the settings common to every media type.
`VideoIterator` requires the video-specific subclass,
[`VideoIterationParameters`](./iterators/video/README.md), instead.

| Parameter | Description |
|-----------|-------------|
| pre_sampled_freq | The sampling frequency of the pre-sampled time series data |
| sampling_freq | The sampling frequency from input data of the time series data |
| raw_sampling_rate | The sampling rate of the raw time series data (e.g. fps) |
| index_base | The base index of the time series data - ZERO or ONE |
| start_time_id | The start time id of the time series data (counted from index_base) |
| end_time_id | The end time id of the time series data (-1 indicates iteration until the last time) |

