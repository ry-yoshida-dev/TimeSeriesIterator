# time_series_iterator

## Overview

Iterators over time-series data (images, videos and numeric signals) with configurable sampling and indexing.

## Components

| Component | Description |
|-----------|-------------|
| [iterator.py](./iterator.py) | `TimeSeriesIterator`, the base iterator and its factory `TimeSeriesIterator.build` |
| [parameters.py](./parameters.py) | `TimeSeriesIterationParameters`, the sampling, start/end ids and index base shared by every source type |
| [types.py](./types.py) | Type aliases for the numeric samples iterators yield (`NumericArray`) |
| [utils/](./utils/README.md) | Enums for the source type and index base |
| [iterators/](./iterators/README.md) | Concrete iterators for image, video and signal inputs |
