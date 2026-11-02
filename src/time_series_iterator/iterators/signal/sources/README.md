# sources

## Overview

Concrete `SignalSource` implementations.

## Components

| Component | Description |
|-----------|-------------|
| [array.py](./array.py) | `ArraySignalSource`, a signal held in memory as an array or tensor |
| [numpy_file.py](./numpy_file.py) | `NumpyFileSignalSource`, a `.npy` file (memory-mapped) or a `.npz` member |
| [delimited_text.py](./delimited_text.py) | `DelimitedTextSignalSource`, a CSV-like table with optional header and column selection |
| [concatenated.py](./concatenated.py) | `ConcatenatedSignalSource`, several sources read as one continuous signal |
