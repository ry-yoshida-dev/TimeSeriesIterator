from __future__ import annotations

from enum import Enum
from pathlib import Path

from .source import SignalSource
from .sources import DelimitedTextSignalSource, NumpyFileSignalSource


class SignalFileFormat(Enum):
    """
    Signal file formats recognized from a file suffix.

    Attributes
    ----------
    NPY: NumPy array file.
    NPZ: NumPy archive holding a single array.
    CSV: Comma-separated table without a header.
    TSV: Tab-separated table without a header.
    TXT: Whitespace-separated table without a header.
    """

    NPY = ".npy"
    NPZ = ".npz"
    CSV = ".csv"
    TSV = ".tsv"
    TXT = ".txt"

    @classmethod
    def from_path(cls, path: str) -> SignalFileFormat:
        """
        Recognize the format of a file from its suffix, ignoring case.

        Parameters
        ----------
        path: str
            The path to the file.

        Returns
        -------
        SignalFileFormat: The format.

        Raises
        ------
        ValueError: If the suffix is not recognized.
        """
        suffix = Path(path).suffix.lower()
        supported_suffixes = {file_format.value for file_format in cls}
        if suffix not in supported_suffixes:
            raise ValueError(
                f"Unsupported signal file: {path}, supported: {sorted(supported_suffixes)}"
            )
        return cls(suffix)

    def open(self, path: str) -> SignalSource:
        """
        Open a file of this format with the format's default options.

        Parameters
        ----------
        path: str
            The path to the file.

        Returns
        -------
        SignalSource: The source reading the file.
        """
        match self:
            case SignalFileFormat.NPY | SignalFileFormat.NPZ:
                return NumpyFileSignalSource(path)
            case SignalFileFormat.CSV:
                return DelimitedTextSignalSource(path, delimiter=",")
            case SignalFileFormat.TSV:
                return DelimitedTextSignalSource(path, delimiter="\t")
            case SignalFileFormat.TXT:
                return DelimitedTextSignalSource(path, delimiter=None)
