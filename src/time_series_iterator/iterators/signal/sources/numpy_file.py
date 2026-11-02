import numpy as np

from .array import ArraySignalSource

class NumpyFileSignalSource(ArraySignalSource):
    """
    Signal stored in a `.npy` or `.npz` file, with its first axis as time.

    A `.npy` file is memory-mapped, so files larger than memory iterate fine.
    A `.npz` member is loaded whole, since NumPy cannot memory-map archives.
    """
    def __init__(
        self,
        path: str,
        array_name: str | None = None,
        ):
        """
        Initialize the NumpyFileSignalSource.

        Parameters:
        ----------
        path: str
            The path to the file.
        array_name: str | None
            The `.npz` member to read, optional when it holds a single array.

        Raises:
        -------
        FileNotFoundError: If the file is not found.
        ValueError: If `array_name` does not resolve to one array of the file.
        """
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                data = loaded[self._resolve_array_name(path, loaded.files, array_name)]
        elif array_name is not None:
            raise ValueError(f"array_name only applies to .npz files, given for: {path}")
        else:
            data = loaded
        super().__init__(data)

    def _resolve_array_name(
        self,
        path: str,
        available_names: list[str],
        array_name: str | None,
        ) -> str:
        """
        Resolve which member of a `.npz` file to read.

        Parameters:
        ----------
        path: str
            The path to the file, for the error message.
        available_names: list[str]
            The members of the file.
        array_name: str | None
            The requested member.

        Returns:
        -------
        str: The member to read.

        Raises:
        -------
        ValueError: If the member is unknown, or none was requested and the
            file does not hold exactly one.
        """
        if array_name is None:
            if len(available_names) != 1:
                raise ValueError(f"array_name is required for {path}, available: {available_names}")
            return available_names[0]
        if array_name not in available_names:
            raise ValueError(f"Unknown array {array_name!r} in {path}, available: {available_names}")
        return array_name
