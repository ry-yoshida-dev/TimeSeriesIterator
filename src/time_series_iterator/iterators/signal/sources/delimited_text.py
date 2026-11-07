import numpy as np

from .array import ArraySignalSource


class DelimitedTextSignalSource(ArraySignalSource):
    """
    Signal stored in a delimited text table such as CSV, with one row per sample.

    The whole table is loaded on construction. A single selected column yields
    zero-dimensional samples, and several columns yield `(columns,)` samples.
    Every selected field must be numeric, so select around timestamps or labels
    with `columns`.
    """

    def __init__(
        self,
        path: str,
        delimiter: str | None = ",",
        is_header_present: bool = False,
        columns: list[int] | list[str] | None = None,
        encoding: str = "utf-8-sig",
    ):
        """
        Initialize the DelimitedTextSignalSource.

        Parameters
        ----------
        path: str
            The path to the file.
        delimiter: str | None
            The field separator, or None for any whitespace.
        is_header_present: bool
            Whether the first line holds column names rather than a sample.
        columns: list[int] | list[str] | None
            Zero-based indices or header names of the columns to read, or None
            for every column.
        encoding: str
            The text encoding. The default also strips a UTF-8 byte order mark.

        Raises
        ------
        FileNotFoundError: If the file is not found.
        ValueError: If a column cannot be resolved or a field is not numeric.
        """
        column_indices = self._resolve_column_indices(
            path, delimiter, is_header_present, columns, encoding
        )
        table = np.loadtxt(
            path,
            dtype=np.float64,
            delimiter=delimiter,
            skiprows=1 if is_header_present else 0,
            usecols=column_indices,
            ndmin=2,
            encoding=encoding,
            quotechar='"',
        )
        super().__init__(table[:, 0] if table.shape[1] == 1 else table)

    def _resolve_column_indices(
        self,
        path: str,
        delimiter: str | None,
        is_header_present: bool,
        columns: list[int] | list[str] | None,
        encoding: str,
    ) -> list[int] | None:
        """
        Resolve the selected columns into indices.

        Parameters
        ----------
        path: str
            The path to the file.
        delimiter: str | None
            The field separator, or None for any whitespace.
        is_header_present: bool
            Whether the first line holds column names.
        columns: list[int] | list[str] | None
            The selected columns.
        encoding: str
            The text encoding.

        Returns
        -------
        list[int] | None: The column indices, or None for every column.

        Raises
        ------
        ValueError: If names are given without a header, or a name is unknown.
        """
        if columns is None:
            return None
        names = [column for column in columns if isinstance(column, str)]
        if len(names) == 0:
            return [column for column in columns if isinstance(column, int)]
        if len(names) != len(columns):
            raise ValueError(
                f"Columns must be all indices or all names, given: {columns}"
            )
        if not is_header_present:
            raise ValueError(
                f"Column names require is_header_present=True, given: {names}"
            )

        with open(path, encoding=encoding) as file:
            header = file.readline().strip().split(delimiter)
        index_by_name = {
            name.strip().strip('"'): index for index, name in enumerate(header)
        }
        unknown_names = [name for name in names if name not in index_by_name]
        if len(unknown_names) > 0:
            raise ValueError(
                f"Unknown columns {unknown_names} in {path}, available: {list(index_by_name)}"
            )
        return [index_by_name[name] for name in names]
