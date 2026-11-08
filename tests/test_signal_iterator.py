import tempfile
import unittest
from functools import cached_property
from pathlib import Path
from typing import override

import numpy as np
import torch

from time_series_iterator import (
    ArraySignalSource,
    ConcatenatedSignalSource,
    DelimitedTextSignalSource,
    IndexBase,
    NumpyFileSignalSource,
    SignalFileFormat,
    SignalIterator,
    SignalSource,
    SourceType,
    TimeSeriesIterationParameters,
    TimeSeriesIterator,
)
from time_series_iterator.types import NumericArray


class SquareSignalSource(SignalSource):
    @override
    def _read_sample(self, index: int) -> NumericArray:
        return np.asarray(float(index**2))

    @override
    def __len__(self) -> int:
        return 4


class SignalIteratorTest(unittest.TestCase):
    @cached_property
    def directory(self) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        return Path(temporary_directory.name)

    def _pairs(self, iterator: SignalIterator) -> list[tuple[int, float]]:
        return [(time_id, float(sample)) for time_id, sample in iterator]

    def test_iterates_with_time_ids_and_sampling(self) -> None:
        iterator = SignalIterator(
            ArraySignalSource([10.0, 11.0, 12.0, 13.0, 14.0]),
            TimeSeriesIterationParameters(sampling_freq=2),
        )
        self.assertEqual(iterator.remaining_step_count, 3)
        self.assertEqual(self._pairs(iterator), [(1, 10.0), (3, 12.0), (5, 14.0)])

    def test_honors_pre_sampling_index_base_and_end(self) -> None:
        iterator = SignalIterator(
            ArraySignalSource(np.arange(5)),
            TimeSeriesIterationParameters(
                pre_sampled_freq=10,
                index_base=IndexBase.ZERO,
                start_time_id=10,
                end_time_id=30,
            ),
        )
        self.assertEqual(self._pairs(iterator), [(10, 1.0), (20, 2.0), (30, 3.0)])

    def test_get_sample_matches_iteration_and_keeps_position(self) -> None:
        iterator = SignalIterator(
            ArraySignalSource([5.0, 6.0, 7.0]), TimeSeriesIterationParameters()
        )
        self.assertEqual(float(iterator.get_sample(3)), 7.0)
        self.assertEqual(float(next(iterator)[1]), 5.0)
        with self.assertRaises(ValueError):
            _ = iterator.get_sample(4)

    def test_skip_advances_like_next(self) -> None:
        iterator = SignalIterator(
            ArraySignalSource([1.0, 2.0]), TimeSeriesIterationParameters()
        )
        self.assertEqual(iterator.skip(), 1)
        self.assertEqual(float(next(iterator)[1]), 2.0)
        with self.assertRaises(StopIteration):
            _ = iterator.skip()

    def test_sample_shapes(self) -> None:
        scalar = next(
            SignalIterator(
                ArraySignalSource([1.0, 2.0]), TimeSeriesIterationParameters()
            )
        )[1]
        self.assertIsInstance(scalar, np.ndarray)
        self.assertEqual(scalar.shape, ())
        channels = next(
            SignalIterator(
                ArraySignalSource(np.zeros((4, 3))), TimeSeriesIterationParameters()
            )
        )[1]
        self.assertEqual(channels.shape, (3,))

    def test_tensor_stays_a_tensor(self) -> None:
        sample = next(
            SignalIterator(
                ArraySignalSource(torch.arange(3)), TimeSeriesIterationParameters()
            )
        )[1]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape, torch.Size([]))

    def test_custom_source(self) -> None:
        iterator = SignalIterator(
            SquareSignalSource(), TimeSeriesIterationParameters(sampling_freq=3)
        )
        self.assertEqual(self._pairs(iterator), [(1, 0.0), (4, 9.0)])

    def test_array_source_rejects_invalid_data(self) -> None:
        for data, error in (
            (3.0, ValueError),
            ([], ValueError),
            ([[1.0], [2.0, 3.0]], ValueError),
            (["a", "b"], TypeError),
            ([1j, 2j], TypeError),
            ([True, False], TypeError),
            (torch.tensor([True]), TypeError),
        ):
            with self.subTest(data=data), self.assertRaises(error):
                _ = ArraySignalSource(data)

    def test_source_read_rejects_out_of_range(self) -> None:
        source = ArraySignalSource([1.0])
        for index in (-1, 1):
            with self.subTest(index=index), self.assertRaises(IndexError):
                _ = source.read(index)

    def test_concatenated_addresses_across_sources(self) -> None:
        source = ConcatenatedSignalSource(
            [
                ArraySignalSource([0.0, 1.0]),
                ArraySignalSource([2.0]),
                ArraySignalSource([3.0, 4.0]),
            ]
        )
        self.assertEqual(len(source), 5)
        self.assertEqual(
            [float(source.read(index)) for index in range(5)], [0.0, 1.0, 2.0, 3.0, 4.0]
        )

    def test_concatenated_rejects_invalid_sources(self) -> None:
        with self.assertRaises(ValueError):
            _ = ConcatenatedSignalSource([])
        with self.assertRaises(ValueError):
            _ = ConcatenatedSignalSource(
                [
                    ArraySignalSource(np.zeros((2, 3))),
                    ArraySignalSource(np.zeros((2, 4))),
                ]
            )

    def test_npy_is_memory_mapped(self) -> None:
        path = self.directory / "signal.npy"
        np.save(path, np.arange(3.0))
        source = NumpyFileSignalSource(str(path))
        assert isinstance(source.data, np.ndarray)
        self.assertIsInstance(source.data.base, np.memmap)
        self.assertEqual(float(source.read(2)), 2.0)

    def test_npz_member_selection(self) -> None:
        path = self.directory / "signal.npz"
        np.savez(path, x=np.arange(3.0), y=np.arange(3.0) * 2)
        self.assertEqual(
            float(NumpyFileSignalSource(str(path), array_name="y").read(2)), 4.0
        )
        with self.assertRaises(ValueError):
            _ = NumpyFileSignalSource(str(path))
        with self.assertRaises(ValueError):
            _ = NumpyFileSignalSource(str(path), array_name="z")

    def test_npy_rejects_array_name(self) -> None:
        path = self.directory / "signal.npy"
        np.save(path, np.arange(3.0))
        with self.assertRaises(ValueError):
            _ = NumpyFileSignalSource(str(path), array_name="x")

    def test_csv_columns_by_name_and_index(self) -> None:
        path = self.directory / "signal.csv"
        _ = path.write_text(
            '﻿time,"x",y\n2026-01-01,1,2\n2026-01-02,3,4\n', encoding="utf-8"
        )
        by_name = DelimitedTextSignalSource(
            str(path), is_header_present=True, columns=["y", "x"]
        )
        self.assertTrue(np.array_equal(by_name.read(1), np.array([4.0, 3.0])))
        by_index = DelimitedTextSignalSource(
            str(path), is_header_present=True, columns=[1]
        )
        self.assertEqual(by_index.read(1).shape, ())
        self.assertEqual(float(by_index.read(1)), 3.0)

    def test_csv_rejects_unresolvable_columns(self) -> None:
        path = self.directory / "signal.csv"
        _ = path.write_text("x\n1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            _ = DelimitedTextSignalSource(str(path), columns=["x"])
        with self.assertRaises(ValueError):
            _ = DelimitedTextSignalSource(
                str(path), is_header_present=True, columns=["z"]
            )

    def test_csv_shape_is_independent_of_row_count(self) -> None:
        path = self.directory / "signal.csv"
        _ = path.write_text("1,2,3\n", encoding="utf-8")
        source = DelimitedTextSignalSource(str(path))
        self.assertEqual(len(source), 1)
        self.assertEqual(source.sample_shape, (3,))

    def test_build_joins_files_of_mixed_formats(self) -> None:
        np.save(self.directory / "a.npy", np.array([1.0, 2.0]))
        _ = (self.directory / "b.CSV").write_text("3\n4\n", encoding="utf-8")
        _ = (self.directory / "c.tsv").write_text("5\n", encoding="utf-8")
        _ = (self.directory / "d.txt").write_text("6\n", encoding="utf-8")
        paths = [
            str(self.directory / name) for name in ("a.npy", "b.CSV", "c.tsv", "d.txt")
        ]
        with TimeSeriesIterator.build(SourceType.SIGNAL, paths) as iterator:
            self.assertEqual(iterator.source_type, SourceType.SIGNAL)
            self.assertEqual(len(iterator), 6)
            self.assertEqual(
                [float(sample) for _, sample in iterator],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )

    def test_tsv_and_whitespace_delimiters(self) -> None:
        _ = (self.directory / "a.tsv").write_text("1\t2\n", encoding="utf-8")
        _ = (self.directory / "a.txt").write_text("1  2\n", encoding="utf-8")
        for name in ("a.tsv", "a.txt"):
            with self.subTest(name=name):
                source = SignalFileFormat.from_path(name).open(
                    str(self.directory / name)
                )
                self.assertEqual(source.sample_shape, (2,))

    def test_from_paths_rejects_invalid_paths(self) -> None:
        params = TimeSeriesIterationParameters()
        with self.assertRaises(ValueError):
            _ = SignalIterator.from_paths([], params)
        with self.assertRaises(ValueError):
            _ = SignalIterator.from_paths(
                [str(self.directory / "signal.parquet")], params
            )
        with self.assertRaises(FileNotFoundError):
            _ = SignalIterator.from_paths([str(self.directory / "missing.npy")], params)


if __name__ == "__main__":
    _ = unittest.main()
