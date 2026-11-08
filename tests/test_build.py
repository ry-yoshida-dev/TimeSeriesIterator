import tempfile
import unittest
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from time_series_iterator import (
    ImageIterator,
    SignalIterator,
    SourceType,
    TimeSeriesIterationParameters,
    TimeSeriesIterator,
    VideoIterationParameters,
    VideoIterator,
)


class BuildTest(unittest.TestCase):
    @cached_property
    def directory(self) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        return Path(temporary_directory.name)

    def _write_video(self, frame_count: int) -> str:
        path = str(self.directory / "video.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (16, 8))
        for _ in range(frame_count):
            _ = writer.write(np.zeros((8, 16, 3), dtype=np.uint8))
        writer.release()
        return path

    def test_builds_the_iterator_of_each_source_type(self) -> None:
        image_path = str(self.directory / "image.png")
        _ = cv2.imwrite(image_path, np.zeros((4, 4, 3), dtype=np.uint8))
        signal_path = str(self.directory / "signal.npy")
        np.save(signal_path, np.arange(3.0))

        with TimeSeriesIterator.build(SourceType.IMAGE, [image_path]) as image:
            self.assertIsInstance(image, ImageIterator)
            self.assertEqual(image.source_type, SourceType.IMAGE)
            self.assertEqual(len(image), 1)
        with TimeSeriesIterator.build(
            SourceType.VIDEO, [self._write_video(frame_count=3)]
        ) as video:
            self.assertIsInstance(video, VideoIterator)
            self.assertEqual(video.source_type, SourceType.VIDEO)
            self.assertEqual(len(video), 3)
        with TimeSeriesIterator.build(SourceType.SIGNAL, [signal_path]) as signal:
            self.assertIsInstance(signal, SignalIterator)
            self.assertEqual(signal.source_type, SourceType.SIGNAL)
            self.assertEqual(len(signal), 3)

    def test_passes_parameters_to_the_iterator(self) -> None:
        params = VideoIterationParameters(sampling_freq=2)
        with TimeSeriesIterator.build(
            SourceType.VIDEO, [self._write_video(frame_count=5)], params
        ) as video:
            self.assertIs(video.params, params)
            self.assertEqual([time_id for time_id, _ in video], [1, 3, 5])

    def test_video_rejects_parameters_of_another_source_type(self) -> None:
        with self.assertRaises(TypeError):
            _ = TimeSeriesIterator.build(
                SourceType.VIDEO,
                [self._write_video(frame_count=1)],
                TimeSeriesIterationParameters(),
            )
