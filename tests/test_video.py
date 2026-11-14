"""Tests for decoder cleanup and bounded frame batches.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from persist_sbd import video


def pyav_container(decoded=()):
    stream = SimpleNamespace(average_rate=25, guessed_rate=None, frames=0,
                             codec_context=SimpleNamespace(width=128, height=128))
    container = Mock(streams=SimpleNamespace(video=[stream]))
    container.decode.return_value = iter(decoded)
    return container


class VideoLifecycleTests(unittest.TestCase):
    def test_pyav_closes_even_if_iterator_never_started(self):
        container = pyav_container()
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}):
            with video.open_video("clip.mp4") as (info, frames):
                self.assertEqual(info.fps, 25)
                self.assertEqual(container.streams.video[0].thread_type, "AUTO")
        container.close.assert_called_once_with()
        self.assertEqual(list(frames), [])

    def test_pyav_closes_on_consumer_error(self):
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((128, 128, 3), np.uint8)
        container = pyav_container([frame])
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with video.open_video("clip.mp4") as (_, frames):
                    self.assertEqual(next(frames).shape, (112, 112, 3))
                    raise RuntimeError("inference failed")
        container.close.assert_called_once_with()

    def test_pyav_closes_on_preprocess_error(self):
        container = pyav_container([Mock()])
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}), \
                patch.object(video, "preprocess", side_effect=ValueError("bad pixels")):
            with self.assertRaisesRegex(ValueError, "bad pixels"):
                with video.open_video("clip.mp4") as (_, frames):
                    next(frames)
        container.close.assert_called_once_with()

    def test_pyav_closes_on_decode_error(self):
        container = pyav_container()
        container.decode.side_effect = RuntimeError("broken packet")
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}):
            with self.assertRaisesRegex(RuntimeError, "broken packet"):
                with video.open_video("clip.mp4") as (_, frames):
                    list(frames)
        container.close.assert_called_once_with()

    def test_audio_only_file_fails_and_closes(self):
        container = pyav_container()
        container.streams.video = []
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}):
            with self.assertRaisesRegex(ValueError, "no video stream.*audio.mp4"):
                with video.open_video("audio.mp4"):
                    self.fail("an audio-only file must not open as video")
        container.close.assert_called_once_with()

    def test_empty_video_fails_and_closes(self):
        container = pyav_container()
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: container)}):
            with self.assertRaisesRegex(ValueError, "no video frames.*empty.mp4"):
                with video.open_video("empty.mp4") as (_, frames):
                    list(frames)
        container.close.assert_called_once_with()

    def test_opencv_releases_if_open_fails(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch.object(video.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "cannot open"):
                with video.open_video("missing.mp4", decoder="opencv"):
                    self.fail("an unopened capture must raise")
        capture.release.assert_called_once_with()

    def test_opencv_releases_if_metadata_fails(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = RuntimeError("metadata failed")
        with patch.object(video.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "metadata failed"):
                with video.open_video("clip.mp4", decoder="opencv"):
                    self.fail("invalid metadata must raise")
        capture.release.assert_called_once_with()

    def test_opencv_releases_if_never_iterated(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        with patch.object(video.cv2, "VideoCapture", return_value=capture):
            with video.open_video("clip.mp4", decoder="opencv"):
                pass
        capture.release.assert_called_once_with()


class FakeDecordReader:
    def __init__(self, shape, count, batches):
        self.shape = shape
        self.count = count
        self.batches = batches

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return SimpleNamespace(shape=self.shape)

    def get_avg_fps(self):
        return 25

    def get_batch(self, indices):
        self.batches.append(indices)
        # Small stand-in frames preserve order without allocating real 4K data.
        values = np.array(indices, dtype=np.uint8).reshape(-1, 1, 1, 1)
        return SimpleNamespace(asnumpy=lambda: values)


class DecordBatchTests(unittest.TestCase):
    def test_batch_budget_preserves_every_frame_in_order(self):
        for height, width, expected_batch in ((128, 128, 64), (2160, 3840, 2), (4320, 7680, 1)):
            with self.subTest(resolution=(height, width)):
                batches = []
                reader = FakeDecordReader((height, width, 3), 70, batches)
                module = SimpleNamespace(VideoReader=lambda *a, **kw: reader)
                with patch.dict(sys.modules, {"decord": module}), \
                        patch.object(video, "preprocess", side_effect=lambda f, *args: f):
                    with video.open_video("clip.mp4", decoder="decord") as (info, frames):
                        values = [int(f[0, 0, 0]) for f in frames]
                self.assertEqual(values, list(range(70)))
                self.assertEqual(max(map(len, batches)), expected_batch)
                self.assertEqual((info.height, info.width), (height, width))

    def test_decord_reader_released_when_iteration_stops_early(self):
        references = []

        def factory(*args, **kwargs):
            reader = FakeDecordReader((128, 128, 3), 70, [])
            references.append(weakref.ref(reader))
            return reader

        with patch.dict(sys.modules, {"decord": SimpleNamespace(VideoReader=factory)}), \
                patch.object(video, "preprocess", side_effect=lambda f, *args: f):
            with video.open_video("clip.mp4", decoder="decord") as (_, frames):
                next(frames)
                self.assertIsNotNone(references[0]())
            self.assertIsNone(references[0]())

    def test_decord_empty_reader_fails(self):
        reader = FakeDecordReader((128, 128, 3), 0, [])
        with patch.dict(sys.modules, {"decord": SimpleNamespace(VideoReader=lambda *a, **kw: reader)}):
            with self.assertRaisesRegex(ValueError, "no video frames"):
                with video.open_video("empty.mp4", decoder="decord"):
                    self.fail("an empty reader must raise")


if __name__ == "__main__":
    unittest.main()
