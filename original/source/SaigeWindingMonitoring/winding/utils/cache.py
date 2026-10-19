from __future__ import annotations

import os
from typing import TypeVar, Generic, Literal, overload

import pickle
import gzip
import zstandard as zstd

from ..error import SegmentationModelInferenceError
from ..data.data_class import SegmentationResults, SideSegmentationResults
from ..SaigeSegmentation2.segmentation import InferenceHandler


CLASS_NAMES = ["left_drum", "right_drum", "winding_cable", "drum_body"]
DEFAULT_KEYS = {
    "segmentation_masks": CLASS_NAMES,
    "segmentation_contours": CLASS_NAMES,
    "bounding_boxes": CLASS_NAMES,
}


def save_data_pkl(file_path: str, data: dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_pkl(file_path: str) -> dict:
    with gzip.open(file_path, "rb") as f:
        loaded = pickle.load(f)
    return loaded


SegmentationResultT = TypeVar("SegmentationResultT", SegmentationResults, SideSegmentationResults)


class SegmentationCacheManager(Generic[SegmentationResultT]):

    @overload
    def __init__(
        self: SegmentationCacheManager[SegmentationResults],
        inference_handler: InferenceHandler,
        video_path: str,
        cache_path: str | None,
        side: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self: SegmentationCacheManager[SideSegmentationResults],
        inference_handler: InferenceHandler,
        video_path: str,
        cache_path: str | None,
        side: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        inference_handler: InferenceHandler,
        video_path: str,
        cache_path: str | None,
        side: bool = False,
    ):
        self._inference_handler = inference_handler
        self._video_path = video_path
        self._cache_path = cache_path
        self._side = side
        self._video_decoder = None
        self._result_class: type[SegmentationResultT] = (
            SideSegmentationResults if side else SegmentationResults
        )  # type: ignore[assignment]

        self._init_cache()

        from ..modules.segmentation import (
            SegmentationEngine,
            CLASS_INDEX_TO_NAME,
            CLASS_INDEX_TO_NAME_SIDE,
        )

        class_index_to_name = CLASS_INDEX_TO_NAME_SIDE if self._side else CLASS_INDEX_TO_NAME

        def _parse(result):
            return SegmentationEngine.parse_segmentation_results(result, class_index_to_name)

        self._parse = _parse

    def _init_cache(self):
        if self._cache_path is not None and os.path.exists(self._cache_path):
            try:
                with zstd.open(self._cache_path, mode="rb") as f:
                    cache = pickle.load(f)

                self._seg_results_dict = cache["seg_results"]
                self._initial_keys = set(self._seg_results_dict.keys())
                self._total_frames = cache["total_frames"]
                self._fps = cache["fps"]
            except Exception:
                self._seg_results_dict = {}
                self._initial_keys = set()
                self._build_video_decoder()

        else:
            self._seg_results_dict = {}
            self._initial_keys = set()
            self._build_video_decoder()

    def _build_video_decoder(self):
        from ..utils.video_decoder import VideoDecoder

        self._video_decoder = VideoDecoder(self._video_path)
        self._total_frames = len(self._video_decoder)
        self._fps = self._video_decoder.fps

    @property
    def fps(self) -> int:
        return self._fps

    def __len__(self) -> int:
        return self._total_frames

    def _infer(self, index: int) -> SegmentationResultT:
        if self._video_decoder is None:
            self._build_video_decoder()

        assert self._video_decoder is not None

        frame = self._video_decoder.get_frame_at(index)
        error_code, message, results = self._inference_handler.infer_and_postprocess(images=[frame.data])
        if error_code != 0 or results is None:
            raise SegmentationModelInferenceError(f"Error code {error_code} : {message}")

        return self._result_class.from_dict(self._parse(results))

    def get(self, index: int) -> SegmentationResultT:
        if index < 0 or index >= self._total_frames:
            raise IndexError(f"Index out of range: {index}")

        if index in self._seg_results_dict:
            return self._seg_results_dict[index]

        results = self._infer(index)
        if self._cache_path is not None:
            self._seg_results_dict[index] = results

        return results

    def __getitem__(self, index: int) -> SegmentationResultT:
        return self.get(index)

    def save(self):
        if self._cache_path is None:
            return

        new_keys = set(self._seg_results_dict.keys()) - self._initial_keys
        if len(new_keys) == 0 or len(self._seg_results_dict) == 0:
            return

        data = {
            "seg_results": self._seg_results_dict,
            "total_frames": self._total_frames,
            "fps": self._fps,
        }

        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with zstd.open(self._cache_path, mode="wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __del__(self):
        self.save()
