from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np
import tifffile as tiff

from project.utils.camera.camera_params import Intrinsics
from project.utils.camera.view import View


def _read_tiff_video(path: Path) -> list[np.ndarray]:
    vid: list[np.ndarray] = []

    for i in count():
        frame_path = path / f"{i}.tiff"

        if not frame_path.exists():
            break

        frame = tiff.imread(frame_path)

        vid.append(frame)

    return vid


def _save_tiff_video(path: Path, vid: list[np.ndarray]) -> None:
    for i, frame in enumerate(vid):
        tiff.imwrite(path / f"{i}.tiff", frame)


@dataclass(frozen=True)
class Video:
    color: list[np.ndarray]
    intrinsics: Intrinsics
    depth: list[np.ndarray] | None = None

    def __post_init__(self):
        if self.depth is not None:
            assert len(self.depth) == len(self.color)
            assert self.depth[0].shape[:2] == self.color[0].shape[:2]

    def __getitem__(self, idx: int):
        return View(
            color=self.color[idx], depth=self.depth[idx] if self.depth is not None else None, intrinsics=self.intrinsics
        )

    def __len__(self) -> int:
        return len(self.color)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def load(path: Path) -> Video:
        assert path.exists()
        color_path = path / "color"
        depth_path = path / "depth"
        intrinsics_path = path / "intrinsics.json"

        assert color_path.exists()

        color_vid: list[np.ndarray] = _read_tiff_video(color_path)

        depth_vid: list[np.ndarray] | None = None
        if depth_path.exists():
            depth_vid = _read_tiff_video(depth_path)

        intrinsics = Intrinsics.from_json_dict(json.loads(intrinsics_path.read_text()))

        return Video(color=color_vid, intrinsics=intrinsics, depth=depth_vid)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)

        color_path = path / "color"
        color_path.mkdir()
        _save_tiff_video(color_path, self.color)

        for i, color_frame in enumerate(self.color):
            tiff.imwrite(color_path / f"{i}.tiff", color_frame)

        if self.depth is not None:
            depth_path = path / "depth"
            depth_path.mkdir()
            _save_tiff_video(depth_path, self.depth)

        (path / "intrinsics.json").write_text(json.dumps(self.intrinsics.to_json_dict()))

    @property
    def depth_assumed(self) -> list[np.ndarray]:
        assert self.depth is not None
        return self.depth
