from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import tifffile as tiff

from project.utils.camera.camera_params import Intrinsics


@dataclass(frozen=True, kw_only=True)
class View:
    color: np.ndarray
    # always in millimeters!
    depth: np.ndarray | None
    intrinsics: Intrinsics

    def with_depth(self, depth: np.ndarray) -> View:
        return replace(self, depth=depth)

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=False, parents=True)

        imageio.imwrite(path / "color.png", self.color)

        tiff.imwrite(path / "depth.tiff", self.depth)

        (path / "intrinsics.json").write_text(json.dumps(self.intrinsics.to_json_dict()))

    @staticmethod
    def load(path: Path) -> View:
        depth = None
        if (path / "depth.tiff").exists():
            depth = tiff.imread(path / "depth.tiff")

        color = imageio.imread(path / "color.png")
        intrinsics = Intrinsics.from_json_dict(json.loads((path / "intrinsics.json").read_text()))

        return View(color=color, depth=depth, intrinsics=intrinsics)

    @property
    def depth_assumed(self) -> np.ndarray:
        assert self.depth is not None
        return self.depth
