"""Utils for using the SymForce optimization library

Docs:
https://symforce.org/

Git:
https://github.com/symforce-org/symforce
"""

import numpy as np
import symforce.symbolic as sf

from project.utils.spatial import Pose


def pose_to_sf(pose: Pose) -> sf.Pose3:
    return sf.Pose3(
        sf.Rot3.from_rotation_matrix(pose.rot_mat),  # type: ignore
        sf.V3(*pose.tvec.tolist()),  # type: ignore
    )


def sf_to_pose(pose: sf.Pose3) -> Pose:
    return Pose(np.array(pose.to_homogenous_matrix()).reshape(4, 4))


def set_darkmode():
    from IPython.core.display import HTML
    from IPython.display import display

    display(
        HTML("""
    <style>
    div.highlight {
        background: #1e1e1e !important;  /* Dark background */
        color: #dcdcdc !important;  /* Light text */
    }
    </style>
    """)
    )
