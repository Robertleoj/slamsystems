# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
from project.foundation.spatial import SE3_log
from project.utils.spatial import Pose
import project.utils.rerun_utils as rr_utils
from project.utils.image import Colors
import rerun as rr
import rerun.blueprint as rrb
import numpy as np

# %%
rr_utils.init("app")
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                origin="/poses",
                name="poses",
                contents=["+ $origin/**"]
            ),
            rrb.Spatial3DView(
                origin="/rotvecs",
                name="rotvecs",
                contents=["+ $origin/**"]
            )
        )
    )
)

# %%
pose = Pose.random()

# %%
# rr.log("origin", rr_utils.triad())
rr.log("/poses/origin", rr_utils.triad())
rr.log("/poses/origin/label", rr_utils.label3D("Origin"))

rr.log("/poses/pose_orig", rr_utils.transform(pose))
rr.log("/poses/pose_orig/triad", rr_utils.triad())
rr.log("/poses/pose_orig/label", rr_utils.label3D("original"))

# %%
se3 = pose.log()
se3

# %%
recovered = Pose.exp(se3)
rr.log("/poses/pose_recovered", rr_utils.transform(recovered))
rr.log("/poses/pose_recovered/triad", rr_utils.triad())
rr.log("/poses/pose_recovered/label", rr_utils.label3D("recovered"))



# %%
rvec = pose.rvec
rvec

# %%
rr.log("/rotvecs/", rr.Arrows3D(
    vectors=[rvec, se3[3:]], 
    labels=['rvec', 'se3vec'],
    show_labels=True
))

# %%
rvectvec_pose = Pose.from_rvec_tvec(rvec, pose.tvec)
se3_pose = Pose.from_rvec_tvec(se3[3:], pose.tvec)

# %%
rr_utils.log_reference_frame("/poses/pose_rotvec", rvectvec_pose, label="rvec_tvec")
rr_utils.log_reference_frame("/poses/pose_se3", se3_pose, label="se3_pose")

