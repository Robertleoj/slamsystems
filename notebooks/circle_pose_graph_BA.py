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

import symforce

if 'eps_set' not in globals():
    symforce.set_epsilon_to_symbol()
    eps_set = True


from pathlib import Path
import project.utils.rerun_utils as rr_utils
import rerun as rr
from project.utils.spatial import Pose, Rotation
from project.utils.paths import repo_root
from project.utils.symforce_utils import sf_to_pose, pose_to_sf, set_darkmode


from dataclasses import dataclass
import numpy as np

import symforce.symbolic as sf
from symforce.values.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.optimizer_params import OptimizerParams
from symforce.codegen import codegen_util
from symforce import codegen
from symforce.notebook_util import display_code_file
from symforce.notebook_util import set_notebook_defaults

# %%
set_darkmode()

# %%
data_path = repo_root() / 'data/sphere_pose_graph.txt'

# %%
lines = data_path.read_text().strip().split("\n")


# %%
@dataclass
class Edge:
    vertex1_id: int
    vertex2_id: int
    vertex1_to_vertex2: Pose

    @property
    def key(self) -> str:
        return f"measurement_{self.vertex1_id}_{self.vertex2_id}"

@dataclass
class Vertex:
    id: int
    pose: Pose

    @property
    def key(self) -> str:
        return f"pose_{self.id}"


# %%
def parse_pose(data: list[str]) -> Pose:
    tx = float(data[0])
    ty = float(data[1])
    tz = float(data[2])

    qx = float(data[3])
    qy = float(data[4])
    qz = float(data[5])
    qw = float(data[6])
    
    rot = Rotation.from_xyzw(np.array([qx, qy, qz, qw]))
    trans = np.array([tx, ty, tz])

    pose = Pose.from_rot_trans(rot, trans)
    return pose



# %%
vertices: dict[int, Vertex] = {}

edges: list[Edge]  = []

for line in lines:
    line_type, data = line.strip().split(":")

    if line_type == "VERTEX_SE3":
        split = data.split()

        assert split[0] == "QUAT"

        vertex_id = int(split[1])

        pose = parse_pose(split[2:])

        assert vertex_id not in vertices

        vertices[vertex_id] = Vertex(vertex_id, pose)

        pass

    elif line_type == "EDGE_SE3":
        split = data.split()

        assert split[0] == "QUAT"

        vertex1_id = int(split[1])
        vertex2_id = int(split[2])

        pose = parse_pose(split[3:])

        edges.append(Edge(
            vertex1_id,
            vertex2_id,
            pose
        ))

    else:
        raise ValueError(f"Bruh: {line_type}")


# %%
len(vertices), len(edges)

# %%
rr_utils.init("pose_graph")

# %%
trajectory = [v.pose for v in sorted(vertices.values(), key=lambda v: v.id)]

# %%
rr.log(
    "/input_trajectory",
    rr.LineStrips3D([t.tvec for t in trajectory])
)

# %%
values = Values()
for vertex in vertices.values():
    values[vertex.key] = pose_to_sf(vertex.pose)

for edge in edges:
    values[edge.key] = pose_to_sf(edge.vertex1_to_vertex2)

values['epsilon'] = sf.numeric_epsilon

# %%
factors = []

def pose_diff_factor(pose1: sf.Pose3, pose2: sf.Pose3, estimated_movement: sf.Pose3, epsilon: sf.Scalar):
    return sf.V6((estimated_movement.inverse() * pose1.inverse() * pose2).to_tangent(epsilon))

for edge in edges:
    factors.append(Factor(
        residual=pose_diff_factor,
        keys=[vertices[edge.vertex1_id].key, vertices[edge.vertex2_id].key, edge.key, 'epsilon']
    ))
    

# %%
params = OptimizerParams()
optimizer = Optimizer(
    factors=factors,
    optimized_keys=[v.key for v in vertices.values()]
)

# %%
res = optimizer.optimize(values)
new_trajectory = [
    sf_to_pose(res.optimized_values[v.key])
    for v in sorted(vertices.values(), key=lambda v: v.id)
]
rr.log(
    '/optimized_trajectory',
    rr.LineStrips3D([t.tvec for t in new_trajectory])
)
