from dataclasses import dataclass

import numpy as np

from project.utils.spatial import Pose, Rotation


@dataclass(frozen=True)
class State:
    pose: Pose
    angular_velocity: np.ndarray
    translational_velocity: np.ndarray


def update_state(state: State, dt: float, rot_volatility: float, trans_volatility: float) -> State:
    dR = Rotation.from_rvec(state.angular_velocity * dt)
    new_R = dR @ state.pose.rot

    new_t = state.pose.tvec + dt * state.translational_velocity

    new_pose = Pose.from_rot_trans(new_R, new_t)

    new_angular_velocity = state.angular_velocity + np.random.randn(*state.angular_velocity.shape) * rot_volatility * dt

    new_translational_velocity = (
        state.translational_velocity + np.random.randn(*state.translational_velocity.shape) * trans_volatility * dt
    )

    return State(
        pose=new_pose, angular_velocity=new_angular_velocity, translational_velocity=new_translational_velocity
    )


def generate_random_trajectory(
    N: int = 1000, dt: float = 0.5, rot_volatility: float = 0.1, trans_volatility: float = 0.1
) -> list[Pose]:
    initial_state = State(
        pose=Pose.identity(), angular_velocity=np.array([0, 0, 0]), translational_velocity=np.array([0, 0, 0])
    )

    states = [initial_state]
    curr_state = initial_state

    for _ in range(N):
        new_state = update_state(curr_state, dt, rot_volatility, trans_volatility)
        states.append(new_state)
        curr_state = new_state

    traj = [s.pose for s in states]
    return traj
