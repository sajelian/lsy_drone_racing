"""Utility module."""

from __future__ import annotations

import importlib.util
import sys
from math import asin, atan2
from pathlib import Path
from typing import Type

import numpy as np
import pybullet as p

from lsy_drone_racing.controller import BaseController


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert a quaternion into euler angles (roll, pitch, yaw).

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def map2pi(angle: np.ndarray) -> np.ndarray:
    """Map an angle or array of angles to the interval of [-pi, pi].

    Args:
        angle: Number or array of numbers.

    Returns:
        The remapped angles.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def load_controller(path: Path) -> Type[BaseController]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)
    assert hasattr(controller_module, "Controller")
    assert issubclass(controller_module.Controller, BaseController)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def check_gate_pass(
    gate_pose: np.ndarray, drone_pos: np.ndarray, last_drone_pos: np.ndarray
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the y-Axis (pointing from -y to +y). Therefore, we check if y
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the x
    and z box boundaries. First, we linearly interpolate to get the x and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        gate_pose: The pose of the gate in the world frame.
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
    """
    # Transform last and current drone position into current gate frame.
    cos_goal, sin_goal = np.cos(gate_pose[5]), np.sin(gate_pose[5])
    last_dpos = last_drone_pos - gate_pose[0:3]
    last_drone_pos_gate = np.array(
        [
            cos_goal * last_dpos[0] - sin_goal * last_dpos[1],
            sin_goal * last_dpos[0] + cos_goal * last_dpos[1],
            last_dpos[2],
        ]
    )
    dpos = drone_pos - gate_pose[0:3]
    drone_pos_gate = np.array(
        [cos_goal * dpos[0] - sin_goal * dpos[1], sin_goal * dpos[0] + cos_goal * dpos[1], dpos[2]]
    )
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    if last_drone_pos_gate[1] < 0 and drone_pos_gate[1] > 0:  # Drone has passed the goal plane
        alpha = -last_drone_pos_gate[1] / (drone_pos_gate[1] - last_drone_pos_gate[1])
        x_intersect = alpha * (drone_pos_gate[0]) + (1 - alpha) * last_drone_pos_gate[0]
        z_intersect = alpha * (drone_pos_gate[2]) + (1 - alpha) * last_drone_pos_gate[2]
        # TODO: Replace with autodetection of gate width
        if abs(x_intersect) < 0.45 and abs(z_intersect) < 0.45:
            return True
    return False


def draw_trajectory(
    initial_info: dict,
    waypoints: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_z: np.ndarray,
):
    """Draw a trajectory in PyBullet's GUI."""
    for point in waypoints:
        urdf_path = Path(initial_info["urdf_dir"]) / "sphere.urdf"
        p.loadURDF(
            str(urdf_path),
            [point[0], point[1], point[2]],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=initial_info["pyb_client"],
        )
    step = int(ref_x.shape[0] / 50)
    for i in range(step, ref_x.shape[0], step):
        p.addUserDebugLine(
            lineFromXYZ=[ref_x[i - step], ref_y[i - step], ref_z[i - step]],
            lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
            lineColorRGB=[1, 0, 0],
            physicsClientId=initial_info["pyb_client"],
        )
    p.addUserDebugLine(
        lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
        lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
        lineColorRGB=[1, 0, 0],
        physicsClientId=initial_info["pyb_client"],
    )
