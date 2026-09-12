"""Mapping a state vector onto a USD articulation's DOF vector."""
from typing import Callable, List

import numpy as np

from .usd_robot import UsdRobotConfig


def create_joint_mapping(config: UsdRobotConfig) -> Callable[[np.ndarray], np.ndarray]:
    """Build input_state (actuated DOFs) -> usd_state (all DOFs): reorder by joint name, hold a
    joint the state does not carry at its home_config value (0 when unlisted), then compute
    mimic joints from their parent."""
    input_to_idx = {joint: i for i, joint in enumerate(config.input_joint_order)}
    usd_to_idx = {joint: i for i, joint in enumerate(config.usd_joint_order)}

    mimic_joints = config.mimic_joints
    home_state = np.array([config.home.get(joint, 0.0) for joint in config.usd_joint_order])
    driven = [(usd_to_idx[j], i) for j, i in input_to_idx.items() if j in usd_to_idx]
    mimics = [(usd_idx, usd_to_idx[mimic_joints[j].parent], mimic_joints[j])
              for usd_idx, j in enumerate(config.usd_joint_order) if j in mimic_joints]

    def mapping_fn(input_state: np.ndarray) -> np.ndarray:
        if len(input_state) != config.actuated_dofs:
            raise ValueError(
                f"Expected {config.actuated_dofs} DOFs from input, got {len(input_state)}"
            )

        usd_state = home_state.astype(input_state.dtype)
        for usd_idx, input_idx in driven:
            usd_state[usd_idx] = input_state[input_idx]
        # After the driven joints, so a mimic joint never reads a parent that is still unset
        for usd_idx, parent_idx, mimic in mimics:
            usd_state[usd_idx] = usd_state[parent_idx] * mimic.multiplier + mimic.offset

        return usd_state

    return mapping_fn


def check_joint_coverage(config: UsdRobotConfig) -> List[str]:
    """Names in the input order that the articulation has no DOF for (non-empty = config error)."""
    usd_names = set(config.usd_joint_order)
    return [n for n in config.input_joint_order if n not in usd_names]
