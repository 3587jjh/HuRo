"""Robot config loader: reads configs/<robot>.yaml and derives the state layout.
A new robot needs its own configs/<name>.yaml plus the URDF it names."""
from dataclasses import dataclass
from typing import Dict, List

import yaml

from common.paths import ROBOT_CONFIG_DIR, robot_urdf


@dataclass
class RobotConfig:
    """A robot's YAML config plus the state layout derived from its joint groups."""
    name: str
    config: dict
    config_path: str
    urdf_path: str
    state_dim: int                      # total active DOF (len of state_qpos)
    eef_left_dim: int                   # wrist pos3 + rot6d6 + left hand joints
    eef_right_dim: int
    state_slices: Dict[str, slice]      # group name -> slice into state_qpos
    joint_names: List[str]              # flat active joint names, state_qpos order


def load_robot_config(robot_name: str) -> RobotConfig:
    """Load configs/<robot_name>.yaml and compute the derived state layout."""
    config_path = ROBOT_CONFIG_DIR / f"{robot_name}.yaml"
    if not config_path.is_file():
        available = sorted(p.stem for p in ROBOT_CONFIG_DIR.glob("*.yaml"))
        raise NotImplementedError(
            f"robot '{robot_name}' is not supported. Available: {available}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    joints = config["joints"]

    joint_names: List[str] = []
    state_slices: Dict[str, slice] = {}
    offset = 0
    for group_name, group_joints in joints.items():
        joint_names.extend(group_joints)
        state_slices[group_name] = slice(offset, offset + len(group_joints))
        offset += len(group_joints)

    hand_groups = config["eef_hand_joint_groups"]
    return RobotConfig(
        name=config.get("name", robot_name),
        config=config,
        config_path=str(config_path),
        urdf_path=str(robot_urdf(config["robot_urdf_path"])),
        state_dim=len(joint_names),
        eef_left_dim=9 + len(joints[hand_groups["left"]]),
        eef_right_dim=9 + len(joints[hand_groups["right"]]),
        state_slices=state_slices,
        joint_names=joint_names,
    )


def schema_metadata(robot_cfg: RobotConfig) -> Dict[str, str]:
    """Parquet schema-level metadata so a consumer can read state_qpos / state_eef_* standalone."""
    import json

    joints = robot_cfg.config["joints"]
    hand_groups = robot_cfg.config["eef_hand_joint_groups"]
    return {
        "robot_name": robot_cfg.name,
        "state_qpos_joint_names": json.dumps(robot_cfg.joint_names),
        "state_eef_layout": "per_side:wrist_pos3_rot6d6_hand_joints",
        "state_eef_left_hand_joint_names": json.dumps(joints[hand_groups["left"]]),
        "state_eef_right_hand_joint_names": json.dumps(joints[hand_groups["right"]]),
        "state_eef_frame": "robot_base_frame",
        "cam_pose_convention": "opencv_cam_to_base",
    }
