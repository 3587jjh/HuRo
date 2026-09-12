"""How a robot's USD is driven: the joint orders and mimic relationships the renderer needs.
Distinct from common/robot_config.py, which describes the pipeline's state layout.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MimicJoint:
    """A joint whose angle follows another joint's: `multiplier * parent + offset`."""
    parent: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class UsdRobotConfig:
    """What the renderer needs to map a state vector onto a USD articulation.
    usd_joint_order is the articulation's DOF order as PhysX reports it."""
    name: str
    input_joint_order: List[str]
    usd_joint_order: List[str]
    mimic_joints: Dict[str, MimicJoint] = field(default_factory=dict)
    home: Dict[str, float] = field(default_factory=dict)   # home_config, for joints the state lacks
    prim_path: str = "/World/Robot"

    @property
    def actuated_dofs(self) -> int:
        """Width of the input state vector."""
        return len(self.input_joint_order)

    @property
    def total_dofs(self) -> int:
        """Number of DOFs the articulation exposes (driven joints + mimic joints)."""
        return len(self.usd_joint_order)


def usd_robot_config(robot_cfg, usd_joint_order: List[str]) -> UsdRobotConfig:
    """Build the USD-side config from a robot's YAML `overlay` block. usd_joint_order comes
    from the loaded articulation (PhysX's DOF order, not the USD's authoring order)."""
    overlay = robot_cfg.config["overlay"]
    mimic_joints = {
        name: MimicJoint(
            parent=spec["parent"],
            multiplier=spec.get("multiplier", 1.0),
            offset=spec.get("offset", 0.0),
        )
        for name, spec in overlay.get("mimic_joints", {}).items()
    }
    return UsdRobotConfig(
        name=robot_cfg.name,
        input_joint_order=list(robot_cfg.joint_names),
        usd_joint_order=list(usd_joint_order),
        mimic_joints=mimic_joints,
        home=dict(robot_cfg.config.get("home_config") or {}),
        prim_path=overlay.get("prim_path", "/World/Robot"),
    )
