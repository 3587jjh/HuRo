"""Bimanual retargeting of human hand keypoints onto a robot, via PyRoKi.

The offset solve fits a 4-DOF world offset [tx, ty, tz, yaw] on representative frames.
The trajectory solve freezes that offset and solves joints over the whole segment. Both
solves pad to fixed shapes.
"""
# create_conn_tree, RetargetingWeights and the alignment costs are adapted from PyRoKi's
# examples, MIT License, Copyright (c) 2025 Chung Min Kim.
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from loguru import logger

# MediaPipe 21-point hand: fingertip indices (0 = wrist, 4 keypoints per finger)
GLOBAL_TIP_INDICES = [4, 8, 12, 16, 20]

# Offset-solve frame count: sequences are subsampled/padded to this (single JIT shape)
N_REPRESENTATIVE = 32


class RetargetingWeights(TypedDict):
    """Residual scales for the optimization."""
    local_alignment: float
    global_alignment: float
    joint_smoothness: float
    hand_smoothness_scale: float
    ego_view_rot: float
    ego_view_pos: float
    rest_weight_default: float
    rest_weight_stiff: float
    hand_rest_scale: float
    joint_limit: float


DEFAULT_WEIGHTS = RetargetingWeights(
    local_alignment=20.0,
    global_alignment=50.0,
    joint_smoothness=40.0,
    hand_smoothness_scale=0.25,
    ego_view_rot=3.0,
    ego_view_pos=10.0,
    rest_weight_default=0.2,
    rest_weight_stiff=1.2,
    hand_rest_scale=0.50,
    joint_limit=100.0,
)

# Locked joints are pinned to the home pose with a weight above every other residual.
LOCKED_REST_WEIGHT = 200.0

# jaxls's parameter-tolerance stop compares the step with the norm of all variables, padded
# frames included. A long padding inflates that norm and ends the trajectory solve early, so pads
# longer than this run without that stop.
PARAMETER_TOLERANCE_MAX_PAD = 224


def sample_representative_frames(
    keypoint_mask: np.ndarray,
    cam_mask: np.ndarray,
    n_target: int = N_REPRESENTATIVE,
) -> np.ndarray:
    """Uniformly sample up to n_target frames with a visible hand and a valid camera."""
    T = keypoint_mask.shape[0]
    any_hand_valid = keypoint_mask.any(axis=1)
    frame_valid = any_hand_valid & (cam_mask > 0.5)
    valid_indices = np.where(frame_valid)[0]

    if len(valid_indices) == 0:
        valid_indices = np.where(any_hand_valid)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(T)

    n_rep = min(n_target, len(valid_indices))
    selected = valid_indices[np.linspace(0, len(valid_indices) - 1, n_rep).astype(int)]
    return np.unique(selected)


def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray, urdf: yourdfpy.URDF) -> jnp.ndarray:
    """NxN adjacency: 1 where one link is the other's ancestor with no target link between."""
    n = len(link_indices)
    link_names = list(robot.links.names)

    target_link_names = [link_names[int(idx)] for idx in link_indices]
    target_link_set = set(target_link_names)

    link_to_parent = {joint.child: joint.parent for joint in urdf.joint_map.values()}

    def is_adjacent(link1: str, link2: str) -> bool:
        for start, other in ((link1, link2), (link2, link1)):
            current = start
            while current is not None:
                parent = link_to_parent.get(current)
                if parent is None:
                    break
                if parent == other:
                    return True
                if parent in target_link_set:
                    break  # another target link comes first
                current = parent
        return False

    conn_matrix = jnp.zeros((n, n))
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)
        for j in range(i + 1, n):
            if is_adjacent(target_link_names[i], target_link_names[j]):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix


class Offset4DofVar(
    jaxls.Var[jnp.ndarray],
    default_factory=lambda: jnp.zeros(4),
):
    """The segment's single 4-DOF world offset variable [tx, ty, tz, yaw]."""


def offset_4dof_to_se3(params: jnp.ndarray) -> jaxlie.SE3:
    """[tx, ty, tz, yaw] -> T_{base<-world} (yaw-only rotation about Z + translation)."""
    tx, ty, tz = params[0], params[1], params[2]
    yaw = params[3]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    R = jnp.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_matrix(R),
        translation=jnp.array([tx, ty, tz]),
    )


def se3_to_offset_4dof(T: np.ndarray) -> np.ndarray:
    """4x4 T_{base<-world} -> [tx, ty, tz, yaw] (assumes the rotation is yaw-only)."""
    tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
    yaw = np.arctan2(T[0, 1], T[0, 0])
    return np.array([tx, ty, tz, yaw], dtype=np.float64)


@jax.jit
def solve_offset_and_joints(
    robot: pk.Robot,
    local_keypoints_world: jnp.ndarray,
    local_link_indices: jnp.ndarray,
    local_conn_mask: jnp.ndarray,
    local_kpt_mask: jnp.ndarray,
    global_keypoints_world: jnp.ndarray,
    global_link_indices: jnp.ndarray,
    global_kpt_mask: jnp.ndarray,

    joint_mask: jnp.ndarray,
    initial_cfg: jnp.ndarray,
    initial_offset_4dof: jnp.ndarray,

    weights: RetargetingWeights,
    rest_weight_per_joint: jnp.ndarray,

    camera_link_index: int,
    target_cam_se3_world: jaxlie.SE3,
    cam_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Offset solve: jointly optimize the 4-DOF offset and coarse joints on representative frames.
    Targets are in WORLD frame. Padded frames carry mask 0. Only the offset is used downstream."""
    timesteps = local_keypoints_world.shape[0]
    n_local = local_keypoints_world.shape[1]

    weight_local = weights["local_alignment"]
    weight_global = weights["global_alignment"]
    weight_cam_rot = weights["ego_view_rot"]
    weight_cam_pos = weights["ego_view_pos"]
    weight_limit = weights["joint_limit"]

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_offset = Offset4DofVar(0)

    @jaxls.Cost.factory
    def local_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_off: Offset4DofVar,
        keypoints_w: jnp.ndarray,
        kpt_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_base_world = offset_4dof_to_se3(var_values[var_off])
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        robot_pos = T_root_link.translation()[local_link_indices]

        kpts_base = T_base_world.apply(keypoints_w)
        pair_mask = kpt_mask[:, None] * kpt_mask[None, :]
        delta_target = kpts_base[:, None] - kpts_base[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Epsilon on the vector, not the norm: norm(v) has no gradient at v=0 (nan Jacobian)
        delta_target_n = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True)
        delta_robot_n = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True)

        residual = 1 - (delta_target_n * delta_robot_n).sum(axis=-1)
        residual = residual * (1 - jnp.eye(n_local)) * local_conn_mask * pair_mask
        return residual.flatten() * weight_local

    @jaxls.Cost.factory
    def global_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_off: Offset4DofVar,
        keypoints_w: jnp.ndarray,
        kpt_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_base_world = offset_4dof_to_se3(var_values[var_off])
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        robot_pos = T_root_link.translation()[global_link_indices]

        kpts_base = T_base_world.apply(keypoints_w)
        diff = (robot_pos - kpts_base) * (weight_global * kpt_mask)[..., None]
        return diff.flatten()

    @jaxls.Cost.factory
    def camera_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_off: Offset4DofVar,
        target_cam_w: jaxlie.SE3,
        c_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_base_world = offset_4dof_to_se3(var_values[var_off])
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        T_cam_actual = jaxlie.SE3(T_root_link.wxyz_xyz[camera_link_index])

        target_base = T_base_world @ target_cam_w
        # SE3.log() is [translation (3), rotation (3)]
        err_log = (target_base.inverse() @ T_cam_actual).log()
        pos_res = err_log[:3] * weight_cam_pos
        rot_res = err_log[3:] * weight_cam_rot
        return jnp.concatenate([pos_res, rot_res]) * c_mask

    @jaxls.Cost.factory
    def joint_limit_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        cfg_full = robot.joints.get_full_config(cfg_eff)
        upper_viol = jnp.maximum(0.0, cfg_full - robot.joints.upper_limits_all)
        lower_viol = jnp.maximum(0.0, robot.joints.lower_limits_all - cfg_full)
        return jnp.concatenate([upper_viol, lower_viol]).flatten() * weight_limit

    costs = [
        local_alignment_cost(
            var_joints, var_offset, local_keypoints_world, local_kpt_mask),
        global_alignment_cost(
            var_joints, var_offset, global_keypoints_world, global_kpt_mask),
        camera_alignment_cost(
            var_joints, var_offset, target_cam_se3_world, cam_mask),
        joint_limit_cost(var_joints),
    ]

    # No temporal smoothness: the offset solve runs on a subsampled frame set
    locked_mask = (1 - joint_mask).astype(bool)
    costs.append(pk.costs.rest_cost(
        var_joints, rest_pose=initial_cfg[None],
        weight=jnp.where(locked_mask, LOCKED_REST_WEIGHT, 0.0)[None],
    ))
    costs.append(pk.costs.rest_cost(
        var_joints, rest_pose=initial_cfg[None],
        weight=(rest_weight_per_joint * joint_mask)[None],
    ))

    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var_joints, var_offset])
        .analyze()
        .solve(
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            termination=jaxls.TerminationConfig(
                max_iterations=1000, early_termination=True),
            initial_vals=jaxls.VarValues.make([
                var_joints.with_value(
                    jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))),
                var_offset.with_value(initial_offset_4dof),
            ]),
            verbose=False,
        )
    )

    return solution[var_offset], solution[var_joints]


@jax.jit
def solve_retargeting(
    robot: pk.Robot,
    local_keypoints: jnp.ndarray,
    local_link_indices: jnp.ndarray,
    local_conn_mask: jnp.ndarray,
    local_kpt_mask: jnp.ndarray,
    global_keypoints: jnp.ndarray,
    global_link_indices: jnp.ndarray,
    global_kpt_mask: jnp.ndarray,

    joint_mask: jnp.ndarray,
    initial_cfg: jnp.ndarray,

    weights: RetargetingWeights,
    rest_weight_per_joint: jnp.ndarray,
    padding_mask: jnp.ndarray,

    camera_link_index: int,
    target_cam_se3: jaxlie.SE3,
    cam_mask: jnp.ndarray,

    hand_joint_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Trajectory solve: joints only, with the offset frozen. All targets are in base frame."""
    timesteps = local_keypoints.shape[0]
    n_local = local_keypoints.shape[1]

    init_guess = jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))

    weight_local = weights["local_alignment"]
    weight_global = weights["global_alignment"]
    weight_smooth = weights["joint_smoothness"]
    hand_smooth_scale = weights["hand_smoothness_scale"]
    weight_cam_rot = weights["ego_view_rot"]
    weight_cam_pos = weights["ego_view_pos"]
    weight_limit = weights["joint_limit"]

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))

    @jaxls.Cost.factory
    def local_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
        kpt_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        robot_pos = T_root_link.translation()[local_link_indices]

        pair_mask = kpt_mask[:, None] * kpt_mask[None, :]
        delta_target = keypoints[:, None] - keypoints[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Epsilon on the vector, not the norm. See the offset-solve cost.
        delta_target_n = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True)
        delta_robot_n = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True)

        residual = 1 - (delta_target_n * delta_robot_n).sum(axis=-1)
        residual = residual * (1 - jnp.eye(n_local)) * local_conn_mask * pair_mask
        return residual.flatten() * weight_local

    @jaxls.Cost.factory
    def global_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
        kpt_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        robot_pos = T_root_link.translation()[global_link_indices]

        diff = (robot_pos - keypoints) * (weight_global * kpt_mask)[..., None]
        return diff.flatten()

    @jaxls.Cost.factory
    def camera_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        target_se3: jaxlie.SE3,
        c_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        T_cam_actual = jaxlie.SE3(T_root_link.wxyz_xyz[camera_link_index])

        # SE3.log() is [translation (3), rotation (3)]
        err_log = (target_se3.inverse() @ T_cam_actual).log()
        pos_res = err_log[:3] * weight_cam_pos
        rot_res = err_log[3:] * weight_cam_rot
        return jnp.concatenate([pos_res, rot_res]) * c_mask

    @jaxls.Cost.factory
    def joint_limit_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        cfg_full = robot.joints.get_full_config(cfg_eff)
        upper_viol = jnp.maximum(0.0, cfg_full - robot.joints.upper_limits_all)
        lower_viol = jnp.maximum(0.0, robot.joints.lower_limits_all - cfg_full)
        return jnp.concatenate([upper_viol, lower_viol]).flatten() * weight_limit

    costs = [
        local_alignment_cost(var_joints, local_keypoints, local_kpt_mask),
        global_alignment_cost(var_joints, global_keypoints, global_kpt_mask),
        camera_alignment_cost(var_joints, target_cam_se3, cam_mask),
        joint_limit_cost(var_joints),
    ]

    # 1st-order smoothness, broken at the padding boundary. Hand joints get a reduced weight.
    if timesteps > 1:
        pair_real = padding_mask[1:] * padding_mask[:-1]
        per_joint_smooth = weight_smooth * joint_mask * (
            1.0 - hand_joint_mask + hand_joint_mask * hand_smooth_scale)
        smooth_w = pair_real[:, None] * per_joint_smooth[None, :]
        costs.append(
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                smooth_w,
            )
        )

    # Locked joints held at the home pose
    locked_mask = (1 - joint_mask).astype(bool)
    costs.append(
        pk.costs.rest_cost(
            var_joints,
            rest_pose=initial_cfg[None],
            weight=jnp.where(locked_mask, LOCKED_REST_WEIGHT, 0.0)[None],
        )
    )

    costs.append(
        pk.costs.rest_cost(
            var_joints,
            rest_pose=initial_cfg[None],
            weight=(rest_weight_per_joint * joint_mask)[None],
        )
    )

    solution, summary = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var_joints])
        .analyze()
        .solve(
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            termination=jaxls.TerminationConfig(
                max_iterations=1000, early_termination=True,
                parameter_tolerance=1e-6 if timesteps <= PARAMETER_TOLERANCE_MAX_PAD else 0.0),
            initial_vals=jaxls.VarValues.make([
                var_joints.with_value(init_guess),
            ]),
            verbose=False,
            return_summary=True,
        )
    )

    final_cost = jax.lax.dynamic_index_in_dim(
        summary.cost_history, summary.iterations - 1, keepdims=False)

    return solution[var_joints], final_cost


class Retargeter:
    """Retargets human hand keypoints to robot joint angles for one robot config."""

    def __init__(self, robot_cfg):
        self.config = robot_cfg.config
        logger.disable("jaxls")

        urdf_path = Path(robot_cfg.urdf_path)

        def filename_handler(fname: str) -> str:
            return yourdfpy.filename_handler_magic(fname, dir=urdf_path.parent)

        self.urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
        self.robot = pk.Robot.from_urdf(self.urdf)

        self.weights = RetargetingWeights(**DEFAULT_WEIGHTS)

        self._build_joint_mappings()
        self._build_keypoint_mappings()

    def _build_joint_mappings(self):
        """Split actuated joints into optimized (config groups) and locked. Mark hand/stiff subsets."""
        actuated_names = list(self.robot.joints.actuated_names)
        n_actuated = len(actuated_names)

        self.joint_groups = self.config["joints"]
        self.active_joint_names = []
        for group_joints in self.joint_groups.values():
            self.active_joint_names.extend(group_joints)

        self.joint_mask = np.zeros(n_actuated, dtype=np.float32)  # 1 = optimize, 0 = lock
        active_joint_indices = []
        for name in self.active_joint_names:
            assert name in actuated_names, f"config joint not actuated in URDF: {name}"
            idx = actuated_names.index(name)
            self.joint_mask[idx] = 1.0
            active_joint_indices.append(idx)
        self.active_joint_indices = np.array(active_joint_indices, dtype=np.int64)

        stiff_joint_names = []
        for group_name in self.config.get("stiff_groups") or []:
            stiff_joint_names.extend(self.joint_groups.get(group_name, []))
        self.stiff_joint_indices = np.array([
            actuated_names.index(n) for n in stiff_joint_names if n in actuated_names
        ], dtype=np.int64)

        self.hand_joint_mask = np.zeros(n_actuated, dtype=np.float32)
        for group_name in self.config.get("hand_groups") or []:
            for name in self.joint_groups.get(group_name, []):
                if name in actuated_names:
                    self.hand_joint_mask[actuated_names.index(name)] = 1.0

        print(f"Active joints: {len(self.active_joint_indices)} / {n_actuated}")
        for group_name, group_joints in self.joint_groups.items():
            print(f"  {group_name}: {len(group_joints)}")

    def _build_keypoint_mappings(self):
        """Resolve the config's keypoint mapping, camera link and EEF links to URDF link indices."""
        link_names = list(self.robot.links.names)
        mapping = self.config["keypoint_mapping"]

        for side in ("left", "right"):
            side_map = mapping[side]
            local_kpt, local_link = [], []
            global_kpt, global_link = [], []

            for kpt_idx, link_name in sorted(side_map.items(), key=lambda x: int(x[0])):
                kpt_idx = int(kpt_idx)
                assert link_name in link_names, f"{side} link not in URDF: {link_name}"
                lidx = link_names.index(link_name)
                local_kpt.append(kpt_idx)
                local_link.append(lidx)
                if kpt_idx in GLOBAL_TIP_INDICES:
                    global_kpt.append(kpt_idx)
                    global_link.append(lidx)

            setattr(self, f"{side}_local_kpt_indices", np.array(local_kpt, dtype=np.int64))
            setattr(self, f"{side}_local_link_indices", np.array(local_link, dtype=np.int64))
            setattr(self, f"{side}_global_kpt_indices", np.array(global_kpt, dtype=np.int64))
            setattr(self, f"{side}_global_link_indices", np.array(global_link, dtype=np.int64))

        for side in ("left", "right"):
            n_tips = len(getattr(self, f"{side}_global_kpt_indices"))
            assert n_tips == 5, f"expected 5 global tips for {side}, got {n_tips}"

        print(f"Local keypoints: {len(self.left_local_kpt_indices)}/hand (left), "
              f"{len(self.right_local_kpt_indices)}/hand (right). Global tips: 5/hand")

        all_local = np.concatenate([self.left_local_link_indices, self.right_local_link_indices])
        self.conn_mask = np.array(create_conn_tree(self.robot, jnp.array(all_local), self.urdf))

        self.global_link_indices = np.concatenate([
            self.left_global_link_indices, self.right_global_link_indices])

        camera_link = self.config["camera_link"]
        assert camera_link in link_names, f"camera link not in URDF: {camera_link}"
        self.camera_link_index = link_names.index(camera_link)

        eef_cfg = self.config["eef_link_names"]
        self.left_eef_link_index = link_names.index(eef_cfg["left"])
        self.right_eef_link_index = link_names.index(eef_cfg["right"])

        # Hand joint positions within the active state vector (config-driven EEF layout)
        hand_groups = self.config["eef_hand_joint_groups"]
        self.left_hand_active_indices = np.array([
            self.active_joint_names.index(n) for n in self.joint_groups[hand_groups["left"]]],
            dtype=np.int64)
        self.right_hand_active_indices = np.array([
            self.active_joint_names.index(n) for n in self.joint_groups[hand_groups["right"]]],
            dtype=np.int64)
        self.eef_dim = (9 + len(self.left_hand_active_indices)) + \
                       (9 + len(self.right_hand_active_indices))

    def get_neutral_config(self) -> np.ndarray:
        """Home configuration over the URDF's actuated joints (unlisted joints are 0)."""
        actuated_names = list(self.robot.joints.actuated_names)
        cfg = np.zeros(len(actuated_names), dtype=np.float32)
        for name, val in (self.config.get("home_config") or {}).items():
            if name in actuated_names:
                cfg[actuated_names.index(name)] = val
        return cfg

    def retarget_bimanual_world(
        self,
        left_kpts_world: np.ndarray,
        right_kpts_world: np.ndarray,
        left_valid: np.ndarray,
        right_valid: np.ndarray,
        target_cam_poses_world: np.ndarray,
        cam_valid: np.ndarray,
        initial_offset_4dof: np.ndarray,
        max_seq_len: int = 224,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run both solves and return joints (T, N_actuated), offset (4,) and the trajectory cost.
        kpts: (T, 21, 3) world-frame. Cam poses: (T, 4, 4) OpenCV cam-to-world. T <= max_seq_len."""
        T = left_kpts_world.shape[0]
        assert left_kpts_world.shape == (T, 21, 3)
        assert right_kpts_world.shape == (T, 21, 3)
        assert target_cam_poses_world.shape == (T, 4, 4)
        assert T <= max_seq_len, f"T={T} > max_seq_len={max_seq_len}"

        initial_cfg = self.get_neutral_config()

        left_valid = np.asarray(left_valid, dtype=np.float32)
        right_valid = np.asarray(right_valid, dtype=np.float32)
        cam_valid = np.asarray(cam_valid, dtype=np.float32)

        # Rest prior: default for arms, stiff for the config's stiff groups, reduced for hands
        rest_weight_per_joint = np.full_like(
            self.joint_mask, self.weights["rest_weight_default"])
        rest_weight_per_joint[self.stiff_joint_indices] = self.weights["rest_weight_stiff"]
        hand_rest = self.weights["rest_weight_default"] * self.weights["hand_rest_scale"]
        rest_weight_per_joint = np.where(
            self.hand_joint_mask > 0.5, hand_rest, rest_weight_per_joint)

        left_local_kpts = left_kpts_world[:, self.left_local_kpt_indices, :]
        right_local_kpts = right_kpts_world[:, self.right_local_kpt_indices, :]
        local_kpts_world = np.concatenate([left_local_kpts, right_local_kpts], axis=1)
        local_link_indices = np.concatenate([
            self.left_local_link_indices, self.right_local_link_indices])

        left_global_kpts = left_kpts_world[:, self.left_global_kpt_indices, :]
        right_global_kpts = right_kpts_world[:, self.right_global_kpt_indices, :]
        global_kpts_world = np.concatenate([left_global_kpts, right_global_kpts], axis=1)

        # Per-hand validity broadcast to every keypoint of that hand
        local_kpt_mask = np.concatenate([
            np.tile(left_valid[:, None], (1, len(self.left_local_kpt_indices))),
            np.tile(right_valid[:, None], (1, len(self.right_local_kpt_indices))),
        ], axis=1)
        global_kpt_mask = np.concatenate([
            np.tile(left_valid[:, None], (1, len(self.left_global_kpt_indices))),
            np.tile(right_valid[:, None], (1, len(self.right_global_kpt_indices))),
        ], axis=1)

        target_cam_se3_world = jaxlie.SE3.from_matrix(
            jnp.array(target_cam_poses_world, dtype=jnp.float32))

        # ---- Offset solve: offset on representative frames, padded to N_REPRESENTATIVE ----
        rep_indices = sample_representative_frames(
            local_kpt_mask, cam_valid, n_target=N_REPRESENTATIVE)
        n_rep = len(rep_indices)

        s1_local_kpts = local_kpts_world[rep_indices]
        s1_local_mask = local_kpt_mask[rep_indices]
        s1_global_kpts = global_kpts_world[rep_indices]
        s1_global_mask = global_kpt_mask[rep_indices]
        s1_cam_se3 = jaxlie.SE3(target_cam_se3_world.wxyz_xyz[rep_indices])
        s1_cam_mask = cam_valid[rep_indices]

        s1_pad = N_REPRESENTATIVE - n_rep
        if s1_pad > 0:
            s1_local_kpts = np.pad(s1_local_kpts, ((0, s1_pad), (0, 0), (0, 0)), mode='edge')
            s1_local_mask = np.pad(s1_local_mask, ((0, s1_pad), (0, 0)), mode='constant')
            s1_global_kpts = np.pad(s1_global_kpts, ((0, s1_pad), (0, 0), (0, 0)), mode='edge')
            s1_global_mask = np.pad(s1_global_mask, ((0, s1_pad), (0, 0)), mode='constant')
            last_cam_se3 = jaxlie.SE3(jnp.tile(s1_cam_se3.wxyz_xyz[-1:], (s1_pad, 1)))
            s1_cam_se3 = jaxlie.SE3(jnp.concatenate([
                s1_cam_se3.wxyz_xyz, last_cam_se3.wxyz_xyz], axis=0))
            s1_cam_mask = np.pad(s1_cam_mask, (0, s1_pad), mode='constant')

        optimized_offset, _ = solve_offset_and_joints(
            robot=self.robot,
            local_keypoints_world=jnp.array(s1_local_kpts, dtype=jnp.float32),
            local_link_indices=jnp.array(local_link_indices),
            local_conn_mask=jnp.array(self.conn_mask),
            local_kpt_mask=jnp.array(s1_local_mask, dtype=jnp.float32),
            global_keypoints_world=jnp.array(s1_global_kpts, dtype=jnp.float32),
            global_link_indices=jnp.array(self.global_link_indices),
            global_kpt_mask=jnp.array(s1_global_mask, dtype=jnp.float32),
            joint_mask=jnp.array(self.joint_mask),
            initial_cfg=jnp.array(initial_cfg, dtype=jnp.float32),
            initial_offset_4dof=jnp.array(initial_offset_4dof, dtype=jnp.float32),
            weights=self.weights,
            rest_weight_per_joint=jnp.array(rest_weight_per_joint, dtype=jnp.float32),
            camera_link_index=self.camera_link_index,
            target_cam_se3_world=s1_cam_se3,
            cam_mask=jnp.array(s1_cam_mask, dtype=jnp.float32),
        )

        optimized_offset_4dof = np.array(optimized_offset)
        T_base_world_se3 = offset_4dof_to_se3(optimized_offset)

        # ---- Trajectory solve: targets into base frame, then joints padded to max_seq_len ----
        local_kpts_base = np.array(T_base_world_se3.apply(
            jnp.array(local_kpts_world, dtype=jnp.float32)))
        global_kpts_base = np.array(T_base_world_se3.apply(
            jnp.array(global_kpts_world, dtype=jnp.float32)))
        target_cam_se3_base = T_base_world_se3 @ target_cam_se3_world

        pad_len = max_seq_len - T
        padding_mask = np.ones(max_seq_len, dtype=np.float32)

        if pad_len > 0:
            local_kpts_padded = np.pad(local_kpts_base,
                                       ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            local_mask_padded = np.pad(local_kpt_mask,
                                       ((0, pad_len), (0, 0)), mode='constant')
            global_kpts_padded = np.pad(global_kpts_base,
                                        ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            global_mask_padded = np.pad(global_kpt_mask,
                                        ((0, pad_len), (0, 0)), mode='constant')
            last_cam_se3 = jaxlie.SE3(jnp.tile(
                target_cam_se3_base.wxyz_xyz[-1:], (pad_len, 1)))
            cam_se3_padded = jaxlie.SE3(jnp.concatenate([
                target_cam_se3_base.wxyz_xyz, last_cam_se3.wxyz_xyz], axis=0))
            cam_mask_padded = np.pad(cam_valid, (0, pad_len), mode='constant')
            padding_mask[T:] = 0.0
        else:
            local_kpts_padded = local_kpts_base
            local_mask_padded = local_kpt_mask
            global_kpts_padded = global_kpts_base
            global_mask_padded = global_kpt_mask
            cam_se3_padded = target_cam_se3_base
            cam_mask_padded = cam_valid

        joints_padded, solver_cost = solve_retargeting(
            robot=self.robot,
            local_keypoints=jnp.array(local_kpts_padded, dtype=jnp.float32),
            local_link_indices=jnp.array(local_link_indices),
            local_conn_mask=jnp.array(self.conn_mask),
            local_kpt_mask=jnp.array(local_mask_padded, dtype=jnp.float32),
            global_keypoints=jnp.array(global_kpts_padded, dtype=jnp.float32),
            global_link_indices=jnp.array(self.global_link_indices),
            global_kpt_mask=jnp.array(global_mask_padded, dtype=jnp.float32),
            joint_mask=jnp.array(self.joint_mask),
            initial_cfg=jnp.array(initial_cfg, dtype=jnp.float32),
            weights=self.weights,
            rest_weight_per_joint=jnp.array(rest_weight_per_joint, dtype=jnp.float32),
            padding_mask=jnp.array(padding_mask, dtype=jnp.float32),
            camera_link_index=self.camera_link_index,
            target_cam_se3=cam_se3_padded,
            cam_mask=jnp.array(cam_mask_padded, dtype=jnp.float32),
            hand_joint_mask=jnp.array(self.hand_joint_mask),
        )

        return np.array(joints_padded[:T]), optimized_offset_4dof, float(solver_cost)

    def extract_joints(self, full_cfg: np.ndarray) -> np.ndarray:
        """(..., N_actuated) -> (..., N_active) in the config's group order."""
        actuated_names = list(self.robot.joints.actuated_names)
        indices = np.array([
            actuated_names.index(n) for n in self.active_joint_names if n in actuated_names],
            dtype=np.int64)
        return full_cfg[..., indices]

    def inject_active_joints(self, state: np.ndarray) -> np.ndarray:
        """(..., N_active) -> (..., N_actuated), locked joints filled from the home pose."""
        neutral = self.get_neutral_config()
        if state.ndim == 1:
            full_cfg = neutral.copy()
            full_cfg[self.active_joint_indices] = state
            return full_cfg
        full_cfg = np.tile(neutral, (state.shape[0], 1))
        full_cfg[:, self.active_joint_indices] = state
        return full_cfg

    def compute_eef_batch(self, state: np.ndarray) -> np.ndarray:
        """EEF in base frame via FK: per side [wrist pos3 | wrist rot6d6 | hand joints], left first."""
        from common.geometry import wxyz_xyz_to_matrix

        full_cfg = self.inject_active_joints(state)
        T = state.shape[0]
        n_lh = len(self.left_hand_active_indices)
        n_rh = len(self.right_hand_active_indices)
        L = 9 + n_lh
        state_eef = np.zeros((T, self.eef_dim), dtype=np.float32)

        for t in range(T):
            fk_result = np.array(self.robot.forward_kinematics(cfg=full_cfg[t]))

            T_left_base = wxyz_xyz_to_matrix(fk_result[self.left_eef_link_index])
            T_right_base = wxyz_xyz_to_matrix(fk_result[self.right_eef_link_index])

            state_eef[t, 0:3] = T_left_base[:3, 3]
            state_eef[t, 3:9] = T_left_base[:3, :3][:2].flatten()
            state_eef[t, 9:L] = state[t, self.left_hand_active_indices]

            state_eef[t, L:L + 3] = T_right_base[:3, 3]
            state_eef[t, L + 3:L + 9] = T_right_base[:3, :3][:2].flatten()
            state_eef[t, L + 9:L + 9 + n_rh] = state[t, self.right_hand_active_indices]

        return state_eef

    def compute_fk_positions_for_diagnostics(self, state: np.ndarray) -> dict:
        """Diagnostic FK in base frame: "global_pos" (T, 10, 3) fingertips, "palm_pos" (T, 2, 3)
        wrists, "local_pos" (T, N_local, 3) dense links, "cam_se3" (T, 4, 4)."""
        from common.geometry import wxyz_xyz_to_matrix

        full_cfg = self.inject_active_joints(state)
        T = state.shape[0]

        n_left_local = len(self.left_local_link_indices)
        n_local = n_left_local + len(self.right_local_link_indices)

        global_pos = np.zeros((T, 10, 3), dtype=np.float64)
        palm_pos = np.zeros((T, 2, 3), dtype=np.float64)
        local_pos = np.zeros((T, n_local, 3), dtype=np.float64)
        cam_se3 = np.zeros((T, 4, 4), dtype=np.float64)

        # Where the wrist (MediaPipe index 0) sits inside each side's local link list
        left_wrist_local_idx = int(np.where(self.left_local_kpt_indices == 0)[0][0])
        right_wrist_local_idx = int(np.where(self.right_local_kpt_indices == 0)[0][0])

        all_local_link_indices = np.concatenate([
            self.left_local_link_indices, self.right_local_link_indices])

        for t in range(T):
            fk_result = np.array(self.robot.forward_kinematics(cfg=full_cfg[t]))

            for k, li in enumerate(self.left_global_link_indices):
                global_pos[t, k] = wxyz_xyz_to_matrix(fk_result[li])[:3, 3]
            for k, li in enumerate(self.right_global_link_indices):
                global_pos[t, 5 + k] = wxyz_xyz_to_matrix(fk_result[li])[:3, 3]

            for k, li in enumerate(all_local_link_indices):
                local_pos[t, k] = wxyz_xyz_to_matrix(fk_result[li])[:3, 3]

            palm_pos[t, 0] = local_pos[t, left_wrist_local_idx]
            palm_pos[t, 1] = local_pos[t, n_left_local + right_wrist_local_idx]

            cam_se3[t] = wxyz_xyz_to_matrix(fk_result[self.camera_link_index])

        return {
            "global_pos": global_pos,
            "palm_pos": palm_pos,
            "local_pos": local_pos,
            "cam_se3": cam_se3,
        }

    def compute_local_dir_error(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        kpt_mask: np.ndarray,
    ) -> np.ndarray:
        """Per-hand mean cosine distance over valid connectivity pairs. Returns (T, 2) [left, right]."""
        T = robot_pos.shape[0]
        conn = np.array(self.conn_mask)
        n_left = len(self.left_local_link_indices)
        n_total = robot_pos.shape[1]
        result = np.zeros((T, 2), dtype=np.float64)

        for hand_idx, (start, end) in enumerate([(0, n_left), (n_left, n_total)]):
            r = robot_pos[:, start:end]
            g = target_pos[:, start:end]
            m = kpt_mask[:, start:end]
            c = conn[start:end, start:end]
            n_kpts = end - start

            for t in range(T):
                total_dist = 0.0
                n_pairs = 0
                for i in range(n_kpts):
                    for j in range(i + 1, n_kpts):
                        if c[i, j] < 0.5:
                            continue
                        if m[t, i] < 0.5 or m[t, j] < 0.5:
                            continue
                        dr = r[t, j] - r[t, i]
                        dg = g[t, j] - g[t, i]
                        nr = np.linalg.norm(dr)
                        ng = np.linalg.norm(dg)
                        if nr < 1e-8 or ng < 1e-8:
                            continue
                        cos_sim = np.clip(np.dot(dr / nr, dg / ng), -1.0, 1.0)
                        total_dist += 1.0 - cos_sim
                        n_pairs += 1
                result[t, hand_idx] = total_dist / max(n_pairs, 1)

        return result
