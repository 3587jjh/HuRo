"""Frame-level driver for the Isaac Sim robot overlay."""
import logging
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from .camera import CameraParams, calculate_camera_params_from_intrinsics
from .renderer import IsaacSimRobotRenderer

logger = logging.getLogger(__name__)


class RobotOverlayProcessor:
    """Renders a robot overlay onto pre-decoded video frames via the Isaac Sim renderer."""

    def __init__(
        self,
        robot_path: str,
        camera_params: CameraParams,
        render_width: Optional[int] = None,
        render_height: Optional[int] = None,
        robot_prim_path: str = "/World/Robot",
        joint_mapping: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hide_links: Optional[set] = None,
        usd_cache_path: Optional[str] = None,
    ):
        """joint_mapping maps input joint states into the articulation's DOF order."""
        self.robot_path = robot_path
        self.camera_params = camera_params
        self.joint_mapping = joint_mapping

        render_width = render_width or camera_params.width
        render_height = render_height or camera_params.height

        self.renderer = IsaacSimRobotRenderer(
            robot_path=robot_path,
            camera_params=camera_params,
            render_width=render_width,
            render_height=render_height,
            robot_prim_path=robot_prim_path,
            hide_links=hide_links,
            usd_cache_path=usd_cache_path,
        )
        
        logger.info(f"Robot articulation has {self.renderer.num_dof} DOFs")
        if self.renderer.robot and hasattr(self.renderer.robot, 'dof_names'):
            joint_names = self.renderer.robot.dof_names
            if len(joint_names) > 10:
                logger.info(f"  Joint names (first 10): {joint_names[:10]}")
            else:
                logger.info(f"  Joint names: {joint_names}")
        
        logger.info("Robot overlay processor initialized")
    
    def update_camera_and_resolution(self, camera_params: CameraParams, width: int, height: int):
        """Update camera and resolution between clips (only the render product is resized)."""
        self.camera_params = camera_params
        self.renderer.update_resolution(width, height)
        self.renderer.update_camera_pose(camera_params, update_intrinsics=True)

    def set_robot_base_pose(self, position, orientation):
        """Set robot base pose in world frame (delegates to renderer)."""
        self.renderer.set_robot_base_pose(position, orientation)

    def process_frames(
        self,
        frames: List[np.ndarray],
        robot_trajectories: np.ndarray,
        camera_poses: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        is_camera_to_world: bool = True,
        alpha: float = 1.0,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[np.ndarray]:
        """Overlay the robot onto each frame and return the composited RGB frames.
        frames: RGB uint8, robot_trajectories: [T, input DOFs], camera_poses: [T, 4, 4]."""
        num_frames = min(len(frames), len(robot_trajectories))
        overlay_frames = []

        for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
            frame_rgb = frames[frame_idx]
            robot_cfg = robot_trajectories[frame_idx]

            if self.joint_mapping is not None:
                robot_cfg = self.joint_mapping(robot_cfg)

            if camera_poses is not None and frame_idx < len(camera_poses):
                extrinsics = camera_poses[frame_idx]
                intrinsics = camera_intrinsics if camera_intrinsics is not None else None

                if intrinsics is not None:
                    camera_params_frame = calculate_camera_params_from_intrinsics(
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        width=self.renderer.render_width,
                        height=self.renderer.render_height,
                        is_world_to_camera=(not is_camera_to_world),
                        verbose=False,
                    )
                    self.renderer.update_camera_pose(camera_params_frame, update_intrinsics=True)

            overlay_rgb = self.renderer.create_overlay(
                robot_cfg=robot_cfg,
                video_frame=frame_rgb,
                alpha=alpha,
            )
            overlay_frames.append(overlay_rgb)
            if progress_cb is not None:
                progress_cb(frame_idx)

        return overlay_frames
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.renderer.cleanup()
        except Exception as e:
            logger.warning(f"Error during processor cleanup: {e}")

