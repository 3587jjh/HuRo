"""Isaac Sim robot renderer: RGB + semantic rendering and compositing onto video frames."""

import numpy as np
import cv2
import time
import logging
import os
from typing import Tuple, Optional

from .camera import CameraParams

logger = logging.getLogger(__name__)


# Override via env vars: OVERLAY_RENDER_WIDTH/HEIGHT, OVERLAY_RENDERER, OVERLAY_ANTI_ALIASING
ISAAC_SIM_CONFIG = {
    "headless": True,
    "width": int(os.environ.get("OVERLAY_RENDER_WIDTH", 1280)),
    "height": int(os.environ.get("OVERLAY_RENDER_HEIGHT", 720)),
    "renderer": os.environ.get("OVERLAY_RENDERER", "PathTracing"),
    "anti_aliasing": int(os.environ.get("OVERLAY_ANTI_ALIASING", 4)),
}



class IsaacSimRobotRenderer:
    """Renders the robot with Isaac Sim and composites it via a semantic soft-alpha matte."""

    def __init__(
        self,
        robot_path: str,
        camera_params: CameraParams,
        render_width: int = 1280,
        render_height: int = 720,
        robot_prim_path: str = "/World/Robot",
        hide_links: Optional[set] = None,
        usd_cache_path: Optional[str] = None,
    ):
        """robot_path may be URDF (imported into a USD at usd_cache_path) or USD."""
        self.robot_path = robot_path
        self.camera_params = camera_params
        self.render_width = render_width
        self.render_height = render_height
        self.robot_prim_path = robot_prim_path
        self._hide_links = hide_links
        self._usd_cache_path = usd_cache_path

        # Physics instability tracking
        self._last_good_joint_positions = None
        self._total_resets = 0
        self._failed_frame_indices = []  # per-video list of frame indices that had render failures
        self._current_frame_index = 0

        self.is_usd = robot_path.lower().endswith(('.usd', '.usda', '.usdc', '.usdz'))

        self.simulation_app = None

        # OVERLAY_NO_SUPPRESS=1 keeps Isaac Sim init output visible (debugging crashes)
        _suppress = not os.environ.get("OVERLAY_NO_SUPPRESS")
        _stdout_fd = os.dup(1)
        _stderr_fd = os.dup(2)
        if _suppress:
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, 1)
            os.dup2(_devnull, 2)
            os.close(_devnull)
        try:
            from isaacsim import SimulationApp
            self.simulation_app = SimulationApp(ISAAC_SIM_CONFIG)

            self._import_isaac_modules()

            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import get_current_stage

            self.world = World(stage_units_in_meters=1.0)
            self.stage = get_current_stage()

            self._setup_scene()

            self._load_robot()

            self._setup_camera()

            self.world.reset()

            self.robot.initialize()

            self._robot_base_position = np.array([0.0, 0.0, 0.0])
            self._robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
            self.robot.set_world_pose(
                position=self._robot_base_position,
                orientation=self._robot_base_orientation,
            )

            self.joint_names = self.robot.dof_names
            self.num_dof = self.robot.num_dof
            if not self.num_dof:
                raise RuntimeError(
                    f"PhysX built no articulation from {self.robot_path}: its joints are not "
                    f"part of one (rigid bodies must be siblings under a single articulation "
                    f"root, and the root API must sit on the root joint). Driving a 0-DOF "
                    f"articulation crashes the simulator, so stopping here."
                )

            self._last_good_joint_positions = np.zeros(self.num_dof)

            self._setup_replicator()
        finally:
            os.dup2(_stdout_fd, 1)
            os.close(_stdout_fd)
            os.dup2(_stderr_fd, 2)
            os.close(_stderr_fd)

        # stderr stays redirected to /dev/null for the renderer's lifetime (restored in cleanup)
        if _suppress:
            self._saved_stderr_fd = os.dup(2)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, 2)
            os.close(_devnull)

        logger.info(f"IsaacSim robot renderer initialized ({self.num_dof} DOF)")

    def _import_isaac_modules(self):
        """Import Isaac Sim modules after SimulationApp is initialized."""
        import omni.replicator.core as rep
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.articulations import Articulation
        from pxr import Gf, UsdGeom, UsdLux, Sdf

        self._rep = rep
        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._Sdf = Sdf
        self._add_reference_to_stage = add_reference_to_stage
        self._Articulation = Articulation

    def _setup_scene(self):
        """Scene lighting, with intensities calibrated for Kit 107 physical light units."""
        distant_light = self._UsdLux.DistantLight.Define(
            self.stage, self._Sdf.Path("/World/DistantLight")
        )
        distant_light.CreateIntensityAttr(3000)
        distant_light.AddOrientOp().Set(self._Gf.Quatf(0.707, 0.707, 0, 0))

        dome_light = self._UsdLux.DomeLight.Define(
            self.stage, self._Sdf.Path("/World/DomeLight")
        )
        dome_light.CreateIntensityAttr(400)

        logger.info("Scene setup complete")

    def _load_robot(self):
        """Load the robot into the scene, importing a URDF into a USD first if needed."""
        if not self.is_usd:
            from .urdf_import import ensure_overlay_usd
            logger.info(f"Importing robot URDF: {self.robot_path}")
            self.robot_path = ensure_overlay_usd(self.robot_path, self._usd_cache_path)
            self.is_usd = True
        logger.info(f"Loading robot from USD: {self.robot_path}")
        self._load_robot_from_usd()

        articulation_path = self._find_articulation_root()
        self.robot = self._Articulation(prim_path=articulation_path)
        logger.info(f"Created robot articulation at {articulation_path}")

        self._apply_semantic_labels()

        if self._hide_links:
            self._hide_link_prims()

    def _load_robot_from_usd(self):
        """Load robot directly from USD file."""
        self._add_reference_to_stage(
            usd_path=self.robot_path,
            prim_path=self.robot_prim_path
        )
        logger.info(f"Loaded USD robot at {self.robot_prim_path}")

    def _apply_semantic_labels(self):
        """Apply semantic labels to robot for semantic segmentation."""
        try:
            from isaacsim.core.utils.semantics import add_update_semantics

            robot_prim = self.stage.GetPrimAtPath(self.robot_prim_path)

            if not robot_prim.IsValid():
                logger.warning(f"Robot prim not found at {self.robot_prim_path}")
                return

            # Semantics are inherited by descendants: labeling the root covers all link geometry
            add_update_semantics(robot_prim, "robot")

            logger.info(f"Applied semantic label 'robot' to {self.robot_prim_path}")

        except Exception as e:
            logger.warning(f"Failed to apply semantic labels: {e}")

    def _find_articulation_root(self) -> str:
        """First prim under robot_prim_path with ArticulationRootAPI, else robot_prim_path."""
        from pxr import Usd, UsdPhysics
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.robot_prim_path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        return self.robot_prim_path

    def _hide_link_prims(self):
        """Hide matched links' own geometry children only. MakeInvisible is inherited, so hiding
        a nested link prim would hide the rest of the chain. Unmatched names are a silent no-op."""
        from pxr import Usd, UsdPhysics
        robot_prim = self.stage.GetPrimAtPath(self.robot_prim_path)
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() not in self._hide_links:
                continue
            for child in prim.GetChildren():
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue                                   # the next link in the chain
                if "Joint" in str(child.GetTypeName()):
                    continue                                   # joints carry no geometry
                if child.IsA(self._UsdGeom.Xformable):
                    self._UsdGeom.Imageable(child).MakeInvisible()

    def _setup_camera(self):
        """Setup camera with calibrated intrinsics and extrinsics."""
        camera_path = "/World/Camera"

        camera_prim = self.stage.DefinePrim(camera_path, "Camera")
        camera = self._UsdGeom.Camera(camera_prim)

        camera.CreateFocalLengthAttr(self.camera_params.focal_length)
        h_ap = self.camera_params.horizontal_aperture
        camera.CreateHorizontalApertureAttr(h_ap)

        # Vertical aperture from fy (correct even when fx != fy)
        vertical_aperture = self.camera_params.focal_length * self.render_height / self.camera_params.fy
        camera.CreateVerticalApertureAttr(vertical_aperture)

        # Principal point offsets (mm). Image y-down → USD y-up.
        off_x = (self.camera_params.cx - 0.5 * self.render_width) * (h_ap / self.render_width)
        off_y = (0.5 * self.render_height - self.camera_params.cy) * (vertical_aperture / self.render_height)
        camera.CreateHorizontalApertureOffsetAttr(off_x)
        camera.CreateVerticalApertureOffsetAttr(off_y)

        camera.CreateClippingRangeAttr(self._Gf.Vec2f(0.00001, 100.0))

        xform = self._UsdGeom.Xformable(camera_prim)

        translate_op = xform.AddTranslateOp()
        translate_op.Set(self._Gf.Vec3d(*self.camera_params.pos))

        w, x, y, z = self.camera_params.ori_wxyz
        quat = self._Gf.Quatf(w, x, y, z)
        rotate_op = xform.AddOrientOp()
        rotate_op.Set(quat)

        self.camera_path = camera_path
        self.camera_prim = camera_prim
        self.camera_xform = xform

        logger.info(f"Camera setup at {camera_path}")
        logger.info(f"  Position: {self.camera_params.pos}")
        logger.info(f"  Orientation (WXYZ): {self.camera_params.ori_wxyz}")

    def _update_camera_intrinsics_on_prim(self, camera_params):
        """Update camera USD prim intrinsics (focal length, aperture, offsets)."""
        camera = self._UsdGeom.Camera(self.camera_prim)
        fl = camera_params.focal_length
        h_ap = camera_params.horizontal_aperture
        v_ap = fl * self.render_height / camera_params.fy
        camera.GetFocalLengthAttr().Set(fl)
        camera.GetHorizontalApertureAttr().Set(h_ap)
        camera.GetVerticalApertureAttr().Set(v_ap)
        off_x = (camera_params.cx - 0.5 * self.render_width) * (h_ap / self.render_width)
        off_y = (0.5 * self.render_height - camera_params.cy) * (v_ap / self.render_height)
        camera.GetHorizontalApertureOffsetAttr().Set(off_x)
        camera.GetVerticalApertureOffsetAttr().Set(off_y)

    def update_camera_pose(self, camera_params: CameraParams, update_intrinsics=False):
        """Update the camera pose (and optionally intrinsics) for a new frame."""
        self.camera_params = camera_params

        xform = self._UsdGeom.Xformable(self.camera_prim)
        xform_ops = xform.GetOrderedXformOps()

        translate_op = xform_ops[0]
        translate_op.Set(self._Gf.Vec3d(*camera_params.pos))

        w, x, y, z = camera_params.ori_wxyz
        quat = self._Gf.Quatf(w, x, y, z)
        rotate_op = xform_ops[1]
        rotate_op.Set(quat)

        if update_intrinsics:
            self._update_camera_intrinsics_on_prim(camera_params)

    def _setup_replicator(self):
        """Setup replicator for synthetic data generation."""
        logger.info(f"Setting up replicator for camera: {self.camera_path}")

        self.render_product = self._rep.create.render_product(
            self.camera_path,
            resolution=(self.render_width, self.render_height)
        )

        self.rgb_annotator = self._rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

        self.semantic_annotator = self._rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        self.semantic_annotator.attach([self.render_product])

        logger.info("Replicator setup complete")

    def _teardown_replicator(self):
        """Detach annotators and destroy render product (keeps world/robot/camera intact)."""
        if hasattr(self, 'rgb_annotator') and self.rgb_annotator is not None:
            try:
                self.rgb_annotator.detach()
            except Exception:
                pass

        if hasattr(self, 'semantic_annotator') and self.semantic_annotator is not None:
            try:
                self.semantic_annotator.detach()
            except Exception:
                pass

        if hasattr(self, 'render_product') and self.render_product is not None:
            try:
                self._rep.destroy(self.render_product)
            except Exception:
                pass

        try:
            self._rep.orchestrator.stop()
        except Exception:
            pass

    def update_resolution(self, width: int, height: int):
        """Recreate the render product/annotators at a new resolution (world/robot/camera kept)."""
        if width == self.render_width and height == self.render_height:
            return
        _stdout_fd = os.dup(1)
        _stderr_fd = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 1)
        os.dup2(_devnull, 2)
        os.close(_devnull)
        try:
            self._teardown_replicator()
            self.render_width = width
            self.render_height = height
            self._setup_replicator()
        finally:
            os.dup2(_stdout_fd, 1)
            os.close(_stdout_fd)
            os.dup2(_stderr_fd, 2)
            os.close(_stderr_fd)

    def _reset_physics(self, joint_positions: np.ndarray):
        """Reset physics and re-apply the base pose + joint_positions."""
        self._total_resets += 1
        logger.warning(
            f"Resetting physics simulation (reset #{self._total_resets}). "
            f"Re-applying joint positions."
        )

        try:
            self.world.reset()

            self.robot.initialize()

            self.robot.set_world_pose(
                position=self._robot_base_position,
                orientation=self._robot_base_orientation,
            )

            self.robot.set_joint_positions(joint_positions)
            self.world.step(render=False)
            self.robot.set_joint_positions(joint_positions)
            self.world.render()

            logger.info("Physics reset completed successfully")

        except Exception as e:
            logger.error(f"Physics reset failed: {e}")

    def _check_render_output(self, rgb: np.ndarray) -> bool:
        """True unless the render is empty or nearly black."""
        if rgb is None or rgb.size == 0:
            return False
        if rgb.mean() < 2.0:
            return False
        return True

    def update_robot_state(self, joint_positions: np.ndarray):
        """Drive the articulation to joint_positions (padded/truncated to num_dof)."""
        input_dof = len(joint_positions)

        if input_dof != self.num_dof:
            logger.debug(f"DOF mismatch: input={input_dof}, robot={self.num_dof}")

            if input_dof < self.num_dof:
                padded_positions = np.zeros(self.num_dof)
                padded_positions[:input_dof] = joint_positions
                joint_positions = padded_positions
            else:
                joint_positions = joint_positions[:self.num_dof]
                logger.warning(f"Truncated joint positions from {input_dof} to {self.num_dof} DOFs")

        # Set -> step -> re-set: the physics step propagates transforms but may perturb positions.
        # Zeroing joint velocities prevents PhysX velocity blow-up. Disable it with OVERLAY_ZERO_VEL=0.
        _zero_vel = os.environ.get("OVERLAY_ZERO_VEL", "1") != "0"
        _zv = np.zeros(self.num_dof, dtype=joint_positions.dtype) if _zero_vel else None
        self.robot.set_joint_positions(joint_positions)
        if _zero_vel:
            try: self.robot.set_joint_velocities(_zv)
            except Exception: pass
        self.world.step(render=False)
        self.robot.set_joint_positions(joint_positions)
        if _zero_vel:
            try: self.robot.set_joint_velocities(_zv)
            except Exception: pass
        self.world.render()
        self._last_good_joint_positions = joint_positions.copy()

    def _capture_frame(self):
        """Flush one frame to the annotators: only rep.orchestrator.step fills annotator buffers
        (world.render never does). pause_timeline=False keeps world.step propagating joint writes."""
        self._rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)

    def render_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Render the scene and return (rgb uint8 [H, W, 3], semantic_mask uint32 [H, W] or None)."""
        self._capture_frame()

        rgb = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rgb_data = self.rgb_annotator.get_data()

                if isinstance(rgb_data, dict):
                    rgb = rgb_data.get("data", rgb_data.get("rgb", None))
                else:
                    rgb = rgb_data

                if rgb is None or rgb.size == 0:
                    if attempt < max_retries - 1:
                        time.sleep(0.05)
                        self._capture_frame()
                        continue
                    else:
                        rgb = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
                        break

                rgb = np.array(rgb, dtype=np.uint8)

                if len(rgb.shape) == 3 and rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]

                break

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.05)
                    self._capture_frame()
                else:
                    logger.error(f"Error getting RGB data: {e}")
                    rgb = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

        if not self._check_render_output(rgb):
            logger.warning("Blank render output detected, attempting recovery...")
            if self._last_good_joint_positions is not None:
                self._reset_physics(self._last_good_joint_positions)
                self._capture_frame()
                try:
                    rgb_data = self.rgb_annotator.get_data()
                    if isinstance(rgb_data, dict):
                        rgb = rgb_data.get("data", rgb_data.get("rgb", None))
                    else:
                        rgb = rgb_data
                    if rgb is not None and rgb.size > 0:
                        rgb = np.array(rgb, dtype=np.uint8)
                        if len(rgb.shape) == 3 and rgb.shape[-1] == 4:
                            rgb = rgb[..., :3]
                    else:
                        rgb = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

        semantic_mask = None
        try:
            semantic_data = self.semantic_annotator.get_data()
            if isinstance(semantic_data, dict):
                semantic_mask = semantic_data.get("data", None)
            else:
                semantic_mask = semantic_data
            if semantic_mask is not None:
                semantic_mask = np.array(semantic_mask, dtype=np.uint32)
        except Exception as e:
            logger.warning(f"Error getting semantic data: {e}")

        return rgb, semantic_mask

    def create_overlay(
        self,
        robot_cfg: np.ndarray,
        video_frame: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Composite the robot at robot_cfg onto video_frame (alpha 0=video only, 1=full robot)."""
        self.update_robot_state(robot_cfg)

        robot_rgb, semantic_mask = self.render_frame()

        if robot_rgb is None or robot_rgb.size == 0:
            logger.warning("Render failed, returning original frame")
            return video_frame

        if robot_rgb.dtype != np.uint8:
            robot_rgb = robot_rgb.astype(np.uint8)
        if len(robot_rgb.shape) == 2:
            robot_rgb = cv2.cvtColor(robot_rgb, cv2.COLOR_GRAY2RGB)
        elif robot_rgb.shape[-1] == 4:
            robot_rgb = robot_rgb[..., :3]

        if robot_rgb.shape[:2] != video_frame.shape[:2]:
            try:
                robot_rgb = cv2.resize(
                    robot_rgb,
                    (video_frame.shape[1], video_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                if semantic_mask is not None and semantic_mask.size > 0:
                    # uint32 -> int32: cv2.resize cannot handle uint32
                    if semantic_mask.dtype == np.uint32:
                        semantic_mask = semantic_mask.astype(np.int32)
                    semantic_mask = cv2.resize(
                        semantic_mask,
                        (video_frame.shape[1], video_frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
            except Exception as e:
                logger.error(f"Error resizing rendered images: {e}")
                return video_frame

        # The rgb annotator's alpha is opaque everywhere: build a soft matte from semantic
        # distance-to-boundary instead (0-1.5px edge band ramps alpha 0->1).
        sem_binary = None
        if semantic_mask is not None:
            sem_binary = (semantic_mask > 0).astype(np.uint8)
        if semantic_mask is None or robot_rgb.mean() < 2.0 or sem_binary.sum() == 0:
            logger.warning("Render failed (no semantic mask, black frame, or empty mask), returning original frame")
            self._failed_frame_indices.append(self._current_frame_index)
            self._current_frame_index += 1
            return video_frame
        dist_in = cv2.distanceTransform(sem_binary, cv2.DIST_L2, 3).astype(np.float32)
        robot_alpha = np.clip(dist_in / 1.5, 0.0, 1.0)

        robot_alpha *= float(alpha)
        robot_alpha = np.clip(robot_alpha, 0.0, 1.0)

        alpha_3ch = robot_alpha[..., None]  # [H, W, 1]
        overlay_frame = (
            robot_rgb.astype(np.float32) * alpha_3ch +
            video_frame.astype(np.float32) * (1.0 - alpha_3ch)
        )
        overlay_frame = np.clip(overlay_frame, 0, 255).astype(np.uint8)

        self._current_frame_index += 1
        return overlay_frame

    @property
    def failed_frame_indices(self) -> list:
        """Frame indices that had render failures during the current video."""
        return self._failed_frame_indices

    def reset_instability_tracking(self):
        """Reset per-video render-failure tracking. Call between videos."""
        self._failed_frame_indices = []
        self._current_frame_index = 0

    def set_robot_base_pose(self, position, orientation):
        """Set robot base pose in world frame. The orientation is a WXYZ quaternion."""
        self._robot_base_position = np.asarray(position, dtype=np.float64)
        self._robot_base_orientation = np.asarray(orientation, dtype=np.float64)
        self.robot.set_world_pose(
            position=self._robot_base_position,
            orientation=self._robot_base_orientation,
        )

    def cleanup(self):
        """Clean up Isaac Sim resources."""
        # Restore stderr before cleanup so errors during teardown are visible
        if hasattr(self, '_saved_stderr_fd'):
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            del self._saved_stderr_fd

        if self._total_resets > 0:
            logger.info(
                f"Physics stability summary: {self._total_resets} total resets performed"
            )

        try:
            if hasattr(self, 'rgb_annotator') and self.rgb_annotator is not None:
                try:
                    self.rgb_annotator.detach()
                except:
                    pass

            if hasattr(self, 'semantic_annotator') and self.semantic_annotator is not None:
                try:
                    self.semantic_annotator.detach()
                except:
                    pass

            # Destroy render product before clearing world
            if hasattr(self, 'render_product') and self.render_product is not None:
                try:
                    self._rep.destroy(self.render_product)
                except Exception:
                    pass

            try:
                self._rep.orchestrator.stop()
            except Exception:
                pass

            if hasattr(self, 'world') and self.world is not None:
                try:
                    self.world.clear()
                except:
                    pass

            # Explicitly close SimulationApp to prevent segfaults on exit
            if self.simulation_app is not None:
                try:
                    self.simulation_app.close()
                except Exception:
                    pass

            logger.info("IsaacSim renderer cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
