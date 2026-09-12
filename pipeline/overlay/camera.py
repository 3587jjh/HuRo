"""Camera parameter handling and OpenCV -> USD coordinate conversion for Isaac Sim."""

import numpy as np
import math
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class CameraParams:
    """Camera calibration parameters for Isaac Sim."""
    name: str
    pos: np.ndarray  # [3] position in world frame
    ori_wxyz: np.ndarray  # [4] quaternion in WXYZ format
    fx: float  # focal length x (pixels)
    fy: float  # focal length y (pixels)
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)
    width: int  # image width
    height: int  # image height
    horizontal_aperture: float  # in mm
    focal_length: float  # in mm


def calculate_camera_params_from_intrinsics(
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    is_world_to_camera: bool = True,
    verbose: bool = True,
) -> CameraParams:
    """Convert OpenCV intrinsics/extrinsics to Isaac Sim CameraParams (USD/OpenGL convention).
    extrinsics is world-to-camera when is_world_to_camera, else camera-to-world."""
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    
    # Fixed 50mm focal length, with apertures chosen so fx = focal_length * width / horizontal_aperture
    focal_length_mm = 50.0
    horizontal_aperture = focal_length_mm * width / fx
    vertical_aperture = focal_length_mm * height / fy
    
    if is_world_to_camera:
        T_cam_to_world_opencv = np.linalg.inv(extrinsics)
    else:
        T_cam_to_world_opencv = extrinsics.copy()
    
    # OpenCV (+Z forward, +Y down) -> USD/OpenGL (+Z backward, +Y up): 180° rotation about X
    opencv_to_opengl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # Flip Y: down -> up
        [0,  0, -1, 0],  # Flip Z: forward -> backward
        [0,  0,  0, 1]
    ])
    
    T_cam_to_world_usd = T_cam_to_world_opencv @ opencv_to_opengl
    
    position = T_cam_to_world_usd[:3, 3]
    rotation_matrix = T_cam_to_world_usd[:3, :3]

    r = Rotation.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    camera_z_axis = T_cam_to_world_usd[:3, 2]
    looking_direction = -camera_z_axis  # camera looks along -Z in USD/OpenGL
    
    if verbose:
        h_fov_rad = 2 * math.atan(horizontal_aperture / (2 * focal_length_mm))
        v_fov_rad = 2 * math.atan(vertical_aperture / (2 * focal_length_mm))
        h_fov_deg = math.degrees(h_fov_rad)
        v_fov_deg = math.degrees(v_fov_rad)
        
        logger.info(f"\nCamera Setup:")
        logger.info(f"  Input format: {'World->Camera' if is_world_to_camera else 'Camera->World'} (OpenCV)")
        logger.info(f"  Applied OpenCV->USD coordinate conversion")
        logger.info(f"  ")
        logger.info(f"  Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  ")
        logger.info(f"  Physical camera model:")
        logger.info(f"    Focal length: {focal_length_mm:.2f} mm")
        logger.info(f"    H aperture: {horizontal_aperture:.2f} mm -> FOV: {h_fov_deg:.1f} deg")
        logger.info(f"    V aperture: {vertical_aperture:.2f} mm -> FOV: {v_fov_deg:.1f} deg")
        logger.info(f"  ")
        logger.info(f"  Camera position (world): {position}")
        logger.info(f"  Camera orientation (WXYZ): [{quat_wxyz[0]:.3f}, {quat_wxyz[1]:.3f}, {quat_wxyz[2]:.3f}, {quat_wxyz[3]:.3f}]")
        logger.info(f"  Camera looking direction: [{looking_direction[0]:.3f}, {looking_direction[1]:.3f}, {looking_direction[2]:.3f}]")
    
    return CameraParams(
        name="main_camera",
        pos=position,
        ori_wxyz=quat_wxyz,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        horizontal_aperture=horizontal_aperture,
        focal_length=focal_length_mm,
    )

