"""Check a robot config's frames against the pipeline's conventions, before running a stage.

Resolves configs/<robot>.yaml against its URDF at the home pose and reports the MANO wrist
frames in anatomical terms and the camera frame against OpenCV. See README.md in this
directory for the conventions. Every `want` in the output should come out satisfied.

  python configs/check_frames.py <robot_name>

It needs numpy, PyYAML and the URDF's own XML, and neither jax nor Isaac, so it runs anywhere
the repo is checked out. Besides the frames it checks the finger keypoint spacing and the home
pose. Stage 8 validates the joint lists.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.robot_config import load_robot_config

# MediaPipe hand landmarks the checks need: wrist, thumb CMC, and the index/middle/pinky MCPs.
WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 1, 5, 9, 17
# The five keypoint chains, base to tip. Two neighbours on one chain whose links sit closer than
# MIN_KEYPOINT_GAP_MM usually mean that one keypoint went to the link a joint too early.
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
MIN_KEYPOINT_GAP_MM = 10.0


def rpy_matrix(r, p, y):
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def parse_joints(urdf_path):
    """child link name -> its joint (parent, origin, axis, type, mimic). One joint per link."""
    joints = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        origin, axis = j.find("origin"), j.find("axis")
        attr = lambda k, default: (origin.get(k, default) if origin is not None else default)
        limit = j.find("limit")
        joints[j.find("child").get("link")] = dict(
            parent=j.find("parent").get("link"), name=j.get("name"), type=j.get("type"),
            xyz=np.array([float(v) for v in attr("xyz", "0 0 0").split()]),
            R=rpy_matrix(*[float(v) for v in attr("rpy", "0 0 0").split()]),
            axis=(np.array([float(v) for v in axis.get("xyz").split()])
                  if axis is not None else np.array([1.0, 0.0, 0.0])),
            limit=((float(limit.get("lower")), float(limit.get("upper")))
                   if limit is not None and limit.get("lower") is not None else None),
            mimic=((mimic.get("joint"), float(mimic.get("multiplier", 1.0)),
                    float(mimic.get("offset", 0.0))) if (mimic := j.find("mimic")) is not None
                   else None))
    return joints


def fk(joints, link, q):
    """Pose of `link` in the root frame with joint angles `q` (missing names are 0). A <mimic>
    joint follows its parent, as it does in the IK."""
    chain = []
    while link in joints:
        chain.append(link)
        link = joints[link]["parent"]
    T = np.eye(4)
    for name in reversed(chain):
        j = joints[name]
        L = np.eye(4)
        L[:3, :3], L[:3, 3] = j["R"], j["xyz"]
        if j["type"] in ("revolute", "continuous"):
            Rj = np.eye(4)
            if j["mimic"]:
                parent, multiplier, offset = j["mimic"]
                angle = multiplier * q.get(parent, 0.0) + offset
            else:
                angle = q.get(j["name"], 0.0)
            Rj[:3, :3] = axis_angle(j["axis"], angle)
            L = L @ Rj
        T = T @ L
    return T


def unit(v):
    return v / np.linalg.norm(v)


def check_link(joints, link, what):
    """A link the config names must exist, or `fk` walks no chain and returns the identity,
    which reads as a frame that happens to sit at the base."""
    if link not in joints and link not in {j["parent"] for j in joints.values()}:
        sys.exit(f"config error: {what} names '{link}', which is not a link of this URDF")
    return link


def main(robot_name):
    cfg = load_robot_config(robot_name)
    missing = [k for k in ("keypoint_mapping", "eef_link_names", "camera_link")
               if k not in cfg.config]
    if missing:
        sys.exit(f"config error: {robot_name}.yaml has no {', '.join(missing)}")
    if not Path(cfg.urdf_path).is_file():
        sys.exit(f"config error: robot_urdf_path points at {cfg.urdf_path}, which is not there")
    config, joints = cfg.config, parse_joints(cfg.urdf_path)
    home = config.get("home_config", {}) or {}
    print(f"{robot_name}: {cfg.state_dim} active DOF, {Path(cfg.urdf_path).name}, at the home pose\n")

    ok = True
    for side, x_sign in (("left", +1.0), ("right", -1.0)):
        mapping = {int(k): v for k, v in config["keypoint_mapping"][side].items()}
        needed = (WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)
        absent = [k for k in needed if k not in mapping]
        if absent:
            sys.exit(f"config error: keypoint_mapping[{side}] has no entry for {absent}, "
                     f"which these checks read")
        P = {k: fk(joints, check_link(joints, mapping[k], f"keypoint_mapping[{side}][{k}]"),
                   home)[:3, 3] for k in needed}
        along = unit(P[MIDDLE_MCP] - P[WRIST])       # wrist -> middle MCP
        across = unit(P[INDEX_MCP] - P[PINKY_MCP])   # pinky MCP -> index MCP (thumb side)
        normal = unit(np.cross(along, across))
        # `normal` is palmar on a left hand and dorsal on a right one. The thumb tells us which.
        dorsal = normal * (-1.0 if unit(P[THUMB_CMC] - P[WRIST]) @ normal > 0 else 1.0)

        T = fk(joints, check_link(joints, config["eef_link_names"][side],
                                  f"eef_link_names[{side}]"), home)
        checks = [("x", float(T[:3, 0] @ along) * x_sign,
                   "wrist->fingers" if x_sign > 0 else "fingers->wrist"),
                  ("y", float(T[:3, 1] @ dorsal), "back of the hand"),
                  ("z", float(T[:3, 2] @ across), "thumb side")]
        print(f"  {side:5s} {config['eef_link_names'][side]}")
        for axis_name, cos, meaning in checks:
            good = cos > 0.9
            ok &= good
            print(f"    {'ok  ' if good else 'FAIL'} +{axis_name} -> {meaning:16s} cos={cos:+.3f} (want > +0.9)")
        offset = np.linalg.norm(T[:3, 3] - P[WRIST]) * 1000
        print(f"    {'note' if offset > 20 else 'ok  '} origin sits {offset:.1f} mm from the "
              f"keypoint-0 anchor ({mapping[WRIST]})")

        # A link sits at the joint that moves it, so a keypoint on the link one joint too early
        # lands next to its neighbour.
        gaps = []
        for chain in FINGER_CHAINS:
            ks = [k for k in chain if k in mapping]
            pos = {k: fk(joints, check_link(joints, mapping[k], f"keypoint_mapping[{side}][{k}]"),
                         home)[:3, 3] for k in ks}
            gaps += [(np.linalg.norm(pos[a] - pos[b]) * 1000, a, b) for a, b in zip(ks, ks[1:])]
        close = [f"{a}-{b} {g:.0f} mm" for g, a, b in gaps if g < MIN_KEYPOINT_GAP_MM]
        if gaps:
            print(f"    {'warn' if close else 'ok  '} neighbouring finger keypoints "
                  f"{min(gaps)[0]:.0f} mm apart or more (want >= {MIN_KEYPOINT_GAP_MM:.0f})"
                  f"{': ' + ', '.join(close) if close else ''}")
        if close:
            print("         map each keypoint to the link that starts at its joint "
                  "(README.md, keypoint_mapping)")

    # The base frame is assumed x-forward, y-left, z-up. OpenCV is then x=-base_y, y=-base_z,
    # z=+base_x. Mounting pitch about the camera's own x is free, so only two properties are
    # checked: +x is image-right, and the camera looks forward-and-down rather than
    # backward-and-up (which is what an OpenGL frame, +x right but y and z flipped, would do).
    fwd, left, up = np.eye(3)
    R = fk(joints, check_link(joints, config["camera_link"], "camera_link"), home)[:3, :3]
    opencv = np.column_stack([-left, -up, fwd])
    tilt = np.degrees(np.arccos(np.clip((np.trace(opencv.T @ R) - 1) / 2, -1, 1)))
    cam_checks = [("x", float(R[:, 0] @ -left), 0.9, "image right"),
                  ("y", float(R[:, 1] @ -up), 0.0, "downward (not up: an OpenGL frame fails here)"),
                  ("z", float(R[:, 2] @ fwd), 0.0, "forward (the view direction)")]
    print(f"\n  camera {config['camera_link']}  "
          f"(base frame assumed x-forward, y-left, z-up)")
    for axis_name, cos, floor, meaning in cam_checks:
        good = cos > floor
        ok &= good
        print(f"    {'ok  ' if good else 'FAIL'} +{axis_name} -> {meaning:44s} "
              f"cos={cos:+.3f} (want > {floor:+.1f})")
    print(f"    note {tilt:.1f} deg from level-forward OpenCV — mounting pitch is free, "
          f"the axis convention is not")

    # The home pose is the rest prior, the solver's initial guess, and where locked joints
    # sit. A joint resting on its own limit is legal but rarely intended.
    limits = {j["name"]: j["limit"] for j in joints.values() if j["limit"]}
    actuated = {j["name"] for j in joints.values()
                if j["type"] in ("revolute", "continuous") and not j["mimic"]}
    active = set(cfg.joint_names)
    on_edge = [n for n, (lo, hi) in limits.items()
               if min(abs(home.get(n, 0.0) - lo), abs(hi - home.get(n, 0.0))) < 1e-6]
    print(f"\n  home pose ({len(home)} entries; the rest of the URDF rests at 0.0)")
    # A home_config name the URDF has no actuated joint for is skipped in silence upstream.
    unknown = sorted(set(home) - actuated)
    if unknown:
        ok = False
        print(f"    FAIL {len(unknown)} home_config name(s) are not an actuated joint of this "
              f"URDF, and are ignored in silence: {unknown[:4]}")
    outside = [f"{n}={home[n]:+.2f} not in [{limits[n][0]:+.2f},{limits[n][1]:+.2f}]"
               for n in home if n in limits and not limits[n][0] <= home[n] <= limits[n][1]]
    if outside:
        ok = False
        print(f"    FAIL {len(outside)} home value(s) outside their joint limits: {outside[:4]}")
    else:
        print(f"    ok   every home value is inside its joint limits")
    # Fingers legitimately rest at one end of their range (an open hand), so only the
    # non-hand groups are worth flagging. A straight arm at its stop is the real mistake.
    hand = {n for g in config.get("hand_groups") or [] for n in config["joints"].get(g, [])}
    arm_on_edge = [n for n in on_edge if n in active and n not in hand]
    n_arm = len(active - hand)
    print(f"    {'warn' if arm_on_edge else 'ok  '} {len(arm_on_edge)} of {n_arm} driven "
          f"non-hand joints rest ON a limit"
          f"{': ' + ', '.join(arm_on_edge[:4]) if arm_on_edge else ''}"
          f"{' ...' if len(arm_on_edge) > 4 else ''}")
    if arm_on_edge:
        print("         an arm at its hard stop is a poor initial guess and fights the limit "
              "cost. See README.md on home_config")

    print("\nall frame checks passed" if ok else "\nSOME FRAME CHECKS FAILED. See README.md")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
