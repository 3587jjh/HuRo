"""Import a robot URDF into an Isaac-ready USD, cached on disk (regenerated when the URDF is
newer). Must be called with a live SimulationApp, because the importer is a Kit extension.
"""
import os
import re
import shutil
from pathlib import Path


def _stage_for_import(urdf_path: Path, staging_dir: Path) -> Path:
    """The URDF to hand the importer, with mesh URIs resolvable. Unresolved ones make it
    silently yield a robot with no visual geometry. A URDF using package:// is copied next to
    <pkg> symlinks, and one using paths relative to itself is imported where it lies."""
    pkgs = sorted(set(re.findall(r"package://([^/]+)/", urdf_path.read_text())))
    if not pkgs:
        return urdf_path

    pkg_root = urdf_path.parent.parent
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / urdf_path.name
    shutil.copy2(urdf_path, staged)
    for pkg in pkgs:
        link = staging_dir / pkg
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pkg_root)
    return staged


def ensure_overlay_usd(urdf_path, usd_path) -> str:
    """Path to the USD for `urdf_path`, importing it into `usd_path` if that is missing or
    older than the URDF. Returns the USD path as a string."""
    urdf_path, usd_path = Path(urdf_path), Path(usd_path)
    if usd_path.is_file() and usd_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return str(usd_path)

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    from isaacsim.asset.importer.urdf import _urdf

    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = False   # keep every link: the config hides some of them by name
    cfg.fix_base = True              # the robot is posed by the pipeline, not by physics
    cfg.import_inertia_tensor = True
    cfg.self_collision = False
    cfg.make_default_prim = True
    cfg.create_physics_scene = False  # the renderer's World owns the scene

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    staged = _stage_for_import(urdf_path, usd_path.parent / "urdf")
    iface = _urdf.acquire_urdf_interface()
    model = iface.parse_urdf(str(staged.parent), staged.name, cfg)
    iface.import_robot(str(staged.parent), staged.name, model, cfg, str(usd_path), False)

    if not usd_path.is_file():
        raise RuntimeError(f"URDF import produced no USD at {usd_path} (from {urdf_path})")
    os.utime(usd_path, None)
    return str(usd_path)
