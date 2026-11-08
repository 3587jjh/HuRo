"""Import a robot URDF into an Isaac-ready USD, cached on disk (regenerated when the URDF is
newer). Must be called with a live SimulationApp, because the importer is a Kit extension.
"""
import fcntl
import os
import re
import shutil
from pathlib import Path


def _stage_for_import(urdf_path: Path, staging_dir: Path) -> Path:
    """The URDF to hand the importer, with mesh URIs resolvable. Unresolved ones make it
    silently yield a robot with no visual geometry. A URDF using package:// is copied next to
    <pkg> symlinks, and one using paths relative to itself is imported where it lies."""
    pkgs, pkg_root = _package_root(urdf_path)
    if not pkgs:
        return urdf_path

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / urdf_path.name
    shutil.copy2(urdf_path, staged)
    for pkg in pkgs:
        link = staging_dir / pkg
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pkg_root)
    return staged


def _package_root(urdf_path: Path):
    """(package names, root) for a URDF's package:// mesh URIs, or ([], None)."""
    pkgs = sorted(set(re.findall(r"package://([^/]+)/", urdf_path.read_text())))
    return pkgs, (urdf_path.parent.parent if pkgs else None)


def _ensure_overlay_usd_v6(urdf_path: Path, usd_path: Path, URDFImporter, URDFImporterConfig) -> str:
    """Isaac Sim >= 6.0: the importer is a different tool (urdf-usd-converter) that writes a
    USD package <dir>/<robot>/<robot>.usda, not a single file. It is cached beside the 5.x
    file: build/overlay/<robot>.usd -> build/overlay/<robot>_isaac6/<URDF stem>/<stem>.usda."""
    out_dir = usd_path.with_name(usd_path.stem + "_isaac6")
    cached = next(iter(sorted(out_dir.glob("*/*.usda"))), None)
    if cached is not None and cached.stat().st_mtime >= urdf_path.stat().st_mtime:
        return str(cached)

    # The importer appends _1, _2, ... instead of overwriting an existing output directory.
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkgs, pkg_root = _package_root(urdf_path)
    cfg = URDFImporterConfig(
        urdf_path=str(urdf_path),
        usd_path=str(out_dir),
        merge_fixed_joints=False,   # keep every link: the config hides some of them by name
        fix_base=True,              # the robot is posed by the pipeline, not by physics
        allow_self_collision=False,
        # package://<pkg>/... resolves to the URDF's parent directory (same as _stage_for_import)
        ros_package_paths=[{"name": pkg, "path": str(pkg_root)} for pkg in pkgs],
    )
    final = Path(URDFImporter(cfg).import_urdf())
    if not final.is_file():
        raise RuntimeError(f"URDF import produced no USD at {final} (from {urdf_path})")
    os.utime(final, None)
    return str(final)


def ensure_overlay_usd(urdf_path, usd_path) -> str:
    """Path to the USD for `urdf_path`, importing it into `usd_path` if that is missing or
    older than the URDF. Returns the USD path as a string."""
    urdf_path, usd_path = Path(urdf_path), Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    # Workers on other GPUs share this file, so one imports while the rest wait for it.
    with open(usd_path.parent / f"{usd_path.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _import_if_stale(urdf_path, usd_path)


def _import_if_stale(urdf_path, usd_path) -> str:
    """The body of ensure_overlay_usd, run while holding its lock."""
    if usd_path.is_file() and usd_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return str(usd_path)

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    except ImportError:            # Isaac Sim <= 5.1: the C++ importer behind _urdf
        URDFImporter = None
    if URDFImporter is not None:   # Isaac Sim >= 6.0: _urdf and its ImportConfig are gone
        return _ensure_overlay_usd_v6(urdf_path, usd_path, URDFImporter, URDFImporterConfig)
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
