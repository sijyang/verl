from importlib.metadata import PackageNotFoundError, version

from .atom_rollout import ServerAdapter
from .atom_async_server import ATOMReplica, ATOMHttpServer
from .constants import ATOMDefaults, SleepLevel


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


atom_package_name = "atom"
atom_package_version = get_version(atom_package_name)
if atom_package_version is None:
    raise PackageNotFoundError(
        "To use ATOM rollout, please ensure the 'atom' package is properly installed. "
        "See ATOM documentation for installation instructions."
    )

__all__ = [
    "ServerAdapter",
    "ATOMReplica",
    "ATOMHttpServer",
    "ATOMDefaults",
    "SleepLevel",
]
