from importlib.metadata import PackageNotFoundError, version

from .atom_rollout import ServerAdapter
from .atom_async_server import ATOMReplica, ATOMHttpServer
from .constants import ATOMDefaults, SleepLevel


def ensure_atom_installed():
    """Check that the 'atom' package is installed. Raises PackageNotFoundError if not."""
    try:
        version("atom")
    except PackageNotFoundError:
        raise PackageNotFoundError(
            "To use ATOM rollout, please ensure the 'atom' package is properly installed. "
            "See ATOM documentation for installation instructions."
        ) from None


__all__ = [
    "ServerAdapter",
    "ATOMReplica",
    "ATOMHttpServer",
    "ATOMDefaults",
    "SleepLevel",
    "ensure_atom_installed",
]
