from ._version import __version__
import sys


_python_version = sys.version_info
if _python_version < (3,):
    raise ValueError("Pipeline now requires python 3")


__all__ = ["__version__"]
