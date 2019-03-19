from .shallow_predict import categorical_predict
import sys
import os

_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_parent_directory)


__all__ = ["categorical_predict"]
