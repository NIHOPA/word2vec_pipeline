"""
Utility file to open the pipeline's config file, and check for errors
"""

import os
from configobj import ConfigObj
from validate import Validator, VdtTypeError

import logging

logger = logging.getLogger(__name__)


def validate_errors(errors, name_stack=None):

    # No errors found if errors object is True
    if errors:
        return False

    is_error = False

    if name_stack is None:
        name_stack = []

    for key, item in errors.items():
        stack = name_stack + [key]

        # If item is True, then object validated
        if item:
            continue

        # If item is a known typeError report it
        if isinstance(item, VdtTypeError):
            logging.warning("ConfigError: {} {}".format("/".join(stack), item))
            is_error = True

        # If item is a dict, recurse into the config stack
        elif isinstance(item, dict):
            is_error += validate_errors(item, stack)

    return is_error


def load(f_config="config.ini"):

    # Raise Error if configfile not found
    if not os.path.exists(f_config):
        msg = "{} not found".format(f_config)
        raise IOError(msg)

    # The config spec should be located in the same directory as simple_config
    local_path = os.path.dirname(os.path.realpath(__file__))
    f_config_spec = os.path.join(local_path, "config_validation.ini")

    assert os.path.exists(f_config_spec)

    config = ConfigObj(f_config, configspec=f_config_spec)

    errors = config.validate(Validator(), preserve_errors=True)

    if validate_errors(errors):
        msg = "{} failed to parse.".format(f_config)
        raise SyntaxError(msg)

    return config

    # Loading a subset is depreciated since type checking isn't done.
    """
    # Load all the global keys
    output = ConfigObj()

    for key, val in config.items():
        if not isinstance(val, Section):
            output[key] = val

    # Add in the local subset information
    output.update(config[subset])

    return output
    """
