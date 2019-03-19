"""
Utililty files to create directories for the pipeline, as well as grab
files found in given directories. Included are functions to help mannipulate
h5 files.
"""

import os
import glob
import h5py
import logging

logger = logging.getLogger(__name__)


def mkdir(directory):
    """
    Tries to create a directory if it doesn't exist.
    Equivalent to UNIX 'mkdir -p directory'

    Args:
        directory: string with the directory to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def grab_files(pattern, directory=""):
    """
    Function to return all files found in a given directory

    Args:
        pattern (str): Identify the proper files to grab
        directory (str): Points to the location of the directory
    """
    g_pattern = os.path.join(directory, pattern)
    FILES = sorted(glob.glob(g_pattern))

    msg = "Found {} files to import in directory {}."
    logger.debug((msg.format(len(FILES), directory)))

    return sorted(FILES)


def load_h5_file(f_h5, *args):
    """
    Generically loads a h5 files top level data structures (assumes
    no nesting). If *args is specified, only the *args subset will be loaded.

    Args:
        f_h5: an h5 file
        *args: additional args

    Returns:
        data: data stored in h5 file
    """
    data = {}

    with h5py.File(f_h5, "r") as h5:
        if not args:
            args = h5.keys()

        for key in args:
            if key not in h5:
                raise ValueError("{} not found in {}".format(key, f_h5))

        for key in args:
            data[key] = h5[key][:]

    return data


def touch_h5(f_db):
    """
    Create the h5 file if it doesn't exist

    Args:
        f_db: string, filename of the h5 file

    Returns
        h5: an h5 file
    """

    if not os.path.exists(f_db):
        h5 = h5py.File(f_db, "w")
    else:
        h5 = h5py.File(f_db, "r+")
    return h5


def get_h5save_object(f_db, method):
    # Returns a usable h5 object to store data
    h5 = touch_h5(f_db)
    g = h5.require_group(method)
    return g


def save_h5(h5, col, data, compression="gzip"):
    # Saves (or overwrites) a column in an h5 object
    if col in h5:
        del h5[col]
    return h5.create_dataset(col, data=data, compression=compression)
