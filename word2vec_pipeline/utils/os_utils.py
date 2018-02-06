"""
Utililty files to create directories for the pipeline, as well as grab
files found in given directories.
"""

import os
import glob


def mkdir(directory):
    '''
    Tries to create a directory if it doesn't exist.
    Equivalent to UNIX 'mkdir -p directory'

    Args:
        directory: string with the directory to be created
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def grab_files(pattern, directory="", verbose=True):
    '''
    Function to return all files found in a given directory

    Args:
        pattern (str): Identify the proper files to grab
        directory (str): Points to the location of the directory
        verbose (bool): If True, print description of importation process
    '''
    g_pattern = os.path.join(directory, pattern)
    FILES = sorted(glob.glob(g_pattern))

    if verbose:
        msg = "Found {} files to import in {}."
        print((msg.format(len(FILES), directory)))

    return sorted(FILES)
