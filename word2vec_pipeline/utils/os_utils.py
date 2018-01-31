import os
import glob

"""
Utililty files to create directories for the pipeline, as well as grab files found in given directories.
"""


def mkdir(directory):
    '''
    Tries to create a directory if it doesn't exist. Equivalent to UNIX 'mkdir -p directory'

    Args:
        directory: string with the directory to be created
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def grab_files(pattern, directory="", verbose=True):
    '''
    Function to return all files found in a given directory

    Args:
        pattern: a string that is used to identify the proper files to grab
        directory: a string that points to the location of the directory
        verbose: a boolean to flag whether to print description of file importation process.
    '''
    g_pattern = os.path.join(directory, pattern)
    FILES = sorted(glob.glob(g_pattern))

    if verbose:
        msg = "Found {} files to import in {}."
        print((msg.format(len(FILES), directory)))

    return sorted(FILES)
