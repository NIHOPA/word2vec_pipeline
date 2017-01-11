import os
import glob

def mkdir(directory):
    # Tries to create a directory if it doesn't exist
    # Equivalent to UNIX 'mkdir -p directory'

    if not os.path.exists(directory):
        os.makedirs(directory)

def grab_files(pattern, directory="", verbose=True):
    g_pattern = os.path.join(directory, pattern)
    FILES = sorted(glob.glob(g_pattern))

    if verbose:
        msg = "Found {} files to import in {}."
        print (msg.format(len(FILES),directory))

    return sorted(FILES)
