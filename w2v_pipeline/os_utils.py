import os

def mkdir(directory):
    # Tries to create a directory if it doesn't exist
    # Equivalent to UNIX 'mkdir -p directory'

    if not os.path.exists(directory):
        os.makedirs(directory)
