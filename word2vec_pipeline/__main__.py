#! /usr/bin/env python
"""
Usage:
  word2vec_pipeline import_data
  word2vec_pipeline parse
"""

import os
import time
from docopt import docopt


def main():
    args = docopt(__doc__)

    # Check config.ini is found here ...
    print args


if __name__ == "__main__":
    main()
