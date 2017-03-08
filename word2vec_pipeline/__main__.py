#! /usr/bin/env python
"""
Usage:
  word2vec_pipeline import_data
  word2vec_pipeline parse
"""

import os
import time
from docopt import docopt

import simple_config

from import_data import import_data_from_config
from phrases_from_abbrs import phrases_from_config

def main():
    args = docopt(__doc__)
    config = simple_config.load()


    if args["import_data"]:
        import_data_from_config(config)
        phrases_from_config(config)

    

    print args


if __name__ == "__main__":
    main()
