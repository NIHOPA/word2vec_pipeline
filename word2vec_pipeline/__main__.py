#! /usr/bin/env python
"""
Usage:
  word2vec_pipeline import_data
  word2vec_pipeline parse
  word2vec_pipeline embed
  word2vec_pipeline score
  word2vec_pipeline predict
"""

import os
import time
from docopt import docopt

import simple_config

from import_data import import_data_from_config
from phrases_from_abbrs import phrases_from_config
from parse import parse_from_config
from embed import embed_from_config
from score import score_from_config
from predict import predict_from_config

def main():
    args = docopt(__doc__)
    config = simple_config.load()


    if args["import_data"]:
        import_data_from_config(config)
        phrases_from_config(config)

    if args["parse"]:
        parse_from_config(config)

    if args["embed"]:
        embed_from_config(config)

    if args["score"]:
        score_from_config(config)
        
    if args["predict"]:
        predict_from_config(config)


if __name__ == "__main__":
    main()
