#! /usr/bin/env python
"""
Usage:
  word2vec_pipeline import_data
  word2vec_pipeline phrase
  word2vec_pipeline parse
  word2vec_pipeline embed
  word2vec_pipeline score
  word2vec_pipeline predict
  word2vec_pipeline metacluster
  word2vec_pipeline analyze

  The code that is run by each command is found in the filename in current
  directory that corresponds to each command. The function
  BLANK_from_config(config) is the entry point for each file,
  and can be found at the bottom of each file.
"""

from docopt import docopt
import simple_config
import logging
import sys


_python_version = sys.version_info
if _python_version < (3,):
    raise ValueError(
        "Pipeline now requires python 3, you have", _python_version
    )

logging.basicConfig(level=logging.INFO)


def main():
    args = docopt(__doc__)
    config = simple_config.load()

    if args["import_data"]:
        from import_data import import_data_from_config

        import_data_from_config(config)

    elif args["phrase"]:
        from phrase import phrases_from_config

        phrases_from_config(config)

    if args["parse"]:
        from parse import parse_from_config

        parse_from_config(config)

    if args["embed"]:
        from embed import embed_from_config

        embed_from_config(config)

    if args["score"]:
        from score import score_from_config

        score_from_config(config)

    if args["predict"]:
        from predict import predict_from_config

        predict_from_config(config)

    if args["metacluster"]:
        from metacluster import metacluster_from_config

        metacluster_from_config(config)

    if args["analyze"]:

        import postprocessing.analyze_metaclusters as pam

        pam.analyze_metacluster_from_config(config)

        # elif func == "LIME":
        #    import postprocessing.lime_explainer as le
        # le.explain_metaclusters(config)
        # else:
        #    raise KeyError("Analyze Function {} not known".format(func))


if __name__ == "__main__":
    main()
