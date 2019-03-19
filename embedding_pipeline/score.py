"""
Score each document imported into the pipeline using a trained word2vec
model. These scores are used in the pipeline to perform classification
and and clustering. The output scores and word vectors can also be exported
for further analysis.
"""

import os
from utils.os_utils import mkdir
import document_scoring as ds
import utils.db_utils as db

import logging

logger = logging.getLogger(__name__)


def _load_model(name, config):
    # Load any kwargs in the config file
    kwargs = config.copy()

    if name in config:
        kwargs.update(config[name])

    return getattr(ds, name)(**kwargs), kwargs


def score_from_config(global_config):

    config = global_config["score"]
    mkdir(config["output_data_directory"])

    # Run the functions that can sum over the data (eg. TF counts)
    for name in config["count_commands"]:

        model, kwargs = _load_model(name, config)
        logger.info("Starting mapreduce {}".format(model.function_name))
        list(map(model, db.text_iterator()))
        model.save(**kwargs)

    # Load the reduced representation model
    RREP = ds.reduced_representation()

    # Run the functions that act per documnet (eg. word2vec)
    for name in config["score_commands"]:

        model, kwargs = _load_model(name, config)
        f_db = os.path.join(kwargs["output_data_directory"], kwargs["f_db"])

        logger.info("Starting score model {}".format(model.method))

        for f_csv in db.get_section_filenames("parse"):
            data = {}
            for row in db.text_iterator([f_csv]):
                data[row["_ref"]] = model(row["text"])

            model.save(data, f_csv, f_db)

        # If required, compute the reduced representation
        if kwargs["compute_reduced_representation"]:
            nc = kwargs["reduced_representation"]["n_components"]
            rdata = RREP.compute(model.method, n_components=nc)
            RREP.save(model.method, rdata, f_db)


if __name__ == "__main__":

    import simple_config

    config = simple_config.load()
    score_from_config(config)
