"""
Train a word2vec embedding of the documents imported into the pipeline.
This creates a gensim word2vec file, which can be then used for NLP tasks.
In this pipeline, this model is used to cluster documents based on similarity,
as well as run document classification.

The code that performs this embedding is found in model_building/,
which creates the word2vec model itself.
"""

import os
from utils.os_utils import mkdir
import model_building as mb
from utils.db_utils import text_iterator


def embed_from_config(config):
    """
    Args:
        config (dict): Import parameters
    """

    # Only load options from the embedding section
    target_column = config["target_column"]
    econfig = config["embed"]

    # Create any missing directories
    d_out = econfig["output_data_directory"]
    mkdir(d_out)

    # Train each embedding model
    for name in econfig["embedding_commands"]:

        # Load any kwargs in the config file
        kwargs = econfig.copy()

        if name in kwargs:
            kwargs.update(kwargs[name])

        model = getattr(mb, name)(**kwargs)
        model.set_iterator_function(text_iterator)
        model.compute(target_column)

        f_save = os.path.join(d_out, kwargs[name]["f_db"])
        model.save(f_save)


if __name__ == "__main__":

    import simple_config

    config = simple_config.load()
    embed_from_config(config)
