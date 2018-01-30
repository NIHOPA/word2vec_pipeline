from utils.os_utils import mkdir
import model_building as mb
from utils.db_utils import item_iterator

"""
Driver file to train a  word2vec embedding of the documents imported into the pipeline. This creates a 
gensim word2vec file, which can be then used for NLP tasks. In this pipeline, this model is used to cluster documents
based on similarity, as well as run document classification.

The code that performs this embedding is found in word2vec_pipeline/model_building, which creates the word2vec
model itself

"""
def embed_from_config(config):

    mkdir(config["embedding"]["output_data_directory"])

    # If there is a whitelist only keep the matching filename
    try:
        whitelist = config["score"]["input_file_whitelist"]
    except:
        whitelist = []

    #
    # Run the functions that act globally on the data

    for name in config["embedding"]["embedding_commands"]:
        obj = getattr(mb, name)

        # Load any kwargs in the config file
        kwargs = config["embedding"].copy()

        if name in kwargs:
            kwargs.update(kwargs[name])
        kwargs['target_column'] = config['target_column']

        func = obj(**kwargs)
        func.set_iterator_function(item_iterator,
                                   config["embedding"],
                                   whitelist,
                                   section="parse")
        func.compute(**kwargs)


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    embed_from_config(config)
