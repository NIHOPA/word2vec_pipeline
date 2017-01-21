import itertools
from utils.os_utils import grab_files
from utils.os_utils import mkdir
import document_scoring as ds
import simple_config
from utils.db_utils import item_iterator

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("score")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    n_jobs = -1 if _PARALLEL else 1
    mkdir(config["output_data_directory"])

    #
    # Fill the pipeline with function objects

    mapreduce_functions = []
    for name in config["mapreduce_commands"]:

        obj = getattr(ds, name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in config:
            kwargs = config[name]

        # Add in the embedding configuration options
        kwargs["embedding"] = simple_config.load("embedding")
        kwargs["score"] = simple_config.load("score")

        val = name, obj(**kwargs)
        mapreduce_functions.append(val)

    col = config['target_column']

    for name, func in mapreduce_functions:
        print("Starting mapreduce {}".format(func.table_name))
        INPUT_ITR = item_iterator(config, text_column=col,
                                  progress_bar=True)

        ITR = itertools.imap(func, INPUT_ITR)
        map(func.reduce, ITR)

        func.save(config)

    #
    # Run the functions that act globally on the data

    for name in config["globaldata_commands"]:
        obj = getattr(ds, name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])

        # Add in the embedding configuration options

        func = obj(**kwargs)
        func.set_iterator_function(item_iterator, config)
        func.compute()
        func.save()
