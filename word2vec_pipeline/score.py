import itertools
from utils.os_utils import mkdir
import document_scoring as ds
import utils.db_utils as db


def score_from_config(global_config):

    config = global_config["score"]

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
        kwargs["embedding"] = global_config["embedding"]
        kwargs["score"] = global_config["score"]

        val = name, obj(**kwargs)
        mapreduce_functions.append(val)

    col = global_config['target_column']

    # Run the functions that can act like mapreduce (eg. TF counts)    
    for name, func in mapreduce_functions:
        print("Starting mapreduce {}".format(func.table_name))
        exit()
        
        INPUT_ITR = db.text_iterator()
        ITR = itertools.imap(func, INPUT_ITR)
        map(func.reduce, ITR)

        func.save(config)
    
    # Run the functions that act globally on the data

    for name in config["globaldata_commands"]:
        obj = getattr(ds, name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])

        # Add in the embedding configuration options
        func = obj(**kwargs)

        F_CSV = db.get_section_filenames("parse")

        for f_csv in F_CSV:
            ITR = db.single_file_item_iterator(f_csv)
            func.compute_single(ITR)
            func.save_single()

        func.compute_reduced_representation()


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    score_from_config(config)
