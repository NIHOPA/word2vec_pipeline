import itertools
from utils.os_utils import mkdir
import document_scoring as ds
import utils.db_utils as db

"""
Driver file to score each document imported into the pipeline using a trained word2vec model. These scores are used
in the pipeline to perform classification and and clustering. The output scores and word vectors can also be exported
for further analysis.
"""

#DOCUMENTATION_UNKNOWN
#in other files the parameter is 'config', rather than 'global_config'

def score_from_config(global_config):
    '''
    Score each document imported into the pipeline using a gensim word2vec model. The config file has the
    parameters of what scoring methods to use

    Args:
        global_config: a config file
    '''

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
        INPUT_ITR = db.item_iterator(
            config,
            text_column=col,
            progress_bar=True,
            include_filename=True,
        )

        ITR = itertools.imap(func, INPUT_ITR)
        map(func.reduce, ITR)

        func.save(config)

    # Run the functions that act globally on the data
    # DOCUMENTATION_UNKNOWN
    # what is obj used for?
    for name in config["globaldata_commands"]:
        obj = getattr(ds, name)


#DOCUMENTATION_UNKNOWN
#is this function ever used?
def parse_from_config(config):

    _PARALLEL = config.as_bool("_PARALLEL")

    import_config = config["import_data"]
    parse_config = config["parse"]

    input_data_dir = import_config["output_data_directory"]
    output_dir = parse_config["output_data_directory"]

    mkdir(output_dir)

    for name in parse_config["pipeline"]:
        obj = getattr(nlpre, name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in parse_config:
            kwargs = dict(parse_config[name])

        # Handle the special case of the precomputed acronyms
        if name == "replace_acronyms":
            f_abbr = os.path.join(
                config["phrase_identification"]["output_data_directory"],
                config["phrase_identification"]["f_abbreviations"]
            )
            ABBR = load_phrase_database(f_abbr)
            kwargs["counter"] = ABBR

        parser_functions.append(obj(**kwargs))

    col = config["target_column"]
    F_CSV = grab_files("*.csv", input_data_dir)

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, col, include_filename=True, progress_bar=False)

    ITR = jobmap(dispatcher, INPUT_ITR, _PARALLEL,
                 batch_size=_global_batch_size,
                 target_column=col,)

    F_CSV_OUT = {}
    F_WRITERS = {}

    for k, row in enumerate(ITR):
        f = row.pop("_filename")

        # Create a CSV file object for all outputs
        if f not in F_CSV_OUT:
            f_csv_out = os.path.join(output_dir, os.path.basename(f))

            F = open(f_csv_out, 'w')
            F_CSV_OUT[f] = F
            F_WRITERS[f] = csv.DictWriter(F, fieldnames=['_ref', col])
            F_WRITERS[f].writeheader()

        F_WRITERS[f].writerow(row)

    # Close the open files
    for F in F_CSV_OUT.values():
        F.close()


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
