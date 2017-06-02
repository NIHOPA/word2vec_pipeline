import os
from utils.os_utils import mkdir, grab_files
import utils.db_utils as db_utils
import csv
import nlpre

from utils.parallel_utils import jobmap

_global_batch_size = 500

# This must be global for parallel to work properly
parser_functions = []

#import logging
#nlpre.logger.setLevel(logging.INFO)

def dispatcher(row, target_column):
    text = row[target_column] if target_column in row else None

    for f in parser_functions:
        text = unicode(f(text))

    row[target_column] = text
    return row

    '''
    meta = {}
    for f in parser_functions:
        result = f(text)
        text   = unicode(result)

        if hasattr(result,"meta"):
            meta.update(result.meta)

    # Convert the meta information into a unicode string for serialization
    #meta = unicode(meta)
    '''

def load_phrase_database(f_abbreviations):

    P = {}
    with open(f_abbreviations,'r') as FIN:
        CSV = csv.DictReader(FIN)
        for row in CSV:
            key = (tuple(row['phrase'].split()), row['abbr'])
            val = int(row['count'])
            P[key] = val
    return P


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
    INPUT_ITR = dfunc(F_CSV, col, include_filename=True)
    
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


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    parse_from_config(config)
