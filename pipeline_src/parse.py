"""
Parse imported documents using the NLPre pre-processing library.
These commands strips out punctuation, unimportant words, identifies acronyms,
as well as other processings steps.
"""

import os
from utils.os_utils import mkdir, grab_files
import utils.db_utils as db_utils
import csv
import sys
import nlpre

from utils.parallel_utils import jobmap

import logging

logger = logging.getLogger(__name__)

# NLPre is too noisy at the info level
logging.getLogger("nlpre").setLevel(logging.WARNING)

# Fix for pathological csv files
csv.field_size_limit(2147483647)

_global_batch_size = 500

# This must be global for parallel to work properly
parser_functions = []


def dispatcher(row, target_column):
    """
    Perform the operation of each step of the NLPre pre-processing
    specified in the config file. Requires a global list of parser_functions
    to be defined.

    Args:
        row (dict): A dictionary where the target column is defined as a key
        target_column (str): The column is to be processed.

    Returns:
         Dict: The dictionary after processing
    """

    text = row[target_column] if target_column in row else None

    for func in parser_functions:
        text = func(text)

    row[target_column] = text
    return row


def load_phrase_database(f_abbreviations):
    """
    Load the dictionary of abbreviated steps created in the "import_data" step

    Args:
        f_abbreviations (str): Filename of the abbreviation dictionary

    Returns:
         A dictionary of abbreviations.
    """

    P = {}
    with open(f_abbreviations, "r") as FIN:
        CSV = csv.DictReader(FIN)
        for row in CSV:
            key = (tuple(row["phrase"].split()), row["abbr"])
            val = int(row["count"])
            P[key] = val
    return P


def parse_from_config(config):
    global _global_batch_size

    _PARALLEL = config.as_bool("_PARALLEL")

    if not _PARALLEL:
        _global_batch_size = 1

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
                config["phrase_identification"]["f_abbreviations"],
            )
            ABBR = load_phrase_database(f_abbr)
            kwargs["counter"] = ABBR

        logger.info(f"Loading {obj}")
        parser_functions.append(obj(**kwargs))

    col = config["target_column"]
    F_CSV = grab_files("*.csv", input_data_dir)

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, col, include_filename=True, progress_bar=False)

    ITR = jobmap(
        dispatcher,
        INPUT_ITR,
        _PARALLEL,
        batch_size=_global_batch_size,
        target_column=col,
    )

    F_CSV_OUT = {}
    F_WRITERS = {}

    for k, row in enumerate(ITR):
        f = row.pop("_filename")

        # Create a CSV file object for all outputs
        if f not in F_CSV_OUT:
            f_csv_out = os.path.join(output_dir, os.path.basename(f))

            F = open(f_csv_out, "w")
            F_CSV_OUT[f] = F
            F_WRITERS[f] = csv.DictWriter(F, fieldnames=["_ref", col])
            F_WRITERS[f].writeheader()

        F_WRITERS[f].writerow(row)

    # Close the open files
    for F in F_CSV_OUT.values():
        F.close()


if __name__ == "__main__":

    import simple_config

    config = simple_config.load()
    parse_from_config(config)
