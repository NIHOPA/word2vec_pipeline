import os
from utils.os_utils import mkdir, grab_files
import utils.db_utils as db_utils
import preprocessing as pre
import csv

from utils.parallel_utils import jobmap

_global_batch_size = 1000


def dispatcher(row, target_column):
    text = row[target_column] if target_column in row else None

    for f in parser_functions:
        text = unicode(f(text))

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

    row[col] = text
    return row


if __name__ == "__main__":

    import simple_config
    config = simple_config.load("parse")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    import_config = simple_config.load("import_data")
    input_data_dir = import_config["output_data_directory"]
    output_dir = config["output_data_directory"]

    import_column = import_config["output_table"]

    mkdir(output_dir)

    # Fill the pipeline with function objects
    parser_functions = []
    for name in config["pipeline"]:
        obj = getattr(pre, name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in config:
            kwargs = config[name]

        parser_functions.append(obj(**kwargs))

    col = config["target_column"]
    F_CSV = grab_files("*.csv", input_data_dir)

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, col, include_filename=True)
    ITR = jobmap(dispatcher, INPUT_ITR, _PARALLEL,
                 # batch_size=_global_batch_size,
                 target_column=col)

    F_CSV_OUT = {}
    F_WRITERS = {}

    for row in ITR:
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
