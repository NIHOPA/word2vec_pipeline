import os
import itertools
from utils.os_utils import mkdir, grab_files
from utils.parallel_utils import jobmap

from unidecode import unidecode
import csv

from tqdm import tqdm

# Create a global reference ID for each item
_ref_counter = itertools.count()


def map_to_unicode(s):
    # Helper function to fix input format
    s = str(s)
    return s.decode('utf-8', errors='replace')


def clean_row(row):
    '''
    Maps all keys through a unicode and unidecode fixer.
    '''
    for key, val in row.iteritems():
        row[key] = unidecode(map_to_unicode(val))
    return row


def csv_iterator(f_csv, clean=True, _PARALLEL=False):
    '''
    Creates and iterator over a CSV file, optionally cleans it.
    '''
    with open(f_csv) as FIN:
        CSV = csv.DictReader(FIN)

        if clean and _PARALLEL:
            CSV = jobmap(clean_row, CSV, FLAG_PARALLEL=_PARALLEL)
        elif clean and not _PARALLEL:
            CSV = itertools.imap(clean_row, CSV)

        for row in CSV:
            yield row


def import_directory_csv(d_in, d_out, output_table):
    '''
    Takes a input_directory and output_directory and builds
    and cleaned (free of encoding errors) CSV for all input
    and attaches unique _ref numbers to each entry.
    '''

    F_CSV = []
    F_CSV_OUT = {}
    F_CSV_OUT_HANDLE = {}

    INPUT_FILES = grab_files("*.csv", d_in)

    if not INPUT_FILES:
        print("No matching CSV files found, exiting")
        exit(2)

    for f_csv in INPUT_FILES:
        f_csvx = os.path.join(d_out, os.path.basename(f_csv))

        if os.path.exists(f_csvx) and not _FORCE:
            print("{} already exists, skipping".format(f_csvx))
            continue

        F_CSV.append(f_csv)
        F_CSV_OUT[f_csv] = open(f_csvx, 'w')
        F_CSV_OUT_HANDLE[f_csv] = None

    for f_csv in F_CSV:

        for k, row in tqdm(enumerate(csv_iterator(f_csv))):
            row["_ref"] = _ref_counter.next()

            if F_CSV_OUT_HANDLE[f_csv] is None:
                F_CSV_OUT_HANDLE[f_csv] = csv.DictWriter(F_CSV_OUT[f_csv],
                                                         sorted(row.keys()))
                F_CSV_OUT_HANDLE[f_csv].writeheader()

            F_CSV_OUT_HANDLE[f_csv].writerow(row)

        msg = "Imported {}, {} entries"
        print(msg.format(f_csv, k))


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    data_out = config["import_data"]["output_data_directory"]
    mkdir(data_out)

    output_table = config["import_data"]["output_table"]

    # Require `input_data_directories` to be a list
    data_in_list = config["import_data"]["input_data_directories"]
    assert(isinstance(data_in_list, list))

    for d_in in data_in_list:
        import_directory_csv(d_in, data_out, output_table)
