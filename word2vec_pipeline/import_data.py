import os
import sys
import csv
import itertools
import collections

import nlpre
import pandas as pd

from utils.os_utils import mkdir, grab_files
from utils.parallel_utils import jobmap
import utils.db_utils as db_utils

from tqdm import tqdm

# Fix for pathological csv files
csv.field_size_limit(sys.maxsize)

# Create a global reference ID for each item
_ref_counter = itertools.count()

parser_parenthetical = nlpre.identify_parenthetical_phrases()
def func_parenthetical(data,**kwargs):
    text = data[kwargs["col"]]
    return parser_parenthetical(text)

parser_unicode = nlpre.unidecoder()
def map_to_unicode(s):
    # Helper function to fix input format
    s = str(s)
    return s.decode('utf-8', errors='replace')

def clean_row(row):
    '''
    Maps all keys through a unicode and unidecode fixer.
    '''
    for key, val in row.iteritems():
        row[key] = parser_unicode(map_to_unicode(val))
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

        if os.path.exists(f_csvx):
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


def import_data_from_config(config):

    data_out = config["import_data"]["output_data_directory"]
    mkdir(data_out)

    output_table = config["import_data"]["output_table"]

    # Require `input_data_directories` to be a list
    data_in_list = config["import_data"]["input_data_directories"]
    assert(isinstance(data_in_list, list))

    for d_in in data_in_list:
        import_directory_csv(d_in, data_out, output_table)


def dedupe_abbr(ABR):

    df = pd.DataFrame()
    df['phrase'] = [' '.join(x[0]) for x in ABR.keys()]
    df['abbr'] = [x[1] for x in ABR.keys()]
    df['count'] = ABR.values()

    # Match phrases on lowercase and remove trailing 's'
    df['reduced_phrase'] = df.phrase.str.strip()
    df['reducedphrase'] = df.reduced_phrase.str.lower()
    df['reduced_phrase'] = df.reduced_phrase.str.rstrip('s')

    data = []
    for phrase, dfx in df.groupby('reduced_phrase'):
        top = dfx.sort_values("count", ascending=False).iloc[0]

        item = {}
        item["count"] = dfx["count"].sum()
        item["phrase"] = top["phrase"]
        item["abbr"] = top["abbr"]
        data.append(item)

    df = pd.DataFrame(data).set_index("phrase")
    return df.sort_values("count", ascending=False)



def phrases_from_config(config):

    _PARALLEL = config.as_bool("_PARALLEL")
    output_dir = config["phrase_identification"]["output_data_directory"]

    target_column = config["target_column"]

    import_config = config["import_data"]
    input_data_dir = import_config["output_data_directory"]

    F_CSV = grab_files("*.csv", input_data_dir)
    ABR = collections.Counter()

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, target_column, progress_bar=True)
    ITR = jobmap(func_parenthetical, INPUT_ITR, _PARALLEL, col=target_column)

    for result in ITR:
        ABR.update(result)

    msg = "\n{} total abbrs found."
    print(msg.format(len(ABR)))

    # Merge abbreviations that are similar
    print("Deduping abbr list.")
    df = dedupe_abbr(ABR)
    print("{} abbrs remain after deduping".format(len(df)))

    # Output top phrase
    print("Top 5 abbreviations")
    print(df[:5])

    mkdir(output_dir)
    f_csv = os.path.join(output_dir,
                         config["phrase_identification"]["f_abbreviations"])
    df.to_csv(f_csv)



if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    
    #import_data_from_config(config)
    phrases_from_config(config)
