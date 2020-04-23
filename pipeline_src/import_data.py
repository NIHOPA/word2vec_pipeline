"""
Import documents into the pipeline, concatenate target fields into a
specifc field, and tag each document with a unique reference ID.
Identifies common phrases found in the document.
"""

import sys
import csv
import os
import itertools

from utils.os_utils import mkdir, grab_files

import logging

logger = logging.getLogger(__name__)

# Fix for pathological csv files
csv.field_size_limit(2147483647)
_ref_counter = itertools.count()


def csv_iterator(f_csv, clean=True, _PARALLEL=False):
    """
    Creates an iterator over a CSV file, optionally cleans it.
    Reads in file using utf-8 encoding.

    Args
        f_csv (str): Filename of the csv to open and iterate over
        clean (bool): Set whether to clean the csv file
        PARALLEL (bool): Set whether the iterator should be run in parallel
    """

    with open(f_csv, encoding="utf-8") as FIN:
        CSV = csv.DictReader(FIN)

        for row in CSV:
            yield row


def import_csv(item):
    """
    Import a csv file, optionally merging select fields into a single field.

    Args:
        item: a list containing function paramters
            f_csv (str): Filename of the csv file to open
            f_csv_out (str): Filename of output csv
            f_target_column (str): Name of the column with concatenated text
            merge_columns (list): Names of the text columns that are
              to be concatenated
    """
    (f_csv, f_csv_out, target_column, merge_columns) = item
    has_checked_keys = False

    if not merge_columns:
        raise ValueError("merge_columns must not be empty")

    with open(f_csv_out, "w") as FOUT:
        CSV_HANDLE = None
        total_rows = 0

        for row in csv_iterator(f_csv):

            output = {"_ref": next(_ref_counter)}

            if not has_checked_keys:
                for key in merge_columns:
                    if key not in row.keys():
                        msg = "Column **{}** not in csv file {}"
                        raise KeyError(msg.format(key, f_csv))
                has_checked_keys = True

            if target_column in row.keys():
                msg = "Generated column **{}** already in csv file {}"
                raise KeyError(msg.format(target_column, f_csv))

            text = []
            for key in merge_columns:
                val = row[key].strip()
                if not val:
                    continue
                if val[-1] not in ".?!,":
                    val += "."
                text.append(val)

            output[target_column] = "\n".join(text).strip()

            if CSV_HANDLE is None:
                CSV_HANDLE = csv.DictWriter(FOUT, sorted(output.keys()))
                CSV_HANDLE.writeheader()

            CSV_HANDLE.writerow(output)
            total_rows += 1

        logger.info("Imported {}, {} entries".format(f_csv, total_rows))


def import_directory_csv(d_in, d_out, target_column, merge_columns):
    """
    Takes a input_directory and output_directory and builds
    and cleaned (free of encoding errors) CSV for all input
    and attaches unique _ref numbers to each entry.

    Args:
        d_in (str): Directory of the csv file to open
        d_out (str): Directory of where the input document should be saved to
        target_column (str): Name of the column with concatenated text
        merge_columns (list): Names of the text columns that are to be
           concatenated
    """

    INPUT_FILES = grab_files("*.csv", d_in)

    if not INPUT_FILES:
        logger.warning("No matching CSV files found, exiting")
        exit(2)

    for f_csv in INPUT_FILES:
        f_csv_out = os.path.join(d_out, os.path.basename(f_csv))
        vals = (f_csv, f_csv_out, target_column, merge_columns)
        import_csv(vals)


def import_data_from_config(config):
    """
    Import parameters from the config file. import_data_from_config()
    and phrases_from_config() are the entry points for this step of the
    pipeline.

    Args:
        config: a config file
    """

    merge_columns = config["import_data"]["merge_columns"]

    if not isinstance(merge_columns, list):
        msg = "merge_columns (if used) must be a list"
        raise ValueError(msg)

    data_out = config["import_data"]["output_data_directory"]
    mkdir(data_out)

    # Require 'input_data_directories' to be a list
    data_in_list = config["import_data"]["input_data_directories"]
    if not isinstance(data_in_list, list):
        msg = "input_data_directories must be a list"
        raise ValueError(msg)

    target_column = config["target_column"]

    for d_in in data_in_list:
        import_directory_csv(d_in, data_out, target_column, merge_columns)


if __name__ == "__main__":

    import simple_config

    config = simple_config.load()

    import_data_from_config(config)
