"""
Utility file to assist in parsing .csv files, as well as display the
process of parsing these files.
"""

import tqdm
import csv
import os
import sys
from .os_utils import grab_files
import simple_config

# Fix for pathological csv files
csv.field_size_limit(sys.maxsize)


def CSV_list_columns(f_csv):
    """
    Iterate over the columns of a .csv file.

    Args:
        f_csv (string): Location of the .csv file

    Returns:
        return tuple(reader.next()): DOCUMENTATION_UNKNOWN
    """
    if not os.path.exists(f_csv):
        msg = "File not found {}".format(f_csv)
        raise IOError(msg)
    with open(f_csv, "r") as FIN:
        reader = csv.reader(FIN)
        return tuple(next(reader))


class CSV_database_iterator(object):

    """
    Class to open .csv files, check that they are valid, and iterate
    through them.

    DOCUMENTATION_UNKNOWN : the name "F_CSV" makes it seem like one file,
    when it actually works on a list of multiple files. It might make the
    code clearer to rename this variable
    """

    def __init__(
        self,
        F_CSV,
        target_column=None,
        progress_bar=True,
        include_filename=False,
    ):
        """
        Initialize the iterator

        Args:
            F_CSV: a list of .csv files to iterate over
            target_column: string, the column name that is being parsed
            progress_bar: boolean, a flag to display a progress bar
            include_filename: boolean, a flag to save the documentation
               location in imported document
        """
        self.F_CSV = sorted(F_CSV)
        self.col = target_column

        self.progress_bar = tqdm.tqdm() if progress_bar else None
        self.include_filename = include_filename

        # Raise Exception if column is missing in a CSV
        if self.col is not None:
            for f in F_CSV:

                # If a file is empty, skip it
                if os.stat(f).st_size == 0:
                    continue

                f_cols = CSV_list_columns(f)
                if self.col not in f_cols:
                    msg = "Missing column {} in {}"
                    raise SyntaxError(msg.format(target_column, f))

    def _update_progress_bar(self):
        if self.progress_bar is not None:
            self.progress_bar.update()

    def __iter__(self):
        self.iter_state = self._iterate_items()
        return self

    def __next__(self):
        self._update_progress_bar()
        return next(self.iter_state)

    def _iterate_items(self):
        """
        Iterate through each document in the F_CSV list
        """

        for f in self.F_CSV:
            with open(f, "r") as FIN:
                reader = csv.DictReader(FIN)
                for k, row in enumerate(reader):

                    # Return only the _ref and target_column if col is set
                    # Otherwise return the whole row
                    if self.col is not None:
                        row = {k: row[k] for k in ("_ref", self.col)}

                    if self.include_filename:
                        row["_filename"] = os.path.basename(f)

                    row["_ref"] = int(row["_ref"])
                    yield row

        if self.progress_bar is not None:
            self.progress_bar.close()


def text_iterator(F_CSV=None, progress_bar=False):
    """
    Returns a generator that loops the indicated files,
    if F_CSV is None or blank, loops over the parsed text data.
    """

    if F_CSV is None:
        F_CSV = get_section_filenames("parse")

    for x in CSV_database_iterator(
        F_CSV, target_column="text", progress_bar=progress_bar
    ):
        yield x


def get_section_filenames(section="parse"):
    """
    Grab filenames in given section of pipeline.

    Args:
        section (str): The section to grab the filenames (default: parse)

    Returns:
         list: files found in directory specified in config
    """

    config = simple_config.load()
    input_data_dir = config[section]["output_data_directory"]
    return grab_files("*.csv", input_data_dir)
