import random
import tqdm
import simple_config
import csv
import os
from os_utils import grab_files

"""
Utility file to assist in parsing .csv files, as well as display the process of parsing these files.
"""

#DOCUMENTATION_UNKNOWN
#this function is never used
def pretty_counter(C, min_count=1):
    """
    Counts
    :param C:
    :param min_count:
    :return:
    """
    for item in C.most_common():
        (phrase, abbr), count = item
        if count > min_count:
            s = "{:10s} {: 10d} {}".format(abbr, count, ' '.join(phrase))
            yield s


def CSV_list_columns(f_csv):
    """
    iterate through the columns of a .csv file.

    Args:
        f_csv: string location of the .csv file

    Returns:
        return tuple(reader.next()): DOCUMENTATION_UNKNOWN



    """
    if not os.path.exists(f_csv):
        msg = "File not found {}".format(f_csv)
        raise IOError(msg)
    with open(f_csv, 'rb') as FIN:
        reader = csv.reader(FIN)
        return tuple(reader.next())


class CSV_database_iterator(object):

    def __init__(self,
                 F_CSV,
                 target_column=None,
                 progress_bar=True,
                 shuffle=False,
                 limit=0,
                 offset=0,
                 include_meta=False,
                 include_table_name=False,
                 include_filename=False,
                 ):
        self.F_CSV = sorted(F_CSV)
        self.col = target_column

        # Raise Exception if column is missing in a CSV
        if self.col is not None:
            for f in F_CSV:
                f_cols = CSV_list_columns(f)
                if self.col not in f_cols:
                    msg = "Missing column {} in {}"
                    raise SyntaxError(msg.format(target_column, f))

        # Functions that may be added later (came from SQLite iterator)
        if shuffle:
            raise NotImplementedError("CSV_database_iterator shuffle")
        if include_table_name:
            raise NotImplementedError(
                "CSV_database_iterator include_table_name")
        if include_meta:
            raise NotImplementedError("CSV_database_iterator include_meta")

        self.progress_bar = tqdm.tqdm() if progress_bar else None
        self.limit = limit
        self.include_filename = include_filename

    def _update_progress_bar(self):
        if self.progress_bar is not None:
            self.progress_bar.update()

    def __iter__(self):
        self.iter_state = self._iterate_items()
        return self

    def next(self):
        self._update_progress_bar()
        return self.iter_state.next()

    def _iterate_items(self):

        for f in self.F_CSV:
            with open(f, 'rb') as FIN:
                reader = csv.DictReader(FIN)
                for k, row in enumerate(reader):

                    if self.limit and k > self.limit:
                        raise StopIteration

                    # Return only the _ref and target_column if col is set
                    # Otherwise return the whole row
                    if self.col is not None:
                        row = {k: row[k] for k in ('_ref', self.col)}

                    if self.include_filename:
                        row["_filename"] = os.path.basename(f)

                    yield row

        if self.progress_bar is not None:
            self.progress_bar.close()


def item_iterator(
        config=None,
        randomize_file_order=False,
        whitelist=[],
        section='parse',
        progress_bar=False,
        text_column=None,
        include_filename=False,
):
    '''
    Iterates over the parsed corpus items and respects a given whitelist.
    '''

    if config is None:
        config = simple_config.load()

    config = simple_config.load()
    input_data_dir = config['parse']["output_data_directory"]
    F_CSV = grab_files("*.csv", input_data_dir, verbose=False)

    if whitelist:
        assert(isinstance(whitelist, list))

        F_CSV2 = set()
        for f_csv in F_CSV:
            for token in whitelist:
                if token in f_csv:
                    F_CSV2.add(f_csv)
        F_CSV = F_CSV2

    # Randomize the order of the input files each time we get here
    if randomize_file_order:
        F_CSV = random.sample(sorted(F_CSV), len(F_CSV))

    INPUT_ITR = CSV_database_iterator(
        F_CSV,
        config["target_column"],
        progress_bar=progress_bar,
        include_filename=include_filename,
    )

    for row in INPUT_ITR:
        if text_column is not None:
            row['text'] = row[text_column]
        yield row



def get_section_filenames(section):
    config = simple_config.load()
    input_data_dir = config['parse']["output_data_directory"]
    return grab_files("*.csv", input_data_dir, verbose=False)

def single_file_item_iterator(
        f_csv,
        config=None,
        section='parse',
        progress_bar=False,
        text_column=None,
        include_filename=True,
):
    '''
    Iterates over a single file
    '''
    

    if config is None:
        config = simple_config.load()

    config = simple_config.load()

    # Make sure the file we requested exists
    assert(f_csv in get_section_filenames(section))
    

    INPUT_ITR = CSV_database_iterator(
        [f_csv],
        config["target_column"],
        progress_bar=progress_bar,
        include_filename=include_filename,
    )

    for row in INPUT_ITR:
        if text_column is not None:
            row['text'] = row[text_column]
        yield row
