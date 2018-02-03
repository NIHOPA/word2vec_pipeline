"""
Utility file to assist in parsing .csv files, as well as display the process of parsing these files.
"""

import tqdm
import simple_config
import csv
import os
from os_utils import grab_files


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
    with open(f_csv, 'rb') as FIN:
        reader = csv.reader(FIN)
        return tuple(reader.next())


class CSV_database_iterator(object):
    '''
    Class to open .csv files, check that they are valid, and iterate 
    through them.
    
    DOCUMENTATION_UNKNOWN : the name "F_CSV" makes it seem like one file, 
    when it actually works on a list of multiple files. It might make the 
    code clearer to rename this variable
    '''

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
        
        '''
        Initialize the iterator

        Args:
            F_CSV: a list of .csv files to iterate over
            target_column: string, the column name that is being parsed
            progress_bar: boolean, a flag to display a progress bar
            shuffle: DOCUMENTATION_UNKNOWN
            limit: DOCUMENTATION_UNKNOWN
            offset: DOCUMENTATION_UNKNOWN
            include_meta: DOCUMENTATION_UNKNOWN
            include_table_name: DOCUMENTATION_UNKNOWN
            include_filename: boolean, a flag to save the documentation 
               location in imported document
        '''

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
        '''
        Iterate through each document in the F_CSV list
        '''

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

                    row['_ref'] = int(row['_ref'])
                    yield row

        if self.progress_bar is not None:
            self.progress_bar.close()


def text_iterator(
    F_CSV=None,
    progress_bar=True,
):
    '''
    Returns a generator that loops the indicated files,
    if F_CSV is None or blank, loops over the parsed text data.
    '''

    if F_CSV is None:
        F_CSV = get_section_filenames('parse')

    for x in CSV_database_iterator(F_CSV, target_column='text',
                                   progress_bar=progress_bar,):
        yield x


def get_section_filenames(section='parse'):
    '''
    Grab filenames in given section of pipeline.

    Args:
        section (str): The section to grab the filenames (default: parse)

    Returns:
         list: files found in directory specified in config
    '''

    config = simple_config.load()
    input_data_dir = config[section]["output_data_directory"]
    return grab_files("*.csv", input_data_dir, verbose=False)


