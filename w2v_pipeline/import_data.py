import os
import itertools
from utils.os_utils import mkdir, grab_files

import pandas as pd
from unidecode import unidecode

from utils.parallel_utils import jobmap

def map_to_unicode(s):
    # Helper function to fix input format
    s = str(s)
    return s.decode('utf-8',errors='replace')

def clean_dataframe(df):
    '''
    Changes all columns of type object into strings.
    All string types are mapped through unidecode.
    '''
    
    for col,dtype in zip(df.columns,df.dtypes):
        if dtype=="object":

            all_types = set(df[col].map(type).values)

            if str in all_types:
                df[col] = df[col].map(map_to_unicode).map(unidecode)
            elif float in all_types:
                df[col] = df[col].astype(float)
    return df

def load_csv(f_csv, clean=True):
    '''
    Loads a CSV file into a pandas DataFrame. 
    Runs the dataframe through a cleaner.
    '''

    print "Starting import of", f_csv

    df = pd.read_csv(f_csv)
    
    if clean:
        df = clean_dataframe(df)

    return f_csv, df

def import_directory_csv(d_in, d_out, output_table):
    '''
    Takes a input_directory and output_directory and builds
    and cleaned (free of encoding errors) CSV for all input
    and attaches unique _ref numbers to each entry.
    '''

    F_CSV     = []
    F_CSV_OUT = {}

    INPUT_FILES = grab_files("*.csv",d_in)
    
    if not INPUT_FILES:
        print "No matching CSV files found, exiting"
        exit(2)

    for f_csv in INPUT_FILES:
        f_csvx = os.path.join(d_out, os.path.basename(f_csv))

        if os.path.exists(f_csvx) and not _FORCE:
            print "{} already exists, skipping".format(f_csvx)
            continue

        F_CSV.append(f_csv)
        F_CSV_OUT[f_csv] = f_csvx

    # Create the output directory if needed
    mkdir(d_out)
    ITR = jobmap(load_csv, F_CSV, _PARALLEL)

    # Create a reference ID for each item
    _ref_counter = itertools.count()

    for (f_csv,df) in ITR:

        df["_ref"] = list(itertools.islice(_ref_counter, len(df)))
        df.set_index("_ref",inplace=True)

        df.to_csv(F_CSV_OUT[f_csv])
        
        msg = "Imported {} to {}, {}, {}"
        print msg.format(f_csv, F_CSV_OUT[f_csv], len(df), list(df.columns))

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("import_data")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    data_out = config["output_data_directory"]
    output_table = config["output_table"]

    # Require `input_data_directories` to be a list
    data_in_list  = config["input_data_directories"]
    assert(type(data_in_list) == list)
  
    for d_in in data_in_list:
        import_directory_csv(d_in, data_out, output_table)


