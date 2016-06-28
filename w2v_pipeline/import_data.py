import os
import itertools
from utils.os_utils import mkdir, grab_files

import pandas as pd
from sqlalchemy import create_engine
from unidecode import unidecode

from utils.parallel_utils import jobmap

def map_to_unicode(s):
    # Helper function to fix input format
    s = str(s)
    return s.decode('utf-8',errors='replace')

def clean_dataframe(df):
    # Changes all columns of type object into strings
    # mapped through unidecode
    for col,dtype in zip(df.columns,df.dtypes):
        if dtype=="object":

            all_types = set(df[col].map(type).values)

            if str in all_types:
                df[col] = df[col].map(map_to_unicode).map(unidecode)
            elif float in all_types:
                df[col] = df[col].astype(float)
    return df

def load_csv(f_csv):
    # Loads a CSV file into a pandas DataFrame

    print "Starting", f_csv

    df = pd.read_csv(f_csv)
    df = clean_dataframe(df)

    return f_csv, df

def import_directory_csv(d_in, d_out, output_table):

    F_CSV = []
    F_SQL = {}

    INPUT_FILES = grab_files("*.csv",d_in)
    
    if not INPUT_FILES:
        print "No matching CSV files found, exiting"
        exit(2)

    for f_csv in INPUT_FILES:
        f_sql = '.'.join(os.path.basename(f_csv).split('.')[:-1])
        f_sql += ".sqlite"                        
        f_sql = os.path.join(d_out,f_sql)

        if os.path.exists(f_sql) and not _FORCE:
            print "{} already exists, skipping".format(f_sql)
            continue

        F_CSV.append(f_csv)
        F_SQL[f_csv] = f_sql


    # Create the output directory if needed
    mkdir(d_out)
    ITR = jobmap(load_csv, F_CSV, _PARALLEL)

    # Create a reference ID for each item
    _ref_counter = itertools.count()

    for (f_csv,df) in ITR:

        f_sql = F_SQL[f_csv]
        engine = create_engine('sqlite:///'+f_sql)

        n_data_items = len(df)
        df["_ref"] = [_ref_counter.next()
                      for _ in range(n_data_items)]
        df.set_index("_ref",inplace=True)

        df.to_sql(output_table,
                  engine,
                  if_exists='replace')

        print "Finished {}, {}, {}".format(f_csv, len(df), list(df.columns))


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


