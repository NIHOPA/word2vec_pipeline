import os
from os_utils import mkdir, grab_files

import pandas as pd
from sqlalchemy import create_engine
from unidecode import unidecode

_DEFAULT_IMPORT_DIRECTORY = "csv_data"
_DEFAULT_EXPORT_DIRECTORY = "sql_data"
_DEBUG = False

output_table = "original"

def map_to_unicode(s):
    # Helper function to fix input format
    s = str(s)
    return s.decode('utf-8')

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


def import_directory_csv(d_in=_DEFAULT_IMPORT_DIRECTORY,
                         add_column_from_filename=[]):

    FILES = grab_files("*.csv",d_in)   

    if not FILES:
        print "No matching CSV files found, exiting"
        exit(2)

    if _DEBUG:
        import itertools
        ITR = itertools.imap(load_csv, FILES)

    if not _DEBUG:
        import multiprocessing
        P = multiprocessing.Pool()
        ITR = P.imap(load_csv, FILES)

    mkdir(_DEFAULT_EXPORT_DIRECTORY)

    for (f_csv,df) in ITR:

        for func in add_column_from_filename:
            name,val = func(f_csv)
            df[name] = val

        f_sql = '.'.join(os.path.basename(f_csv).split('.')[:-1])
        f_sql += ".sqlite"                        
        f_sql = os.path.join('sql_data',f_sql)
        engine = create_engine('sqlite:///'+f_sql)

        df.to_sql(output_table,
                  engine,
                  if_exists='replace')

        print "Finished", f_csv


if __name__ == "__main__":

    def get_year(f_csv):
        base = os.path.basename(f_csv).split('.csv')[-2]
        year = base.split('-')[-1]
        return "year", int(year)
        

    import_directory_csv(add_column_from_filename=[get_year,])

