import sqlite3, glob, os, itertools
from utils.db_utils import database_iterator
from utils.os_utils import mkdir
import preprocessing as pre

global_limit = 10

def dispatcher(item):
    idx,x  = item
    return idx, reduce(lambda x, f: f(x), parser_functions, x)

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("parse")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    if _PARALLEL:
        import multiprocessing

    import_config = simple_config.load("import_data")
    input_data_dir = import_config["output_data_directory"]
    output_dir = config["output_data_directory"]

    mkdir(output_dir)

    # Fill the pipeline with function objects
    parser_functions = []
    for name in config["pipeline"]:
        obj  = getattr(pre,name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in config:
            kwargs = config[name]

        parser_functions.append( obj(**kwargs) )

    F_SQL = glob.glob(os.path.join(input_data_dir,'*'))
    DB_ITR = itertools.product(F_SQL, config["target_columns"])

    for f_sql, target_col in DB_ITR:
        print "Parsing on {}:{}".format(f_sql, target_col)
        
        conn = sqlite3.connect(f_sql, check_same_thread=False)

        args = {
            "column_name":target_col,
            "table_name":import_config["output_table"],
            "conn":conn,
            "limit":global_limit,
        }
            
        INPUT_ITR = database_iterator(**args)

        if not _PARALLEL:
            ITR = itertools.imap(dispatcher, INPUT_ITR)

        if _PARALLEL:
            import multiprocessing
            P = multiprocessing.Pool()
            ITR = P.imap(dispatcher, INPUT_ITR,chunksize=5)

        mkdir(output_dir)
        f_sql_out = os.path.join(output_dir, os.path.basename(f_sql))
        conn_out  = sqlite3.connect(f_sql_out)

        #engine = sqlalchemy.create_engine('sqlite:///'+f_sql)

        cmd_create = '''
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE IF NOT EXISTS {table_name} (
        [index] INTEGER PRIMARY KEY,
        text STRING
        );
        '''.format(table_name=target_col)
        
        conn_out.executescript(cmd_create)

        cmd_insert = '''
        INSERT INTO {table_name} ([index],text)
        VALUES (?,?)
        '''.format(table_name=target_col)

        conn_out.executemany(cmd_insert, ITR)
        conn_out.commit()
        conn_out.close()

