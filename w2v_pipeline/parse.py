import sqlite3, glob, os, itertools
from utils.db_utils import database_iterator, list_tables, count_rows
from utils.os_utils import mkdir
import preprocessing as pre
import gc

from utils.parallel_utils import jobmap

global_limit = 0

def dispatcher(item):
    text,idx  = item
    meta = {}
    
    for f in parser_functions:
        result = f(text)
        text   = unicode(result)
        
        if hasattr(result,"meta"):
            meta.update(result.meta)

    # Convert the meta information into a unicode string for serialization
    meta = unicode(meta)

    return idx, text, meta

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("parse")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    import_config = simple_config.load("import_data")
    input_data_dir = import_config["output_data_directory"]
    output_dir = config["output_data_directory"]

    import_column = import_config["output_table"]

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

        f_sql_out = os.path.join(output_dir, os.path.basename(f_sql))
        mkdir(output_dir)
        conn_out  = sqlite3.connect(f_sql_out)
        conn = sqlite3.connect(f_sql, check_same_thread=False)

        tables = list_tables(conn_out)

        if target_col in tables:

            if not _FORCE:

                row_n_conn = count_rows(conn, import_column)
                row_n_conn_out = count_rows(conn_out, target_col)

                if row_n_conn == row_n_conn_out:
                    msg = "{}:{} already exists, skipping"
                    print msg.format(f_sql,target_col)
                    continue

                msg = "{} already exists but there is a size mismatch {} to {}"
                print msg.format(target_col, row_n_conn, row_n_conn_out)

            # Remove the table if it exists
            print "Removing table {}:{}".format(f_sql,target_col)
            conn_out.execute("DROP TABLE {}".format(target_col))
        
        
        print "Parsing {}:{}".format(f_sql, target_col)       

        args = {
            "column_name":target_col,
            "table_name":import_config["output_table"],
            "conn":conn,
            "limit":global_limit,
            "progress_bar":True,
        }
            
        INPUT_ITR = database_iterator(**args)

        ITR = jobmap(dispatcher, INPUT_ITR, _PARALLEL)

        cmd_create = '''
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE IF NOT EXISTS {table_name} (
        _ref INTEGER PRIMARY KEY,
        text STRING,
        meta STRING
        );
        '''.format(table_name=target_col)
        
        conn_out.executescript(cmd_create)

        cmd_insert = '''
        INSERT INTO {table_name} (_ref,text,meta)
        VALUES (?,?,?)
        '''.format(table_name=target_col)

        conn_out.executemany(cmd_insert, ITR)
        conn_out.commit()
        conn_out.close()
        conn.close()
            
        del INPUT_ITR, ITR, conn, conn_out
        gc.collect()
