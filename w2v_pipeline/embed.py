import sqlite3, glob, os, itertools, random
from utils.os_utils import mkdir
import model_building as mb
from utils.db_utils import database_iterator
import simple_config

_global_limit = 0

def item_iterator(name,cmd_config=None):

    score_config = simple_config.load("parse")
    input_data_dir = score_config["output_data_directory"]

    F_SQL = glob.glob(os.path.join(input_data_dir,'*'))

    # If there is a whitelist only keep the matching filename
    try:
        whitelist = cmd_config["command_whitelist"].strip()
    except:
        whitelist = None
    if whitelist:
        assert(type(whitelist)==list)

        F_SQL2 = set()
        for f_sql in F_SQL:
            for token in whitelist:
                if token in f_sql:
                    F_SQL2.add(f_sql)
        F_SQL = F_SQL2

    
    # Randomize the order of the input files
    F_SQL = random.sample(sorted(F_SQL), len(F_SQL))  
    DB_ITR = itertools.product(F_SQL, config["target_columns"])

    for f_sql, target_col in DB_ITR:

        #print ("Computing {}:{}".format(f_sql, target_col))
        
        conn = sqlite3.connect(f_sql, check_same_thread=False)

        args = {
            "column_name":"text",
            "table_name" :target_col,
            "conn":conn,
            "limit":_global_limit,
            "shuffle":False,
            "include_table_name":True,
        }

        INPUT_ITR = database_iterator(**args)
        for item in INPUT_ITR:
            yield list(item) + [f_sql,]

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("embedding")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    mkdir(config["output_data_directory"])

    if _PARALLEL:
        import multiprocessing

    ###########################################################
    # Run the functions that act globally on the data

    for name in config["embedding_commands"]:
        obj  = getattr(mb,name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])
            
        func = obj(**kwargs)
        func.set_iterator_function(item_iterator,name,config[name])
        func.compute(config)        
