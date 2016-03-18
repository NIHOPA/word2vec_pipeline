import sqlite3, glob, os, itertools, random
from utils.os_utils import mkdir
import model_building as mb
from utils.db_utils import database_iterator
import simple_config

global_limit = 0

def item_iterator(cmd_config=None):

    train_config = simple_config.load("train")
    input_data_dir = train_config["input_data_directory"]

    F_SQL = glob.glob(os.path.join(input_data_dir,'*'))

    # If there is a whitelist only keep the matching filename
    if "command_whitelist" in cmd_config:
        whitelist = cmd_config["command_whitelist"]
        
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

        print f_sql

        #print ("Computing {}:{}".format(f_sql, target_col))
        
        conn = sqlite3.connect(f_sql, check_same_thread=False)

        args = {
            "column_name":"text",
            "table_name" :target_col,
            "conn":conn,
            "limit":global_limit,
            "shuffle":False,
        }


        print cmd_config

        if "require_meta" in cmd_config:
            args["include_meta"] = True
            INPUT_ITR = database_iterator(**args)
            print INPUT_ITR.next()
            for idx,text,meta in INPUT_ITR:
                yield (text,meta,idx,f_sql)

        else:
            INPUT_ITR = database_iterator(**args)

            for idx,text in INPUT_ITR:
                yield (text,idx,f_sql)
                


if __name__ == "__main__":

    import simple_config
    config = simple_config.load("train")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    if _PARALLEL:
        import multiprocessing

    ###########################################################
    # Fill the pipeline with function objects

    mapreduce_functions = []
    for name in config["mapreduce_commands"]:
        obj  = getattr(mb,name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in config:
            kwargs = config[name]

        val = name, obj(**kwargs)
        mapreduce_functions.append(val)

    for name, func in mapreduce_functions:

        INPUT_ITR = item_iterator(config[name])
        
        if _PARALLEL:
            MP = multiprocessing.Pool()
            ITR = MP.imap(func, INPUT_ITR, chunksize=200)
        else:
            ITR = itertools.imap(func, INPUT_ITR)

        for item in ITR:
            result,idx,f_sql = item
            func.reduce(result)

        func.save(config)

    ###########################################################
    # Run the functions that act globally on the data

    for name in config["globaldata_commands"]:
        obj  = getattr(mb,name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])
            
        func = obj(**kwargs)
        func.set_iterator_function(item_iterator,config[name])
        func.compute(config)        
