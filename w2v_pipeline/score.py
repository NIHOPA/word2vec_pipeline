import sqlite3, glob, os, random
from utils.os_utils import mkdir
import document_scoring as ds
from utils.db_utils import database_iterator, count_rows
import simple_config
import tqdm

_global_limit = 0

class item_iterator(object):

    def __init__(self, name, cmd_config=None, yield_single=False):

        # yield_single returns one item at a time,
        # not in chunks like (table_name, f_sql)
        
        self.yield_single = yield_single

        score_config = simple_config.load("parse")
        input_data_dir = score_config["output_data_directory"]

        F_SQL = sorted(glob.glob(os.path.join(input_data_dir,'*')))

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

        # Randomize the order of the input files (why? not needed for scoring)
        # F_SQL = random.sample(sorted(F_SQL), len(F_SQL))

        DB_ITR = itertools.product(F_SQL, config["target_columns"])

        # Get database sizes for progress bar
        self.total_items = 0
        for f_sql, target_col in DB_ITR:
            conn = sqlite3.connect(f_sql, check_same_thread=False)
            self.total_items += count_rows(conn, target_col)
            conn.close()
        
        self.F_SQL = F_SQL
        self.config = config

    def __len__(self):
        return self.total_items

    def __iter__(self):

        # Setup the progress bar
        progress_bar = tqdm.tqdm(total=self.total_items)

        # Rebuild the iterator
        DB_ITR = itertools.product(self.F_SQL,
                                   self.config["target_columns"])

        for f_sql, target_col in DB_ITR:

            conn = sqlite3.connect(f_sql, check_same_thread=False)

            args = {
                "column_name":"text",
                "table_name" :target_col,
                "conn":conn,
                "limit":_global_limit,
                "shuffle":False,
                "include_table_name":True,
            }

            requires_meta = []
            requires_ref  = ["document_scores",]

            if name in requires_meta:
                args["include_meta"] = True

            INPUT_ITR = database_iterator(**args)

            if not self.yield_single:
                data = []
                for item in INPUT_ITR:
                    val = list(item) + [f_sql,]
                    data.append(val)

            if self.yield_single:
                for item in INPUT_ITR:
                    val = list(item) + [f_sql,]
                    yield val
                    progress_bar.update()

            if not self.yield_single:
                yield data

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("score")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")

    n_jobs = -1 if _PARALLEL else 1

    mkdir(config["output_data_directory"])

    ###########################################################
    # Fill the pipeline with function objects

    mapreduce_functions = []
    for name in config["mapreduce_commands"]:

        obj  = getattr(ds,name)

        # Load any kwargs in the config file
        kwargs = {}
        if name in config:
            kwargs = config[name]

        # Add in the embedding configuration options
        kwargs["embedding"] = simple_config.load("embedding")
        kwargs["score"] = simple_config.load("score")

        val = name, obj(**kwargs)
        mapreduce_functions.append(val)

    for name, func in mapreduce_functions:
        print "Starting mapreduce {}".format(func.table_name)
        INPUT_ITR = item_iterator(name, config[name], yield_single=True)
        ITR = itertools.imap(func, INPUT_ITR)
        for item in ITR:
            result = item[0]
            func.reduce(result)

        func.save(config)

    ###########################################################
    # Run the functions that act globally on the data

    for name in config["globaldata_commands"]:
        obj  = getattr(ds,name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])

        # Add in the embedding configuration options
        kwargs["score"] = simple_config.load("score")
        kwargs["embedding"] = simple_config.load("embedding")
        
        func = obj(**kwargs)
        
        func.set_iterator_function(item_iterator,name,config[name])
        func.compute(config)
        
