import sqlite3
import itertools, multiprocessing
import db_utils

def parse_item((item, func)):
    # This function has to be outside the pipeline class for 
    # multiprocessing to work
    index, text = item
    return func(text), index

class text_pipeline(object):
    def __init__(self, target_column, func, 
                 input_table, output_table,
                 verbose=False, debug=False, 
                 limit=0,
                 offset=0,
    ):
        self.target_column = target_column
        self.t_in  = input_table
        self.t_out = output_table
        self.func  = func
        self.verbose = verbose
        self.limit = limit
        self.offset = offset
        self.debug = debug
        self.insert_size = 400

        self.conn = None

    def get_input_iter(self):
        return db_utils.database_iterator(
            self.target_column,
            self.t_in,
            self.conn,
            limit=self.limit,
            offset=self.offset,
            verbose=self.verbose)

    def connect(self, f_sqlite):
        self.conn = sqlite3.connect(f_sqlite,check_same_thread=False)

    def __call__(self, f_sqlite):
        t_out = self.t_out
        target_column = self.target_column

        print "Starting", f_sqlite, target_column
        self.connect(f_sqlite)
        
        cmd_template = '''
        CREATE TABLE {out_table} (
        [index] INTEGER PRIMARY KEY);
        '''.format(out_table=t_out)
        try:
            self.conn.execute(cmd_template)
            FLAG_new_table = True
        except sqlite3.OperationalError:
            FLAG_new_table = False

        # Insert the index from one column into another
        if FLAG_new_table:
            cmd_insert = '''
            INSERT INTO {out_table} ([index])
            SELECT [index] FROM {in_table};
            '''.format(in_table=self.t_in, out_table=t_out)
            self.conn.execute(cmd_insert)

        # Determine if the column is new or not
        cmd = "SELECT * FROM {out_table} LIMIT 0".format(out_table=t_out)
        col_names = self.conn.execute(cmd)
        col_names = zip(*col_names.description)[0]

        # Alter the table to allow the column (if needed)
        cmd = "ALTER TABLE {out_table} ADD COLUMN {target_column} TEXT;"
        if target_column not in col_names:
            cmd = cmd.format(out_table=t_out, target_column=target_column)
            self.conn.execute(cmd)
            self.conn.commit()

        INPUT_ITR = self.get_input_iter()

        # Group the data and function together
        func_iter = itertools.cycle([self.func])
        data_iter = itertools.izip(INPUT_ITR, func_iter)

        ITR = itertools.imap(parse_item, data_iter)

        cmd_update = '''
        UPDATE {out_table} SET {target_column}=(?)
        WHERE [index]=(?)
        '''.format(out_table=t_out,target_column=target_column)

        if not self.debug:
            self.conn.executemany(cmd_update, ITR)
            self.conn.commit()
        else:
            for _ in ITR: pass
