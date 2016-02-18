import sqlite3
import itertools, multiprocessing
import db_utils

class text_pipeline(object):
    def __init__(self, target_column, func, 
                 input_table, output_table,
                 verbose=False, debug=False, force=False,
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
        self.force = force
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

        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        table_names = zip(*self.conn.execute(cmd).fetchall())[0]

        if t_out not in table_names:
            cmd_template = '''
            CREATE TABLE {out_table} (
            [index] INTEGER PRIMARY KEY);
            '''.format(out_table=t_out)

            self.conn.execute(cmd_template)

        # Insert the index from one column into another
        cmd_insert = '''
        INSERT OR IGNORE INTO {out_table} ([index])
        SELECT [index] FROM {in_table};
        '''.format(in_table=self.t_in, out_table=t_out)
        self.conn.execute(cmd_insert)
        self.conn.commit()

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

        # Find out which rows have been completed
        cmd_search = '''SELECT [index] FROM {out_table}
        WHERE {target_column} IS NOT NULL'''.format(out_table=t_out, 
                                                    target_column=target_column)

        cursor = self.conn.execute(cmd_search)
        known_index = cursor.fetchall()
        if known_index:
            known_index = set(zip(*known_index)[0])

        # If forcing, compute all columns
        if self.force:
            known_index = []                    

        INPUT_ITR = ((index,text) for (index,text) in
                     self.get_input_iter() if index not in known_index)

        # Group the data and function together
        func_iter = itertools.cycle([self.func])
        data_iter = itertools.izip(INPUT_ITR, func_iter)

        def parse_item((item, func)):
            index, text = item
            return func(text), index

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
