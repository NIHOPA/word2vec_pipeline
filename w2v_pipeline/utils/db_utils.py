import sqlite3, random, tqdm

def list_tables(conn):
    cmd = "SELECT name FROM sqlite_master WHERE type='table';"
    result = conn.execute(cmd).fetchall()
    
    if not result:
        return []

    tables = zip(*result)[0]
    return tables

def list_columns(conn, table_name):
    cmd = "SELECT * FROM {} LIMIT 0;".format(table_name)
    cursor = conn.execute(cmd)
    col_names = zip(*cursor.description)[0]
    return col_names

def count_rows(conn, table):
    '''
    Not the true row count, but the maxID. Useful for sanity checks.
    '''
    if table not in list_tables(conn):
        return None
        
    #cmd = "SELECT MAX(_ROWID_) FROM {} LIMIT 1;"
    cmd = "SELECT COUNT(*) FROM {}"
    cursor = conn.execute(cmd.format(table))
    result = cursor.fetchall()[0][0]
    return result

class database_iterator(object):

    def __init__(self,
                 column_name,
                 table_name,
                 conn,
                 progress_bar=False,
                 shuffle=False,
                 limit=0,
                 offset=0,
                 include_meta=False,
                 include_table_name=False,
    ):

        # Raise an exception if the column isn't found
        
        if column_name not in list_columns(conn, table_name):
            msg = 'Column "{}" not found in table "{}"'
            raise SyntaxError(msg.format(column_name,table_name))

        meta_field = "" if not include_meta else ",meta"
        cmd  = "SELECT {},[index] {} FROM {}".format(column_name,
                                                     meta_field,
                                                     table_name)
        # Adjust the limits and offset
        
        if limit: cmd  += " LIMIT  {} ".format(limit)
        if offset: cmd += " OFFSET {} ".format(offset)

        if not limit and offset:
            msg = "If offset is > 0, limit must be set"
            raise SyntaxError(msg)

        if progress_bar:
            total = count_rows(conn, table_name)

            if limit: 
                total = min(limit, total)

            progress_bar = tqdm.tqdm(total=total)

        self.cmd = cmd
        self.progress_bar = progress_bar
        self.table_name = table_name
        self.conn = conn
        self.shuffle = shuffle
        self.include_table_name = include_table_name

    def _update_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.update()

    def __iter__(self):
        
        cursor = self.conn.execute(self.cmd)

        # If shuffle is true, load the entire set selection into memory, 
        # then give permuted results
        if self.shuffle:
            cursor = random.shuffle(cursor.fetchall())

        for k,item in enumerate(cursor):

            # If the table name is required, pass this through
            if self.include_table_name:
                item = list(item) + [self.table_name,]

            yield item
            self._update_progress_bar()



def pretty_counter(C,min_count=1):
    for item in C.most_common():
        (phrase, abbr),count = item
        if count>min_count:
            s = "{:10s} {: 10d} {}".format(abbr,count,' '.join(phrase))
            yield s
