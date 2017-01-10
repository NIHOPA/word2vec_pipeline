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
        msg = "Table {} not database.".format(table)
        raise ValueError(msg)
        
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
        cmd  = "SELECT {},_ref {} FROM {}".format(column_name,
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


#######################################################################
import csv, os

def CSV_list_columns(f_csv):
    if not os.path.exists(f_csv):
        msg = "File not found {}".format(f_csv)
        raise IOError(msg)
    with open(f_csv,'rb') as FIN:
        reader = csv.reader(FIN)
        return tuple(reader.next())

class CSV_database_iterator(object):

    def __init__(self,
                 F_CSV,
                 target_column=None,
                 progress_bar=True,
                 shuffle=False,
                 limit=0,
                 offset=0,
                 include_meta=False,
                 include_table_name=False,
                 include_filename=False,
    ):
        self.F_CSV = sorted(F_CSV)
        self.col = target_column
        
        # Raise Exception if column is missing in a CSV
        if self.col is not None:
            for f in F_CSV:
                if self.col not in CSV_list_columns(f):
                    msg = "Missing column {} in {}"
                    raise SyntaxError(msg.format(column_name,f))

        # Functions that may be added later (came from SQLite iterator)
        if shuffle:
            raise NotImplementedError("CSV_database_iterator shuffle")
        if include_table_name:
            raise NotImplementedError("CSV_database_iterator include_table_name")
        if include_meta:
            raise NotImplementedError("CSV_database_iterator include_meta")

        self.progress_bar = tqdm.tqdm() if progress_bar else None
        self.limit = limit
        self.include_filename = include_filename

    def _update_progress_bar(self):
        if self.progress_bar is not None:
            self.progress_bar.update()

    def __iter__(self):
        self.iter_state = self._iterate_items()
        return self

    def next(self):
        self._update_progress_bar()
        return self.iter_state.next()
        
    def _iterate_items(self):

        for f in self.F_CSV:
            with open(f,'rb') as FIN:
                reader = csv.DictReader(FIN)
                for k,row in enumerate(reader):
                    
                    if self.limit and k>self.limit:
                        raise StopIteration
                    
                    # Return only the _ref and target_column if col is set
                    # Otherwise return the whole row
                    if self.col is not None:
                        row = {k: row[k] for k in ('_ref', self.col)}

                    if self.include_filename:
                        row["_filename"] = f
                        
                    yield row                        
 
        if self.progress_bar is not None:
            self.progress_bar.close()
