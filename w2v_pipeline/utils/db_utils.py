import sqlite3, random, tqdm

def list_tables(conn):
    cmd = "SELECT name FROM sqlite_master WHERE type='table';"
    result = conn.execute(cmd).fetchall()
    
    if not result:
        return []

    tables = zip(*result)[0]
    return tables

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

def database_iterator(
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

    meta_field = "" if not include_meta else ",meta"
    cmd  = "SELECT {},[index] {} FROM {}".format(column_name,
                                                 meta_field,
                                                 table_name)

    if limit: cmd  += " LIMIT {}  ".format(limit)
    if offset: cmd += " OFFSET {} ".format(offset)
    
    if not limit and offset:
        msg = "If offset is > 0, limit must be set"
        raise SyntaxError(msg)

    if progress_bar:
        count_cmd = "SELECT MAX(ROWID) FROM {}".format(table_name)
        total = conn.execute(count_cmd).next()[0]

        if limit: 
            total = min(limit, total)
                
        progress_bar = tqdm.tqdm(total=total)

    try:
        cursor = conn.execute(cmd)
    except Exception as Ex:
        print Ex
        raise(Ex)
    
    # If shuffle is true, load the entire set selection into memory, then
    # give permuted results
    if shuffle:
        cursor = random.shuffle(cursor.fetchall())
    
    for k,item in enumerate(cursor):

        if include_table_name:
            item = list(item) + [table_name,]
        
        yield item
            
        if progress_bar:
            progress_bar.update()


def pretty_counter(C,min_count=1):
    for item in C.most_common():
        (phrase, abbr),count = item
        if count>min_count:
            s = "{:10s} {: 10d} {}".format(abbr,count,' '.join(phrase))
            yield s
