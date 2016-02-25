import sqlite3, random

def database_iterator(
        column_name,
        table_name,
        conn,
        verbose=False,
        shuffle=False,
        limit=0,
        offset=0,
):

    cmd  = "SELECT [index],{} FROM {}".format(column_name, table_name)
    
    if limit: cmd  += " LIMIT {}  ".format(limit)
    if offset: cmd += " OFFSET {} ".format(offset)

    if not limit and offset:
        msg = "If offset is > 0, limit must be set"
        raise SyntaxError(msg)

    cursor = conn.execute(cmd)
    
    # If shuffle is true, load the entire set selection into memory, then
    # give permuted results
    if shuffle:
        results = cursor.fetchall()
        random.shuffle(results)
        for k,item in enumerate(results):
            yield item
    else:
        for k,item in enumerate(cursor):
            yield item



def pretty_counter(C,min_count=1):
    for item in C.most_common():
        (phrase, abbr),count = item
        if count>min_count:
            s = "{:10s} {: 10d} {}".format(abbr,count,' '.join(phrase))
            yield s
