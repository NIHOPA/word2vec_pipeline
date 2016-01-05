import sqlite3

def database_iterator(
        column_name,
        table_name,
        conn,
        verbose=False, 
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
    for k,item in enumerate(cursor):
        if verbose and k and k%10000==0:
            print k
        yield item


def pretty_counter(C,min_count=1):
    for item in C.most_common():
        (phrase, abbr),count = item
        if count>min_count:
            s = "{:10s} {: 10d} {}".format(abbr,count,' '.join(phrase))
            yield s
