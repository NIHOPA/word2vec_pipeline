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

    if limit: cmd += " LIMIT {}".format(limit)
    if offset: cmd += " OFFSET {}".format(offset)

    cursor = conn.execute(cmd)
    for k,item in enumerate(cursor):
        if verbose and k and k%1000==0:
            print k
        yield item
