'''
Builds the TF database for quick reference.
'''
import sqlite3, os, itertools
import collections

from utils.db_utils import database_iterator
from utils.os_utils import grab_files, mkdir

#####################################################################

input_table = "parsed5_pos_tokenized"

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEFAULT_EXPORT_DIRECTORY = "collated"

from utils.config_reader import load_config
cargs = load_config()
target_columns = cargs["target_columns"]
_DEBUG = cargs["debug"]

global_limit = 0
f_db_TF = "TF.sqlite"

######################################################################

F_SQL = grab_files("*.sqlite", _DEFAULT_IMPORT_DIRECTORY)
mkdir(_DEFAULT_EXPORT_DIRECTORY)

######################################################################

def word_counter(item):
    f_sqlite, target_column = item

    conn = sqlite3.connect(f_sqlite, check_same_thread=False)
    ITR  = database_iterator(target_column,
                             input_table, 
                             conn, 
                             limit=global_limit)


    C = collections.Counter()
    for k,(idx,text) in enumerate(ITR):
        tokens = text.split()
        C.update(tokens)

    conn.close()   
    return C

######################################################################

f_db = os.path.join(_DEFAULT_EXPORT_DIRECTORY, f_db_TF)
conn = sqlite3.connect(f_db, check_same_thread=False)

cmd = '''
DROP TABLE IF EXISTS TF;
DROP INDEX IF EXISTS idx_TF_word;
CREATE TABLE IF NOT EXISTS TF ( 
    rank INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT,
    count INTEGER
);
'''
conn.executescript(cmd)

######################################################################

INPUT_ITR = itertools.product(F_SQL, target_columns)
ITR = itertools.imap(word_counter, INPUT_ITR)

C = collections.Counter()

if not _DEBUG:
    import multiprocessing
    P = multiprocessing.Pool()
    ITR = P.imap(word_counter, INPUT_ITR)

for C_result in ITR:
    C.update(C_result)

print "Found {} tokens".format(len(C))


cmd_insert = '''
INSERT INTO TF (word, count) VALUES (?,?)
'''

data = C.most_common()
print "Inserting"
conn.executemany(cmd_insert, data)

print "Building index"
conn.executescript('''
CREATE INDEX IF NOT EXISTS 
idx_tf_word ON TF(word)
''')

print "Illustrating top 10 words"
cmd = '''
SELECT word,rank,count FROM TF
ORDER BY rank
LIMIT 10
'''

for item in conn.execute(cmd):
    print "{:15s} {:2d} {:10d}".format(*item)




