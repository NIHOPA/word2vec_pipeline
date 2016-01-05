from utils.pipeline import text_pipeline
from utils.os_utils import grab_files

input_table  = "parsed2_no_paren"
output_table = "parsed3_removed_special_tokens"

target_columns = ["abstract", "specificAims"]

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEBUG    = False
_PARALLEL = True

global_limit = 0

class token_replacement(object):
    def __init__(self):
        pass
                                        
    def __call__(self,doc):
        doc = doc.replace('&', ' and ')
        doc = doc.replace('%', ' percent ')
        doc = doc.replace('>', ' greater-than ')
        doc = doc.replace('<', ' less-than ')
        doc = doc.replace('=', ' equals ')
        doc = doc.replace('#',  ' ')
        doc = doc.replace('~', ' ')
        doc = doc.replace('/' , ' ')
        doc = doc.replace('\\', ' ')
        doc = doc.replace('|', ' ')
        doc = doc.replace('$', '')

        # Remove possesive splits
        doc = doc.replace(" 's ", ' ')

        doc = doc.replace("'", '')
        doc = doc.replace('"', '')

        return doc

target_function = token_replacement()

######################################################################
 
def compute(f_sqlite):

    func = target_function

    for col in target_columns:
        pipeline = text_pipeline(col, func,
                                 input_table, output_table,
                                 debug=_DEBUG,
                                 limit=global_limit, 
                                 verbose=True)
        pipeline(f_sqlite)

    return f_sqlite

if __name__ == "__main__":
    F_SQL = grab_files("*.sqlite", _DEFAULT_IMPORT_DIRECTORY)

    import itertools
    ITR = itertools.imap(compute, F_SQL)

    if _PARALLEL:
        import multiprocessing
        MP  = multiprocessing.Pool(len(F_SQL))
        ITR = MP.imap(compute, F_SQL)

    for f_sqlite in ITR:
        print "Completed", f_sqlite
