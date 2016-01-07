from utils.pipeline import text_pipeline
from utils.os_utils import grab_files
import pattern.en

input_table  = "parsed3_removed_special_tokens"
output_table = "parsed4_decaps_text"


from utils.config_reader import load_config
cargs = load_config()
target_columns = cargs["target_columns"]
_DEBUG = cargs["debug"]

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
global_limit = 0

word_tokenizer = lambda x:pattern.en.parse(x,chunks=False,tags=False)

class decaps_text(object):

    def diffn(self,s1,s2):
        return len([a for a,b in zip(s1,s2) if a!=b])
    
    def __init__(self):
        pass

    def modify_word(self,org):
        
        lower = org.lower()
        
        if self.diffn(org,lower) > 1:
            return org
        else:
            return lower
                                        
    def __call__(self,doc):

        sentences = word_tokenizer(doc).split()
        doc2 = []

        for sent in sentences:
            sent = [' '.join(x) for x in sent]
            if sent:
                sent[0] = self.modify_word(sent[0])

            doc2.append(' '.join(sent))
        doc2 = '\n'.join(doc2)
        return doc2

target_function = decaps_text()

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

    if not _DEBUG:
        import multiprocessing
        MP  = multiprocessing.Pool(len(F_SQL))
        ITR = MP.imap(compute, F_SQL)

    for f_sqlite in ITR:
        print "Completed", f_sqlite
