import sqlite3, glob
import pattern.en
from utils.pipeline import text_pipeline
from utils.os_utils import grab_files, mkdir

input_table  = "original"
output_table = "parsed1_abbr_tokens"

from utils.config_reader import load_config
cargs = load_config()
target_columns = cargs["target_columns"]
_DEBUG = cargs["debug"]
_FORCE = cargs["force"]

f_abbreviations = "collated/abbreviations.sqlite"

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
global_limit = 0

word_tokenizer = lambda x:pattern.en.parse(x,chunks=False,tags=False)

class phrase_replacement(object):
    def __init__(self):
        self.load_phrase_database()

    def load_phrase_database(self):
        # Load the phrases from abbrs
        conn = sqlite3.connect(f_abbreviations,check_same_thread=False)
        cmd  = "SELECT phrase,abbr,count FROM abbreviations"
        cursor = conn.execute(cmd)

        self.P = {}
        self.max_n = 0
        self.min_n = 10**10

        for phrase,abbr,count in cursor:
            phrase = tuple(phrase.split(' '))
            self.P[phrase] = abbr
            self.max_n = max(self.max_n,len(phrase))
            self.min_n = min(self.min_n,len(phrase))


    def ngram_tokens(self, tokens, n):

        for k in range(len(tokens)-n):
            block = tokens[k:k+n]
            lower_block = tuple([x.lower() for x in block])
            substring = ' '.join(block)
            yield lower_block, substring

    def phrase_sub(self, phrase):
        return '_'.join(["PHRASE"]+list(phrase))
                                       
    def __call__(self,org_doc):

        doc = org_doc
        tokens = unicode(word_tokenizer(doc)).split()

        # First pass, identify which phrases are used
        iden_abbr = {}
        replacements = {}
        for n in range(self.min_n, self.max_n+1):
            for phrase, substring in self.ngram_tokens(tokens, n):
                
                if phrase in self.P:
                    abbr = self.P[phrase]
                    iden_abbr[phrase] = abbr
                    replacements[substring] = self.phrase_sub(phrase)

        # Replace these with a phrase token
        for substring, newstring in replacements.items():
            doc = doc.replace(substring,newstring)

        # Now find any abbrs used in the document and replace them       
        tokens = unicode(word_tokenizer(doc)).split()
        
        for phrase,abbr in iden_abbr.items():
            tokens = [self.phrase_sub(phrase) 
                      if x==abbr else x for x in tokens]

        # This returns word split phrase string
        doc = ' '.join(tokens)

        return doc

target_function = phrase_replacement()

######################################################################

def compute(f_sqlite):

    func = target_function

    for col in target_columns:
        pipeline = text_pipeline(col, func,
                                 input_table, output_table,
                                 debug=_DEBUG,
                                 force=_FORCE,
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

    
