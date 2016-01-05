from utils.pipeline import text_pipeline
from utils.os_utils import grab_files
import pattern.en

input_table  = "parsed4_decaps_text"
output_table = "parsed5_POS_tokenized"

target_columns = ["abstract", "specificAims"]

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEBUG    = False
_PARALLEL = True

global_limit = 0

class POS_tokenizer(object):
    
    def __init__(self):

        self.parse = lambda x:pattern.en.parse(x,chunks=False,tags=True)

        # connectors = conjunction,determiner,infinitival to,
        #              interjection,predeterminer
        # w_word = which, what, who, whose, when, where & there ...

        POS = {
            "connector"   : ["CC","IN","DT","TO","UH","PDT"],
            "cardinal"    : ["CD","LS"],
            "adjective"   : ["JJ","JJR","JJS"],
            "noun"        : ["NN","NNS","NNP","NNPS"],
            "pronoun"     : ["PRP","PRP$"],
            "adverb"      : ["RB","RBR","RBS","RP"],
            "symbol"      : ["SYM",'$',],
            "punctuation" : [".",",",":",')','('],
            "modal_verb"  : ["MD"],
            "verb"        : ["VB","VBZ","VBP","VBD","VBG","VBN"],
            "w_word"      : ["WDT","WP","WP$","WRB","EX"],
            "unknown"     : ["FW"],
        }

        self.filtered_POS = set(("connector", 
                                 "cardinal",
                                 "pronoun",
                                 "adverb",
                                 "symbol",
                                 "verb",
                                 "punctuation",
                                 "modal_verb",
                                 "w_word",))

        self.POS_map = {}
        for pos,L in POS.items():
            for y in L: self.POS_map[y]=pos
                                        
    def __call__(self,doc,force_lemma=True):
        tokens = self.parse(doc)
        doc2 = []
        for sentence in tokens.split():
            sent2 = []
            for word,tag in sentence:

                if "PHRASE_" in word:
                    sent2.append(word)
                    continue

                tag = tag.split('|')[0].split('-')[0].split("&")[0]
                
                try:
                    pos = self.POS_map[tag]
                except:
                    print "UNKNOWN POS", tag
                    pos = "unknown"

                if pos in self.filtered_POS:
                    continue

                org_word = word

                word = pattern.en.singularize(word,pos)

                if pos == "verb" or force_lemma:
                    lem = pattern.en.lemma(word,parse=False)
                    if lem is not None: word = lem

                sent2.append(word)
            doc2.append(' '.join(sent2))

        return '\n'.join(doc2)

target_function = POS_tokenizer()

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
