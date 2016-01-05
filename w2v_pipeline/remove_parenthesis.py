import pyparsing as pypar
from utils.pipeline import text_pipeline
from utils.os_utils import grab_files

input_table = "parsed1_abbr_tokens"
output_table = "parsed2_no_paren"
target_columns = ["abstract", "specificAims"]

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEBUG    = False
_PARALLEL = True

global_limit = 0

class parenthesis_nester(object):
    def __init__(self):
        nest = pypar.nestedExpr
        g = pypar.Forward()
        nestedParens   = nest('(', ')')
        nestedBrackets = nest('[', ']')
        nestedCurlies  = nest('{', '}')
        nest_grammar = nestedParens|nestedBrackets|nestedCurlies
        
        parens = "(){}[]"
        letters = ''.join([x for x in pypar.printables
                    if x not in parens])
        word = pypar.Word(letters)

        g = pypar.OneOrMore(word | nest_grammar)
        self.grammar = g

                                        
    def __call__(self,line):
        try:
            tokens = self.grammar.parseString(line)
        except pypar.ParseException:
            # On fail simply remove all parens
            line = line.replace('(','')
            line = line.replace(')','')
            line = line.replace('[','')
            line = line.replace(']','')
            line = line.replace('{','')
            line = line.replace('}','')
            tokens = line.split()

        # Remove nested parens
        tokens = [x for x in tokens if type(x) in [str,unicode]]
        text = ' '.join(tokens)
        return text

target_function = parenthesis_nester()

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
