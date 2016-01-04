import glob, sqlite3, string, os, collections
from utils.os_utils import grab_files, mkdir
import utils.db_utils

import pandas as pd
from sqlalchemy import create_engine
import pyparsing as pypar

input_table    = "original"
output_table   = "abbreviations"
target_columns = ["abstract", "specificAims"]

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEFAULT_EXPORT_DIRECTORY = "collated"
_DEBUG = False

global_limit  = 0
global_offset = 0

output_filename = "abbreviations.sqlite"

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
        except:
            return []
        return tokens


def is_valid_abbr(item):
    if type(item) == unicode:
        return False
    if len(item) != 1:
        return False
    
    word = item[0]

    # Break if we are doubly nested
    if type(word) != unicode:
        return False


    # Check if there are any capital letters
    if word.lower() == word:
        return False

    return word

def check_matching(word, k, tokens):
    # Identify the capital letters
    caps = [let for let in word if 
            let in string.ascii_uppercase.upper()]

    # Don't try to match with only a single letter (to noisy!)
    if len(caps)<2:
        return False

    # This may fail if used too early in doc or if nested parens
    # this shouldn't be a match so it's OK!

    try:
        subtokens = tokens[k-len(caps):k]
        subtoken_let = [let.upper()[0] for let in subtokens]
    except:
        return False

    if subtoken_let != caps:
        return False

    return tuple(subtokens)

def evaluate_document(item):
    idx,doc = item
    doc = doc.replace('-',' ')
    doc = doc.replace("'",'')
    doc = doc.replace('"','')
    tokens = P(doc)

    results = collections.Counter()

    for k,item in enumerate(tokens):
        word = is_valid_abbr(item)
        if word:
            subtokens = check_matching(word,k,tokens)
            if subtokens:
                results[(tuple(subtokens),word)] += 1
    
    #if results:
    #    print "Found {} abbrs in doc idx {}".format(len(results),idx)

    return results

def dedupe_abbr(ABR):
    data = {}

    for key1,c1 in ABR.most_common():
        phrase,abbr1 = key1
        p1 = ' '.join(phrase)
        FLAG_continue = False
        
        for key2,c2 in data.items():
            p2,abbr2  = key2
            save_key2 = (p2,abbr2)

            # Only merge when abbreviations match
            if abbr1 != abbr2:
                continue
        
            # If lower cased phrases match merge them
            if p1.lower() == p2.lower():
                data[save_key2] = c1+c2
                FLAG_continue = True
                print "Merging case '{}', '{}'".format(p1, p2)
                break

            # If phrase without trailing 's' matches, merge
            if p1.rstrip('s') == p2.rstrip('s'):
                data[save_key2] = c1+c2
                FLAG_continue = True
                print "Merging plural '{}', '{}'".format(p1, p2)
                break

        if FLAG_continue: continue

        save_key1 = (p1,abbr1)
        data[save_key1] = c1

    ABR = collections.Counter(data)
    return ABR
    

if __name__ == "__main__":
    F_SQL = grab_files("*.sqlite", _DEFAULT_IMPORT_DIRECTORY)

    ABR = collections.Counter()
    P = parenthesis_nester()
    dfunc = utils.db_utils.database_iterator

    import itertools
    import multiprocessing
    MP = multiprocessing.Pool()

    FILE_COL_ITR = itertools.product(F_SQL, target_columns)

    for f_sql,column_name in FILE_COL_ITR:

        conn = sqlite3.connect(f_sql,check_same_thread=False)

        INPUT_ITR = dfunc(column_name,
                          input_table,
                          conn,
                          limit=global_limit,
                          offset=global_offset)

        ITR = itertools.imap(evaluate_document, INPUT_ITR)

        if not _DEBUG:
            ITR = MP.imap(evaluate_document, INPUT_ITR)

        for result in ITR:
            ABR.update(result)

        msg = "Completed {} {}. {} total abbrs found."
        print msg.format(f_sql,column_name,len(ABR))

    # Merge abbreviations that are similar
    print "Deduping list"
    ABR = dedupe_abbr(ABR)

    # Convert abbrs to a list
    data_insert = [(phrase,abbr,count) 
                   for (phrase,abbr),count in ABR.most_common()]

    # Convert the list to a dataframe for insert
    df = pd.DataFrame(data_insert, 
                      columns=("phrase","abbr","count"))

    
    mkdir(_DEFAULT_EXPORT_DIRECTORY)
    
    f_sql  = os.path.join(_DEFAULT_EXPORT_DIRECTORY, output_filename)
    engine = create_engine('sqlite:///'+f_sql)

    # Save the abbrs to a table
    df.to_sql(output_table,
              engine,
              if_exists='replace')
