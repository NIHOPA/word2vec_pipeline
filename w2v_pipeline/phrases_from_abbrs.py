import glob, sqlite3, string, os, collections, itertools
from utils.os_utils import grab_files, mkdir
import utils.db_utils

import pandas as pd
from sqlalchemy import create_engine
import pyparsing as pypar
import tqdm

from utils.parallel_utils import jobmap

global_limit  = 0
global_offset = 0

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
    doc,idx = item
    
    doc = unicode(doc)
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


def dedupe_item(item):
    global ABR

    (phrase, abbr), count = item
    p1 = ' '.join(phrase)
    match_keys = []

    for key2 in ABR.keys():
        phrase2, abbr2 = key2
        p2 = ' '.join(phrase2)

        # Only merge when abbreviations match
        if abbr != abbr2:
            continue

        # If lower cased phrases match merge them
        if p1.lower() == p2.lower():
            match_keys.append(key2)

        # If phrase without trailing 's' matches, merge
        elif p1.rstrip('s') == p2.rstrip('s'):
            match_keys.append(key2)
            
    return match_keys


def dedupe_abbr(ABR):
    data = {}
    
    ITR = jobmap(dedupe_item, tqdm.tqdm(ABR.items()), True)
    for result in ITR:

        # Only add the most common result
        max_val,max_item = 0, None
        total_counts = 0
        for item in result:
            current_val = ABR[item]
            total_counts += current_val
            if current_val > max_val:
                max_val = current_val
                max_item = item
                
        data[(' '.join(max_item[0]), max_item[1])] = total_counts

    ABR = collections.Counter(data)

    return ABR
    

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("phrase_identification")
    _PARALLEL = config.as_bool("_PARALLEL")
    _FORCE = config.as_bool("_FORCE")
    output_dir = config["output_data_directory"]

    target_columns = config["target_columns"]

    import_config = simple_config.load("import_data")
    input_data_dir = import_config["output_data_directory"]
    input_table = import_config["output_table"]
    
    F_SQL = grab_files("*.sqlite", input_data_dir)

    ABR = collections.Counter()
    P = parenthesis_nester()

    dfunc = utils.db_utils.database_iterator

    FILE_COL_ITR = itertools.product(F_SQL, target_columns)

    for f_sql,column_name in FILE_COL_ITR:

        conn = sqlite3.connect(f_sql,check_same_thread=False)

        INPUT_ITR = dfunc(column_name,
                          input_table,
                          conn,
                          limit=global_limit,
                          offset=global_offset,
                          progress_bar=True,
        )

        ITR = jobmap(evaluate_document, INPUT_ITR, _PARALLEL)

        for result in ITR:
            ABR.update(result)

        msg = "Completed {} {}. {} total abbrs found."
        print msg.format(f_sql,column_name,len(ABR))

    # Merge abbreviations that are similar
    print "Deduping list"    
    ABR = dedupe_abbr(ABR)
    print "{} abbrs remain after deduping".format(len(ABR))


    # Convert abbrs to a list
    data_insert = [(phrase,abbr,count) 
                   for (phrase,abbr),count in ABR.most_common()]

    # Convert the list to a dataframe for insert
    df = pd.DataFrame(data_insert, 
                      columns=("phrase","abbr","count"))

    mkdir(output_dir)
    f_sql = os.path.join(output_dir, config["f_abbreviations"])
    engine = create_engine('sqlite:///'+f_sql)

    # Save the abbrs to a table
    df.to_sql(config["output_table"],
              engine,
              if_exists='replace')
