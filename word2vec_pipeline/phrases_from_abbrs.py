import string
import os
import collections
from utils.os_utils import grab_files, mkdir
import utils.db_utils as db_utils

import pandas as pd
import pyparsing as pypar
import tqdm

from utils.parallel_utils import jobmap

global_limit = 0
global_offset = 0


class parenthesis_nester(object):

    def __init__(self):
        nest = pypar.nestedExpr
        g = pypar.Forward()
        nestedParens = nest('(', ')')
        nestedBrackets = nest('[', ']')
        nestedCurlies = nest('{', '}')
        nest_grammar = nestedParens | nestedBrackets | nestedCurlies

        parens = "(){}[]"
        letters = ''.join([x for x in pypar.printables
                           if x not in parens])
        word = pypar.Word(letters)

        g = pypar.OneOrMore(word | nest_grammar)
        self.grammar = g

    def __call__(self, line):
        try:
            tokens = self.grammar.parseString(line)
        except:
            return []
        return tokens


def is_valid_abbr(item):
    if isinstance(item, unicode):
        return False
    if len(item) != 1:
        return False

    word = item[0]

    # Break if we are doubly nested
    if not isinstance(word, unicode):
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
    if len(caps) < 2:
        return False

    # This may fail if used too early in doc or if nested parens
    # this shouldn't be a match so it's OK!

    try:
        subtokens = tokens[k - len(caps):k]
        subtoken_let = [let.upper()[0] for let in subtokens]
    except:
        return False

    if subtoken_let != caps:
        return False

    return tuple(subtokens)


def evaluate_document(row, col):
    doc = row[col]

    doc = unicode(doc)
    doc = doc.replace('-', ' ')
    doc = doc.replace("'", '')
    doc = doc.replace('"', '')

    P = parenthesis_nester()
    tokens = P(doc)

    results = collections.Counter()

    for k, item in enumerate(tokens):
        word = is_valid_abbr(item)
        if word:
            subtokens = check_matching(word, k, tokens)
            if subtokens:
                results[(tuple(subtokens), word)] += 1

    # if results:
    #    print "Found {} abbrs in doc idx {}".format(len(results),idx)

    return results


def dedupe_abbr(ABR):

    df = pd.DataFrame()
    df['phrase'] = [' '.join(x[0]) for x in ABR.keys()]
    df['abbr'] = [x[1] for x in ABR.keys()]
    df['count'] = ABR.values()

    # Match phrases on lowercase and remove trailing 's'
    df['reduced_phrase'] = df.phrase.str.strip()
    df['reduced_phrase'] = df.reduced_phrase.str.lower()
    df['reduced_phrase'] = df.reduced_phrase.str.rstrip('s')

    data = []
    for phrase, dfx in df.groupby('reduced_phrase'):
        top = dfx.sort_values("count",ascending=False).iloc[0]
        
        item = {}
        item["count"] = dfx["count"].sum()
        item["phrase"] = top["phrase"]
        item["abbr"] = top["abbr"]
        data.append(item)

    df = pd.DataFrame(data).set_index("phrase")
    return df.sort_values("count",ascending=False)


def phrases_from_config(config):

    _PARALLEL = config.as_bool("_PARALLEL")
    output_dir = config["phrase_identification"]["output_data_directory"]

    target_column = config["target_column"]

    import_config = config["import_data"]
    input_data_dir = import_config["output_data_directory"]

    F_CSV = grab_files("*.csv", input_data_dir)

    ABR = collections.Counter()

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, target_column, progress_bar=True)
    ITR = jobmap(evaluate_document, INPUT_ITR, _PARALLEL, col=target_column)

    for result in ITR:
        ABR.update(result)

    msg = "\n{} total abbrs found."
    print(msg.format(len(ABR)))

    # Merge abbreviations that are similar
    print("Deduping abbr list.")
    df = dedupe_abbr(ABR)
    print("{} abbrs remain after deduping".format(len(df)))

    # Output top phrase
    print("Top 5 abbreviations")
    print(df[:5])

    mkdir(output_dir)
    f_csv = os.path.join(output_dir,
                         config["phrase_identification"]["f_abbreviations"])
    df.to_csv(f_csv)


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    phrases_from_config(config)
