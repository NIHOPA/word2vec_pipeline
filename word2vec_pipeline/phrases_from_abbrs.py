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
    doc = row[target_column]

    doc = unicode(doc)
    doc = doc.replace('-', ' ')
    doc = doc.replace("'", '')
    doc = doc.replace('"', '')
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
        max_val, max_item = 0, None
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
    config = simple_config.load()
    _PARALLEL = config.as_bool("_PARALLEL")
    output_dir = config["phrase_identification"]["output_data_directory"]

    target_column = config["target_column"]

    import_config = config["import_data"]
    input_data_dir = import_config["output_data_directory"]

    F_CSV = grab_files("*.csv", input_data_dir)

    ABR = collections.Counter()
    P = parenthesis_nester()

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, target_column, progress_bar=True)
    ITR = jobmap(evaluate_document, INPUT_ITR, _PARALLEL, col=target_column)

    for result in ITR:
        ABR.update(result)

    msg = "\n{} total abbrs found."
    print(msg.format(len(ABR)))

    # Merge abbreviations that are similar
    print("Deduping abbr list.")
    ABR = dedupe_abbr(ABR)
    print("{} abbrs remain after deduping".format(len(ABR)))

    # Convert abbrs to a list
    data_insert = [(phrase, abbr, count)
                   for (phrase, abbr), count in ABR.most_common()]

    # Convert the list to a dataframe and sort
    df = pd.DataFrame(data_insert,
                      columns=("phrase", "abbr", "count"))
    df = df.sort_values(
        ["count", "phrase"], ascending=False).set_index("phrase")

    # Output top phrase
    print("Top 5 abbreviations")
    print(df[:5])

    mkdir(output_dir)
    f_csv = os.path.join(output_dir,
                         config["phrase_identification"]["f_abbreviations"])
    df.to_csv(f_csv)
