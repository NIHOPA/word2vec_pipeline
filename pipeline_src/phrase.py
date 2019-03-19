"""
Identifies phrases from abbreivations for documents in the pipeline.
Saves the result to disk for use with other parsers.
"""

# from utils.os_utils import grab_files, mkdir

from utils.parallel_utils import jobmap
import utils.db_utils as db_utils
from utils.os_utils import grab_files, mkdir

import pandas as pd
import collections
import nlpre
import os

import logging

logger = logging.getLogger(__name__)

# NLPre is too noisy at the info level
logging.getLogger("nlpre").setLevel(logging.WARNING)

parser_parenthetical = nlpre.identify_parenthetical_phrases()


def phrases_from_config(config):
    """
    Identify parenthetical phrases in the documents as they are being
    imported to the pipeline.

    import_data_from_config() and phrases_from_config() are the entry
    points for this step of the pipeline.

    Args:
        config: a config file
    :return:
    """

    _PARALLEL = config.as_bool("_PARALLEL")
    output_dir = config["phrase_identification"]["output_data_directory"]

    target_column = config["target_column"]

    import_config = config["import_data"]
    input_data_dir = import_config["output_data_directory"]

    F_CSV = grab_files("*.csv", input_data_dir)
    ABBR = collections.Counter()

    INPUT_ITR = db_utils.CSV_database_iterator(
        F_CSV, target_column, progress_bar=True
    )

    ITR = jobmap(func_parenthetical, INPUT_ITR, _PARALLEL, col=target_column)

    for result in ITR:
        ABBR.update(result)

    logger.info("{} total abbrs found.".format(len(ABBR)))

    # Merge abbreviations that are similar
    logger.debug("Deduping abbr list.")
    df = dedupe_abbr(ABBR)
    logger.info("{} abbrs remain after deduping.".format(len(df)))

    # Output top phrase
    logger.info("Top 5 abbreviations")
    msg = "({}) {}, {}, {}"
    for k, (_, row) in enumerate(df[:5].iterrows()):
        logger.info(msg.format(k + 1, row.name, row["abbr"], row["count"]))

    mkdir(output_dir)
    f_csv = os.path.join(
        output_dir, config["phrase_identification"]["f_abbreviations"]
    )
    df.to_csv(f_csv)


def dedupe_abbr(ABBR):
    """
    Remove duplicate entries in dictionary of abbreviations

    Args:
        ABBR: a dictionary of abbreviations and corresponding phrases

    Returns:
        df: a DataFrame of sorted abbreviations
    """

    df = pd.DataFrame()
    df["phrase"] = [" ".join(x[0]) for x in ABBR.keys()]
    df["abbr"] = [x[1] for x in ABBR.keys()]
    df["count"] = ABBR.values()

    # Match phrases on lowercase and remove trailing 's'
    df["reduced_phrase"] = df.phrase.str.strip()
    df["reduced_phrase"] = df.reduced_phrase.str.lower()
    df["reduced_phrase"] = df.reduced_phrase.str.rstrip("s")

    data = []
    for phrase, dfx in df.groupby("reduced_phrase"):
        top = dfx.sort_values("count", ascending=False).iloc[0]

        item = {}
        item["count"] = dfx["count"].sum()
        item["phrase"] = top["phrase"]
        item["abbr"] = top["abbr"]
        data.append(item)

    df = pd.DataFrame(data).set_index("phrase")
    return df.sort_values("count", ascending=False)


def func_parenthetical(data, **kwargs):
    """
    Identify paranthetical phrases in the data

    Args:
        data: a text document
        kwargs: additional arguments
    Returns:
        parser_parenthetical(text): A collections.counter object with
                                    count of parenthetical phrases
    """
    text = data[kwargs["col"]]
    return parser_parenthetical(text)
