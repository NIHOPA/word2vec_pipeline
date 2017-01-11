import random
from utils.os_utils import mkdir, grab_files
import model_building as mb
import simple_config

import utils.db_utils as db_utils

def item_iterator(
        randomize_file_order=False,
        whitelist=[],
        section='parse',
):
    '''
    Iterates over the parsed corpus items and respects a given whitelist.
    '''
    

    parse_config = simple_config.load(section)
    input_data_dir = parse_config["output_data_directory"]
    F_CSV = grab_files("*.csv", input_data_dir, verbose=False)

    if whitelist:
        assert(type(whitelist)==list)

        F_CSV2 = set()
        for f_csv in F_CSV:
            for token in whitelist:
                if token in f_csv:
                    F_CSV2.add(f_csv)
        F_CSV = F_CSV2
    
    # Randomize the order of the input files each time we get here
    if randomize_file_order:
        F_CSV = random.sample(sorted(F_CSV), len(F_CSV))

    dfunc = db_utils.CSV_database_iterator
    INPUT_ITR = dfunc(F_CSV, config["target_column"],
                      progress_bar=False)

    for item in INPUT_ITR:
        yield item
    
if __name__ == "__main__":

    import simple_config
    config = simple_config.load("embedding")
    _FORCE = config.as_bool("_FORCE")

    mkdir(config["output_data_directory"])

    score_config = simple_config.load("score")
    # If there is a whitelist only keep the matching filename
    try:
        whitelist = score_config["input_file_whitelist"]
    except:
        whitelist = []
    
    ###########################################################
    # Run the functions that act globally on the data

    for name in config["embedding_commands"]:
        obj  = getattr(mb,name)

        # Load any kwargs in the config file
        kwargs = config
        if name in config:
            kwargs.update(config[name])
            
        func = obj(**kwargs)
        func.set_iterator_function(item_iterator,whitelist,section="parse")
        func.compute(config)
