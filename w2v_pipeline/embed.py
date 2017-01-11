import random
from utils.os_utils import mkdir, grab_files
import model_building as mb
import simple_config
from utils.db_utils import item_iterator

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
        func.set_iterator_function(item_iterator,
                                   config,
                                   whitelist,
                                   section="parse")
        func.compute(config)
