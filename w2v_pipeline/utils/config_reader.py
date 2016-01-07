import ConfigParser, os

_DEFAULT_INT = "config_w2v_pipeline.ini"

def load_config(f_config=_DEFAULT_INT):
    if not os.path.exists(f_config):
        print "{} not found! Need to create one!".format(f_config)
        exit(1)
        
    config = ConfigParser.ConfigParser()
    config.read(f_config)


    args = {
        "debug" : config.getboolean('PIPELINE_ARGS',"debug"),
        "target_columns" : config.get('PIPELINE_ARGS',
                                      "target_columns",
                                      "text"),
    }

    args["target_columns"] = args["target_columns"].split()
    return args
