import ConfigParser, os

_DEFAULT_INT = "config_w2v_pipeline.ini"

def load_configfile(f_config):
    if not os.path.exists(f_config):
        print "{} not found! Need to create one!".format(f_config)
        exit(1)
        
    config = ConfigParser.ConfigParser()
    config.read(f_config)
    return config

def load_config(f_config=_DEFAULT_INT):

    config = load_configfile(f_config)

    args = {
        "debug" : config.getboolean('PIPELINE_ARGS',"debug"),
        "force" : config.getboolean('PIPELINE_ARGS',"force"),
        "target_columns" : config.get('PIPELINE_ARGS',
                                      "target_columns",
                                      "text"),
    }

    args["target_columns"] = args["target_columns"].split()
    return args

def load_kSVD_config(f_config=_DEFAULT_INT):
    config = load_configfile(f_config)

    try:
        FORCE = config.getboolean('kSVD',"FLAG_FORCE"),
    except:
        FORCE = False

    args = {
        "basis_size" : config.getint('kSVD',"basis_size"),
        "sparsity"   : config.getint('kSVD',"sparsity"),
        "iterations" : config.getint('kSVD',"iterations"),
        "samples"    : config.getint('kSVD',"samples"),
        "FLAG_FORCE" : FORCE,
    }

    return args    
