import os
import configobj

def load(subset=None, f_config = "config.ini"):

    # Raise Error if configfile not found
    if not os.path.exists(f_config):
        msg = "{} not found".format(f_config)
        raise IOError(msg)

    config = configobj.ConfigObj(f_config)
    
    if subset is None:
        return config

    # Load all the global keys
    output = configobj.ConfigObj()

    for key,val in config.items():
        if type(val) is not configobj.Section:
            output[key] = val

    # Add in the local subset information
    output.update(config[subset])

    return output
    


