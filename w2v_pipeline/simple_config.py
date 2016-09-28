import os
from configobj import ConfigObj, Section
from validate import Validator, VdtTypeError

def validate_errors(errors, name_stack=None):
      
    # No errors found if errors object is True
    if errors == True:
        return False

    is_error = False

    if name_stack is None:
        name_stack = []
    
    for key,item in errors.items():
        stack = name_stack + [key,]
        
        # If item is True, then object validated
        if item == True:
            continue

        # If item is a known typeError report it
        if type(item) == VdtTypeError:           
            print "ConfigError: {} {}".format('/'.join(stack), item)
            is_error = True

        # If item is a dict, recurse into the config stack
        elif type(item) == dict:
            is_error += validate_errors(item, stack)

    return is_error


def load(subset=None, f_config = "config.ini"):

    # Raise Error if configfile not found
    if not os.path.exists(f_config):
        msg = "{} not found".format(f_config)
        raise IOError(msg)

    # The config spec should be located in the same directory as simple_config
    local_path = os.path.dirname(os.path.realpath(__file__))
    f_config_spec = os.path.join(local_path, 'config_validation.ini')

    print f_config_spec

    config = ConfigObj(f_config, configspec=f_config_spec)

    errors = config.validate(Validator(), preserve_errors=True)
    
    if validate_errors(errors):
        msg = "{} failed to parse.".format(f_config)
        raise SyntaxError(msg)

    if subset is None:
        return config

    # Load all the global keys
    output = ConfigObj()

    for key,val in config.items():
        if type(val) is not Section:
            output[key] = val

    # Add in the local subset information
    output.update(config[subset])

    return output
    


