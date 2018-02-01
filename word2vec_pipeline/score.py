import itertools
from utils.os_utils import mkdir
import document_scoring as ds
import utils.db_utils as db

'''
Remove dependence on config from document scores
Can probably merge the two score functions into one, 
no need for a mapreduce+global

Can test speed difference in making parallel
'''

def _load_model(name, config):
    # Load any kwargs in the config file
    kwargs = config.copy()
    
    if name in config:
        kwargs.update(config[name])

    return getattr(ds, name)(**kwargs), kwargs


def score_from_config(global_config):

    config = global_config["score"] 
    mkdir(config["output_data_directory"])

    # Run the functions that can sum over the data (eg. TF counts)
    for name in config["count_commands"]:
 
        model, kwargs = _load_model(name, config)
        print("Starting mapreduce {}".format(model.function_name))
        map(model, db.text_iterator())
        model.save(**kwargs)
        
    
    # Run the functions that act per documnet (eg. word2vec)
    for name in config["score_commands"]:

        model, kwargs = _load_model(name, config)
        print("Starting score model {}".format(model.method))
    
        for f_csv in db.get_parsed_filenames():
            data = {}
            for row in db.text_iterator([f_csv,]):
                data[row["_ref"]] = model(row['text'])

            model.save(data, f_csv)

        if kwargs["compute_reduced_representation"]:
            nc = kwargs['reduced_representation']['n_components']
            model.compute_reduced_representation(n_components=nc)



if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    score_from_config(config)
