import itertools
from utils.os_utils import mkdir
import document_scoring as ds
import utils.db_utils as db

'''
Can probably merge the two score functions into one, 
no need for a mapreduce+global

Can test speed difference in making parallel
'''


def score_from_config(global_config):

    config = global_config["score"]
    dout = config["output_data_directory"]
    mkdir(dout)

    # Run the functions that can sum over the data (eg. TF counts)
    for name in config["mapreduce_commands"]:
        
        # Load any kwargs in the config file
        kwargs = {"output_data_directory":dout}
        if name in config:
            kwargs.update(config[name])

        model = getattr(ds, name)(**kwargs)
        
        print("Starting mapreduce {}".format(model.function_name))
        map(model, db.text_iterator())
        model.save(**kwargs)
        
    
    # Run the functions that act per documnet (eg. word2vec)

    for name in config["globaldata_commands"]:

        # Load any kwargs in the config file
        kwargs = config.copy()
        if name in config:
            kwargs.update(config[name])

        # Add in the embedding configuration options
        model = getattr(ds, name)(**kwargs)

        print("Starting score model {}".format(model.method))

    
        for f_csv in db.get_parsed_filenames():
            data = {}
            for row in db.text_iterator([f_csv,]):
                data[row["_ref"]] = model(row['text'])

            model.save(data, f_csv)
        
        model.compute_reduced_representation()



if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    score_from_config(config)
