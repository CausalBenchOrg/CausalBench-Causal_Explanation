import os
from causalbench.modules import Run, Model


def process_run(zip_file, hp_dtype, model_cache):
    # Get run
    run: Run = Run(zip_file=zip_file)

    for result in run.results:
        # Get and cache model
        model_key = (result.model.id, result.model.version)
        if not model_key in model_cache:
            model_cache[model_key] = Model(*model_key)
        model = model_cache[model_key]

        for hp in model.hyperparameters.keys():
            hp_dtype[hp] = model.hyperparameters[hp].data

    return hp_dtype


def get_hp_dtypes(zip_dir):
    """
    Get data types for hyperparamters
    
    :param zip_dir: Description
    """
    hp_dtype = dict()
    model_cache = dict()

    # Loop over each .zip file in the specified directory
    for filename in os.listdir(zip_dir):
        if filename.endswith('.zip'):
            zip_file_path = os.path.join(zip_dir, filename)
            process_run(zip_file_path, hp_dtype, model_cache)
    
    return hp_dtype
