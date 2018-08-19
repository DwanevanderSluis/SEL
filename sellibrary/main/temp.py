import pickle
import logging
from sellibrary.locations import FileLocations
from sellibrary.util.first_model_value import FirstValueModel
import numpy as np


if __name__ == '__main__':

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    model = FirstValueModel()

    dropbox_intermediate_path = FileLocations.get_dropbox_intermediate_path()
    output_filename = dropbox_intermediate_path + 'return_first_value.pickle'
    logger.info('About to write %s', output_filename)
    with open(output_filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('file written = %s', output_filename)

    with open(output_filename, 'rb') as handle:
        model = pickle.load(handle)

    logger.info(model.predict(np.ones(shape=(3,7))))

