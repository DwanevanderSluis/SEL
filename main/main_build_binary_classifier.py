import logging

import numpy as np

from sel.binary_classifier import BinaryClassifierTrainer

# set up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

def find_hyper_prameters():
    bct = BinaryClassifierTrainer()
    X, y, entity_id_array = bct.load_light_parameter_data_fmt_03()
    bct.find_hyper_parameters(X, y)

def convert_to_decimal(data):
    filtered_list = []
    for d in data:
        if d is not None:
            if d:
                filtered_list.append(1.0)
            else:
                filtered_list.append(0.0)
        else:
            filtered_list.append(0.0)
    n = np.array([filtered_list])
    return n

def train_model():
    bct = BinaryClassifierTrainer()
    X, y, entity_id_array = bct.load_light_parameter_data_fmt_03()
    bct.train_model(X, y, entity_id_array, 2)

def train_and_save_model():
    bct = BinaryClassifierTrainer()
    X, y, entity_id_array = bct.load_light_parameter_data_fmt_03()
    bct.train_model(X, y, entity_id_array, 2)
    bct.save_model()

def tune_model():
    bct = BinaryClassifierTrainer()
    X, y, entity_id_array = bct.load_light_parameter_data_fmt_03()
    bct.tune_model(X, y, entity_id_array)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    #train_model()
    # find_hyper_prameters()
    # tune_model()
    train_and_save_model()
    # pass_values_into_model()
