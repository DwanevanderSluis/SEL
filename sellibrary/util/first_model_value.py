import pickle
import logging
from sellibrary.locations import FileLocations
import numpy as np


# when asked to predict, just returns the first value
class FirstValueModel():

    def __init__(self):
        pass

    def predict(self, x):
        return x[:,0]



