import os
import logging
import pickle
import numpy as np


class SimpleGBRT:
    # class variables - shared across all instances
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)


    def __init__(self, model_filename ):
        self._model = None
        self._load_model(model_filename)
        self._model_filename = model_filename

    def _load_model(self, filename):
        if os.path.isfile(filename):
            self.logger.info('loading from %s', filename)
            with open(filename, 'rb') as handle:
                self._model = pickle.load(handle)
            self.logger.info('loaded')
        else:
            self.logger.warning('file not found, not loading: %s', filename)


    # the signature of this routine must be consistent across all models
    def get_salient(self,list_of_features):
        data = np.array(list_of_features)
        data = data.reshape(1,-1)
        if self._model is not None:
            saliency = self._model.predict(data)[0]
        else:
            self.logger.warning('model is Non, returning 0:%s', self._model_filename)
            saliency = 0.0
        return saliency

    def get_model(self):
        return self._model

    # the signature of this routine must be consistent across all models
    def get_salient_from_numpy_matrix(self,numpy_matrix):
        if self._model is not None:
            saliency = self._model.predict(numpy_matrix)
        else:
            self.logger.warning('model is Non, returning 0:%s', self._model_filename)
            saliency = np.zeros(shape=[numpy_matrix.shape[0],1])
        return saliency