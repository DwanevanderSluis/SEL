import os
import logging
import pickle
import numpy as np


class SelFeatToSent:
    # class variables - shared across all instances
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)


    def __init__(self, model_filename ):
        self._load_model(model_filename)

    def _load_model(self, heavy_filename):
        if os.path.isfile(heavy_filename):
                        self.logger.info('loading from %s', heavy_filename)
                        with open(heavy_filename, 'rb') as handle:
                            self._model = pickle.load(handle)
                        self.logger.info('loaded')

    # the signature of this routine must be consistent across all models
    def get_salient(self,list_of_features):
        data = np.array(list_of_features)
        data = data.reshape(1,-1)
        try:
            saliency = self._model.predict(data)[0]
        except ValueError as e:
            self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, err=%s',
                                list_of_features, e)
        except IndexError as e:
            self.logger.warning('could not calc gbrt, returning 0. entity_id=%d, err=%s',
                                list_of_features, e)

        return saliency