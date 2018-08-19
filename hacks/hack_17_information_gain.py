from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
import numpy as np

import logging
import math
import numpy as np
import pandas as pd


class EntropyCalculator:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def __init__(self):
        pass
    #


    # from https://stackoverflow.com/questions/47241525/python-information-gain-implementation
    def information_gain(self, X, y):

        def _entropy(labels):
            labels = labels.astype(int)
            counts = np.bincount(labels)
            return entropy(counts, base=None)

        def _ig(x, y):
            # indices where x is set/not set
            x_set = np.nonzero(x)[0]
            x_not_set = np.delete(np.arange(x.shape[0]), x_set)
            h_x_set = _entropy(y[x_set])
            h_x_not_set = _entropy(y[x_not_set])
            entropy_for_feature = (((len(x_set) / f_size) * h_x_set)
                                   + ((len(x_not_set) / f_size) * h_x_not_set))

            self.logger.info('entropy_for_feature = %f', entropy_for_feature)

            return entropy_full - entropy_for_feature

        entropy_full = _entropy(y)
        self.logger.info('entropy_full = %f', entropy_full)
        f_size = float(X.shape[0])
        scores = np.array([_ig(x, y) for x in X.T])
        return scores


    def unit_test_entropy_3(self):

        X, y = self.get_data()

        self.logger.info('____________________')
        self.logger.info('X Shape = %s', X.shape)
        self.logger.info('y Shape = %s', y.shape)


        ig = self.information_gain(X, y)

        self.logger.info(' %s', ig)




    def get_data(self):
        # taken from https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjdhNGPuancAhUDZ8AKHTFjCLsQFggpMAA&url=https%3A%2F%2Fhomes.cs.washington.edu%2F~shapiro%2FEE596%2Fnotes%2FInfoGain.pdf&usg=AOvVaw2dVa94ThGfcbTtCp-FdwJc
        # slide 7, circles == 1, crosses == 0

        # y is 30 values, the first 14 are 0's, the next 16 are 1s
        y = np.ones(30)
        for i in range(14):
            y[i] = 0

        X = np.zeros(30).reshape([30, -1])
        # X is 30 values, 4 of its 1 co-inside with y's 0's, the other 12 of its ones, co-inside with y's 1s
        for i in range(4):
            X[i, 0] = 1
            # X[i, 1] = 1

        for i in range(18, 30):
            X[i, 0] = 1
            # X[i, 1] = 1

        return X, y


    def unit_test_entropy_2(self):
        X, y = self.get_data()

        self.logger.info('____________________')
        self.logger.info('X Shape = %s', X.shape)
        self.logger.info('y Shape = %s', y.shape)

        self.logger.info('X %s sum %d', X, np.sum(X))
        self.logger.info('y %s sum %s', y, np.sum(y))
        self.logger.info('____________________')


        s = mutual_info_classif(X, y, discrete_features=False)

        self.logger.info('mutual_class_info %s', s)




app = EntropyCalculator()
#app.unit_test_entropy_2()
app.unit_test_entropy_3()