from sel.binary_classifier import BinaryClassifierTrainer

import numpy as np
import logging

def get_rand_x_y(row_count=1000):
    X = np.random.rand(row_count, 23)
    y = np.ones(row_count)
    y[0:int(row_count/2)] = 0
    return X, y

def unit_test_can_train_with_random_data():
    bct = BinaryClassifierTrainer()
    X, y = get_rand_x_y()
    logging.info('sum_x', np.sum(np.isreal(X)),X.shape)
    logging.info('sum_y', np.sum(np.isreal(y)),y.shape)
    bct.train_model(X, y)


def pass_values_into_model_raw():
    features = [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3]
    n2 = np.array([features])
    print(n2.shape)
    bct = BinaryClassifierTrainer()
    bct.predict_raw(n2)

def pass_values_into_model():
    features = [1,2,3,4,5,6,7,8,9,0,1,2,3,4,None,6,7,8,9,0,1,2,3]
    bct = BinaryClassifierTrainer()
    bct.predict(features)


if __name__ == "__main__":
    unit_test_can_train_with_random_data()

    #pass_values_into_model()

    #pass_values_into_model_raw()

