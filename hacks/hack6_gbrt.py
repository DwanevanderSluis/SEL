#not needed using different impl -> pip install gbdt
#or https://stackoverflow.com/questions/47627478/how-to-obtain-whole-decision-process-of-a-sklearn-gbdt

import logging

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sel.gbrt import GBRT


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def ex1():

    clf = GradientBoostingClassifier(n_estimators=4)
    iris = load_iris()

    clf = clf.fit(iris.data, iris.target)

    for i in range(4):
        print(clf.estimators_[i, 0].tree_)

    clf.predict( np.array([0,1,2,3]).reshape(1, -1) )

    print(clf.apply( np.array([0,1,2,3]).reshape(1, -1) ) )




def test_2():
    rt = GBRT()
    iris = load_iris()
    logger.info('iris.data')
    logger.info(iris.data)
    logger.info('iris.target')
    logger.info(iris.target)
    rt.train(iris.data, iris.target)


def test_3_train():
    rt = GBRT()
    data = np.random.rand(2,36)
    target = np.array([1,0])
    logger.info('iris.data')
    logger.info(data)
    logger.info('iris.target')
    logger.info(target)

    rt.train(data, target)
    rt.save_model()