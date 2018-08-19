import numpy as np


a = np.ones(shape=[1,4])
b = np.ones(shape=[1,3])

c = np.concatenate([a,b],axis=1)

print(c)
print(list(c[0]))


