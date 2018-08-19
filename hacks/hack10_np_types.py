import numpy as np




a = [1,None]


b = np.array(a)

x = b.dtype.kind in {'U', 'S'}

print(x)

print( np.argwhere(np.isnan(b)))

print(np.issctype(np.array([1])))