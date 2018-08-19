
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# mu, sigma = 100, 15
# x = mu + sigma*np.random.randn(10000)
x = [0,0,0,2,5,5,5,5,5,7,7,8,7,8,13,13,13,16,16,6,16,5,16,6,16,15,15,9,9,13,16,9,13,16,13,12,6]


# the histogram of the data
n, bins, patches = plt.hist(x, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()



