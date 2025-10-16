import numpy as np
from numpy.polynomial import Polynomial as Pln
from matplotlib import pyplot as plt
import pandas as pd

x = np.array([i for i in range(0,10)])
y = 2*x  + 1

series = Pln.fit(x,y, deg=1, domain=None)
y_10 = series(99/2)
print(y_10)

a = ['3',5]
print(type(a))