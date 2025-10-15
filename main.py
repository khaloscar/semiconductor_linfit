import numpy as np
from numpy.polynomial import Polynomial as Pln
from matplotlib import pyplot as plt

class Data:
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplot()

    def fit(self, deg=1, domain=None):
        self.domain = domain
        self.series = Pln.fit(self.x,self.y, domain=domain, deg=deg)
        if not self.domain:
            self.fit_y = self.series(domain)
        else:
            self.fit_y = self.series(self.x)

    def data_plot(self):
        self.ax.plot(self.x, self.y, 'o', label='Data')

    def fit_plot(self):
        if not self.domain:
            self.ax.plot(self.domain, self.fit_y, label='Fit')
        else:
            self.ax.plot(self.x, self.fit_y, label='Fit')

    def get_coef(self):
        self.coef = self.series.convert().coef
        self.y_0 = self.coef[0]
        self.slope = self.coef[1]

    def get_params(self, V_ideal_forward, I_high):
        self.get_coef()
        q = 1 # = e elementary charge
        T = 300 # 300 K
        k = 8.617333262E-5 # eV/k
        self.ideality_constant = q /(k*T*self.slope)*V_ideal_forward

        # Just need to extract the series Resistanc, finish this code
        yy_idx = np.argmin(np.abs(yy-I_high))
        y_idx = np.argmin(np.abs(y-I_high))
        deltax = x[y_idx] - x[yy_idx]


def main():
    x = np.array([i for i in range(0,40)])
    yy = (x+0.01)**2
    y = 3*x

    value = 100
    yy_idx = np.argmin(np.abs(yy-value))
    y_idx = np.argmin(np.abs(y-value))
    deltax = x[y_idx] - x[yy_idx]
    print(deltax)


        
if __name__ == "__main__":
    main()