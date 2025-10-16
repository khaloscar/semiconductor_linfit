import numpy as np
from numpy.polynomial import Polynomial as Pln
from matplotlib import pyplot as plt
import pandas as pd

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
        y_idx = np.argmin(np.abs(self.y-I_high))
        fit_y_idx = np.argmin(np.abs(self.fit_y-I_high))
        deltax = self.x[fit_y_idx] - self.x[y_idx]
        self.Rs = deltax/I_high

        # y_0 axis intercept
        # ideality constant
        # Rs thats it ? 


def main():
    if False:
        names = ["diodeA_Light","diodeA_Dark","diodeB_Light","diodeB_Dark"]
    # Read the file, skip the header lines
        filename = "diodeB_Light"
        name = "Diode B Light"
        df = pd.read_csv(
            f'Data/{filename}.txt',
            skiprows=3,          # skips the first 4 header lines
            delim_whitespace=True,  # columns separated by spaces
            decimal=','          # interpret commas as decimal points
        )

    

        # Extract columns into numpy arrays
        x = df.iloc[:, 0].values  # Voltage (V)
        y = df.iloc[:, 1].values  # Current (A)

        # --- 1. Normal linear plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o-', markersize=4)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"{name} IV Characteristic")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename}_linear.png", dpi=600, bbox_inches='tight')
        #plt.show()

        # --- 2. Semilog (log y-axis) plot ---
        plt.figure(figsize=(8, 5))
        plt.semilogy(x, abs(y), 'o-', markersize=4)
        plt.xlabel("Voltage (V)")
        plt.ylabel("ln(I) (A)")
        plt.title(f"{name} IV Characteristic")
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(f"{filename}_semilog.png", dpi=600, bbox_inches='tight')
        #plt.show()
        print(f'yay')

    if False:
        names = ["diodeA_Light","diodeA_Dark","diodeB_Light","diodeB_Dark"]
        # Read the file, skip the header lines
        filename = "MOSCAP_CV_Light"
        name = "MOSCAP Light"
        df = pd.read_csv(
            f'Data/{filename}.txt',
            skiprows=3,          # skips the first 4 header lines
            delim_whitespace=True,  # columns separated by spaces
            decimal=','          # interpret commas as decimal points
        )

    

        # Extract columns into numpy arrays
        x = df.iloc[:, 0].values  # Voltage (V)
        y = df.iloc[:, 1].values  # Current (A)

        # --- 1. Normal linear plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o-', markersize=4)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Capacitance (F)")
        plt.title(f"{name} CV Characteristic")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename}_linear.png", dpi=600, bbox_inches='tight')
        #plt.show()

        # --- 2. Semilog (log y-axis) plot ---
        plt.figure(figsize=(8, 5))
        plt.semilogy(x, abs(y), 'o-', markersize=4)
        plt.xlabel("Voltage (V)")
        plt.ylabel("ln(C) (F)")
        plt.title(f"{name} CV Characteristic")
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(f"{filename}_semilog.png", dpi=600, bbox_inches='tight')
        #plt.show()
        print(f'yay')
    # read data and transform to log
    # create linear fit,
    # extract values 

    
if __name__ == "__main__":
    main()