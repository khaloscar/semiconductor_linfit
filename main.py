import numpy as np
from numpy.polynomial import Polynomial as Pln
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter
import os
import shutil
from pathlib import Path

class Data:
    
    def __init__(self,x,y, filename, name):
        self.x = x
        self.y = y
        self.filename = filename
        self.name = name
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=150)

    def fit(self, deg=1, domain=None):
        self.domain = domain
        # fitting has to occur on a domain
        if type(domain) is list:
            indices = np.where((self.x >= self.domain[0]) & (self.x <= self.domain[-1]))
            x_sub = self.x[indices]
            y_sub = self.y[indices]
        else:
            x_sub = self.x
            y_sub = self.y

        # Get the series solution
        y_sub = np.log(np.abs(y_sub))
        self.series = Pln.fit(x_sub, y_sub, deg=deg)
        # calculate the linear fit y_data extended over the whole thing
        indices = self.x >= 0
        self.fit_y = self.series(self.x[indices])
        self.fit_y = np.exp(self.fit_y)
    
    def data_plot(self):
        self.ax.plot(self.x, np.abs(self.y), '-o', color='red', label="Data plot")

    def fit_plot(self):
        indices = self.x >= 0
        self.ax.plot(self.x[indices], self.fit_y, color='blue', label='Fit plot')

    def draw_plot(self):
        self.data_plot()
        self.fit_plot()
        self.ax.set_yscale("log")
        self.ax.set_xscale("linear")

                # 3) force proper log ticks + grid (for base-10)
        self.ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        self.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=100))
        self.ax.yaxis.set_minor_formatter(NullFormatter())

        self.ax.grid(which="major", axis='y', linestyle="--", alpha=0.7)
        self.ax.grid(which="minor", axis='y', linestyle=":",  alpha=0.5)

        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_xlim(xmin=0, xmax=max(self.x))
        self.ax.grid()
        self.ax.set_title(self.name)

        self.fig.tight_layout()
        self.ax.legend()
        self.ax.legend(loc='upper left')

    def get_coef(self):
        self.coef = self.series.convert().coef
        self.y_0 = np.exp(self.coef[0])
        self.slope = self.coef[1]
    
    def  get_params(self, I_high):
        self.get_coef()
        q = 1 # = e elementary charge
        T = 300 # 300 K
        k = 8.617333262E-5 # eV/k
        self.ideality_constant = q /(k*T*self.slope)

        # Just need to extract the series Resistanc, finish this code
        y_idx = np.argmin(np.abs(self.y-I_high))
        fit_y_idx = np.argmin(np.abs(self.fit_y-I_high))
        deltax =  self.x[y_idx] - self.x[fit_y_idx]
        self.Rs = deltax/np.exp(I_high)

def copy_figures_to_folder():
    # === SETTINGS ===
    main_folder = Path(r"data")  # <-- change this
    output_folder = main_folder / "All figures"         # <-- your destination folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    out_resolved = output_folder.resolve()

    # === MAIN LOOP ===
    count = 0
    for root, dirs, files in os.walk(main_folder):
        # Prevent descending into the output folder
        dirs[:] = [d for d in dirs if (Path(root) / d).resolve() != out_resolved]

        # Process images in this directory
        for fname in files:
            if Path(fname).suffix.lower() in image_extensions:
                src = Path(root) / fname

                # Skip if the source is already in the output folder (extra safety)
                if src.resolve().parent == out_resolved:
                    continue

                dest = output_folder / fname

                # Overwrite duplicates by filename (no counters)
                shutil.copy2(src, dest)
                count += 1

    print(f"✅ Copied {count} images to '{output_folder}'")

def plot_data():

    if True:
        diodes = ['DiodeA', 'DiodeA', 'DiodeB', 'DiodeB']
        filenames = ["diodeA_Light","diodeA_Dark","diodeB_Light","diodeB_Dark"]
        names = ['Diode A Light', 'Diode A Dark', 'Diode B Light', 'Diode B Dark']
        # Read the file, skip the header lines
        for i, filename in enumerate(filenames):
            print(f'Reading {filename}...')
            df = pd.read_csv(
                f'Data/{filename}.txt',
                skiprows=3,          # skips the first 4 header lines
                sep=r'\s+',           # columns separated by spaces
                decimal=','          # interpret commas as decimal points
            )

            name = names[i]
            folder = diodes[i]
            # Create folder if it doesn't exist
            if not os.path.exists(f"Data/{folder}"):
                os.makedirs(f"Data/{folder}")

            # Extract columns into numpy arrays
            x = df.iloc[:, 0].values  # Voltage (V)
            y = df.iloc[:, 1].values  # Current (A)

            # --- 1. Normal linear plot ---
            plt.figure(figsize=(6, 6))
            plt.plot(x, y, 'o-', markersize=4)
            plt.xlabel("Voltage (V)")
            plt.ylabel("Current (A)")
            plt.title(f"{name} IV Characteristic")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"Data/{folder}/{filename}_linear.png", dpi=150, bbox_inches='tight')
            plt.close()
            #plt.show()

            # --- 2. Semilog (log y-axis) plot ---
            plt.figure(figsize=(6, 6))
            plt.semilogy(x, abs(y), 'o-', markersize=4)
            plt.xlabel("Voltage (V)")
            plt.ylabel("Current (A)")
            plt.title(f"{name} IV Characteristic")
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            plt.savefig(f"Data/{folder}/{filename}_semilog.png", dpi=150, bbox_inches='tight')
            #plt.show()
    print(f'yay')


    if True:
        filenames = ["MOSCAP_CV_Light", "MOSCAP_CV_Dark"]
        names = ['MOSCAP Light', 'MOSCAP Dark']
        folder = 'MOSCAP'
        if not os.path.exists(f"Data/{folder}"):
            os.makedirs(f"Data/{folder}")
        # Read the file, skip the header lines
        for i, filename in enumerate(filenames):
            print(f'Reading {filename}...')
            name = names[i]
            df = pd.read_csv(
                f'Data/{filename}.txt',
                skiprows=3,          # skips the first 4 header lines
                sep=r'\s+',  # columns separated by spaces
                decimal=','          # interpret commas as decimal points
            )



            # Extract columns into numpy arrays
            x = df.iloc[:, 0].values  # Voltage (V)
            y = df.iloc[:, 1].values  # Current (A)

            # --- 1. Normal linear plot ---
            plt.figure(figsize=(6, 6))
            plt.plot(x, y, '-', markersize=4)
            plt.xlabel("Voltage (V)")
            plt.ylabel("Capacitance (F)")
            plt.title(f"{name} CV Characteristic")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"Data/{folder}/{filename}_linear.png", dpi=150, bbox_inches='tight')
            #plt.show()
            if False:
            # --- 2. Semilog (log y-axis) plot ---
                plt.figure(figsize=(8, 5))
                plt.semilogy(x, abs(y), '-', markersize=4)
                plt.xlabel("Voltage (V)")
                plt.ylabel("Capacitance (F)")
                plt.title(f"{name} CV Characteristic")
                plt.grid(True, which='both', ls='--')
                plt.tight_layout()
                plt.savefig(f"Data/{folder}/{filename}_semilog.png", dpi=150, bbox_inches='tight')
                #plt.show()
    print(f'yay2')
    print(mpl.rcParams['lines.markersize'])

def get_lin_fit():

    filenames = ["diodeA_Light","diodeA_Dark","diodeB_Light","diodeB_Dark"]
    diodes = ['DiodeA', 'DiodeA', 'DiodeB', 'DiodeB']
    names = ['Diode A Light', 'Diode A Dark', 'Diode B Light', 'Diode B Dark']
    I_highs = [0.016, 0.016, 0.010, 0.010] 
    fit_domains = [[0.35,0.6], [0.25,0.6], [0.039,0.22], [0.04,0.24]]
    for i, filename in enumerate(filenames):
        df = pd.read_csv(
        f'Data/{filename}.txt',
        skiprows=3,          # skips the first 4 header lines
        sep=r'\s+',          # columns separated by spaces
        decimal=','          # interpret commas as decimal points
        )

        x = df.iloc[:, 0].values  # Voltage (V)
        y = df.iloc[:, 1].values  # Current (A)
        data = Data(x,y, filename=filename, name=names[i])
        data.fit(domain=fit_domains[i])
        if True:
            data.get_params(I_high = I_highs[i])

            info = (
                f"Name: {data.name}\n"
                f"Intercept: {data.y_0}\n"
                f"Slope: {data.slope}\n"
                f"Ideality constant: {data.ideality_constant}\n"
                f"Series Rs: {data.Rs}\n"
                f"Current backward bias (I0): {data.y[0]}\n"
                )
            print(info)

            with open(f'Data/{diodes[i]}/calculated_values_{data.filename}_fit.txt', 'w') as f:
                f.write(info)

        data.draw_plot()
        #plt.show()
        data.fig.savefig(f'Data/{diodes[i]}/{data.filename}_fit.png', dpi=150)

def a_vs_b_plot():
    plt.figure(figsize=(6, 6))

    diodes = ['Diode A', 'Diode B']
    filenames = ["diodeA_Light","diodeB_Light"]
    names = ['Diode A Light', 'Diode B Light']
    # Read the file, skip the header lines
    for i, filename in enumerate(filenames):
        print(f'Reading {filename}...')
        df = pd.read_csv(
            f'Data/{filename}.txt',
            skiprows=3,          # skips the first 4 header lines
            sep=r'\s+',           # columns separated by spaces
            decimal=','          # interpret commas as decimal points
        )

        name = names[i]
        folder = diodes[i]
        # Create folder if it doesn't exist
        if not os.path.exists(f"Data/Comparison"):
            os.makedirs(f"Data/Comparison")

        # Extract columns into numpy arrays
        x = df.iloc[:, 0].values  # Voltage (V)
        y = df.iloc[:, 1].values  # Current (A)

        # --- 2. Semilog (log y-axis) plot ---
        plt.semilogy(x, abs(y), 'o-', markersize=4, label=folder)

    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    #plt.show()
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Diode IV Comparison")
    plt.legend()
    plt.savefig(f"Data/Comparison/Comparison_semilog.png", dpi=150, bbox_inches='tight')
    print(f'yay')


def main():

    plot_data()
    get_lin_fit()
    a_vs_b_plot()
    copy_figures_to_folder()
    

if __name__ == "__main__":
    main()