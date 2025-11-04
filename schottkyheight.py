import math
import os
import numpy as np
import pandas as pd

kb = 8.62*10**(-5) #eV/K
KK = 100 # ??
q = 1.6*10**(-19)
T = 300
A = ((0.05)**2*3.1415926535)
I_0 = 2.58*10**(-7)
ktq = 0.0259 # V

phiB = ktq * math.log(A*KK*T**3/I_0)
print(f'phiB {phiB}')

epsi = 1.04*10**(-12)
A = (0.05)**2*math.pi
Cmin = 2.920670E-11
Cox = 3.27*10**(-10)
dep = 1/Cmin - 1/Cox
wdep = dep*A*epsi
ni = 10**10
print(f'wdep {wdep}')


def closest_value(N):

    diodes = ['MOSCAP_CV_Dark']
    filenames = ["MOSCAP_CV_Dark"]
    names = ['MOSCAP_CV_Dark']
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
        V = df.iloc[:, 0].values  # Voltage (V)
        C = df.iloc[:, 1].values  # Capacitnce (C)

        print(x)

    Cfb = (1/Cox + 1/A*math.sqrt((ktq/(q*N*epsi))))**(-1)
    print(f'Cfb is {Cfb}')
    idx = np.argmin(np.abs(C - Cfb))
    closest_V = V[idx]
    print(f"Closest V {closest_V}")
def theoretical_Vfb(N):
    Nc = 2.8*10**(19)
    N = ni**2/N
    Ecf = ktq * math.log(N/Nc)
    phiM = 4.1 # eV
    Xsi = 4.05 # eV
    Vfb = phiM - Xsi - Ecf
    print(f'Theoretical Vfb {Vfb}')


def Wdep(N):
    return math.sqrt(4*epsi*ktq*math.log(N/ni)/(q*N))-wdep

def Wdep_prim(N):
    a = 4*epsi*ktq/q
    return a*(1-math.log(N/ni)) / (2*N**2*Wdep(N))

print(Wdep(10**17))
def bisection(f, a, b, tol=10**(-12), max_iter=100):
    fa, fb = f(a), f(b)
    if fa == 0: return a,0
    if fb == 0: return b,0
    if fa*fb > 0: print('ass')
    for k in range(1, max_iter+1):
        print(k)
        m = (a+b) / 2
        fm = f(m)
        if  abs(fm) <=tol:
            print('Hello I return N and iteration k')
            return m,k
        if fa*fm < 0:
            b, fb= m, fm
        else:
            a, fa = m, fm
    
    return (a+b) / 2, max_iter

(x, f) = bisection(Wdep, 10**12, 10**20)
print(x,f)
print(x/(1E13))

closest_value(x)
theoretical_Vfb(x)