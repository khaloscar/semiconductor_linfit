import math
kb = 8.62*10**(-5) #eV/K
KK = 100 # ??
q = 1.6*10**(-19)
T = 300
A = (0.1)**2*3.1415926535
I_0 = 2.58*10**(-7)
ktq = 0.0259 # V

phiB = ktq * math.log(A*KK*T**3/I_0)

epsi = 1.04*10**(-12)
A = 0.1*10**(-3) * math.pi
Cmin = 2.920670E-11
Cox = 3.27*10**(-10)
dep = 1/Cmin - 1/Cox
wdep = dep*A*epsi
ni = 10**10
print(wdep)

def Wdep(N):
    return math.sqrt(4*epsi*ktq*math.log(N/ni)/(q*N))-wdep
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