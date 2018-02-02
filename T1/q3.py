import numpy as np
import matplotlib.pyplot as plt
import math
import sys

K_true = 10

def BIC(sd, K, x, coeffs, chi, N): 
  third = 0.5 * chi 
  fourth = (K+1)/2 * np.log(N) 
  return third + fourth

def part3(N):
  coeffs = np.random.uniform(-1,1, K_true+1) 
  x = np.random.uniform(-5,5,N+1)
  y = []
  mx = 0 
  mn = 10000000000000  
  for i in range(N+1): 
    tmp = np.poly1d(coeffs)(x[i])
    y.append(tmp)
    mx = max(mx, tmp) 
    mn = min(mn, tmp) 
  sd = ((mx - mn)/10) ** 0.5
  eps = np.random.normal(0, sd, N+1)
  y = [a+b for a,b in zip(y, eps)]
  min_bic = sys.maxint
  opt_K = None 
  for K in range(1, 20): 
    a = np.polyfit(x,y,K, full=True)
    if a[1]: 
      err = a[1][0]/(sd**2)
    else: 
      err = 0.
    bic_val = BIC(sd, K, x, coeffs, err, N) 
    if bic_val < min_bic: 
      min_bic = bic_val
      opt_K = K 
  return(opt_K)

opt_K_values = [] 
for _ in range(500): 
  opt_K_values.append(part3(20))
opt_K_values = np.array(opt_K_values) 
print(np.var(opt_K_values))
print(np.mean(opt_K_values))

n_range = 3*np.logspace(0,4,40)
opt_K = []
for i in range(500):
  opt_Kn = [] 
  for n in n_range: 
    opt_Kn.append(part3(int(n))) 
  opt_K.append(opt_Kn)
  print("Trial: " + str(i))
opt_K = np.array(opt_K)
mn = []
vr = [] 
for arr in opt_K.T: 
  mn.append(np.mean(arr)) 
  vr.append(np.var(arr)) 
mn = np.array(mn) 
vr = np.array(vr) 
plt.figure(1) 
plt.errorbar(np.vectorize(np.log)(n_range), mn, yerr=vr, marker='o')
plt.xlabel("N") 
plt.ylabel("Optimal K") 
plt.show() 
