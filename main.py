"""
Illustrative plot: fall height vs probability of serious injury and death.
This is illustrative only — based on literature trend points (see citations in chat).
Requires: numpy, scipy, matplotlib
Run: python plot_fall_risks.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k*(x - x0)))


data = np.array([
    [0.5, 0.01, 0.001],   
    [1.5, 0.02, 0.002],
    [3.0, 0.08, 0.01],    
    [5.0, 0.20, 0.05],    
    [7.0, 0.35, 0.12],   
    [10.0, 0.55, 0.30],  
    [15.0, 0.80, 0.60],  
    [20.0, 0.92, 0.85],  
    [25.0, 0.98, 0.98], 
])

heights = data[:,0]
serious = data[:,1]
death = data[:,2]


p0_ser = [1.0, 8.0, 0.5] 
p0_dea = [1.0, 15.0, 0.5]

bounds = ([0.5, 0, 0.01], [1.0, 100.0, 5.0])

popt_ser, _ = curve_fit(logistic, heights, serious, p0=p0_ser, bounds=bounds, maxfev=20000)
popt_dea, _ = curve_fit(logistic, heights, death, p0=p0_dea, bounds=bounds, maxfev=20000)


x = np.linspace(0, 30, 601)
ser_curve = logistic(x, *popt_ser)
dea_curve = logistic(x, *popt_dea)

plt.figure(figsize=(10,6))
plt.plot(x, ser_curve, label='Prob. of serious injury (ISS high)', linewidth=2)
plt.plot(x, dea_curve, label='Prob. of death', linewidth=2)
plt.scatter(heights, serious, marker='o', s=60, label='Reference points: serious injury')
plt.scatter(heights, death, marker='s', s=60, label='Reference points: death')


for hx in [1, 5, 10, 15, 20]:
    plt.axvline(hx, color='gray', linestyle='--', alpha=0.25)

plt.title('Drago mi je so ti sum dokazaf da ne mamim! A i da ne mi je muka')
plt.xlabel('Fall height (meters)')
plt.ylabel('Probability')
plt.ylim(-0.02, 1.02)
plt.xlim(0, 30)
plt.grid(alpha=0.4, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("not_a_liar.png", dpi=300)
plt.show()

print("Fitted logistic params (serious injury): L,x0,k =", popt_ser)
print("Fitted logistic params (death):           L,x0,k =", popt_dea)
