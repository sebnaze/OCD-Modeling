import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm

def H(x, a=270, b=108, d=0.154):
    """ Average synaptic gating
        -----------------------
            x: local activity
            a: slope (n/C); default=270
            b: offset (Hz); default=108
            d: decay (s);   default=0.154
    """
    return (a*x-b)/(1-np.exp(-d*(a*x-b)))

x = np.arange(-1,2,0.01)

n = 10 

plt.figure(figsize=[20,4])
plt.subplot(1,3,1)
for i,a in enumerate(np.linspace(0,540,n)):
    plt.plot(x, H(x,a=a), color=cm.jet(i/n))
plt.plot(x, H(x), color='black', lw=2)
plt.title('a')
plt.xlabel('x')
plt.ylabel('H(x)')

plt.subplot(1,3,2)
for i,b in enumerate(np.linspace(0,216,n)):
    plt.plot(x, H(x,b=b), color=cm.jet(i/n))
plt.plot(x, H(x), color='black', lw=2)
plt.title('b')
plt.xlabel('x')
plt.ylabel('H(x)')

plt.subplot(1,3,3)
for i,d in enumerate(np.linspace(0.01,0.06,n)):
    plt.plot(x, H(x,d=d), color=cm.jet(i/n))
plt.plot(x, H(x), color='black', lw=2)
plt.title('d')
plt.xlabel('x')
plt.ylabel('H(x)')
