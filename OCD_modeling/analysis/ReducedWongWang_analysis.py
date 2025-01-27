# Early exploratory analysis of the original Reduced Wong-Wang model
# showing effects of parameters on dynamics, i.e. mosltly nullclines.

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from OCD_modeling.models import ReducedWongWang

x = np.arange(-1,2,0.01)
n = 100 # number of sampling point

# effect of I_0
plt.figure(figsize=[20,4])
for i,i0 in enumerate(np.linspace(0,1,n)):
    rww = ReducedWongWang(I_0=i0)
    S = rww.S_i(x)
    dS = rww.dS_i(x)

    plt.subplot(1,3,1)
    plt.plot(x,dS, color=mpl.cm.jet(i/n))
    
    plt.subplot(1,3,2)
    plt.plot(S,dS, color=mpl.cm.jet(i/n))

    plt.subplot(1,3,3)
    plt.plot(x,S, color=mpl.cm.jet(i/n))

plt.subplot(1,3,1)
plt.plot(x,dS_0, '--', color='black', lw=2)
plt.xlabel('x_i')
plt.ylabel('dS_i')

plt.subplot(1,3,2)
plt.plot(S,dS_0, '--', color='black', lw=2)
plt.xlabel('S_i')
plt.ylabel('dS_i')

plt.subplot(1,3,3)
plt.plot(x,dS_0, '--', color='black', lw=2)
plt.xlabel('x_i')
plt.ylabel('S_i')
plt.ylim([-0.1, 1])
plt.show()


# effect of J_N
plt.figure(figsize=[20,4])
for i,jN in enumerate(np.linspace(0,0.5,n)):
    rww = ReducedWongWang(J_N=jN)
    S = rww.S_i(x)
    dS = rww.dS_i(x)

    plt.subplot(1,3,1)
    plt.plot(x,dS, color=mpl.cm.jet(i/n))
    
    plt.subplot(1,3,2)
    plt.plot(S,dS, color=mpl.cm.jet(i/n))

    plt.subplot(1,3,3)
    plt.plot(x,S, color=mpl.cm.jet(i/n))

plt.subplot(1,3,1)
plt.plot(x,dS_0, '--', color='black', lw=2)
plt.xlabel('x_i')
plt.ylabel('dS_i')

plt.subplot(1,3,2)
plt.plot(S,dS_0, '--', color='black', lw=2)
plt.xlabel('S_i')
plt.ylabel('dS_i')

plt.subplot(1,3,3)
plt.plot(x,dS_0, '--', color='black', lw=2)
plt.xlabel('x_i')
plt.ylabel('S_i')
plt.ylim([-0.1, 1])
plt.show()


# effect of tau_S
plt.figure(figsize=[20,4])
for i,j in enumerate(np.linspace(50,500,n)):
    rww = ReducedWongWang(tau_S=j)
    S = rww.S_i(x)
    dS = rww.dS_i(x)
    Snc = rww.S_nc(x)

    plt.subplot(1,3,1)
    plt.plot(x,dS, color=mpl.cm.jet(i/n))
    plt.plot(x,Snc, '--', color=mpl.cm.jet(i/n))
    
    plt.subplot(1,3,2)
    plt.plot(S,dS, color=mpl.cm.jet(i/n))
    plt.plot(S,Snc, '--', color=mpl.cm.jet(i/n))

    plt.subplot(1,3,3)
    plt.plot(x,S, color=mpl.cm.jet(i/n))
    plt.plot(x,Snc, '--', color=mpl.cm.jet(i/n))

plt.subplot(1,3,1)
plt.xlabel('x_i')
plt.ylabel('dS_i')

plt.subplot(1,3,2)
plt.xlabel('S_i')
plt.ylabel('dS_i')

plt.subplot(1,3,3)
plt.xlabel('x_i')
plt.ylabel('S_i')
plt.ylim([-0.1, 1])

plt.show()

# plotting nuccline against each other
plt.figure()
plt.plot(x,Snc)
plt.plot(Snc,x)
plt.show()



# effect of tau_S and I_0 together on nullcine intersection
n=5
m=4
fig = plt.figure(figsize=[16,20])
gs = plt.GridSpec(n,m)
for i,j in enumerate(np.linspace(0.2,0.4,n)):
    for k,l in enumerate(np.linspace(100,400,m)):
        fig.add_subplot(gs[i,k])
        rww = ReducedWongWang(I_0=j, tau_S=l)
        S = rww.S_i(x)
        dS = rww.dS_i(x)
        Snc = rww.S_nc(x)
        plt.plot(Snc,S, color=mpl.cm.jet(i/n))
        plt.plot(S,Snc, color=mpl.cm.jet(i/n))
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.xlabel('S_nc')
        plt.ylabel('S_nc')
        plt.title("i0={:.2f}; tauS={}".format(j,int(l)))
        plt.tight_layout()

plt.show()


