### Log-book
##
#

import sympy as sp
from sympy import init_session
init_session()

# model definition
x1,x2,S1,S2,C_12,C_21 = sp.symbols('x1 x2 S1 S2 C_12 C_21')
S = sp.Matrix([S1, S2])
x = sp.Matrix([x1, x2])
C = sp.Matrix([[0, C_12],[C_21, 0]])

w, J_N, I_0, G = sp.symbols('w J_N I_0 G')
X = w*J_N*S + G*J_N*C*S + sp.ones(2,1)*I_0

a,b,d, x_i, theta = sp.symbols('a b d x_i theta')
#H = sp.Matrix([a1/a2 for a1,a2 in zip((a*X-b*sp.ones(2,1)),sp.ones(2,1)-(-d*(a*X-b*sp.ones(2,1))).applyfunc(sp.exp))])
#H = sp.Matrix([a*(X-theta*sp.ones(2,1))])
H_ = sp.Matrix([sp.Piecewise( (0, X[0] <= theta), (a*(X[0]-theta), X[0]>theta)), sp.Piecewise( (0, X[1] <= theta), (a*(X[1]-theta), X[1]>theta))])
#H = H_(X)

tau_S, gamma = sp.symbols('tau_S gamma')
dS = (-S/tau_S) + sp.matrices.dense.matrix_multiply_elementwise((sp.ones(2,1)-S), gamma*H_)
sp.print(dS)


## Case 1: S1=0 ; S2=0

## Case 2: S2=0, S1:
nc_case2a = ( S1**2*(-gamma*a*w*J_N) + S1*(-1/tau_S + gamma*a*w*J_N - gamma*a*I_0 + gamma*a*theta) + gamma*a*I_0 - gamma*a*theta) / (gamma*a*G*J_N*C_12*(S1-1))

dS1 = dS[0].subs({'S2':0})
sp.print(dS1)
case2a_sol = S1**2*(-gamma*a*w*J_N)   +   S1*(-1/tau_S + gamma*a*w*J_N - gamma*a*I_0 + gamma*a*theta)   +   gamma*a*I_0 - gamma*a*theta
sp.print(case2a_sol)
sp.plot(case2a_sol.subs({'theta':0.4, 'a':270, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}), xlim=(-2,2), ylim=(-0.5,0.1))



## Case 3:
nc_case3a = ( S1**2*(-gamma*a*w*J_N) + S1*(-1/tau_S + gamma*a*w*J_N - gamma*a*I_0 + gamma*a*theta) + gamma*a*I_0 - gamma*a*theta) / (gamma*a*G*J_N*C_12*(S1-1))
nc_case3b = ( S2**2*(-gamma*a*w*J_N) + S2*(-1/tau_S + gamma*a*w*J_N - gamma*a*I_0 + gamma*a*theta) + gamma*a*I_0 - gamma*a*theta) / (gamma*a*G*J_N*C_21*(S2-1))
nc_case3a_subs = nc_case3a.subs({'theta':0.4, 'a':270, 'C_12':-1, 'C_21':-1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641})
nc_case3b_subs = nc_case3b.subs({'theta':0.4, 'a':270, 'C_12':-1, 'C_21':-1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641})

#nc_cond3a = (theta - I_0 - G*J_N*C_12*S2) / (w*J_N) # > S1
nc_cond3a = sp.Piecewise(((theta - I_0 - J_N*S1*w) / (C_12*G*J_N), C_12>0), (-(theta - I_0 - J_N*S1*w) / (C_12*G*J_N), C_12<0)) # S1 > theta
#nc_cond3b = (theta - I_0 - G*J_N*C_21*S1) / (w*J_N) # > S2
nc_cond3b = sp.Piecewise(((theta - I_0 - J_N*S2*w) / (C_21*G*J_N), C_21>0), (-(theta - I_0 - J_N*S2*w) / (C_21*G*J_N), C_21<0)) # S2 > theta
nc_cond3a_subs = nc_cond3a.subs({'theta':0.4, 'a':270, 'C_12':-1, 'C_21':-1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641})
nc_cond3b_subs = nc_cond3b.subs({'theta':0.4, 'a':270, 'C_12':-1, 'C_21':-1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641})


## replacing eq1 solution in eq2 equation:
dS[1].subs({'S2':nc_case3a})
sp.plot(dS[1].subs({'S2':nc_case3a}).subs({'theta':0.4, 'a':270, 'C_12':1, 'C_21':1, 'G':1, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}),
        (S1, -3,3), ylim=(-0.1,0.1))

def find_roots(f,x,itv=None, slope_thr=10):
    """ find zero crossings of function f """
    if itv==None:
        itv = [x.min(), x.max()]
    roots = []
    for i,x_i in enumerate(x[:-1]):
        if ( ((f(x_i) > 0) & (f(x[i+1]) < 0)) | ((f(x_i) < 0) & (f(x[i+1]) > 0)) ):
            diff = f(x[i+1]) - f(x_i) 
            if ((diff < 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
            if ((diff > 0) & (np.abs(diff) < slope_thr)):
                roots.append({'x':(x_i+x[i+1])/2, 'slope':diff})
    return roots


## Bifurcation diagrams:
sub_params = {'theta':0.4, 'a':270, 'C_12':1, 'C_21':1, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}
s = np.linspace(-3,3,999)

# 1 param
output = dict()
for c_12 in np.arange(-1,1,0.05):
    sub_params['C_12'] = c_12
    g = sp.lambdify(S1, dS[1].subs({'S2':nc_case3a}).subs(sub_params))
    fp = find_roots(g,s)
    if len(fp)>0:
        output[c_12] = fp


fig = plt.figure(figsize=[6,4])
for c_12,fps in output.items():
    for fp in fps:
        if fp['slope'] < 0:
            opts = {'marker': 'o', 'markersize':4, 'color':'k', 'fillstyle':'full'}
        elif fp['slope'] > 0:
            opts = {'marker': 'o', 'markersize':4, 'color':'k', 'fillstyle':'none'}
        else:
            continue
        plt.plot(c_12, fp['x'], **opts)
plt.xlim(-1,1)

# 2 params
output = dict()
for c_12 in np.arange(-1,1,0.2):
    for c_21 in np.arange(-1,1,0.2):
        sub_params['C_12'] = c_12
        sub_params['C_21'] = c_21
        g = sp.lambdify(S1, dS[1].subs({'S2':nc_case3a}).subs(sub_params))
        fp = find_roots(g,s)
        if len(fp)>0:
            output[c_12,c_21] = fp

from mpl_toolkits import mplot3d

fig = plt.figure(figsize=[8,8])
ax = plt.axes(projection='3d')

for c,fps in output.items():
    for fp in fps:
        if fp['slope'] < 0:
            marker = matplotlib.markers.MarkerStyle('o', fillstyle='full')
        elif fp['slope'] > 0:
            marker = matplotlib.markers.MarkerStyle('o', fillstyle='none')
        else:
            continue
        opts = {'marker':marker, 'color':'k'}
        ax.scatter3D(c[0], c[1], fp['x'], **opts)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('C_12')
plt.ylabel('C_21')
plt.show()

## Stability analysis

jacobian = [[dS[0].diff(S1), dS[0].diff(S2)], [dS[1].diff(S1), dS[1].diff(S2)]]

def get_stability(dS, c_12, c_21, default_params):
        default_params['C_12'] = c_12
        default_params['C_21'] = c_21
        g = sp.lambdify(S1, dS[1].subs({'S2':nc_case3a}).subs(default_params))
        fps = find_roots(g,s)
        for fp in fps: 
            fp['S1'] = fp['x']
            default_params['S1'] = fp['S1']
            fp['S2'] = nc_case3a.subs(default_params)
            default_params['S2'] = fp['S2']
            
            fp['tau'] = sp.trace(sp.Matrix(jacobian).subs(default_params))
            fp['delta'] = sp.det(sp.Matrix(jacobian).subs(default_params))
            fp['lambda1'] = (fp['tau'] - sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
            fp['lambda2'] = (fp['tau'] + sp.sqrt(fp['tau']**2 - 4*fp['delta']))/2
            
            l1_re, l1_im = fp['lambda1'].as_real_imag()
            l2_re, l2_im = fp['lambda2'].as_real_imag()
            
            # real eigenvalues
            if ((l1_im==0) & (l2_im==0)):
                # saddle
                if (((l1_re>0) & (l2_re<0)) | ((l1_re<0) & (l2_re>0))):
                    fp['type'] = 'saddle'
                # unstable
                elif ((l1_re>0) & (l2_re>0)):
                    fp['type'] = 'unstable node'
                elif ((l1_re<0) & (l2_re<0)):
                    fp['type'] = 'stable node'
            # complex eigenvalues
            else:
                if ((l1_re>0) & (l2_re>0)):
                    fp['type'] = 'unstable focus'
                elif ((l1_re<0) & (l2_re<0)):
                    fp['type'] = 'stable focus'
        return (c_12,c_21), fps


sub_params = {'theta':0.4, 'a':270, 'C_12':1, 'C_21':1, 'G':2.5, 'J_N':0.2609, 'I_0':0.3, 'tau_S':100, 'w':0.9, 'gamma':0.000641}
s = np.linspace(-3,3,999)

c_12s, c_21s = np.arange(-1,1,0.025), np.arange(-1,1,0.025)
out = joblib.Parallel(n_jobs=32)(joblib.delayed(get_stability)(dS,c_12,c_21,sub_params.copy()) for c_12,c_21 in itertools.product(c_12s, c_21s))
output = dict((c,fps) for c,fps in out)
