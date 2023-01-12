import importlib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
import sympy as sp

from ..models import ReducedWongWang as RWW
importlib.reload(RWW)

rww2D = RWW.ReducedWongWang2D()

# model definition
x1,x2,S1,S2,C_12,C_21 = sp.symbols('x1 x2 S1 S2 C_12 C_21')
S = sp.Matrix([S1, S2])
x = sp.Matrix([x1, x2])
C = sp.Matrix([[0, C_12],[C_21, 0]])

w, J_N, I_0, G = sp.symbols('w J_N I_0 G')
X = w*J_N*S + G*J_N*C*S + sp.ones(2,1)*I_0

a,b,d, x_i = sp.symbols('a b d x_i')
H = sp.Matrix([a1/a2 for a1,a2 in zip((a*X-b*sp.ones(2,1)),sp.ones(2,1)-(-d*(a*X-b*sp.ones(2,1))).applyfunc(sp.exp))])

tau_S, gamma = sp.symbols('tau_S gamma')
dS = (-S/tau_S) + sp.matrices.dense.matrix_multiply_elementwise((sp.ones(2,1)-S), gamma*C*H)
dS

# using SOLVE
# -----------
# nullclines
n1 = sp.solve(dS[0], S2)
n2 = sp.solve(dS[1], S1)

# plot nullclines:
sp.plot(n1[0].subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:-1, C_21:1}), (S1,-1,0.9), n=1000, title='dS_1=0 curve\n')
sp.plot(n2[0].subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:-1, C_21:1}), (S2, -1,0.9), n=1000, title='dS_2=0 curve\n')

# substitute S1 into S2 nullcline equation
n0 = n1[0].subs({S1:n2[0]})
sp.plot(n0.subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:1, C_21:1}), (S2, 0.18, 0.9), n=1000, title='dS=0 curve\n')

# using SOLVESET (faster)
# -----------------------
# nullclines
nc_set = sp.nonlinsolve(dS, [S1, S2])
# TODO: extract terms automatically
n1_set = sp.solve(sp.Eq(C_12*gamma*tau_S*(1 - S1)*(a*(C_21*G*J_N*S1 + I_0 + J_N*S2*w) - b)*sp.exp(d*(a*(C_21*G*J_N*S1 + I_0 + J_N*S2*w) - b)) - S1*(sp.exp(d*(a*(C_21*G*J_N*S1 + I_0 + J_N*S2*w) - b)) - 1), 0), S2)
sp.plot(n1_set[0].subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:-1, C_21:1}), (S1,-0.5,0.9), n=1000, title='dS_1=0 curve\n')

n2_set = sp.solve(sp.Eq(C_21*gamma*tau_S*(1 - S2)*(a*(C_12*G*J_N*S2 + I_0 + J_N*S1*w) - b)*sp.exp(d*(a*(C_12*G*J_N*S2 + I_0 + J_N*S1*w) - b)) - S2*(sp.exp(d*(a*(C_12*G*J_N*S2 + I_0 + J_N*S1*w) - b)) - 1), 0), S1)
sp.plot(n2_set[0].subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:-1, C_21:1}), (S2, -1,0.9), n=1000, title='dS_2=0 curve\n')

n0_set = n1_set[0].subs({S1:n2_set[0]})
sp.plot(n0_set.subs({a:rww2D.a, b:rww2D.b, d:rww2D.d, tau_S:rww2D.tau_S, I_0:0.3, J_N:rww2D.J_N, w:rww2D.w, gamma:rww2D.gamma, G:rww2D.G, C_12:1, C_21:1}), (S2, 0.18, 0.9), n=1000, title='dS=0 curve\n')

# multivariate Taylor expansion on fixed point (C_12=-1; C_21=1; S1~1.1; S2~0.2)
# on dS1
taylor_dS1_S1 = sp.series(dS[0], S1, x0=1.1, n=2).removeO()
taylor_dS1_S2 = sp.series(dS[0], S2, x0=0.2, n=2).removeO()
taylor_dS1 = taylor_dS1_S1.subs({S2:0.2}) + taylor_dS1_S2.subs({S1:1.1})
dS1_S1_coeff = sp.Poly(taylor_dS1.factor(S1,S2)).as_expr().coeff(S1)
dS1_S2_coeff = sp.Poly(taylor_dS1.factor(S1,S2)).as_expr().coeff(S2)

# on dS2
taylor_dS2_S1 = sp.series(dS[1], S1, x0=1.1, n=2).removeO()
taylor_dS2_S2 = sp.series(dS[1], S2, x0=0.2, n=2).removeO()
taylor_dS2 = taylor_dS2_S1.subs({S2:0.2}) + taylor_dS2_S2.subs({S1:1.1})
dS2_S1_coeff = sp.Poly(taylor_dS2.factor(S1,S2)).as_expr().coeff(S1)
dS2_S2_coeff = sp.Poly(taylor_dS2.factor(S1,S2)).as_expr().coeff(S2)

# combining
taylor_dS = sp.Matrix([[dS1_S1_coeff, dS1_S2_coeff],[dS2_S1_coeff, dS2_S2_coeff]])
