##Scenario 6
import numpy as np
from sklearn.linear_model import LinearRegression


def DD_Function(Alpha, N1, N2):
    #Set parameter values
    Amin=1
    Amax=50
    Bmin=0
    Bmax=0.9
    a=20
    Beta=0.01
    AA=(Alpha-Amin)/(Amax-Amin)
    S=(Bmax-Bmin)*(1-AA)/(1+a*AA) #Survival
    N=N1*(Alpha*np.exp(-Beta*N2)+S)
    return N

def pop_dynamics_6(alpha_res, alpha_inv, genRes, genInv):
    tot_gen=genRes+genInv
    N_res=[0]*tot_gen
    N_inv=[0]*tot_gen
    N_res[0]=1
    N_inv[genRes-1]=1
    for i in range(1,genRes):
        N_res[i]=DD_Function(alpha_res,N_res[i-1],N_res[i-1])
    for i in range(genRes, tot_gen):
        Ntot=N_res[i-1]+N_inv[i-1]
        N_res[i]=DD_Function(alpha_res,N_res[i-1],Ntot)
        N_inv[i]=DD_Function(alpha_inv,N_inv[i-1],Ntot)
    for i in range(genRes+1,len(N_inv)):
        a=N_inv[i]
        N_inv[i]=np.log(a)
    Pop= N_inv[genRes:]
    Generation=np.linspace(0,tot_gen, len(Pop))
    inv_mod=LinearRegression()
    inv_mod.fit(Generation.reshape(-1,1), Pop)
    Elasticity=float(inv_mod.coef_)
    return(Elasticity)
