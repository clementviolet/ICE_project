#%% [markdown]
#
## Introduction to Computational Ecology
# This Jupyter notbebook is the place where all the chapter of Roff's book about
# invasibility analysis is translated from R to Python.

#%%
import pandas as pd
import numpy as np
from scipy import linalg, sparse, optimize # Library for linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
### Introduction
#### Age or stage-sctructure models

Leslie_matrix = np.array([[0.8, 1.2, 1, 0],
                          [0.8, 0, 0, 0],
                          [0, 0.4, 0, 0],
                          [0, 0, 0.25, 0]])

Eigen_data = linalg.eig(Leslie_matrix)
Lambda = Eigen_data[0][1] # Get first eigenvalue
Maxgen = 12 # Number of generations simulation runs

n = np.array([1,0,0,0]) # Initial population

Pop = np.append(n, n.sum()) # Preassign matrix to hold cohort number and total population size

Obs_lambda = np.array([0, 0, 0, 0, 0]) # Preassign storage for observed lambda

for Igen in range(2,Maxgen+1): # Iterate over generations
    n = np.dot(Leslie_matrix, n) # Apply matrix multiplication
    Pop = np.vstack([Pop, np.append(n, n.sum())]) # Store cohorts and total population size
    Obs_lambda = np.vstack([Obs_lambda, (Pop[Igen-1, ]/Pop[Igen-2])]) # Store observed lambda
    # End of Igen loop

print(Obs_lambda[Maxgen-1, 0],Obs_lambda[Maxgen-1, 0]/Lambda)

Generation = np.arange(1, Maxgen+1) # Vector of generation number

ymin = np.min(Pop) # Get minimum and maximum pop
ymax = np.max(Pop)

fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={"wspace": 0.4, "hspace": 0.3})

# Plot population and cohort trajectories
for i in range(0, 4):
    axes[0,0].plot(Generation, Pop[:, i], color="black")

axes[0,0].plot(Generation, Pop[:,4], color="black", linestyle="--")
axes[0,0].set(xlabel="Generation",
              ylabel="Population and cohort sizes",
              ylim=[ymin, ymax])

x = np.matrix.flatten(Pop)
ymin = np.min(np.log(x[x > 0]))
ymax = np.max(np.log(x))

# Plot log of population and cohort trajectories
for i in range(0,4):
    axes[0,1].plot(Generation, np.log(Pop[:, i]), color="black")

axes[0,1].plot(Generation, np.log(Pop[:, 4]), color="black", linestyle="--")
axes[0,1].set(xlabel="Generation",
              ylabel="log Sizes",
              ylim=[ymin, ymax])

# Plot observed lambdas
for i in range(0,4):
    axes[1, 0].plot(Generation, Obs_lambda[:,i], color="black")

axes[1,0].plot(Generation, Obs_lambda[:,4], color="black", linestyle="--")
axes[1,0].set(xlabel="Generation",
              ylabel="Lambda")

# Plot observed r
for i in range(0,4):
    axes[1, 1].plot(Generation, np.log(Obs_lambda[:,i]), color="black")

axes[1,1].plot(Generation, np.log(Obs_lambda[:,4]), color="black", linestyle="--")
axes[1,1].set(xlabel="Generation",
              ylabel="r")

plt.show()

print(Obs_lambda[Maxgen-1, 0],Obs_lambda[Maxgen-1, 0]/Lambda)

#%% [markdown]
####Adding density dependence

#%%

def BH_FUNCTION(n, c1, c2):
    return(c1/(1+c2*n))

def RICKER_FUNCTION(n, alpha, beta):
    return(alpha*np.exp(-beta*n))

c1 = 100
c2 = 2*10**-3 # Beverthon Holt parameters

## Beverton Holt function ##
# Plot N(t) on t
Maxgen = 20

N_t = np.array([1.0])

for i in range(1, Maxgen):
    N_t = np.append(N_t, N_t[i-1]*BH_FUNCTION(N_t[i-1], c1, c2))

fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={"hspace": 0.5, "right": 0.97, "wspace": 0.40})

axes[0,0].plot(np.arange(1, Maxgen+1), N_t, color="black")
axes[0,0].set(xlabel="Generation, t", ylabel="N(t)")

# Plot N(t+1) on N(t)

MaxN = 5000

N_t = np.arange(1, MaxN+1)
N_tplus1 = N_t * np.apply_along_axis(BH_FUNCTION,0 ,N_t , c1, c2)

axes[0,1].plot(N_t, N_tplus1, color="black")
axes[0,1].set(xlabel="N(t)", ylabel="N(t+1)")

## Ricker function ##

alpha = np.array([6,60]) # Parameter values
beta = .0005

# Plot N(t) on t for 2 values of alpha

Maxgen = 40

for j in range(0,2):
    N_t = np.array([1.0])

    for i in range(1,Maxgen):
        N_t = np.append(N_t, N_t[i-1]*RICKER_FUNCTION(N_t[i-1], alpha[j], beta))

    axes[1+j, 0].plot(np.arange(1, Maxgen+1), N_t, color="black")
    axes[1+j, 0].set(xlabel="Generation, t", ylabel="N(t)")

    MaxN = 10000
    N_t = np.arange(1, MaxN+1)
    N_tplus1 = N_t * np.apply_along_axis(RICKER_FUNCTION, 0, N_t, alpha[j], beta)

    axes[1+j, 1].plot(N_t, N_tplus1, color="black")
    axes[1+j, 1].plot(N_t, N_t, color="black")
    axes[1+j, 1].set(xlabel="N(t)", ylabel="N(t+1)")

plt.show()

#%%[markdown]
## Scénario 1 - Comparing approaches
### Resolution as in chapter 2

#%%

def RCALC(x):
    """Function to compute the fitness"""
    Af = 0
    Bf = 16
    As = 1
    Bs = 0.5
    r = np.log(Af+Bf*x+1)-(As+Bs*x)
    return(-r) # max(f(x)) == min(-f(x))

print(optimize.minimize_scalar(RCALC, bounds=[0.1, 3]).x) # There is a small  

#%%[markdown]
# If the fitness function is too complex, we can approach r numericaly

#%%
# def SUMMATION(r,x):
#     Maxage = 50 # Maximum age
#     age = np.arange(1, Maxage+1)
#     # Parameter value
#     Af = 0
#     Bf = 16
#     As = 1
#     Bs = 0.5
#     m = np.repeat(Af+Bf*x, repeats=Maxage) # Number of female offspring
#     l = np.exp(-(As+Bs*x)*age) # Surival to age
#     Sum = sum(np.exp(-r*age)*l*m) #Characteristic equation sum
#     return(1-Sum)

# # Define function to find r given x
# def RCALC(x):
#     return(-optimize.brentq(SUMMATION, -1, 2, args=(x)))

# optimize.minimize_scalar(RCALC, bounds=[0.1, 3])

# Completer ne fonctionne pas

#%%[markdown]
### Solving using the eigenvalue of the Leslie matrix

#%%
def RCALC(x):
    Maxage = 50
    M = Maxage-1
    age = np.arange(1, Maxage+1)
    # Parameter values
    Af = 0
    Bf = 16
    As = 1
    Bs = 0.5
    m = np.repeat(Af+Bf*x, repeats=Maxage)
    l = np.exp(-(As+Bs*x)*age)
    S = np.array([l[0]])
    for i in range(1,M):
        S = np.append(S, l[i]/l[i-1])
    S = np.append(S, 0) # Last value is a zero for the last age class
    Fertility = m*S # Top row of Leslie matrix
    Survival = np.diag(S[0:M]) # Assign survivals to diagonal
    Survival = np.insert(Survival, 49, 0, axis=1) # Create the Leslie Matrix without the fertiity
    # 0 == index where to place the Fertility values, axis=0 place the Fertility values rowise
    Leslie_matrix = np.insert(Survival, 0, Fertility, axis = 0)# Create our Leslie Matrix
    Eigen_data = linalg.eig(Leslie_matrix) # Call for eigenvalues
    Lambda = Eigen_data[0][1] # Get lambda
    r = np.abs(np.log(Lambda))
    return(-r) # Maxium of a functi

Optimum = optimize.minimize_scalar(RCALC, bracket=[1,3], method='Golden') # bracket = interval; method='Golden' is the method used by R stats::optimize()
Best_x = Optimum.x
Best_r = Optimum.fun*-1 # Due to max(f(x)) == min(-f(x))

# # Print out results to 6 significant digits
print("Optimum body size = " + str(round(Optimum.x,6))+"\nMaximum R = " + str(round(Optimum.fun,6)))


x = np.linspace(1, 2.5, 50)
r_est = np.array([])

for i in range(0,50):
    r_est = np.append(r_est, RCALC(x[i]))

r_est = r_est*-1 # Due to use of - for maximize the function

# Plotting r.estimaed vs x
fig, ax = plt.subplots()

ax.plot(x, r_est, color="black")
ax.plot(Best_x, Best_r, color="Black", marker="o", markerfacecolor="None")
ax.set(xlabel="Body size, x", ylabel="r estimated")

plt.show()