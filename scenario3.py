# Librairies

import numpy as np
from scipy import optimize # Library for linear algebra
import matplotlib.pyplot as plt
import random as rd # Use of random numbers

import POP_DYNAMICS as pop # POP_DYNAMICS is the function in the same name module created for working with the two scénarii
#from POP_DYNAMICS import pair
################ Density Dependance Function ################

def DD3_1(Alpha, N1, N2): # DD function for pairwise analysis
    """
    Density-dependence function.
    Density-dependence function for Pairwise Invasibility Analysis in the scenario 3.
    
    Parameters
    ----------
    Alpha: int
        Measure of density independent recruitment.
    N1: int
        Number of individuals at time = t.
    N2: int
        Number of individuals at time = t.
        
    Returns
    -------
    int
        The new population size
    """
    
    Beta = Alpha * 0.001 # Set value of beta
    
    N = N1 * Alpha * np.exp(-Beta * N2) # New population size
    
    return(N)

def DD3_2(x, N_total): # DD function for multiple invasibility analysis
    """
    Density-dependence function.
    Density-dependence function for Multiple Invasibility Analysis in the scenario 3.
    
    Parameters
    ----------
    x: array
        An array containing two values: Alpha the Measure of density independent recruitment and N the population size for this alpha.
    N_total: int
        Total number of individuals at time = t.
        
    Returns
    -------
    int
        The new population size
    """
    Alpha = x[0]
    
    N = x[1]
    
    Beta = Alpha * 0.001
    
    N = N * Alpha * np.exp(-Beta * N_total)
    
    return(N)

####################### Pariwise invasibility analysis #######################
    
##################### MAIN PROGRAM #####################
    

########### PIP Analysis ###########
N1 = 30 # Nos of increments

A_Resident = np.linspace(2, 4, N1) # Resident alpha
A_Invader = A_Resident.copy() # Invader alpha

d = (np.array(np.meshgrid(A_Resident, A_Invader)).reshape(2,len(A_Resident)*len(A_Resident)).T) # Equivalent to R's expand.grid()

z = np.apply_along_axis(pop.pair, 1, d, DD3_1) # Apply POP_DYNAMICS for all X_Resident/X_Invader paired
z_mat = z.copy().reshape(len(A_Resident),len(A_Invader)) # in plt.contour() Z must be a matrix so reshape z vector into a matrix

########### Elasticity Analysis ###########

minA = 1
maxA = 10

Best_Alpha = optimize.brentq(pop.pair, minA, maxA, args=(DD3_1, False, 0.995))
print("Optimum = " + str(round(Best_Alpha,6)))


# Calculs for plotting elasticity vs Alpha

N_int = 30 # Nos of intervals for plot
Alpha = np.linspace(minA, maxA, N_int)

Elasticity = np.array([])

# Calculs for plot Elasticity vs alpha

for i in range(0, len(Alpha)):
    Elasticity = np.append(Elasticity, pop.pair(Alpha[i], DD3_1, False, 0.995))

# Calculs for plot invasion when resident is optimal
Coeff = np.divide(Alpha, Best_Alpha)

Invasion_coeff = np.zeros(shape = (N_int, 1)) # Pre-allocating space

for i in range(0, N_int):
    Invasion_coeff[i] = pop.pair(Best_Alpha, DD3_1, False, Coeff[i]) 
    
# Calculus for Plot N(t+1) on N(t) for optimum alpha
maxN = 1000 # Number of N
    
N_t = np.linspace(1, maxN, maxN) # Values of N(t)
    
N_tplus1 = np.zeros(shape = (maxN, 1))
    
for i in range(0, maxN):
    
    N_tplus1[i] = DD3_1(Best_Alpha, N_t[i], N_t[i])
     
# Calculs for plotting N(t) on t

N = np.repeat(1, 100) # Allocate memory

for i in range(1, 100):
    N[i] = DD3_1(Best_Alpha, N[i-1], N[i-1])

########### Plots ###########
    
### PIP ###
fig = plt.figure()

ax = fig.add_subplot(111)

CS = ax.contour(A_Resident, A_Invader, z_mat, levels=9, colors="black")
ax.clabel(CS) # Labels for the countour plot
ax.scatter(Best_Alpha, Best_Alpha, edgecolors='black', facecolors='black',  s = 150) # s = size
ax.set(title="PIP Plot")

plt.show()

### Elasticity Analysis ###

fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={"wspace": 0.5, "hspace": 0.5}) # Space between two plots

axes[0,0].plot(Alpha, Elasticity, color="black")
axes[0,0].axhline(y=0, color="black")
axes[0,0].set(ylim = [-0.006, 0.006],
              xlabel = "Alpha",
              ylabel = "Elasticity")

axes[0,1].plot(Alpha, Invasion_coeff, color = "black")
axes[0,1].scatter(Best_Alpha, 0, edgecolors='black', facecolors='none', s = 150)
axes[0,1].set(xlabel = "Alpha",
              ylabel = "Invasion coefficient")

axes[1,0].plot(N_t, N_tplus1, color = "black")
axes[1,0].set(xlabel = "N(t)", ylabel = "N(t+1)")

axes[1,1].plot(np.linspace(1, 100, 100), N, color = "black")
axes[1,1].set(xlabel = "Generation", ylabel = "Population")

plt.show()

####################### Multiple invasibility analysis #######################

rd.seed(42) # Initialize the random number seed

pop.mult(Maxgen = 5000, MaxAlpha = 4, Ninc = 50, DD_fun = DD3_2)

