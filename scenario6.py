# Libraries

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import random as rd # Use of random numbers

import POP_DYNAMICS as pop # POP_DYNAMICS is the function in the same name module created for working with the two scénarii

################ Density Dependance Function ################
    
def DD6_1(Alpha, N1, N2): # DD function for pairwise analysis
    """
    Density-dependence function for PIP Analysis and Elasticity Analysis in the sceanrio 6.
    
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
    # Parameters for the survival
    Amin = 1
    Amax = 50
    
    Bmin = 0
    Bmax = 0.9
    
    a = 20
    Beta = 0.01
    
    AA = (Alpha - Amin) / (Amax - Amin)
    
    S = (Bmax - Bmin) * (1 - AA) / (1 + a * AA) # Survival
    
    N = N1 * (Alpha * np.exp(-Beta * N2) + S) # New population
    
    return(N)

    
def DD6_2(x, N_total): # DD function for pairwise analysis
    """
    Density-dependence function.
    Density-dependence function for Multiple Invasibility Analysis in the scenario 6.
    
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
    
    # Parameters for the survival
    Amin = 1
    Amax = 50
    
    Bmin = 0
    Bmax = 0.9
    
    a = 20
    Beta = 0.01
    
    AA = (Alpha - Amin) / (Amax - Amin)
    
    S = (Bmax - Bmin) * (1 - AA) / (1 + a * AA) # Survival
    
    N = N * (Alpha * np.exp(-Beta * N_total) + S) # New population
    
    return(N)
    
####################### Pariwise invasibility analysis #######################

##################### MAIN PROGRAM #####################


########### PIP Analysis ###########
N1 = 30 # Nos of increments

A_Resident = np.linspace(5, 45, N1) # Resident alpha
A_Invader = A_Resident.copy() # Invader alpha

d = (np.array(np.meshgrid(A_Resident, A_Invader))
    .reshape(2,len(A_Resident)*len(A_Resident)).T) # Equivalent to R's expand.grid(). Combinaition of Alpha Resident and Alpha Invader

z = np.apply_along_axis(pop.pair, 1, d, DD6_1) # Apply POP_DYNAMICS for all X_Resident/X_Invader paired
z_mat = z.copy().reshape(len(A_Resident),len(A_Invader)) # in plt.contour() Z must be a matrix so reshape z vector into a matrix

########### Elasticity Analysis ###########

minA = 10
maxA = 40


#Best_Alpha = optimize.brentq(POP_DYNAMICS_EA_6, minA, maxA, args=(0.995))

Best_Alpha =  optimize.brentq(pop.pair, minA, maxA, args=(DD6_1, False, 0.9675499)) # True best value of Invader alpha
print("Optimum = " + str(round(Best_Alpha,6)))


# Calculs for plotting elasticity vs Alpha
N_int = 30 # Nos of intervals for plot
Alpha = np.linspace(minA, maxA, N_int)

Elasticity = np.array([])

# Calculs for plot Elasticity vs alpha

for i in range(0, len(Alpha)):
    Elasticity = np.append(Elasticity, pop.pair(Alpha[i], DD6_1, False, 0.995))

# Calculs for plot invasion when resident is optimal
Coeff = np.divide(Alpha, Best_Alpha)

Invasion_coeff = np.zeros(shape = (N_int, 1)) # Pre-allocating space

for i in range(0, N_int):
    Invasion_coeff[i] = pop.pair(Best_Alpha, DD6_1, False, Coeff[i]) 
    
# Calculus for Plot N(t+1) on N(t) for optimum alpha
maxN = 1000 # Number of N
    
N_t = np.linspace(1, maxN, maxN) # Values of N(t)
    
N_tplus1 = np.zeros(shape = (maxN, 1))
    
for i in range(0, maxN):
    
    N_tplus1[i] = DD6_1(Best_Alpha, N_t[i], N_t[i])
     
# Calculs for plotting N(t) on t

N = np.repeat(1, 100) # Allocate memory

for i in range(1, 100):
    N[i] = DD6_1(Best_Alpha, N[i-1], N[i-1])


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

### Population dynamics ###

Invader_A = 10 # Alpha for invader
Coeff = np.divide(Invader_A, Best_Alpha)

pop.pair(Best_Alpha, DD6_1, coeff = Coeff, pair = False, Plot = True)

####################### Multiple invasibility analysis #######################

rd.seed(1234) # Initialize the random number seed

pop.mult(Maxgen = 10000, MaxAlpha = 50, Ninc = 50, DD_fun = DD6_2, n_gen_plot = 3)
