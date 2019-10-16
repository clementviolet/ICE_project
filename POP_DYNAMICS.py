import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.linear_model import LinearRegression

def pair(alpha, DD_fun, pair = True, coeff = None, Maxgen1 = 50, Maxgen2 = 300, Plot = False):
    """
    Population Dynamics function pairwise analysis.
    
    Parameters
    ----------
    alpha: array or int
        If array alpha[0] is the resident trait value and alpha[1] the invader trait value used in PIP analysis
    DD_fun: function
        Density-dependence function used.
    pair: bool
        If True perform pairwise analysis, if False perfom invasibility analysis. In the last case you MUST specify a value for `coeff`parameter.
    coeff: int
        Coefficient for the invader trait. If supplied, an elasticity analysis is performed.
    Maxgen1: int
        Number of generations before invasion.
    Maxgen2: int
        Number of generations after invasion.
    Plot: bool
        If true, show the population dynamics.
    Returns
    -------
    int
        Elasticity coefficient: the rate of growth of the invader population.
    """

    
    if pair:
        Alpha_resident = alpha[0] # Alpha for resident
        Alpha_invader  = alpha[1] # Alpha for invader
    else:
        Alpha_resident = alpha # Alpha for resident
        Alpha_invader  = alpha * coeff # Alpha for invader

    Tot_gen = Maxgen1 + Maxgen2 # Total number of generations
    
    N_resident = np.zeros(shape = (Tot_gen, 1)) # Allocate space
    N_invader  = N_resident.copy()
    
    N_resident[0]        = 1 # Initial number of residents and invaders
    N_invader[Maxgen1-1] = 1
    
    for i in range(1, Maxgen1): # Iterate over only resident
        N_resident[i] = DD_fun(Alpha_resident, N_resident[i-1], N_resident[i-1])
        
    for j in range(Maxgen1, Tot_gen):
        N_total = N_resident[j-1] + N_invader[j-1] # Total population size
        
        # Resident population size
        N_resident[j] = DD_fun(Alpha_resident, N_resident[j-1], N_total)
        
        # Invader population size
        N_invader[j]  = DD_fun(Alpha_invader, N_invader[j-1], N_total)
    
    Generation = np.linspace(1, Tot_gen, Tot_gen)
    
    # Visualisation of the population dynamics
    if Plot:
        
        fig, axes = plt.subplots(1, 2, gridspec_kw={"wspace": 0.5})
        
        axes[0].plot(Generation, N_resident, color = "black")
        axes[0].set(xlabel = "Generation",
                       ylabel = "# Resident")
        
        axes[1].plot(Generation, N_invader, "black")
        axes[1].set(xlabel = "Generation",
                       ylabel = "# Resident")
        
        print("Elasticity coefficient:")


    Nstart = 10 + Maxgen1
    
    #Linear regression
    Invasion_model = LinearRegression()
    Invasion_model.fit(Generation[Nstart:Tot_gen].reshape(-1,1), np.log(N_invader[Nstart:Tot_gen]))
    Elasticity = Invasion_model.coef_

    return(Elasticity)

def mult(Maxgen, MaxAlpha, Ninc,DD_fun, n_gen_plot = 1):
    """
    Population Dynamics function for multiple invasibility.
        
    Parameters
    ----------
    Maxgen: int
        Number of generations run
    MaxAlpha: int
        Maximum value of alpha
    Ninc: int
        Number of classes for alpha
    DD_fun: function
        Density-dependence function used.
    n_gen_plot: int
        Number of generation to plot
            
    Returns
    -------
    Plots of multiple invasibility analysis
        
    """
    
    Stats = np.zeros(shape = (Maxgen, 3)) # Allocate space for statistics

    Store = np.zeros(shape = (Maxgen, Ninc)) # Allocate space to store data for each generation
    
    Data = np.zeros(shape = (Ninc, 2)) # Allocate space for alpha class and population size
    
    Data[23, 1] = 1 # Initial population size and alpha class
    
    Alpha = np.linspace(2, MaxAlpha, Ninc) # Set up alpha
    
    Data[:,0] = Alpha # Place alpha in 1st column
    
    for i in range(0, Maxgen):
        
        N_total = sum(Data[:, 1]) # Total population size
    
        Data[:,1] = np.apply_along_axis(DD_fun, 1, Data, N_total) # New cohort
    
        Store[i,:] = Data[:, 1].copy() # Store values for this generation
    
        # Keep track of population size, mean trait value and SD of trait
        
        S = sum(Data[:, 1]) # Compute population size
        
        SP = sum(np.multiply(Data[:,0], Data[:, 1])) # Compute the sum of the produce of the two columns
        
        Stats[i, 0] = S # Storing population size
        
        Stats[i, 1] = sum(np.multiply(Data[:,0], Data[:, 1])) / S # Mean trait value
        
        SX1 = sum(np.multiply(np.power(Data[:,0], 2), Data[:,1]))
        
        SX2 = np.power(SP, 2) / S
        
        Stats[i, 2] = np.sqrt((SX1 - SX2) / (S-1))
    
    
        # Introduce a mutant by picking a random integer between 1 and 50
        
        Mutant = rd.randrange(0, 50)
        
        Data[Mutant, 1] = Data[Mutant, 1] + 1 # Add mutant to class
    
    Generation = np.linspace(1, Maxgen, Maxgen)
    
    
    if n_gen_plot == 1: # Handling special case of n_gen_plot == 1. Maybe not the best way to handle it, but I'm too tired to find something cleaner
    
        fig, axes = plt.subplots(nrows = 2, ncols = 2, gridspec_kw={"wspace": 0.5, "hspace": 0.7})
        
    elif n_gen_plot % 2 == 1:
        
        # If we have an odd number of alpha vs number to plot, we must remove two of n_gen_plot for avoiding blanks plots
        
        fig, axes = plt.subplots(nrows = 2+(n_gen_plot-2), ncols = 2, gridspec_kw={"wspace": 0.5, "hspace": 0.7})
    
    else:
        
        # Same principle as above, but we just need to remove one to avoid one blank cell.
        
        fig, axes = plt.subplots(nrows = 2+(n_gen_plot-1), ncols = 2, gridspec_kw={"wspace": 0.5, "hspace": 0.7})
        
    
    count = 0 # Count the number of plot done
    i = 0 # indice for number of rows in plot array
    n_r = n_gen_plot # Reverse order of which indice of generation to plot
    
    while count != n_gen_plot:
        
        axes[i,0].plot(Alpha, Store[Maxgen-n_r, :], color = "black")
        axes[i,0].set(xlabel = ("Alpha (Generation = " + str(Maxgen-n_r+1) + ")"),
                      ylabel = "Number")
        
        count += 1 
        n_r   -= 1
        
        if count == n_gen_plot:
            break
        
        axes[i,1].plot(Alpha, Store[Maxgen-n_r, :], color = "black")
        axes[i,1].set(xlabel = ("Alpha (Generation = " + str(Maxgen-n_r+1) + ")"),
                      ylabel = "Number")
        
        i += 1 # Use a new line of plot
        n_r -=1
        count += 1
        
    if n_gen_plot == 1: # Handling special case of n_gen_plot == 1. Maybe not the best way to handle it, but I'm too tired to find something cleaner
        axes[i, 1].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,0], color = "black")
        axes[i, 1].set(xlabel = "Generation", ylabel = "Population size")
        
        axes[i+1, 0].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,1], color = "black")
        axes[i+1, 0].set(xlabel = "Generation", ylabel = "Mean")
    
        axes[i+1, 1].plot(Generation[(Maxgen-101):(Maxgen-1)], Stats[Maxgen-101:Maxgen-1,2], color = "black")
        axes[i+1, 1].set(xlabel = "Generation", ylabel = "SD")
    
    elif i+1 == n_gen_plot: # If the last line of plot is fulled. i + 1 because i started from 0 and n_gen_plot from 1
        
        axes[i, 0].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,0], color = "black")
        axes[i, 0].set(xlabel = "Generation", ylabel = "Population size")
        axes[i, 1].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,1], color = "black")
        axes[i, 1].set(xlabel = "Generation", ylabel = "Mean")
        axes[i+1, 0].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,2], color = "black")
        axes[i+1, 0].set(xlabel = "Generation", ylabel = "SD")
        
    else:
        
        axes[i, 1].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,0], color = "black")
        axes[i, 1].set(xlabel = "Generation", ylabel = "Population size")
        
        axes[i+1, 0].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,1], color = "black")
        axes[i+1, 0].set(xlabel = "Generation", ylabel = "Mean")
    
        axes[i+1, 1].plot(Generation[Maxgen-101:Maxgen-1], Stats[Maxgen-101:Maxgen-1,2], color = "black")
        axes[i+1, 1].set(xlabel = "Generation", ylabel = "SD")
        
    plt.show()
    
    print("Mean alpha in last gen  = ", str(Stats[Maxgen-1,1]))
    print("SD of alpha in last gen = ",str(Stats[Maxgen-1,2]))
