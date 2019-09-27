# Scenario 4
# Principle : Addition of age structure and density-dependance which affects the reproductive effort
# Assumptions : Population composed of 2 age classes
#               Fecundity increases with reproductive effort
#               Survival decreases with reproductive effort
#               Fecundity is a negative function of population size

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def DD_FUNCTION_4(ALPHA,BETA,F_DI,Ei,n):
    return(np.multiply(Ei,F_DI)*ALPHA*np.exp(-BETA*n))

def POP_DYNAMICS_4(E) :
    E_resident = E[0]
    E_invader = E[1]
    F_DI = [4,10]
    S_DI = [0.6,0.85]
    ALPHA = 1
    BETA = 2*10**-5
    z=6
    Resident_matrix = np.zeros(shape=(2,2))
    Invader_matrix=np.zeros(shape=(2,2))
    Resident_matrix[0,]= np.multiply(E_resident,F_DI)
    Invader_matrix[0,]= np.multiply(E_invader,F_DI)
    Resident_matrix[1,] = np.multiply((1-E_resident**z),S_DI) 
    Invader_matrix[1,] = np.multiply((1-E_invader**z),S_DI)
    
    # Run generations with resident only
    Maxgen = 20 # Nbs of generations
    n_resident = [1,0] # Initial population vector
    
    for Igen in range(1,Maxgen) : # Iterate over generations
            # Calculate the new entries
            N = sum(n_resident) # Pop size of residents
            Resident_matrix[1,] = DD_FUNCTION_4(ALPHA, BETA, F_DI, E_resident, N) # New Fertility
            n_resident = np.dot(Resident_matrix,n_resident) # New pop vector 
    
    # Introduce invader
    Maxgen = 100 # Set nos of generations to run
    Pop_invader = np.zeros(shape=(Maxgen,1)) # pre-allocate space of pop size
    Pop_invader[0,0] = 1 # Initial population size
    n_invader = [1,0] # Initiate invader pop vector
    for Igen in range(1,Maxgen) : # Iterate over generations
            N = sum(n_resident) + sum(n_invader) # Total pop
            # Apply density dependence to fertilities
            Resident_matrix[0,] = DD_FUNCTION_4(ALPHA,BETA,F_DI,E_resident,N) 
            Invader_matrix[0,] = DD_FUNCTION_4(ALPHA,BETA,F_DI,E_invader,N)
            # Calculate new population vectors
            n_resident = np.dot(Resident_matrix,n_resident)
            n_invader = np.dot(Invader_matrix,n_invader)
            Pop_invader[Igen] = sum(n_invader) # Store pop size of invader
            
    Generation = np.linspace(21,Maxgen,Maxgen-20) # Create vector of generations
    # Get growth of invader starting at generation 20
    Pop=Pop_invader[20:]
    Invasion_model=LinearRegression()
    Invasion_model.fit(Generation.reshape(-1,1),np.log(Pop))
    #Invasion_model = lm(log(Pop_invader[20:Maxgen])~Generation[20:Maxgen])
    # Elasticity = slope of regression
    Elasticity = float(Invasion_model.coef_)
    return(Elasticity)
    
####### MAIN PROGRAM ########
N1 = 30 # Nos of increments
E_Resident = np.linspace(0.2,0.9,N1)
E_Invader = E_Resident.copy()
d = np.array(np.meshgrid(E_Resident, E_Invader)).reshape(2,len(E_Resident)*len(E_Resident)).T # Equivalent to R's expand.grid()
z = np.apply_along_axis(POP_DYNAMICS_4, 1, d) 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contour(E_Resident, E_Invader, z.reshape((N1,N1)), color="black")
plt.show()
