#%%[markdown]
## Scenario 2: Adding density dependence
### Solving using $R_0$ as the fitness measure
# Can't do it right now, because I don't understand how optimize.minimize_scalar() works.

### Pairwise invasibility analysis

#%%
import numpy as np
from scipy import linalg, optimize # Library for linear algebra
from sklearn.linear_model import LinearRegression # Linear regression package
import matplotlib.pyplot as plt

#%%
def LESLIE2(x, Maxage):
    """Function to generate the Leslie's matrix"""
    
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
    
    return(Leslie_matrix)

def DD2(alpha, beta, fi, n):
    """Densitydependence function"""
    return(fi*alpha*np.exp(-beta*n))

def POP_DYNAMICS2_1(x):
    
    x_resident = x[0].copy() # Body size of resident population
    x_invader = x[1].copy() # Body size of invader
    
    # Density  dependence parameters
    alpha = 1
    beta = 2*10**-5
    Maxage = 50
    Maxgen = 30 # Number of gen to run the simulation
    
    Resident_matrix = LESLIE2(x_resident, Maxage) # Resident Leslie matrix
    Invader_matrix = LESLIE2(x_invader, Maxage) # Invader leslie matrix
    
    F_resident = Resident_matrix[0,:].copy() # Resident fertility
    F_invader = Invader_matrix[0,:].copy() # Invader fertility
    
    n_resident = np.zeros(shape=(Maxage,1)) # Resident population vector
    n_resident[0] = 1 #initial resident pop size == one
    
    for Igen in range(1,Maxgen): # Iterate over generation
        N = sum(n_resident) # Total pop size
        Resident_matrix[0] = DD2(alpha, beta, F_resident, N)
        #Density Dependent fertility of resident for population at time = Igen
        n_resident = np.dot(Resident_matrix, n_resident)
    
    
    # Introduce invaders
    
    Maxgen = 100 # Number of generations to run the simulation
    n_invader = np.zeros(shape=(Maxage,1))
    n_invader[0] = 1 # Initial number of invader
    Pop_invader = np.array([1]) # Store the total population of invader each generation
    
    for Igen in range(1, Maxgen):
        N = sum(n_resident) + sum(n_invader)
        
        # Density dependent fertility of resident
        Resident_matrix[0,:] = DD2(alpha, beta, F_resident, N)
        # Storing the number of new residents
        n_resident = np.dot(Resident_matrix, n_resident)
        
        # Density dependent fertility of invaders
        Invader_matrix[0,:] = DD2(alpha, beta, F_invader, N)
        # Storing the number of new invaders
        n_invader = np.dot(Invader_matrix, n_invader)
        Pop_invader = np.append(Pop_invader, sum(n_invader))
    
    
    # Now do linear regression of log(Pop.invader) on Generation
    
    Generation = np.arange(1, Maxgen+1)
    Nstart = 19 # Number of generations to ignore (i.e 20 firsts)
    
    # Linera regression
    Invasion_model = LinearRegression()
    Invasion_model.fit(Generation[Nstart:Maxgen].reshape(-1,1), np.log(Pop_invader[Nstart:Maxgen]))
    Elasticity = Invasion_model.coef_

    return(Elasticity)

N1 = 30 # Number of increment

X_Resident = np.linspace(1,3,N1)
X_Invader = X_Resident.copy()

d = (np.array(np.meshgrid(X_Resident, X_Invader))
    .reshape(2,len(X_Resident)*len(X_Resident)).T) # Equivalent to R's expand.grid()
z = np.apply_along_axis(POP_DYNAMICS2_1, 1, d) # Apply POP_DYNAMICS for all X_Resident/X_Invader paired
z_mat = z.copy().reshape(len(X_Resident),len(X_Invader)) # in plt.contour() Z must be a matrix so reshape z vector into a matrix

fig = plt.figure()
ax = fig.add_subplot(111)
CS = ax.contour(X_Resident, X_Invader, z_mat, levels=9, colors="black") # levels = draw *n+1* contour lines
ax.clabel(CS)
ax.set(title="PIP Plot")
plt.show()

#%%[markdown]
### Elasticity analysis

#%%
def POP_DYNAMICS2_2(x, coeff):
    x_resident = x # Body size of resident population
    x_invader = x * coeff # Body size of invader
    
    # Density  dependence parameters
    alpha = 1
    beta = 2*10**-5
    Maxage = 50
    Maxgen = 30 # Number of gen to run the simulation
    
    Resident_matrix = LESLIE2(x_resident, Maxage) # Resident Leslie matrix
    Invader_matrix = LESLIE2(x_invader, Maxage) # Invader leslie matrix
    
    F_resident = Resident_matrix[0,:].copy() # Resident fertility
    F_invader = Invader_matrix[0,:].copy() # Invader fertility
    
    n_resident = np.zeros(shape=(Maxage,1)) # Resident population vector
    n_resident[0] = 1 #initial resident pop size == one
    
    for Igen in range(1,Maxgen): # Iterate over generation
        N = sum(n_resident) # Total pop size
        Resident_matrix[0] = DD2(alpha, beta, F_resident, N)
        #Density Dependent fertility of resident for population at time = Igen
        n_resident = np.dot(Resident_matrix, n_resident)
    
    
    # Introduce invaders
    
    Maxgen = 100 # Number of generations to run the simulation
    n_invader = np.zeros(shape=(Maxage,1))
    n_invader[0] = 1 # Initial number of invader
    Pop_invader = np.array([1]) # Store the total population of invader each generation
    
    for Igen in range(1, Maxgen):
        N = sum(n_resident) + sum(n_invader)
        
        # Density dependent fertility of resident
        Resident_matrix[0,:] = DD2(alpha, beta, F_resident, N)
        # Storing the number of new residents
        n_resident = np.dot(Resident_matrix, n_resident)
        
        # Density dependent fertility of invaders
        Invader_matrix[0,:] = DD2(alpha, beta, F_invader, N)
        # Storing the number of new invaders
        n_invader = np.dot(Invader_matrix, n_invader)
        Pop_invader = np.append(Pop_invader, sum(n_invader))
    
    
    # Now do linear regression of log(Pop.invader) on Generation
    
    Generation = np.arange(1, Maxgen+1)
    Nstart = 19 # Number of generations to ignore (i.e 20 firsts)
    
    # Linera regression
    Invasion_model = LinearRegression()
    Invasion_model.fit(Generation[Nstart:Maxgen].reshape(-1,1), np.log(Pop_invader[Nstart:Maxgen]))
    Elasticity = Invasion_model.coef_

    return(Elasticity)

N_int = 20

X = np.linspace(0.5,3, N_int)

Elasticity = np.array([])

for i in range(0, len(X)):
    Elasticity = np.append(Elasticity, POP_DYNAMICS2_2(X[i], 0.995))

# Calculate the optimum size by calling uniroot

Best_X = optimize.brentq(POP_DYNAMICS2_2, 0.5, 3, args=(0.995))
print("Optimum body size = " + str(round(Best_X,6)))

Coeff = np.divide(X, Best_X)

Invasion_exponent = np.array([])

for i in range(0, len(X)):
    Invasion_exponent = np.append(Invasion_exponent, POP_DYNAMICS2_2(Best_X, Coeff[i]))

#%%
# Plots
fig = plt.figure()

ax = fig.add_subplot(111)

CS = ax.contour(X_Resident, X_Invader, z_mat, levels=9, colors="black")
ax.clabel(CS) # Labels for the countour plot
ax.scatter(Best_X, Best_X, edgecolors='black', facecolors='black',  s = 150) # s = size
ax.set(title="PIP Plot")

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={"wspace": 0.5}) # Space between two plots

axes[0].plot(X, Elasticity, color="black")
axes[0].axhline(y=0, color="black")
axes[0].set(title="Elasticity plot",
            xlabel= "X",
            ylabel="Elasticity")

axes[1].plot(X, Invasion_exponent, color="black")
axes[1].scatter(Best_X, 0, edgecolors='black', facecolors='none', s = 150)
axes[1].set(title="Invasion plot",
            xlabel="X",
            ylabel="Invasion exponent")
plt.show()
