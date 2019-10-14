#%%[markdown]
## Scénario 1 - Comparing approaches
### Resolution as in chapter 2

#%%
import numpy as np
from scipy import linalg, optimize # Library for linear algebra
import matplotlib.pyplot as plt

#%%
def RCALC(x):
    """Function to compute the fitness"""
    Af = 0
    Bf = 16
    As = 1
    Bs = 0.5
    r = np.log(Af+Bf*x+1)-(As+Bs*x)
    return(-r) # max(f(x)) == min(-f(x))

print("Optimal body size:")
print(optimize.minimize_scalar(RCALC, bounds=[0.1, 3]).x) # There is a small  

#%%[markdown]
# If the fitness function is too complex, we can approach r numericaly

#%%
def SUMMATION(r,x):
    Maxage = 50 # Maximum age
    age = np.arange(1, Maxage+1)
    # Parameter value
    Af = 0
    Bf = 16
    As = 1
    Bs = 0.5
    m = np.repeat(Af+Bf*x, repeats=Maxage) # Number of female offspring
    l = np.exp(-(As+Bs*x)*age) # Surival to age
    Sum = sum(np.exp(-r*age)*l*m) #Characteristic equation sum
    return(1-Sum)

# # Define function to find r given x
def RCALC(x):
    return(-1*optimize.brentq(SUMMATION, -1, 2, args=(x))) # -1 to find the minimum of the function
# - 1 is mandatory, if missing it crash. See RCALC2 for more info

# def RCALC2(x):
    # return(optimize.brentq(SUMMATION, -1, 2, args=(x))) # -1 to find the minimum of the function

print("Approached optimal body size:")
optimize.minimize_scalar(RCALC, bracket=[0.1,3], method='Golden').x # bracket = interval; method='Golden' is the method used by R stats::optimize()
# optimize.minimize_scalar(RCALC2, bracket=[0.1,3], method='Golden') # bracket = interval; method='Golden' is the method used by R stats::optimize()


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
    return(-r) # Maxium of a function is given by the minimum of a -f(x)

Optimum = optimize.minimize_scalar(RCALC, bracket=[1,3], method='Golden') # bracket = interval; method='Golden' is the method used by R stats::optimize()
Best_x = Optimum.x
Best_r = Optimum.fun*-1 # Due to max(f(x)) == min(-f(x))

# # Print out results to 6 significant digits
print("Optimum body size = " + str(round(Optimum.x,6))+"\nMaximum R = " + str(round(-1*Optimum.fun,6)))


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