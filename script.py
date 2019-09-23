#%% [markdown]
#
## Introduction to Computational Ecology
# This Jupyter notbebook is the place where all the chapter of Roff's book about
# invasibility analysis is translated from R to Python.

#%%
import pandas as pd
import numpy as np
from scipy import linalg, sparse # Library for linear algebra
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

fig, axes = plt.subplots(nrows=2, ncols=2)

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