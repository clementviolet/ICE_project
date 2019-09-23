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