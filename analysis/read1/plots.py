# import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# establish data set

## number of submitted papers
N = 69

## vocab and takeaway dataset
dataV = {
    'top vocabs' : ["jupyter notebook","pandas/numpy","github","preprocess","readme","deep learning"],
    'count' : [30,19,20,35,11,15]
}

dataT = {
    'top takeaways' : ["one language","passion project","form good habits","learn the jargon","emotion reflection"],
    'count' : [19,8,8,7,4]
}

# convert to pandas dataframes
dfV = pd.DataFrame(dataV)
dfT = pd.DataFrame(dataT)
print(dfV,"\n",dfT)

# make plots
ax = dfT.plot.bar(x='top takeaways', y='count', rot=0)
plt.show()