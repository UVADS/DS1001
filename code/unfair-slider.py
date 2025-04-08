# author: lpa2a
# date: 2025-04-08
# purpose: create an interactive demonstration of the binomial distribution for variable p coin flips

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# set the number of coin flips
flips = 10

# set the initial probability of heads
pheads = 0.5

# create the figure and the histogram subplot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # make room for the slider

# gather initial results of binomial distribution
result = np.random.binomial(n=flips, p=pheads, size=1000)

# plot the initial histogram
hist, bins, patches = ax.hist(result, bins=flips+1, range=(-0.5, flips+0.5), 
                             color='#E57200',     # UVA Orange for bin fill
                             edgecolor='#232D4B', # UVA Navy for bin edges
                             linewidth=2.25)      # Width of the edge lines

# set labels and title with bold and larger font
ax.set_xlabel("Number of Heads", fontsize=14, fontweight='bold')
ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
ax.set_title("1000 trials of 10 unfair coin flips", fontsize=16, fontweight='bold')

# add tick marks on all sides
ax.tick_params(top=True, right=True)
ax.tick_params(direction='in')

# Set explicit tick labels for all integers from 0 to 10 with larger font
ax.set_xticks(range(flips + 1))
ax.set_xticklabels(range(flips + 1), fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# add a slider for the probability
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
p_slider = Slider(
    ax=ax_slider,
    label='Probability of Heads',
    valmin=0.0,
    valmax=1.0,
    valinit=pheads,
    color='#E57200'
)

# function to update the histogram when the slider is moved
def update(val):
    # get the current probability value
    p = p_slider.val
    
    # generate new data with the updated probability
    new_result = np.random.binomial(n=flips, p=p, size=1000)
    
    # clear the current histogram
    ax.clear()
    
    # plot the new histogram
    ax.hist(new_result, bins=flips+1, range=(-0.5, flips+0.5), 
            color='#E57200',     # UVA Orange for bin fill
            edgecolor='#232D4B', # UVA Navy for bin edges
            linewidth=2.25)      # Width of the edge lines
    
    # update labels and title with bold and larger font
    ax.set_xlabel("Number of Heads", fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
    ax.set_title(f"1000 trials of 10 coin flips (p={p:.2f})", fontsize=16, fontweight='bold')
    
    # add tick marks on all sides
    ax.tick_params(top=True, right=True)
    ax.tick_params(direction='in')
    
    # Set explicit tick labels for all integers from 0 to 10 with larger font
    ax.set_xticks(range(flips + 1))
    ax.set_xticklabels(range(flips + 1), fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # redraw the figure
    fig.canvas.draw_idle()

# register the update function with the slider
p_slider.on_changed(update)

plt.show()