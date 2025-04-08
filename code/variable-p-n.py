# author: lpa2a
# date: 2025-04-08
# purpose: create an interactive demonstration of the binomial distribution for variable p and n

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# set the initial number of coin flips
flips = 10

# set the initial probability of heads
pheads = 0.5

# set the initial number of trials
trials = 1000

# create the figure and the histogram subplot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.45)  # make room for all three sliders

# gather initial results of binomial distribution
result = np.random.binomial(n=flips, p=pheads, size=trials)

# plot the initial histogram
hist, bins, patches = ax.hist(result, bins=flips+1, range=(-0.5, flips+0.5), 
                             color='#E57200',     # UVA Orange for bin fill
                             edgecolor='#232D4B', # UVA Navy for bin edges
                             linewidth=2.25)      # Width of the edge lines

# set labels and title with bold and larger font
ax.set_xlabel("Number of Heads", fontsize=14, fontweight='bold')
ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
ax.set_title(f"{trials} trials of {flips} unfair coin flips", fontsize=16, fontweight='bold')

# add tick marks on all sides
ax.tick_params(top=True, right=True)
ax.tick_params(direction='in')

# Set explicit tick labels for all integers from 0 to 10 with larger font
ax.set_xticks(range(flips + 1))
ax.set_xticklabels(range(flips + 1), fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# add a slider for the probability
ax_slider_p = plt.axes([0.25, 0.3, 0.65, 0.03])
p_slider = Slider(
    ax=ax_slider_p,
    label='Probability of Heads',
    valmin=0.0,
    valmax=1.0,
    valinit=pheads,
    color='#E57200'
)

# add a slider for the number of flips
ax_slider_n = plt.axes([0.25, 0.2, 0.65, 0.03])
n_slider = Slider(
    ax=ax_slider_n,
    label='Number of Flips',
    valmin=1,
    valmax=100,  # Increased from 20 to 100
    valinit=flips,
    valstep=1,  # Only allow integer values
    color='#232D4B'
)

# add a slider for the number of trials
ax_slider_trials = plt.axes([0.25, 0.1, 0.65, 0.03])
trials_slider = Slider(
    ax=ax_slider_trials,
    label='Number of Trials',
    valmin=1,
    valmax=10000,
    valinit=trials,
    valstep=1,  # Only allow integer values
    color='#00A3B4'  # UVA Teal
)

# function to update the histogram when any slider is moved
def update(val):
    # get the current probability, number of flips, and number of trials values
    p = p_slider.val
    n = int(n_slider.val)  # Convert to integer
    t = int(trials_slider.val)  # Convert to integer
    
    # generate new data with the updated parameters
    new_result = np.random.binomial(n=n, p=p, size=t)
    
    # clear the current histogram
    ax.clear()
    
    # plot the new histogram
    ax.hist(new_result, bins=n+1, range=(-0.5, n+0.5), 
            color='#E57200',     # UVA Orange for bin fill
            edgecolor='#232D4B', # UVA Navy for bin edges
            linewidth=2.25)      # Width of the edge lines
    
    # update labels and title with bold and larger font
    ax.set_xlabel("Number of Heads", fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
    ax.set_title(f"{t} trials of {n} coin flips (p={p:.2f})", fontsize=16, fontweight='bold')
    
    # add tick marks on all sides
    ax.tick_params(top=True, right=True)
    ax.tick_params(direction='in')
    
    # Set explicit tick labels for all integers from 0 to n with larger font
    # For larger values of n, we'll use fewer tick labels to avoid overcrowding
    if n <= 20:
        ax.set_xticks(range(n + 1))
        ax.set_xticklabels(range(n + 1), fontsize=12)
    else:
        # For n > 20, use fewer tick marks (every 5 or 10)
        step = 5 if n <= 50 else 10
        ax.set_xticks(range(0, n + 1, step))
        ax.set_xticklabels(range(0, n + 1, step), fontsize=12)
    
    ax.tick_params(axis='y', labelsize=12)
    
    # redraw the figure
    fig.canvas.draw_idle()

# register the update function with all sliders
p_slider.on_changed(update)
n_slider.on_changed(update)
trials_slider.on_changed(update)

plt.show()