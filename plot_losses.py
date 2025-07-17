import matplotlib.pyplot as plt
import numpy as np # Used for easier handling if you parse data later

# --- Your Data ---
# Manually extracted from your sample output
# For a real paper, you'd ideally have data for every epoch or more frequent intervals
# if your script logged it. This example uses the provided 200-epoch intervals.

epochs_data = [200, 400, 600, 800, 1000]

# Note: The 'Content Loss' and 'Style Loss' printed in your log are likely
# already multiplied by alpha and beta respectively, or they are the raw losses
# before multiplication. The 'Total Loss' is alpha*content + beta*style.
# For plotting, it's often insightful to plot the *unweighted* content and style losses
# if you have them, or clearly label what is being plotted.

# Assuming the printed "Content Loss" and "Style Loss" are ALREADY WEIGHTED
# by alpha and beta respectively as suggested by your print format:
# "Content Loss: {alpha*current_content_loss_tensor.item():.4f}"
# "Style Loss: {beta*current_style_loss_tensor.item():.4f}"

weighted_content_losses = [
    7.2861,          # Epoch 200
    8.0017,          # Epoch 400
    8.3826,          # Epoch 600
    8.6521,          # Epoch 800
    8.8519           # Epoch 1000
]

weighted_style_losses = [
    90925175000.0000, # Epoch 200
    40000334375.0000, # Epoch 400
    25931417187.5000, # Epoch 600
    18973846875.0000, # Epoch 800
    14756146875.0000  # Epoch 1000
]

total_losses = [
    90925178880.0000, # Epoch 200
    40000335872.0000, # Epoch 400
    25931417600.0000, # Epoch 600
    18973847552.0000, # Epoch 800
    14756147200.0000  # Epoch 1000
]

# --- Plotting ---
plt.figure(figsize=(12, 8)) # Adjust figure size as needed

# Plot Total Loss
plt.plot(epochs_data, total_losses, label='Total Loss ($\mathcal{L}_{total}$)', color='red', marker='o', linestyle='-')

# Plot Weighted Content Loss
plt.plot(epochs_data, weighted_content_losses, label=r'Weighted Content Loss ($\alpha \mathcal{L}_{content}$)', color='blue', marker='s', linestyle='--')

# Plot Weighted Style Loss on a secondary y-axis if scales are very different (which they are here)
# However, since Total Loss is dominated by Style Loss, plotting them on the same primary axis but
# clearly labeling them is also an option. Let's try a secondary axis for style loss.

fig, ax1 = plt.subplots(figsize=(12, 8)) # Create figure and primary axis

color_total = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss / Weighted Content Loss', color='black') # Label for primary y-axis
ax1.plot(epochs_data, total_losses, color=color_total, marker='o', linestyle='-', label='Total Loss ($\mathcal{L}_{total}$)')
ax1.plot(epochs_data, weighted_content_losses, color='tab:blue', marker='s', linestyle='--', label=r'Weighted Content Loss ($\alpha \mathcal{L}_{content}$)')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

# Create a secondary y-axis for the weighted style loss because its scale is vastly different
ax2 = ax1.twinx() 
color_style = 'tab:green'
ax2.set_ylabel(r'Weighted Style Loss ($\beta \mathcal{L}_{style}$)', color=color_style)  # We already handled the x-label with ax1
ax2.plot(epochs_data, weighted_style_losses, color=color_style, marker='^', linestyle=':', label=r'Weighted Style Loss ($\beta \mathcal{L}_{style}$)')
ax2.tick_params(axis='y', labelcolor=color_style)

# Adding title
plt.title('Loss Curves During Neural Style Transfer', fontsize=16)

# Adding legends
# For combined axes, getting legends together can be tricky. One way:
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.tight_layout() # Otherwise the right y-label is slightly clipped
plt.savefig('loss_curves_stylization.png', dpi=300) # Save the figure
plt.show()

print("Loss curve graph saved as loss_curves_stylization.png")