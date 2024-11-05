import matplotlib.pyplot as plt

# Classification sets
import numpy as np

# Breast Cancer

# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [(0.886, 0.896), (0.950, 0.953), (0.649, 0.649)]
set1_values, set2_values = zip(*values)

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, set1_values, width, label='Precision')
bars2 = ax.bar(x + width/2, set2_values, width, label='Accuracy')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Metric Values for Breast Cancer Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Display plot
plt.ylim(0, 1)
plt.savefig('../Figures/breast_bar.svg')
plt.show()

# Glass

# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [(0.715, 0.880), (0.670, 0.880), (0.356, 0.781)]
set1_values, set2_values = zip(*values)

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, set1_values, width, label='Precision')
bars2 = ax.bar(x + width/2, set2_values, width, label='Accuracy')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Metric Values for Glass Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Display plot
plt.ylim(0, 1)
plt.savefig('../Figures/glass_bar.svg')
plt.show()

# Soy

# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [(0.989, 0.990), (0.975, 0.980), (0.41, 0.708)]
set1_values, set2_values = zip(*values)

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, set1_values, width, label='Precision')
bars2 = ax.bar(x + width/2, set2_values, width, label='Accuracy')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Metric Values for Soybean Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Display plot
plt.ylim(0, 1.2)
plt.savefig('../Figures/soy_bar.svg')
plt.show()

# Abalone
# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [0.424, 0.545, 0.766]

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars = ax.bar(x, values, width, label='Relative Mean Squared Error')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Relative Mean Squared Error Values for Abalone Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adding values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Set y-axis limit
plt.ylim(0, 1.2)

# Display plot
plt.savefig('../Figures/abalone_bar.svg')
plt.show()

# Machine
# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [0.233, 1.047, 1.052]

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars = ax.bar(x, values, width, label='Relative Mean Squared Error')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Relative Mean Squared Error Values for Machine Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adding values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Set y-axis limit
plt.ylim(0, 1.3)

# Display plot
plt.savefig('../Figures/machine_bar.svg')
plt.show()


# Forest
# Data
labels = ['0 Hidden Layers', '1 Hidden Layer', '2 Hidden Layers']
values = [1.180, 1.109, 1.109]

# Positions for bars
x = np.arange(len(labels))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots()
bars = ax.bar(x, values, width, label='Relative Mean Squared Error')

# Labels and title
ax.set_ylabel('Values')
ax.set_title('Relative Mean Squared Error Values for Forest Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adding values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Set y-axis limit
plt.ylim(0, 1.4)

# Display plot
plt.savefig('../Figures/forest_bar.svg')
plt.show()
