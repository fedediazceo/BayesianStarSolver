import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math
import re
import numpy as np
from BayesianStarSolver.BayesianEmceeStarSolver.DataRead import zorec_functions

def parse_line(line):
    """Parse a line of numbers into a list of floats."""
    return [float(x) for x in line.split()[1:]]

def read_file(filename):
    data = {}
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break  # End of file

            if line.startswith('Model'):
                # Use the entire line as the key, stripping newline characters
                model_key = line.strip()
                mass_line = parse_line(file.readline())
                xc_line = parse_line(file.readline())
                data[model_key] = (mass_line, xc_line)

    return data

def find_closest_model_value(zorec_value, model_data):
    """Find the closest model value and its corresponding model key for a given zorec value."""
    closest_value = None
    closest_model = None
    min_diff = float('inf')

    for model_key, (mass_values, _) in model_data.items():
        for mass_value in mass_values:
            diff = abs(mass_value - zorec_value)
            if diff < min_diff:
                min_diff = diff
                closest_value = mass_value
                closest_model = model_key

    return closest_model, closest_value

def format_model_key(model_key):
    """Format the model key into the specified string format."""
    # Extract numbers from the model key
    numbers = map(int, re.findall(r'-?\d+', model_key))
    alfa, P, Fe_H = numbers
    formattedModel = r"$\alpha=${}, [Fe/H]={}, $P=${}".format(float(alfa/100), float(Fe_H/100), P)
    return formattedModel

def percent_difference(value1, value2):
    if value1 == 0:
        raise ValueError("Value1 cannot be zero since it's the reference value.")
    return ((value2 - value1) / value1) * 100

# Replace 'your_file.txt' with your file's name
filename = 'test_results.txt'
model_data = read_file(filename)
#print(model_data)

zorecValues, zorecErrors, zorecMassValues, zorecR_Masses, zorecVsini, zorecR_vsini  = zorec_functions.read_zorec("BayesianStarSolver/tests/zorecValues.txt")

zorecMassValues = zorecMassValues[0:100]

plt.figure(figsize=(80, 40), dpi=400)

# For each zorecMassValue, find the closest model value and plot them
for i, zorec_value in enumerate(zorecMassValues):
    model_key, closest_value = find_closest_model_value(zorec_value, model_data)
    # Plot zorec value
    plt.plot(i, zorec_value, 'ro')  # 'ro' for red circle
    plt.text(i, zorec_value, f'Zorec Mass: {zorec_value:.4f}', fontsize=8, verticalalignment='top', horizontalalignment='center')
    # Plot closest model value
    plt.plot(i, closest_value, 'bo') # 'bo' for blue circle
    plt.text(i, closest_value, f'{format_model_key(model_key)} - Model Mass: {closest_value:.4f}', fontsize=8, verticalalignment='bottom', horizontalalignment='center')
    

plt.xlabel('Index')
plt.ylabel('Mass')
plt.title('Comparison of Zorec Mass Values with Closest Model Values')
plt.savefig('ZorecModelComparison.png')
plt.close()

plt.figure(figsize=(80, 40), dpi=400)

# For each zorecMassValue, find the closest model value and plot them
for i, zorec_value in enumerate(zorecMassValues):
    model_key, closest_value = find_closest_model_value(zorec_value, model_data)
    # Plot zorec value
    plt.plot(i, zorec_value - closest_value, 'ro')  # 'ro' for red circle
    alignment = 'top' if i%2==0 else 'bottom'
    plt.text(i, zorec_value - closest_value, f'{format_model_key(model_key)} \n Star index: {i} \n Diff: {percent_difference(zorec_value, closest_value):.4f}%', fontsize=8, verticalalignment=alignment, horizontalalignment='center')
   

plt.xlabel('Index')
plt.ylabel('Mass Difference')
plt.title('Comparison of Zorec Mass Values with Closest Model Values')
plt.savefig('ZorecModelDiffComparison.png')

###############################################################################
###########################        PLOTS     ##################################
###############################################################################

# Define a function to map values to colors
def get_color_from_value(values, cmap=cm.rainbow):
    norm = plt.Normalize(min(values), max(values))
    return cmap(norm(values))

# Ensure /plots directory exists or create it
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set the size of the plot and the DPI for high resolution
plt.figure(figsize=(20, 10), dpi=300)  # You can adjust these values as needed

# Plot each model's mass values
for model_key, (mass_values, _) in model_data.items():
    plt.plot(mass_values, label=model_key)

# Plot the zorecMassValues for comparison
plt.plot(zorecMassValues, label='Zorec Mass Values', linestyle='--')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Mass')
plt.title('Mass Comparison')
plt.legend()

# Save the plot as a file
plt.savefig('zorecComparisonValues.png')

# Optionally, close the plot to free up memory
plt.close()