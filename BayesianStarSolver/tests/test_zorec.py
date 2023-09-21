from concurrent.futures import ProcessPoolExecutor
from BayesianStarSolver.BayesianEmceeStarSolver.DataRead import cefiro_functions
from BayesianStarSolver.BayesianEmceeStarSolver.DataRead import cefiro_interpolation
from BayesianStarSolver.BayesianEmceeStarSolver.MCMC import MCMCStarSolver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import corner


from BayesianStarSolver.BayesianEmceeStarSolver.DataRead import zorec_functions

# Read the cefiro table, add additional parameters, and show ranges for those parameters

cefiro2_all = cefiro_functions.read_cefiro("BayesianStarSolver/tests/cefiro2-24dec2019.txt")
cefiro_functions.additional_cefiro("BayesianStarSolver/tests/roche_radii", cefiro2_all)
cefiro_functions.show_ranges(cefiro2_all, ["M", "feh", "alfa", "p", "ov", "t"])

# Isolate slices
slice_50_0_0 = cefiro_functions.select_slice(cefiro2_all,50, 0, 0)

# extract original data from the grid
keys = ["Teff", "L"]  
masses = sorted(list(dict.fromkeys(slice_50_0_0["M"])))
extracted_values, extracted_grid_points = cefiro_functions.extract_grid_values(slice_50_0_0, masses, keys)

model = cefiro_interpolation.InterpolatedModel(slice_50_0_0, 1000, ["Teff", "L"])

# test_input = [0.279526, 2.050196]
# result = modelInstance.get_stellar_params(test_input)
# print(result)

zorecValues, zorecErrors, zorecMassValues, zorecR_Masses, zorecVsini, zorecR_vsini  = zorec_functions.read_zorec("BayesianStarSolver/tests/zorecValues.txt")

def run_test(i):
    # emcee specifics
    nwalkers = 32
    M_range = [1.5, 2.5]  # range for M
    Xc_range = [0.0, 0.7]  # corrected range for Xc
    runs = 6000
    burnin = 1000  
    parameters = {}
    parameters["initialValues"] = zorecValues[i]
    parameters["errorValues"] = zorecErrors[i]
    parameters["ranges"] = [M_range, Xc_range]
    
    samples, bestEstimates = MCMCStarSolver.estimateStellarParameters(parameters, nwalkers, runs, burnin, model.get_stellar_params, False)
    
    title = f"Original: M={zorecMassValues[i]}"
    
    return samples, bestEstimates, title

TEST = 5  # Update this value as needed
sampledValues = []
best_estimates_masses = []
titles = []

with ProcessPoolExecutor() as executor:
    for samples, bestEstimates, title in executor.map(run_test, range(TEST)):
        sampledValues.append(samples)
        best_estimates_masses.append(bestEstimates[0])
        titles.append(title)

# Generate corner plots sequentially after all tests are done
for i in range(TEST):
    figure = corner.corner(sampledValues[i], labels=[f"$M$({zorecMassValues[i]})", "$Xc$"], quantiles=[0.16, 0.5, 0.84], show_titles=True, title=titles[i])
    figure.savefig(f"plots/corner_plot_{i}.png")
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

# Using pcolormesh for Teff
plt.figure(figsize=(10,7))
c = plt.pcolormesh(model.new_grid_points["xc_points"], model.new_grid_points["M_points"], model.interpolated_data["Teff"].T, cmap='rainbow')
plt.colorbar(c, label='Teff')
plt.title('Interpolated Teff values')
plt.ylabel('M values')
plt.xlabel('Xc values')
plt.xlim(max(model.new_grid_points["xc_points"]), min(model.new_grid_points["xc_points"]))  # Reversing the x-axis
plt.savefig('plots/Interpolated_Teff_values.png')

# Using pcolormesh for Luminosity
plt.figure(figsize=(10,7))
c = plt.pcolormesh(model.new_grid_points["xc_points"], model.new_grid_points["M_points"], model.interpolated_data["L"].T, cmap='rainbow')
plt.colorbar(c, label='Luminosity')
plt.title('Interpolated Luminosity values')
plt.ylabel('M values')
plt.xlabel('Xc values')
plt.xlim(max(model.new_grid_points["xc_points"]), min(model.new_grid_points["xc_points"]))  # Reversing the x-axis
plt.savefig('plots/Interpolated_Luminosity_values.png')

# Using pcolormesh for logg
# plt.figure(figsize=(10,7))
# c = plt.pcolormesh(model.new_grid_points["xc_points"], model.new_grid_points["M_points"], model.interpolated_data["logg"].T, cmap='rainbow')
# plt.colorbar(c, label='logg')
# plt.title('Interpolated logg values')
# plt.ylabel('M values')
# plt.xlabel('Xc values')
# plt.xlim(max(model.new_grid_points["xc_points"]), min(model.new_grid_points["xc_points"]))  # Reversing the x-axis
# plt.savefig('plots/Interpolated_logg_values.png')

# Plot for Teff values
plt.figure(figsize=(10, 6))
colors_teff = get_color_from_value(extracted_values["Teff"])
scatter_teff = plt.scatter(extracted_grid_points["Xc"], extracted_grid_points["M"], c=extracted_values["Teff"], cmap=cm.rainbow, s=10)
cbar_teff = plt.colorbar(scatter_teff)
cbar_teff.set_label('Teff')
plt.xlabel("Xc")
plt.ylabel("Mass")
plt.xlim(max(extracted_grid_points["Xc"]), min(extracted_grid_points["Xc"]))  # Reversing the x-axis
plt.title("Teff vs Xc and Mass")
plt.savefig('plots/Cefiro_Teff-Xc-M.png')

# Plot for L values
plt.figure(figsize=(10, 6))
colors_L = get_color_from_value(extracted_values["L"])
scatter_L = plt.scatter(extracted_grid_points["Xc"], extracted_grid_points["M"], c=extracted_values["L"], cmap=cm.rainbow, s=10)
cbar_L = plt.colorbar(scatter_L)
cbar_L.set_label('L')
plt.xlabel("Xc")
plt.ylabel("Mass")
plt.xlim(max(extracted_grid_points["Xc"]), min(extracted_grid_points["Xc"]))  # Reversing the x-axis
plt.title("L vs Xc and Mass")
plt.savefig('plots/Cefiro_L-Xc-M.png')