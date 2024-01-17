import numpy as np
import pandas as pd
import BayesianStarSolver.BayesianEmceeStarSolver.DataRead.variables as vr
import matplotlib.pyplot as plt
import os

"""
cefiro_functions.py

This module contains the methods to read the cefiro data grid

read_cefiro: 
    reads the table file and returns the grid with some columns renamed for consistency

additional_cefiro:
    this adds crucial information to the grid using a rochefile

show_ranges: 
    this will show ranges of specific parameters of the grid

select_slice:
    With this method you can select a specific slice from the grid, by locking alfa, p, and feh parameters

plot_MS_rough:
    Use this to plot the main sequence values from the grid for all parameters

extract_grid_values:
    Use this method in combination with a list of Masses, and specific keys from the grid, to extract the original 
    values from it. Good for checking data and plotting
"""

##################################################################################

def read_cefiro(table_file):
    """
    Read the CEFIRO2 table built by Antonio
    A few variables get renamed

    Input:  table_file  (string)    path to the ASCII file containing data
    Output: grid        (pandas df) contents of the grid
    """
    try:
        grid = pd.read_csv(table_file, delim_whitespace=True)
    except FileNotFoundError:
        full_path = os.path.abspath(table_file)
        print(f"File not found: {full_path}")


    grid = grid.rename(
        columns={"F0": "n1l0", "F1": "n2l0"}
    )  # Here we rename columns F0 and F1 for consistency
    return grid

def additional_cefiro(rochefile, grid):
    """
    Add crucial information to grid

    Input:  grid        (pandas df) contents of the grid
            rochefile   (string)    path to a Roche model ASCII file
    Output: none                    grid is updated with new data
    """
    roche_model = pd.read_csv(
        rochefile,
        skiprows=1,
        delim_whitespace=True,
        header=None,
        names=["Ok", "Rp_e", "Rm_e", "Vol"],
    )
    # These are the definitions of the constants
    solarzx = 0.0245
    ypr = 0.235
    dydz = 2.2

    # First, computation of Z, then Y and X
    grid["Z0"] = (1.0 - ypr) / ( (1.0 / (solarzx * 10 ** (grid["feh"] / 100))) + (dydz + 1.0) )
    #dydz galactic enrichment
    #here there is an assumption that every metal behaves as iron
    
    # Y = dydz*Z+ypr
    grid["X0"] = 1.0 - ypr - grid["Z0"] * (1 + dydz)

    grid["O"] = 2 * np.pi * grid["nu_rots"] * 1e-6
    # Angular rotation frequency (rad/s)

    grid["Rp"] = grid["R"] / ( 1 + grid["O"]**2 * (grid["R"] * vr.Rsun)**3 / (3 * vr.G * grid["M"] * vr.Msun) )
    # (23) Pérez-Hernández et al. (1999)

    grid["O_c"] = np.sqrt( 8 * vr.G * grid["M"] * vr.Msun / (3 * grid["Rp"] * vr.Rsun) ** 3 )
    # (2) Pérez-Hernández et al. (1999).
    # This is an approximation that assumes that Rp does not change with omega

    grid["omega_c"] = grid["O"] / grid["O_c"]
    # Omega/Omega_c (1) Pérez-Hernández et al. (1999)

    # Computing Re is a bit more complicated.
    # We use the exact calculus through Cardanus' method or the iterative approximative method
    # Using the iterative method is simplier but will it be faster?
    Re = (1 + grid.omega_c**2 / 2) * grid["Rp"]
    Re_0 = grid.Rp
    while np.max(np.abs(Re_0 - Re)) > 1e-5:
        Re_0 = Re
        O_k = np.sqrt(vr.G * grid["M"] * vr.Msun / (Re * vr.Rsun) ** 3)
        omega_k = grid.O / O_k
        Re = (1 + omega_k**2 / 2) * grid["Rp"]

    grid["O_k"] = O_k
    grid["omega_k"] = omega_k #on a different scale
    grid["Re"] = Re
    # grid['Re'] = (1 + grid['omega']**2 / 2) * grid['Rp']
    # (38) Paxton et al. (2019)  !!! NOT WITH OMEGA_C

    grid["rho_spheroid"] = ( grid["M"] * vr.Msun
        / (4 / 3 * np.pi * grid["Rp"] * grid["Re"] ** 2 * vr.Rsun**3) )
    # Another approximation.
    # This is an spheroid, differs from Roche volume at omega > 0.6 (Paxton et al., 2019)

    grid["rho_ss"] = ( grid["M"] * vr.Msun / (4 / 3 * np.pi * grid["R"] ** 3 * vr.Rsun**3) )
    # This is the same as cefiro1_all['rho']*vr.rho_sun*4/3*np.pi,
    # because an incorrect calculus of 'rho' in previous versions of filou_file.py

    grid["vrot"] = grid["O"] * grid["Re"] * vr.Rsun * 1e-5

    grid["Rmean"] = np.interp(grid.omega_k, roche_model.Ok, roche_model.Rm_e) * grid.Re
    

    grid["rho_roche"] = (
        grid["M"] * vr.Msun / (4 / 3 * np.pi * grid["Rmean"] ** 3 * vr.Rsun**3)
    )
    # Mean density of a Roche model

    for i in range(1, 9):
        grid["Q" + str(i)] = np.sqrt(grid["rho_roche"] / vr.rho_sun) / (
            grid["n" + str(i) + "l0"] * 0.0864
        )
        grid["F0" + str(i - 1)] = grid["n1l0"] / grid["n" + str(i) + "l0"]

####################################################################################################3
def show_ranges(grid, parameters):
    """
    Print to screen the parameter ranges used in the grid

    Input:  grid        (pandas df)         contents of the grid
            parameters  (list of strings)   list of parameters used to build the grid
    Output: none
    """
    for param in parameters:
        print("Range for {}:".format(param))
        print(sorted(list(dict.fromkeys(grid[param]))))

####################################################################################################3
def select_slice(grid, current_alfa, current_p, current_feh):
    """
    Select a slice from the grid, using alfa, p and feh values, and isolate the MS

    Input:  grid        (pandas df)         contents of the grid
    Output: a slice of the grid
    """

    print("Slice: {} {} {}".format(current_alfa, current_p, current_feh))

    # Isolate a slice
    subgrid = grid.loc[  (grid["alfa"] == current_alfa)
                        & (grid["p"] == current_p)
                        & (grid["feh"] == current_feh)]

    # Isolate the MS
    if len(subgrid["Xc"]) > 0:
        max_xc = subgrid["Xc"].max()
    else:
        max_xc = 1.0

    print("MAX XC for this slice: ", max_xc, " out of ", len(subgrid["Xc"]))
        
    #subgrid = subgrid.loc[(subgrid["Xc"] < max_xc - 1e-2) & (subgrid["Xc"] > 1e-4)]

    return subgrid


####################################################################################################3
def plot_MS_rough(grid, plotdir, estimateGridDensity = False):
    """
    Plot an HRD and a Xc-M plane for each combination of alphaMLT, [Fe/H] and rotation period

    Input:  grid        (pandas df)         contents of the grid
            plot_dir    (string)            path to directory to save plots
    Output: none
    """
    # Select a specific "slice" of the grid
    for current_alfa in sorted(list(dict.fromkeys(grid["alfa"]))):
        for current_p in sorted(list(dict.fromkeys(grid["p"]))):
            for current_feh in sorted(list(dict.fromkeys(grid["feh"]))):

                print("We do {} {} {}".format(current_alfa, current_p, current_feh))

                # Isolate a slice
                subgrid = grid.loc[  (grid["alfa"] == current_alfa)
                                  & (grid["p"] == current_p)
                                  & (grid["feh"] == current_feh)]

                # Isolate the MS
                if len(subgrid["Xc"]) > 0:
                    max_xc = subgrid["Xc"].max()
                else:
                    max_xc = 1.0
                print("MAX XC", max_xc, "out of", len(subgrid["Xc"]))
                subgrid = subgrid.loc[(subgrid["Xc"] < max_xc - 1e-2) & (subgrid["Xc"] > 1e-4)]

                plt.scatter(np.log10(subgrid["Teff"]),
                            np.log10(subgrid["L"]), c=subgrid["M"])
                plt.gca().invert_xaxis()
                plt.title(r"$\alpha=${}, [Fe/H]={}, $P=${}".format(
                        float(current_alfa) / 100, float(current_feh) / 100, current_p) )
                plt.xlabel(r"$\log( T_{\rm eff} /{\rm K} )$")
                plt.ylabel(r"$\log L/L_\odot$")
                cbar = plt.colorbar()
                cbar.set_label(r"$M/M_\odot$")
                plt.savefig("{}/HRD_alfa{}_feh{}_p{}.jpg".format(
                        plotdir, current_alfa, current_feh, current_p),
                        dpi=300)
                plt.close()

                validGrid = ""
                if(estimateGridDensity == True):
                    if estimate_grid_density(subgrid, "{}/rectangle_alfa{}_feh{}_p{}.jpg".format(
                        plotdir, current_alfa, current_feh, current_p)):
                        validGrid = "_ISVALID"

                plt.scatter(subgrid["Xc"], subgrid["M"], c=subgrid["R"])
                plt.gca().invert_xaxis()
                plt.title(r"$\alpha=${}, [Fe/H]={}, $P=${}".format(
                        float(current_alfa) / 100, float(current_feh) / 100, current_p) )
                plt.xlabel(r"$X_c$")
                plt.ylabel(r"$M/M_\odot$")
                cbar = plt.colorbar()
                cbar.set_label(r"$R/R_\odot$")
                plt.savefig("{}/rectangle_alfa{}_feh{}_p{}{}.jpg".format(
                        plotdir, current_alfa, current_feh, current_p, validGrid),
                        dpi=300)
                plt.close()

####################################################################################################3
def get_valid_density_parameters(grid):
    """
    Get valid parameters from the model that have a good density of values

    Input:  grid        (pandas df)         contents of the grid
    Output: grid parameters list tuples (list[tuple])         alfa, p, feh 
    """

    parametersList = []

    # Select a specific "slice" of the grid
    for current_alfa in sorted(list(dict.fromkeys(grid["alfa"]))):
        for current_p in sorted(list(dict.fromkeys(grid["p"]))):
            for current_feh in sorted(list(dict.fromkeys(grid["feh"]))):

                #print("We do {} {} {}".format(current_alfa, current_p, current_feh))

                # Isolate a slice
                subgrid = grid.loc[  (grid["alfa"] == current_alfa)
                                  & (grid["p"] == current_p)
                                  & (grid["feh"] == current_feh)]

                # Isolate the MS
                if len(subgrid["Xc"]) > 0:
                    max_xc = subgrid["Xc"].max()
                else:
                    max_xc = 1.0
                #print("MAX XC", max_xc, "out of", len(subgrid["Xc"]))
                subgrid = subgrid.loc[(subgrid["Xc"] < max_xc - 1e-2) & (subgrid["Xc"] > 1e-4)]

                validGrid = ""
                if estimate_grid_density(subgrid, "rectangle_alfa{}_feh{}_p{}.jpg".format(
                    current_alfa, current_feh, current_p)):
                    validParameters = current_alfa, current_p, current_feh
                    parametersList.append(validParameters)
    
    return parametersList

def estimate_grid_density(subgrid, gridName):
    # Define the Xc range and the desired coverage threshold
    xc_range = (0.0, 0.7)
    coverage_threshold_xc = 0.8  # 80%

    countCoverage = 0
    massLength = 0
    # Iterate over each unique mass value
    for mass in subgrid['M'].unique():
        xc_values = subgrid[subgrid['M'] == mass]['Xc']
        xc_coverage = np.histogram(xc_values, bins=100, range=xc_range)[0] > 0
        coverage = np.sum(xc_coverage) / len(xc_coverage)
        #print(f"Mass {mass}: Coverage = {coverage*100:.2f}%")
        massLength = massLength + 1
        if coverage >= coverage_threshold_xc:
            countCoverage = countCoverage + 1
 
    if massLength > 0 and countCoverage/massLength >= 0.8:
        #print("{} Is Valid {}".format(gridName,countCoverage/massLength))
        return True
    else:
        #print("{} Is NOT Valid".format(gridName))
        return False

########################################################################################################

def extract_grid_values(cefiro2_all, masses, keys):
    """
    Extract grid values from the cefiro2_all DataFrame based on the provided keys.
    
    Parameters:
    - cefiro2_all (DataFrame): The source data containing the various stellar parameters.
    - masses (list): A list of unique masses to filter the grid.
    - keys (list): The list of keys (column names) from cefiro2_all for which to extract the values.
    
    Returns:
    - values_dict (dict): A dictionary containing extracted values for each provided key.
    - grid_points (dict): A dictionary containing corresponding 'M' (mass) and 'Xc' (central mass fraction) values.
    """
    
    # Isolate the MS from the grid
    if len(cefiro2_all["Xc"]) > 0:
        max_xc = cefiro2_all["Xc"].max()
    else:
        max_xc = 1.0
    
    # Initialize an empty dictionary to store the results
    values_dict = {key: [] for key in keys}
    grid_points = {"M": [], "Xc": []}

    itams = None
    izams = None

    for M in masses:
        subgrid = cefiro2_all.loc[(cefiro2_all["M"] == M)]
        xcs = subgrid["Xc"].tolist()

        for i in range(len(xcs) - 1):
            if ((xcs[i] - (max_xc - 0.0015)) * (xcs[i + 1] - (max_xc - 0.0015))) < 0:
                izams = i - 11
            if ((xcs[i] - 1e-4) * (xcs[i + 1] - 1e-4)) < 0:
                itams = i + 2
                break
        
        if izams is None:
            izams = 0
        
        if itams is None:
            itams = len(xcs) - 1
        
        #print("for mass {} xc length {} izams {} - {},itams {} - {}".format(M,len(xcs), izamsCount, izams, itamsCount, itams))        
        # Extract values for each key and add to the dictionary
        for key in keys:
            values = subgrid[key].tolist()[izams:itams]
            values_dict[key].extend(values)
        
        xcs_range = xcs[izams:itams]
        
        grid_points["M"].extend([M] * len(xcs_range))
        grid_points["Xc"].extend(xcs_range)
    
    return values_dict, grid_points

######################################################################################

