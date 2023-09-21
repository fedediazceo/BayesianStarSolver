import numpy as np
from scipy.interpolate import interp1d, griddata, RegularGridInterpolator

"""
cefiro_interpolation.py

This module contains the class to initialize for the cefiro grid interpolation
Methods
    - init
        Use this to initialize the interpolators. 
        EXAMPLE
            model = InterpolatedModel(cefiro2_all, 1000, ["Teff", "L", "logg"])
            test_input = [0.279526, 2.050196]
            result = model.get_stellar_params(test_input)

    - interpolate_parameters: 
        Use this to interpolate multiple parameters inside the grid.
        The output contains 2 dictionaries, one with the interpolated functions, and another with the parameter space to interpolate to
    - get_stellar_params_interpn
        Use this to get a specific value from the interpolation. 
        Supply it with the interpolated functions from before, and the parameter values from the parameter space
"""

class InterpolatedModel:
    def __init__(self, cefiro2_all, totalValues, parameters):
        """
        Initialize the object
        Input:
            - cefiro2_all (DataFrame): The input data. It's based on the Cefiro data grid. Modify this class if the DF changes
            - totalValues (int): Number of values to interpolate.
            - parameters (list): List of parameters to interpolate. This are keys from the cefiro dictionary that you wish to interpolate
        """
        self.cefiro2_all = cefiro2_all
        self.totalValues = totalValues
        self.parameters = parameters

        # Compute the interpolations and grid points during initialization
        self.interpolated_data, self.new_grid_points = self._interpolate_parameters()

    def _interpolate_parameters(self):
        """
        Interpolates the given parameters based on a set of input data and returns the interpolated values and grid points.

        Input:
        - cefiro2_all (DataFrame): The input data. It's based on the Cefiro data grid. Modify this function if you change the dataframe
        - totalValues (int): Number of values to interpolate.
        - parameters (list): List of parameters to interpolate. This are keys from the cefiro dictionary that you wish to interpolate

        Returns:
        - Tuple: Contains a dictionary with interpolated values for each parameter and a dictionary with grid points.
        """

        # Isolate the MS from the grid
        if len(self.cefiro2_all["Xc"]) > 0:
            max_xc = self.cefiro2_all["Xc"].max()
        else:
            max_xc = 1.0

        # Extract and sort unique mass values from the data
        masses = sorted(list(dict.fromkeys(self.cefiro2_all["M"])))

        # Lists to store results for each mass value
        reg_mass = []
        reg_xcs = []

        # Dictionary to store values for each parameter
        reg_values = {param: [] for param in self.parameters}

        # Loop through each unique mass value
        for M in masses:
            # Filter data for the current mass
            subgrid = self.cefiro2_all.loc[(self.cefiro2_all["M"] == M)]

            # Extract Xc values
            xcs = subgrid["Xc"].tolist()

            # Identify regions in the data for interpolation
            for i in range(len(xcs) - 1):
                if ((xcs[i] - (max_xc - 0.0015)) * (xcs[i + 1] - (max_xc - 0.0015))) < 0:
                    izams = i - 11
                if ((xcs[i] - 1e-4) * (xcs[i + 1] - 1e-4)) < 0:
                    itams = i + 2
                    break

            # Create a linear space for Xc values
            tmp_xcs = np.linspace(1e-4, max_xc - 0.0015, self.totalValues)

            # Loop through each parameter to interpolate values
            for param in self.parameters:
                # Interpolate current parameter values using quadratic interpolation
                f = interp1d(
                    xcs[izams:itams], subgrid[param].tolist()[izams:itams], kind="quadratic", fill_value='extrapolate', bounds_error=False)

                # Store the interpolated values
                reg_values[param].append(f(tmp_xcs))

            # Append current mass and Xc values to the result lists
            reg_mass.append(M * np.ones(self.totalValues))
            reg_xcs.append(tmp_xcs)

        # Initialize a dictionary to store interpolated results for each parameter
        interpolated_values = {}

        # Flatten and combine mass and Xc values
        mass_values = np.concatenate(reg_mass).ravel()
        xc_values = np.concatenate(reg_xcs).ravel()
        original_points = np.vstack([mass_values, xc_values]).T

        # Create a new grid for interpolation
        new_M_points = np.linspace(np.min(mass_values), np.max(mass_values), num=self.totalValues)
        new_xc_points = np.linspace(np.min(xc_values), np.max(xc_values), num=self.totalValues)
        new_points = np.meshgrid(new_M_points, new_xc_points)
        new_points_flat = np.array([new_points[0].ravel(), new_points[1].ravel()]).T

        # Loop through each parameter to populate the interpolated results dictionary
        for param in self.parameters:
            param_values = np.concatenate(reg_values[param]).ravel()
            interpolated_param_values = griddata(original_points, param_values, new_points_flat, method='linear')
            interpolated_values[param] = interpolated_param_values.reshape(new_points[0].shape)

        # Dictionary to store the new grid points
        grid_points = {
            'M_points': new_M_points,
            'xc_points': new_xc_points
        }

        return interpolated_values, grid_points

    def get_stellar_params(self, input_values):
        """
        Returns interpolated stellar parameters based on the input values.
        
        Input:
        - input_values (list): A list containing [M, Xc] for which the parameters are to be interpolated.
        
        Returns:
        - A dictionary containing interpolated values for the input value.
        """
        
        # Create interpolators for each parameter
        interpolators = {}
        for param, values in self.interpolated_data.items():
            interpolators[param] = RegularGridInterpolator(
                (self.new_grid_points['xc_points'], self.new_grid_points['M_points']),
                values, method="linear", bounds_error=False, fill_value=None
            )
        
        # Get the interpolated parameters for the input value
        interpolated_values = []
        for param, interp in interpolators.items():
            interpolated_values.append(interp([input_values[::-1]])[0]) # Extract the single value from the result array
        
        return interpolated_values