import math

def read_zorec(filepath):
    """
    Reads the specified file and extracts relevant Zorec values and errors.

    Input:
    - filepath: Path to the Zorec data file.

    Returns:
    - tuple: A tuple containing zorec T,L values, zorec T,L Errors, zorecMassValues, zorecR_Masses, zorecVsini and zorecR_vsini
    """
    
    # Lists to store the extracted values
    spectralType, logT_L_pairs, e_logT_e_logL_pairs, zorecMassValues, zorecE_Masses, zorecVsini, zorecR_vsini, zorecAgeValue, zorecE_Age = [], [], [], [], [], [], [], [], []

    # Process the file
    with open(filepath, "r") as file:
        for line in file:
            # Skip header lines
            if line.startswith("#") or line.startswith("-") or len(line.strip()) == 0:
                continue

            # Split line into parts
            parts = line.split("|")

            # Check if mass is within the desired range
            mass = float(parts[6].strip())
            if 1.5 <= mass <= 2.5:

                # Append values to respective lists
                logT = float(parts[2].strip())
                e_logT = float(parts[3].strip())
                logL = float(parts[4].strip())
                e_logL = float(parts[5].strip())

                logT_L_pairs.append((logT, logL))
                e_logT_e_logL_pairs.append((e_logT, e_logL))

                zorecMassValues.append(mass)
                zorecE_Masses.append(float(parts[7].strip()))

                zorecAgeValue.append(float(parts[8].strip()))
                zorecE_Age.append(float(parts[9].strip()))

                zorecVsini.append(float(parts[10].strip()))
                zorecR_vsini.append(float(parts[11].strip()))

                spectralType.append(parts[1].strip())

    # Calculate the converted values using the given formulas
    zorecValues = [(10**logT, 10**logL) for logT, logL in logT_L_pairs]

    zorecErrors = [
        (
            math.log(10) * 10**logT * deltaLogT, 
            math.log(10) * 10**logL * deltaLogL
        ) 
        for (logT, logL), (deltaLogT, deltaLogL) in zip(logT_L_pairs, e_logT_e_logL_pairs)
    ]

    return spectralType, zorecValues, zorecErrors, zorecMassValues, zorecE_Masses, zorecVsini, zorecR_vsini, zorecAgeValue, zorecE_Age