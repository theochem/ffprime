import re

def parse_gaussian_connectivities(logfile):
    """Extract bonds and angles with equilibrium values from a Gaussian .log file."""
    with open(logfile, "r") as f:
        text = f.read()

    # Find the Optimized Parameters section
    matches = re.findall(r"Optimized Parameters[\s\S]+?(-{10,}[\s\S]+?)(?:\n\s*\n|\Z)", text)
    if not matches:
        raise ValueError("Could not find Optimized Parameters section in log file.")

    section = matches[-1]  # Use the last occurrence

    # Pattern: R(i,j) <spaces> value
    bond_pattern = re.compile(r"R\((\d+),(\d+)\)\s+([-\d\.Ee]+)")
    angle_pattern = re.compile(r"A\((\d+),(\d+),(\d+)\)\s+([-\d\.Ee]+)")

    bonds = []
    angles = []


    for b in bond_pattern.finditer(section):
        i, j = int(b.group(1))-1, int(b.group(2))-1
        bonds.append([i, j, float(b.group(3))])  
    for a in angle_pattern.finditer(section):
        i, j, k = int(a.group(1))-1, int(a.group(2))-1, int(a.group(3))-1
        angles.append([i, j, k, float(a.group(4))])  

    return bonds, angles


# Example usage:
#bonds, angles= parse_gaussian_connectivities("bonding/lig.log")
#print("Bonds:", bonds)
#print("Angles:", angles)