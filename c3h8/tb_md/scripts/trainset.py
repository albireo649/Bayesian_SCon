import sys
import re
import torch

# read a skf file and find positions
def trainX(skf):
    with open(skf, "r") as file:
        lines = file.readlines()[1]

    parameters = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", lines)]
    sub_para=parameters[1:3]
    train_x = torch.tensor(sub_para, dtype=torch.float64)
    print(f'train_x: {train_x}')
    return train_x

def trainY(out):
    with open(out, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):

        if 'Total energy:' in line:
            energy_match = re.search(r'Total energy:\s*([-\d.]+)', line)
            if energy_match:
                total_energy = float(energy_match.group(1))

        if 'Total Forces' in line:
            for sub_line in lines[i+1:]:
                if 'Total stress tensor' in sub_line:
                            break
        
                if re.search(r'[=+]?\d*\.\d+|\d+', sub_line):
                    forces = re.findall(r'[=+]?\d*\.\d+', sub_line)
                    if forces:
                        force = [float(n) for n in forces]
    train_energy = torch.tensor(total_energy, dtype=torch.float64)                        
    train_force = torch.tensor(force, dtype=torch.float64)                    
    return train_energy, train_force
