# import torch
import random
import sys
import os
import re

Skf_File = os.path.join("../corrupt_pp", "C-C.skf")
Result_File = os.path.join("..","detailed.out")
Data_Dir = "../data"
Initial_Out = os.path.join(Data_Dir, "initial_dataset.out")

N_SAMPLES=10
LOWER = -0.6
UPPER = 0.0

def read(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines

# read the parameters from skf files
lines_skf=read(Skf_File)
line = lines_skf[1]
parameters = list(re.finditer( r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", line ))
Ep = parameters[1]
Es = parameters[2]
SPE= parameters[3]

# read the output data from Result_File
lines_result = read(Result_File)

for i, line in enumerate(lines_result):

    if 'Total energy:' in line:
        energy_match = re.search(r'Total energy:\s*([-\d.]+)', line)
        if energy_match:
            total_energy = float(energy_match.group(1))

    if 'Total Forces' in line:
        for sub_line in lines_result[i+1:]:
            if 'Total stress tensor' in sub_line:
                        break
    
            if re.search(r'[=+]?\d*\.\d+|\d+', sub_line):
                forces = re.findall(r'[=+]?\d*\.\d+', sub_line)
                if forces:
                    force = [float(n) for n in forces]

# tensor form for torch
#parameter_tensor = torch.tensor([Ep,Es,SPE, total_energy, force[0], force[1], force[2]], dtype = torch.float64)
parameter_tensor = [Ep,Es,SPE, total_energy, force[0], force[1], force[2]]

# join the tensors in the initial file
if not os.path.exists(Initial_Out):
    with open(Initial_Out, "w") as f:
        f.write("Ep Es SPE Energy Fx Fy Fz\n")

with open(Initial_Out, "a") as f:
    f.write(
        " ".join(
            f"{x.item():.10f}"
            for x in parameter_tensor
        )
        + "\n"
    )

#create random 
Ep = random.uniform(LOWER, UPPER)
Es = random.uniform(LOWER, UPPER)
SPE = random.uniform(LOWER, UPPER)

replacements= [f"{Ep:.10f}", f"{Es:.10f}", f"{SPE:.10f}"]

for parameter, replacement in reversed(list(zip(parameters[1:4], replacements))): 
        start, end = parameter.span()
        line = line[:start] + replacement + line[end:]
lines_skf[1] = line

with open(Skf_File, "w") as f:
    f.writelines(lines_skf)






