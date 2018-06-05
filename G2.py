import numpy as np

def distance(r1, r2):
    diff = r2-r1
    return np.linalg.norm(diff)

def fc(r_ij, r_c):
    if r_ij < r_c:
        f_c = 0.5 * (np.cos(np.pi * r_ij / r_c) + 1)
    else:
        f_c = 0.0
    return f_c

# Data
xyzs = np.array([[ [0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]])
Zs = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
elements = np.array([1])

# Functions parameters
rs = np.array([0.0, 0.1, 0.2])
eta = 2.0
r_c = 500.0

# Useful numbers
n_samples = xyzs.shape[0]
n_atoms = xyzs.shape[1]

# For the first atom in the first data sample


total_descriptor = []
for sample in range(n_samples):
    sample_descriptor = []
    for main_atom in range(n_atoms):
        atom_descriptor = []
        for rs_value in rs:
                g = 0
                for neighb_atom in range(n_atoms):
                    if main_atom == neighb_atom:
                        continue
                    else:
                        r_ij = distance(xyzs[0, main_atom, :], xyzs[0, neighb_atom, :])
                        cut_off_term = fc(r_ij, r_c)
                        exponent_term = np.exp(-eta * (r_ij - rs_value)**2)
                        g2_term = exponent_term * cut_off_term
                        g += g2_term

                atom_descriptor.append(g)
        sample_descriptor.append(atom_descriptor)
    total_descriptor.append(sample_descriptor)

total_descriptor = np.asarray(total_descriptor)
print(total_descriptor)





