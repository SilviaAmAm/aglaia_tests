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

def get_costheta(xyz_i, xyz_j, xyz_k):
    r_ij = xyz_j - xyz_i
    r_ik = xyz_k - xyz_i
    numerator = np.dot(r_ij, r_ik)
    denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_ik)
    costheta = numerator/denominator
    return costheta


# Data
xyzs = np.array([[ [0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]])

Zs_list = [[7, 2, 1, 1]]

elements_list = [1, 2, 7]
element_pairs_list = [[1, 1], [2, 1], [7, 1], [7, 2]]

# Parameters
radial_cutoff = 500.0
angular_cutoff = 500.0
radial_rs = [0.0, 0.1, 0.2]
angular_rs = [0.0, 0.1, 0.2]
theta_s = [3.0, 2.0]
zeta = 3.0
eta = 2.0


old_settings = np.seterr(all='raise')

# Useful numbers
n_samples = xyzs.shape[0]
n_atoms = xyzs.shape[1]


# Mother of for loops

total_descriptor = []
for sample in range(n_samples):
    sample_descriptor = []
    for i in range(n_atoms):                            # Loop over main atom
        atom_descriptor = []
        for eta_value in eta:                           # Loop over parameters
            for zeta_value in zeta:
                g_sum = 0
                for j in range(n_atoms):                # Loop over 1st neighbour
                    if j == i:
                        continue
                    for k in range(n_atoms):            # Loop over 2nd neighbour
                        if k == j or k == i:
                            continue

                        r_ij = distance(xyzs[sample, i, :], xyzs[sample, j, :])
                        r_ik = distance(xyzs[sample, i, :], xyzs[sample, k, :])
                        cos_theta_ijk = get_costheta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])

                        term_1 = np.power((1.0+lam*cos_theta_ijk), zeta_value)
                        term_2 = np.exp(- eta_value * (r_ij**2 + r_ik**2))
                        term_3 = fc(r_ij, r_c) * fc(r_ik, r_c)
                        g_term = term_1 * term_2 * term_3
                        g_sum += g_term

                atom_descriptor.append(g_sum* np.power(2.0, 1.0-zeta_value))

        sample_descriptor.append(atom_descriptor)
    total_descriptor.append(sample_descriptor)

total_descriptor = np.asarray(total_descriptor)
print(total_descriptor)





