import numpy as np

def distance(r1, r2):
    diff = r2-r1
    return np.linalg.norm(diff)

def fc(r_ij, r_c):
    if r_ij < r_c:
        f_c = 0.5 * (np.cos(np.pi * r_ij / r_c) + 1.0)
    else:
        f_c = 0.0
    return f_c

def get_theta(xyz_i, xyz_j, xyz_k):
    r_ij = xyz_j - xyz_i
    r_ik = xyz_k - xyz_i
    numerator = np.dot(r_ij, r_ik)
    denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_ik)
    costheta = numerator/denominator
    theta = np.arccos(costheta)
    return theta

# Data
xyzs = np.array([[ [0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]])
Zs = np.array([[18, 18, 18, 18]])
elements = np.array([18])

# Parameters
eta = 2.0
zeta = 3.0
theta_s = np.array([3.0, 2.0])
angular_rs = np.array([0.0, 0.1, 0.2])
# angular_rs = np.array([0.1])
angular_cutoff = 500.0

# Useful numbers
n_samples = xyzs.shape[0]
n_atoms = xyzs.shape[1]

total_descriptor = []
for sample in range(n_samples):
    sample_descriptor = []
    for i in range(n_atoms):  # Loop over main atom
        atom_descriptor = []
        for angular_rs_value in angular_rs:
            for theta_s_value in theta_s:
                g_sum = 0
                for j in range(n_atoms):                # Loop over 1st neighbour
                    if j == i:
                        continue
                    for k in range(j+1, n_atoms):            # Loop over 2nd neighbour
                        if k == i:
                            continue

                        r_ij = distance(xyzs[sample, i, :], xyzs[sample, j, :])
                        r_ik = distance(xyzs[sample, i, :], xyzs[sample, k, :])

                        theta_ijk = get_theta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])

                        term1 = np.power((1.0 + np.cos(theta_ijk - theta_s_value)), zeta)
                        exponent = - eta * np.power(0.5*(r_ij + r_ik) - angular_rs_value, 2.0)
                        term2 = np.exp(exponent)
                        term3 = fc(r_ij, angular_cutoff) * fc(r_ik, angular_cutoff)

                        g_term = term1 * term2 * term3
                        g_sum += g_term

                atom_descriptor.append(g_sum * np.power(2.0, 1.0 - zeta))
        sample_descriptor.append(atom_descriptor)
    total_descriptor.append(sample_descriptor)

total_descriptor = np.asarray(total_descriptor)
print(total_descriptor)