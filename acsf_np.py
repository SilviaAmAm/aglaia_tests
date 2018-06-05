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

def get_costheta(xyz_i, xyz_j, xyz_k):
    r_ij = xyz_j - xyz_i
    r_ik = xyz_k - xyz_i
    numerator = np.dot(r_ij, r_ik)
    denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_ik)
    costheta = numerator/denominator
    return costheta

def acsf_rad(xyzs, Zs, elements, radial_cutoff, radial_rs, eta):
    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]
    n_rs = len(radial_rs)
    n_elements = len(elements)

    total_descriptor = []
    for sample in range(n_samples):
        sample_descriptor = []
        for main_atom in range(n_atoms):
            atom_descriptor = np.zeros((n_rs* n_elements,))
            for i, rs_value in enumerate(radial_rs):
                for neighb_atom in range(n_atoms):
                    if main_atom == neighb_atom:
                        continue
                    else:
                        r_ij = distance(xyzs[0, main_atom, :], xyzs[0, neighb_atom, :])
                        cut_off_term = fc(r_ij, radial_cutoff)
                        exponent_term = np.exp(-eta * (r_ij - rs_value) ** 2)
                        g2_term = exponent_term * cut_off_term
                        # Compare the current neighbouring atom to the list of possible neighbouring atoms and then
                        # split the terms accordingly
                        for j in range(len(elements)):
                            if Zs[sample][neighb_atom] == elements[j]:
                                atom_descriptor[i*n_rs + j] += g2_term

            sample_descriptor.append(atom_descriptor)
        total_descriptor.append(sample_descriptor)

    total_descriptor = np.asarray(total_descriptor)

    return total_descriptor

def acsf_ang(xyzs, Zs, element_pairs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]
    n_rs = len(angular_rs)
    n_theta = len(theta_s)
    n_elements_pairs = len(element_pairs)

    total_descriptor = []

    for sample in range(n_samples):
        sample_descriptor = []
        for i in range(n_atoms):  # Loop over main atom
            atom_descriptor = np.zeros((n_rs*n_theta*n_elements_pairs, ))
            counter = 0
            for rs_value in angular_rs:  # Loop over parameters
                for theta_value in theta_s:
                    for j in range(n_atoms):  # Loop over 1st neighbour
                        if j == i:
                            continue
                        for k in range(j+1, n_atoms):  # Loop over 2nd neighbour
                            if k == j or k == i:
                                continue

                            r_ij = distance(xyzs[sample, i, :], xyzs[sample, j, :])
                            r_ik = distance(xyzs[sample, i, :], xyzs[sample, k, :])
                            cos_theta_ijk = get_costheta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])
                            theta_ijk = np.arccos(cos_theta_ijk)

                            term_1 = np.power((1.0 + np.cos(theta_ijk - theta_value)), zeta)
                            term_2 = np.exp(- eta * np.power(0.5*(r_ij + r_ik) - rs_value, 2))
                            term_3 = fc(r_ij, angular_cutoff) * fc(r_ik, angular_cutoff)
                            g_term = term_1 * term_2 * term_3 * np.power(2.0, 1.0 - zeta)
                            # Compare the pair of neighbours to all the possible element pairs, then summ accordingly
                            current_pair = np.flip(np.sort([Zs[sample][j], Zs[sample][k]]), axis=0)     # Sorting the pair in descending order
                            for m, pair in enumerate(element_pairs):
                                if np.all(current_pair == pair):
                                    atom_descriptor[counter * n_elements_pairs + m] += g_term
                    counter += 1

            sample_descriptor.append(atom_descriptor)
        total_descriptor.append(sample_descriptor)

        return np.asarray(total_descriptor)



if __name__ == "__main__":
    xyzs = np.array([[[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]])

    Zs_list = [[7, 2, 1, 1], [7, 2, 1, 1]]

    elements_list = [1, 2, 7]
    element_pairs_list = [[1, 1], [2, 1], [7, 1], [7, 2]]

    radial_cutoff = 500.0
    angular_cutoff = 500.0
    radial_rs = [0.0, 0.1, 0.2]
    angular_rs = [0.0, 0.1, 0.2]
    theta_s = [3.0, 2.0]
    zeta = 3.0
    eta = 2.0

    rad_term = acsf_rad(xyzs, Zs_list, elements_list, radial_cutoff, radial_rs, eta)
    ang_term = acsf_ang(xyzs, Zs_list, element_pairs_list, angular_cutoff, angular_rs, theta_s, zeta, eta)
    print(ang_term)