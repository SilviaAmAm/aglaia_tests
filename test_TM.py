import tensorflow as tf
import numpy as np
from aglaia import symm_funct


def test_coord():
    # xyzs = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5,  0.5, 0.7, 0.5, 0.5, 0.5, 0.8],
    #      [0.1, 0.0, 0.0, 0.9, 0.0, 0.0,  -0.5, -0.5, -0.5,  0.1, 0.5, 0.5, 0.6, 0.5, 0.5, ],
    #      [-0.1, 0.0, 0.0, 1.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.3, 1.5, 2.5, 0.5, ]]
    # Zs = [[1, 1, 6, 6, 7], [1, 1, 6, 6, 7], [1, 1, 6, 6, 7]]
    # elements = [1, 6, 7]
    xyzs = [[[1., 2., 3.], [4., 5., 6.], [1, 1, 1], [2, 2, 2]], [[2, 4, 6], [8, 10, 12], [1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4], [1, 1, 1], [2, 2, 2]]]
    Zs = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    elements = [1]

    return xyzs, Zs, elements


if __name__ == "__main__":
    xyzs_list = [[[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]]

    Zs_list = [[7, 1, 1, 1]]

    elements_list = [1, 7]
    element_pairs_list = [[1, 7], [1, 1]]
    element_triples_list = [[1,1,1], [1,1,7], [1,7,1]]

    Zs = tf.constant(Zs_list)
    xyzs = tf.constant(xyzs_list)
    elements = tf.constant(elements_list)
    element_pairs = tf.constant(element_pairs_list)

    radial_cutoff = tf.constant(500.0, dtype=tf.float32)
    angular_cutoff = tf.constant(500.0, dtype=tf.float32)
    radial_rs = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
    angular_rs = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
    theta_s = tf.constant([3.0, 2.0], dtype=tf.float32)
    zeta = tf.constant(3.0, dtype=tf.float32)
    eta = tf.constant(2.0, dtype=tf.float32)

    n_samples = Zs.get_shape().as_list()[0]

    # Calculating the distance matrix between the atoms of each sample
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.norm(dxyzs, axis=3)  # (n_samples, n_atoms, n_atoms)

    # Tells where there are atoms
    padding_mask = tf.not_equal(Zs, 0)  # shape (n_samples, n_atoms)
    expanded_padding_1 = tf.expand_dims(padding_mask, axis=1)  # (n_samples, 1, n_atoms)
    expanded_padding_2 = tf.expand_dims(padding_mask, axis=-1)  # (n_samples, n_atoms, 1)

    # Where there are distances under the cut-off
    under_cutoff = tf.less(dist_tensor, radial_cutoff)  # (n_samples, n_atoms, n_atoms)
    # If there is an atom AND the distance is < cut-off, then mask2 element is TRUE (done one dimension at a time)
    mask1 = tf.logical_and(under_cutoff, expanded_padding_1)  # (n_samples, n_atoms, n_atoms)
    mask2 = tf.logical_and(mask1, expanded_padding_2)  # (n_samples, n_atoms, n_atoms)

    # All the indices of the atoms-pairs that have distances inside the cut-off and where there are atom pairs
    pair_indices = tf.where(mask2) # (n_atoms*n_atoms, 3)

    # Removing diagonal elements
    identity_mask = tf.where(tf.not_equal(pair_indices[:, 1], pair_indices[:, 2]))
    clean_pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32) # (n_atoms*n_atoms - n_atoms, 3)

    # Modifying the shape of clean_pair_indices to use it to make triple indices
    mol_pair_indices = tf.dynamic_partition(clean_pair_indices, clean_pair_indices[:, 0], n_samples) # (n_samples, n_atoms*n_atoms - n_atoms, 3)

    # Now making the triple indices
    triple_idx = []
    for i in range(n_samples):
        mol_common_pair_indices = tf.where(tf.equal(tf.expand_dims(mol_pair_indices[i][:,1], axis=1),
                                                    tf.expand_dims(mol_pair_indices[i][:,1], axis=0))) #(3*(n_atoms*n_atoms - n_atoms), 2)
        mol_triples_indices = tf.concat([tf.gather(mol_pair_indices[i], mol_common_pair_indices[:, 0]),
                                         tf.gather(mol_pair_indices[i], mol_common_pair_indices[:, 1])[:, -1:]], axis=1)
        permutation_identity_pairs_mask = tf.where(tf.less(mol_triples_indices[:, 2], mol_triples_indices[:, 3]))
        mol_triples_indices = tf.squeeze(tf.gather(mol_triples_indices, permutation_identity_pairs_mask))
        triple_idx.append(mol_triples_indices)

    triples_indices = tf.concat(triple_idx, axis=0)

    # Getting the elements corresponding to the the various indices
    triples_elements = tf.gather_nd(Zs, triples_indices[:, 0:2])

    triples_element_pairs, _ = tf.nn.top_k(tf.stack([tf.gather_nd(Zs, triples_indices[:, 0:3:2]),
                                                     tf.gather_nd(Zs, triples_indices[:, 0:4:3])], axis=-1), k=2)
    sorted_triples_element_pairs = tf.reverse(triples_element_pairs, axis=[-1])


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    something_else = sess.run(b)
    print(something_else)



