"""
This script explores a way of calculating the total energy with a neural network where atomic decomposition is done. The
idea is that there is a dictionary where the key is the atomic number of an element in the system. The value is the list
containing the weights of the neural network for that particular element. Then, one loops through all the elements in
the system. For each set of weight, one calculates the output of the neural network as if all of the atoms had the
atomic number that is currently used. Then, from the zs tensor one figures out which elements actually have that atomic
number. Since zs and the matrix of energies have the same shape, it is then easy to extract the values of the energies
that correspond to the atoms of correct element type. These are then put into the final atomic decomposed energy tensor.
The final step is to sum them within a molecule.
"""

import numpy as np
import tensorflow as tf
from aglaia import symm_funct


def init_weight(n1, n2):
    """
    Generate a tensor of weights of size (n1, n2)

    """

    w = tf.Variable(tf.truncated_normal([n1, n2], stddev=1.0 / np.sqrt(n2)))

    return w

def init_bias(n):
    """
    Generate a tensor of biases of size n.

    """

    b = tf.Variable(tf.zeros([n]))

    return b

def generate_weights(n_features, hidden_layer_sizes, n_out):
    """
    Generates the weights and the biases, by looking at the size of the hidden layers,
    the number of features in the descriptor and the number of outputs. The weights are initialised from
    a zero centered normal distribution with precision :math:`\\tau = a_{m}`, where :math:`a_{m}` is the number
    of incoming connections to a neuron. Weights larger than two standard deviations from the mean is
    redrawn.

    :param n_out: Number of outputs
    :type n_out: integer
    :return: tuple of weights and biases, each being of length (n_hidden_layers + 1)
    :rtype: tuple
    """

    weights = []
    biases = []

    # Weights from input layer to first hidden layer
    weights.append(init_weight(hidden_layer_sizes[0], n_features))
    biases.append(init_bias(hidden_layer_sizes[0]))

    # Weights from one hidden layer to the next
    for i in range(1, hidden_layer_sizes.size):
        weights.append(
            init_weight(hidden_layer_sizes[i],hidden_layer_sizes[i - 1]))
        biases.append(init_bias(hidden_layer_sizes[i]))

    # Weights from last hidden layer to output layer
    weights.append(init_weight(n_out, hidden_layer_sizes[-1]))
    biases.append(init_bias(n_out))

    return weights, biases

def model(x, hidden_layer_sizes, weights, biases):
    """
    Constructs the actual network.

    :param x: Input
    :type x: tf.placeholder of shape (None, n_features)
    :param weights: Weights used in the network.
    :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
    :param biases: Biases used in the network.
    :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
    :return: Output
    :rtype: tf.Variable of size (None, n_targets)
    """
    n_samples = x.get_shape().as_list()[0]

    # Calculate the activation of the first hidden layer
    expanded_weights = tf.tile(tf.expand_dims(tf.transpose(weights[0]), axis=0), multiples=[n_samples, 1, 1])
    z = tf.add(tf.matmul(x, expanded_weights), biases[0])
    h = tf.sigmoid(z)

    # Calculate the activation of the remaining hidden layers
    for i in range(hidden_layer_sizes.size - 1):
        expanded_weights = tf.tile(tf.expand_dims(tf.transpose(weights[i+1]), axis=0), multiples=[n_samples, 1, 1])
        z = tf.add(tf.matmul(h, expanded_weights), biases[i + 1])
        h = tf.sigmoid(z)

    # Calculating the output of the last layer
    expanded_weights = tf.tile(tf.expand_dims(tf.transpose(weights[-1]), axis=0), multiples=[n_samples, 1, 1])
    z = tf.add(tf.matmul(h, expanded_weights), biases[-1], name="output")

    z_squeezed = tf.squeeze(z, axis=[-1])

    return z_squeezed


# Test Data
input_data = "/Volumes/Transcend/repositories/Aglaia/aglaia/tests/data_test_acsf.npz"
data = np.load(input_data)

xyzs = data["arr_0"]
zs_np = data["arr_1"]
elements = data["arr_2"]
element_pairs = data["arr_3"]

acsf = symm_funct.generate_parkhill_acsf(xyzs, zs_np, elements, element_pairs)

n_atoms = acsf.get_shape().as_list()[1]

# Generating the weights
n_features = acsf.get_shape().as_list()[-1]
hidden_layers = np.array([1])

ele_weights = {}
ele_biases = {}

for i in range(elements.shape[0]):
    weights, biases = generate_weights(n_features, hidden_layers, 1)
    ele_weights[elements[i]] = weights
    ele_biases[elements[i]] = biases

# Multiplying things
zs = tf.constant(zs_np)
atomic_energies = tf.zeros(zs.get_shape())
zeros = tf.zeros(zs.get_shape())

for i in range(elements.shape[0]):
    # Calculating the output for every sample and atom
    all_energies = model(acsf, hidden_layers, ele_weights[elements[i]], ele_biases[elements[i]]) # (n_samples, n_atoms)

    # Figuring out which atomic energies correspond to the current element.
    current_element = tf.constant(elements[i], shape=zs.get_shape())
    where_element = tf.equal(zs, current_element) # (n_samples, n_atoms)

    # Extracting the energies corresponding to the right element
    element_energies = tf.where(where_element, all_energies, zeros)

    # Adding the energies of the current element to the final atomic energies tensor
    atomic_energies = tf.add(atomic_energies, element_energies)

total_energies = tf.reduce_sum(atomic_energies, axis=-1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
something = sess.run(total_energies)
print(something.shape)





