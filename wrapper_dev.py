"""
This script is used to try and figure out a way to get the descriptor to be stored in the class rather than being generated everytime.
"""

from aglaia import wrappers
import glob
import numpy as np


import qml

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/examples/qm7_hyperparam_search/qm7/*.xyz")[:100]
#
# x = wrappers.OSPMRMP(representation="slatm")
# y = np.array(range(len(filenames)), dtype=int)
#
# x.generate_compounds(filenames)
# x.set_properties(y)
#
# x.fit(y)


